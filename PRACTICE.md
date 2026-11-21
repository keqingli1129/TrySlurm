# Slurm + torchrun practice: from one process to multi-node DDP

A hands-on path through this lab, ordered so each module builds the mental model the next one
needs. Work top to bottom. Every step says **what to run**, **what to look for**, and **why it
matters** for multi-node GPU training.

Modules:

| # | Topic | Where |
|---|---|---|
| 0 | Get the lab ready | host |
| 1 | torchrun on its own (no Slurm) | host |
| 2 | Slurm basics with a GPU | host Slurm |
| 3 | Batch DDP training on the GPU | host Slurm |
| 4 | A real 2-node cluster | Docker |
| 5 | Launching ranks with srun instead of torchrun | Docker |
| 6 | Failure drills | Docker + host |
| 7 | Moving to real multi-node GPUs | planning |
| — | Self-check questions and a cheat sheet | — |

Keep one picture in mind the whole time. Learning multi-node training mostly means getting it to
fit in your head:

```
Slurm layer      sbatch → allocation (nodes, CPUs, GPUs) → batch script on first node
                   └─ srun: N tasks spread across the nodes
Launcher layer       └─ torchrun (1 per node): spawns nproc_per_node workers,
                        sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT,
                        all torchruns meet at one rendezvous endpoint
Training layer          └─ train.py: init_process_group(nccl|gloo) → DDP → all-reduce on backward()
```

Most multi-node bugs come from two layers disagreeing about a number (nodes, processes per
node, GPUs per node, ports, network interfaces).

---

## Module 0: Get the lab ready

```bash
cd ~/Documents/TrySlurm
sinfo -N -o "%N %P %T %c %m %G"   # expect your node, partition gpu, state idle, Gres gpu:1
./setup-python.sh                 # uv sync → ./.venv with cu128 torch, then runs check_gpu.py twice
```

Look for `sm_120` in the arch list and a `bf16 matmul OK` line, both from the direct run and from
the `srun` run. The second one proves Slurm can hand the GPU to a job.

**Small upgrade used throughout this guide.** The job scripts hardcode `--epochs` and so on. Let
them take extra arguments, so you can change a run from the `sbatch` command line. In each of
the three files in `jobs/`, add `"$@"` to the end of the `train.py ...` line, for example in
`jobs/gpu-1node.sbatch`:

```bash
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt" "$@"
```

Anything after the script name is passed through, and argparse uses the last value it sees:
`sbatch jobs/gpu-1node.sbatch --epochs 50`.

---

## Module 1: torchrun on its own (no Slurm)

Goal: understand exactly what torchrun does before Slurm wraps it.

### 1.1 What torchrun gives each worker

```bash
.venv/bin/torchrun --nproc_per_node=4 --no-python bash -c \
  'echo "RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE MASTER=$MASTER_ADDR:$MASTER_PORT"'
```

You get four lines with RANK 0–3. On a single node, LOCAL_RANK equals RANK. Remember these five
variables. Every launcher (torchrun, srun, mpirun) has to produce them, and `train.py` only
reads them through `init_process_group` and `LOCAL_RANK`.

### 1.2 Single-node DDP on CPU, then on GPU

```bash
mkdir -p logs checkpoints
.venv/bin/torchrun --nproc_per_node=4 train.py --cpu --epochs 2 --samples 40000 --hidden 512 --ckpt /tmp/p1.pt
.venv/bin/torchrun --nproc_per_node=1 train.py --epochs 2 --ckpt /tmp/p1-gpu.pt
```

Compare the first lines: `backend=gloo` against `backend=nccl`, and `global_batch` = per-process batch × world size.
Run the first command again and it prints `Resumed from /tmp/p1.pt ... already complete`. That is
the checkpoint logic. Delete `/tmp/p1*.pt` to start over.

### 1.3 Fake two nodes on one machine (static rendezvous)

Multi-node is the same thing with more torchruns, one per "node", pointed at the same master.
Open two terminals:

```bash
# terminal A ("node 0")
.venv/bin/torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29400 \
  train.py --cpu --epochs 2 --samples 40000 --hidden 512 --ckpt /tmp/p13.pt

# terminal B ("node 1"), the only difference is node_rank
.venv/bin/torchrun --nnodes=2 --node_rank=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29400 \
  train.py --cpu --epochs 2 --samples 40000 --hidden 512 --ckpt /tmp/p13.pt
```

Observe:
- Terminal A **waits** until B starts. Rendezvous blocks until all `nnodes` show up.
- World size is 4. Node 1 gets RANK 2–3 but LOCAL_RANK 0–1.
- In static mode you choose `node_rank` yourself, and two nodes with the same `node_rank` collide.

Why it matters: on a cluster the only thing that changes is that `127.0.0.1` becomes the first
node's hostname, and something has to start a torchrun on every node. That something is `srun`.

### 1.4 The same thing with c10d (dynamic) rendezvous, which the lab's jobs use

```bash
# run this in two terminals, word for word the same command in both
.venv/bin/torchrun --nnodes=2 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29401 --rdzv_id=practice \
  train.py --cpu --epochs 2 --samples 40000 --hidden 512 --ckpt /tmp/p14.pt
```

No `--node_rank`: c10d assigns node ranks for you. That's why the sbatch files can run *one
identical command* on every node through `srun`. `rdzv_id` must match across nodes, and the jobs
use `$SLURM_JOB_ID` for it so two jobs never join each other.

**Try it:** start just one of the two and wait. It hangs at rendezvous (and eventually times out).
Remember what this looks like. "Job hangs at startup" in multi-node almost always means a node
couldn't reach the rendezvous endpoint.

---

## Module 2: Slurm basics with a GPU

### 2.1 See the cluster

```bash
sinfo -N -l                         # node state, CPUs, memory
scontrol show node $(hostname -s)   # look at CfgTRES / AllocTRES and Gres=gpu:1
scontrol show partition gpu
```

`CfgTRES` is what Slurm can hand out (cpu, mem, gres/gpu). Every request you make is checked against it.

### 2.2 srun: run one thing inside an allocation

```bash
srun hostname
srun -n 4 bash -c 'echo task $SLURM_PROCID local $SLURM_LOCALID node $SLURM_NODEID'
srun --gpus=1 bash -c 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi -L'
srun bash -c 'echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}'
```

`SLURM_PROCID` / `SLURM_LOCALID` / `SLURM_NODEID` are Slurm's versions of RANK / LOCAL_RANK / node rank.
That's the key to Module 5. Without `--gpus`, the job isn't given a GPU. Note that with
`proctrack/linuxproc` and no cgroups, the device isn't actually *blocked*. Slurm only controls the env var.

### 2.3 Interactive GPU shell

```bash
srun --gpus=1 --cpus-per-task=4 --mem=8G --pty bash
#   inside:  nvidia-smi; echo $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES; .venv/bin/python check_gpu.py; exit
```

While it's open, run `squeue` in another terminal. Your job is `R` and holds the GPU.

### 2.4 Requests Slurm refuses

```bash
srun --gpus=2 nvidia-smi -L                 # more GPUs than exist → rejected right away
srun --cpus-per-task=999 hostname           # more CPUs than the node has
srun --mem=999G hostname
```

Read the error text. "Requested node configuration is not available" means the job can *never*
run. That's different from *pending*, which means it can't run *yet*.

### 2.5 The queue with one GPU (pending and its reason)

Each job writes a checkpoint named after itself. If you submit the same job three times, jobs 2
and 3 resume from job 1's *finished* checkpoint and exit at once. So give them separate names:

```bash
for i in 1 2 3; do sbatch --job-name=q$i jobs/gpu-1node.sbatch --epochs 30; done
squeue -o "%.8i %.10j %.4T %.12r %.10M %b"
```

One is `RUNNING`. The others are `PENDING` with reason `Resources` (or `Priority`). Then:

```bash
scontrol show job <a pending id> | grep -E 'Reason|TRES'
scancel --name=q3              # cancel by name
scontrol hold <id>; scontrol release <id>
scancel -u $USER               # cancel everything when done
rm -f checkpoints/q*.pt
```

---

## Module 3: Batch DDP training on the GPU

### 3.1 Read the job script line by line

Open `jobs/gpu-1node.sbatch` and match each line to the picture at the top:

| Line | Layer | Must agree with |
|---|---|---|
| `--nodes=1`, `--ntasks-per-node=1` | Slurm | one torchrun per node |
| `--gpus-per-node=1` | Slurm | `GPUS_PER_NODE=1` → `--nproc_per_node` |
| `head_node=$(scontrol show hostnames ... \| head -n 1)` | glue | becomes the rendezvous host |
| `port=$((29500 + SLURM_JOB_ID % 1000))` | glue | different per job, so jobs sharing a node don't clash |
| `--rdzv_id=$SLURM_JOB_ID` | launcher | identical on every node |
| `--ckpt checkpoints/$SLURM_JOB_NAME.pt` | training | shared storage in multi-node |
| `--requeue` | Slurm | train.py resumes from the checkpoint |

### 3.2 Run it and watch NCCL start up

```bash
NCCL_DEBUG=INFO sbatch jobs/gpu-1node.sbatch     # sbatch passes your environment into the job
squeue
tail -f logs/ddp-gpu-<jobid>.out
```

Even with one rank, `NCCL INFO` lines show the NCCL version, the chosen device and the init. On
multi-node you will read those same lines to see which **network interface / transport** NCCL picked.

After it finishes:

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,AllocTRES%40,ExitCode
```

The `.batch` step is your script, and the step with no suffix (`<jobid>.0`) is the `srun` line.
Each `srun` inside a job is a separate **job step**.

### 3.3 Requeue and resume

```bash
rm -f checkpoints/ddp-gpu.pt
sbatch jobs/gpu-1node.sbatch --epochs 60
# wait for a few "epoch" lines, then:
scontrol requeue <jobid>
tail -f logs/ddp-gpu-<jobid>.out      # same job id; output restarts with "Resumed from ..."
```

Why it matters: on shared clusters jobs get preempted and hit time limits. Writing checkpoints
atomically (`.tmp`, then `os.replace`) plus `--requeue` is how long training survives that.

### 3.4 Break the GPU/process match on purpose

```bash
srun --gpus=1 .venv/bin/torchrun --nproc_per_node=2 train.py --epochs 1 --ckpt /tmp/bad.pt
```

Rank 1 calls `torch.cuda.set_device(1)` and fails with *invalid device ordinal*. When you change
GPUs per node, update **all three** places together: `--gpus-per-node`, `GPUS_PER_NODE`, and
the physical GPU count.

---

## Module 4: A real 2-node cluster (Docker)

### 4.1 Bring it up

```bash
cd docker-cluster
docker compose up -d --build
docker compose ps
docker compose exec slurmctld sinfo -N      # c1, c2 idle
docker compose exec -u ubuntu slurmctld bash
```

Everything below (until Module 6) runs **inside** that shell, in `/lab` (this folder, mounted).

### 4.2 Explore how tasks spread across nodes

```bash
cd /lab && mkdir -p logs checkpoints
srun -N2 hostname
srun -N2 -n4 bash -c 'echo "$(hostname) PROCID=$SLURM_PROCID LOCALID=$SLURM_LOCALID NODEID=$SLURM_NODEID"'
srun -N2 --ntasks-per-node=1 bash -c 'echo $(hostname) $SLURM_JOB_NODELIST'
scontrol show hostnames c[1-2]
```

Notice how `--ntasks-per-node=1` gives exactly one task per node, which is how the job scripts
start one torchrun on each node.

### 4.3 Run the 2-node job

```bash
sbatch jobs/multinode-cpu.sbatch
squeue
tail -f logs/ddp-cpu-2node-*.out
```

Check:
- 4 `[rank r/4]` lines, two with `host=c1` and two with `host=c2`.
- `head=c1:<port>`: the batch script runs on the first node, and that node hosts the rendezvous.
- `global_batch=1024` = 256 × 4.

### 4.4 Scaling experiment

Make `PROCS_PER_NODE` overridable in `jobs/multinode-cpu.sbatch`:

```bash
PROCS_PER_NODE=${PROCS_PER_NODE:-2}
export OMP_NUM_THREADS=$(( 4 / PROCS_PER_NODE ))
```

Then run the same training with different layouts. Give each a different `--job-name`, or it
resumes from the last run's checkpoint:

```bash
sbatch --job-name=s-1x1 --nodes=1 --export=ALL,PROCS_PER_NODE=1 jobs/multinode-cpu.sbatch
sbatch --job-name=s-1x2 --nodes=1 --export=ALL,PROCS_PER_NODE=2 jobs/multinode-cpu.sbatch
sbatch --job-name=s-2x1 --nodes=2 --export=ALL,PROCS_PER_NODE=1 jobs/multinode-cpu.sbatch
sbatch --job-name=s-2x2 --nodes=2 --export=ALL,PROCS_PER_NODE=2 jobs/multinode-cpu.sbatch
squeue                      # each node has 4 CPUs, so some of these wait their turn
grep -H "epoch   4" logs/s-*.out
```

Compare samples/s and final loss. Things to think about:
- Global batch changes with world size, so the loss curves are not apples-to-apples. A real
  workflow keeps the global batch fixed (shrink the per-process batch) or scales the learning rate.
- On real GPUs, cross-node all-reduce over Ethernet is the bottleneck. Inside Docker all "nodes"
  share one machine's memory bus, so the numbers here don't reflect real network cost.

Clean up: `rm -f checkpoints/s-*.pt`.

---

## Module 5: Launching ranks with srun instead of torchrun

Many clusters skip torchrun: `srun` starts *every* rank itself, and the script maps Slurm's
variables to PyTorch's. Doing this once is the best way to understand what torchrun was doing for you.

Create `jobs/multinode-cpu-srun.sbatch`:

```bash
#!/bin/bash
# 2 nodes x 2 ranks, every rank started directly by srun (no torchrun).
#SBATCH --job-name=ddp-cpu-srun
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2          # one task PER RANK now, not per node
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
export GLOO_SOCKET_IFNAME=eth0 OMP_NUM_THREADS=2
echo "job $SLURM_JOB_ID  ntasks=$SLURM_NTASKS  master=$MASTER_ADDR:$MASTER_PORT"

# Single quotes: the SLURM_* variables must be expanded inside each task, not once here.
srun --kill-on-bad-exit=1 bash -c '
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
  exec /opt/venv/bin/python train.py --cpu --epochs 3 --samples 60000 --hidden 512 \
       --batch-size 256 --ckpt "checkpoints/$SLURM_JOB_NAME.pt"'
```

```bash
sbatch jobs/multinode-cpu-srun.sbatch && tail -f logs/ddp-cpu-srun-*.out
```

It works without changing `train.py`, because `init_process_group` defaults to `env://` and reads
exactly these variables.

| | torchrun launcher | srun per rank |
|---|---|---|
| `--ntasks-per-node` | 1 | = GPUs per node |
| Who sets RANK etc. | torchrun | your script, from `SLURM_*` |
| Rendezvous | c10d at `head:port` | `MASTER_ADDR:MASTER_PORT` TCP store |
| Worker restarts (`--max-restarts`) | yes | no, Slurm requeue only |
| On GPU nodes | torchrun spawns 1 worker per GPU | add `--gpus-per-task=1` or bind with `LOCAL_RANK` |

**Try it:** change the env line to `WORLD_SIZE=3` and submit. The job hangs until timeout,
because the store waits for a rank 3 that will never come. That is the same mismatch failure as in Module 3, at another layer.

---

## Module 6: Failure drills

Do these deliberately so you recognise them later. Run the Docker ones inside the login shell
unless the step says to use the host.

### 6.1 Wrong network interface (the most common real-world multi-node failure)

The script hardcodes `export GLOO_SOCKET_IFNAME=eth0`, so first make it overridable in
`jobs/multinode-cpu.sbatch`:

```bash
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
```

```bash
GLOO_SOCKET_IFNAME=eth9 sbatch --job-name=f-if jobs/multinode-cpu.sbatch
```

Expect an immediate error about the device. Then try `GLOO_SOCKET_IFNAME=lo`: each rank advertises 127.0.0.1, which the
other node can't reach, so expect a hang or connection error. On GPU clusters the same failure
comes from `NCCL_SOCKET_IFNAME` (or NCCL choosing a docker/virtual bridge NIC).

### 6.2 A node dies in the middle of a job

```bash
sbatch --job-name=f-node jobs/multinode-cpu.sbatch --epochs 40
# once epochs are printing, on the HOST in docker-cluster/:
docker compose stop c2
```

Watch from the login shell: `tail -f logs/f-node-*.out`, `squeue`, `sinfo -N`. The ranks on c1
lose their peers and the collective fails. `--kill-on-bad-exit=1` makes srun take down the other
tasks instead of leaving them hanging. The node goes `down*` after slurmctld stops hearing from
it. Then:

```bash
docker compose start c2                                      # host
docker compose exec slurmctld scontrol update nodename=c2 state=resume   # host, as root, if it stays down
sacct -j <jobid> --format=JobID,State,ExitCode,NodeList      # was it requeued, or did it fail?
sbatch --job-name=f-node jobs/multinode-cpu.sbatch --epochs 40   # resumes from the checkpoint
```

Lesson: in DDP one dead rank kills the whole job, so checkpoint frequency equals how much work you lose.

### 6.3 Time limit, then resume

```bash
sbatch --job-name=f-time --time=00:01:00 jobs/multinode-cpu.sbatch --epochs 200
sacct -j <jobid> --format=JobID,State,Elapsed     # State=TIMEOUT
sbatch --job-name=f-time jobs/multinode-cpu.sbatch --epochs 200   # picks up where it stopped
```

Note: `--requeue` does **not** resubmit after TIMEOUT, only after preemption/node failure/manual
requeue. Long runs on real clusters chain jobs (`--dependency=afterany:<id>`) or use `--signal`
to checkpoint just before the limit.

### 6.4 Draining a node

Changing node state needs root, so those two commands run on the host in `docker-cluster/`:

```bash
docker compose exec slurmctld scontrol update nodename=c2 state=drain reason="practice"   # host
sbatch --job-name=f-drain jobs/multinode-cpu.sbatch          # login shell: PENDING, look at the reason
sinfo -R                                                     # login shell
docker compose exec slurmctld scontrol update nodename=c2 state=resume                    # host
```

### 6.5 Clean up

```bash
scancel -u ubuntu; rm -f checkpoints/f-*.pt     # login shell
docker compose down                             # host, in docker-cluster/
```

---

## Module 7: Moving to real multi-node GPUs

### 7.1 What changes from the Docker job

Compare the two scripts: `diff jobs/multinode-cpu.sbatch jobs/multinode-gpu.sbatch`. The real differences are:

| Docker CPU job | Multi-node GPU job | Why |
|---|---|---|
| no GPU request | `--partition=gpu`, `--gpus-per-node=N` | GRES allocation |
| `PROCS_PER_NODE=2` | `GPUS_PER_NODE=N` | one worker per GPU |
| `--cpu` → gloo | (no flag) → NCCL | GPU collectives |
| `GLOO_SOCKET_IFNAME=eth0` | `NCCL_SOCKET_IFNAME=<NIC>` (if needed) | pick the inter-node NIC |
| `/opt/venv` in an image | `.venv/` from `uv.lock` (shared on NFS or `uv sync --frozen` per node) | identical packages on every node |
| `/lab` bind mount | real NFS export | shared code/checkpoints/logs |

Everything else (srun, torchrun, rendezvous, checkpointing) is identical. That's the point of
Modules 4–6.

### 7.2 Setup checklist for a second machine (README Part 3)

1. One `slurm.conf` on every node listing all nodes, e.g. `NodeName=box1,box2 ... Gres=gpu:1`. Change `SlurmctldHost` to the controller. Each node also needs a `gres.conf`.
2. Copy `/etc/munge/munge.key` from the controller to every node (owner `munge`, mode `400`), then restart munge. Test with `munge -n | ssh box2 unmunge`.
3. Same usernames **and UIDs** on every machine.
4. NFS: export this folder from one machine, mount it at the **same path** on all.
5. Network: hostnames resolve everywhere, ports 6817–6819 open, plus the rendezvous/NCCL ports (on a LAN, allow all traffic between nodes).
6. `sinfo -N` shows both nodes idle, then `srun -N2 --gpus-per-node=1 nvidia-smi -L` prints both GPUs.
7. `NCCL_DEBUG=INFO sbatch jobs/multinode-gpu.sbatch`, and check ranks on both hosts with `backend=nccl`.

### 7.3 NCCL variables worth knowing

| Variable | Use |
|---|---|
| `NCCL_DEBUG=INFO` | prints the transport (`NET/Socket`, `NET/IB`, `P2P`) and NIC NCCL chose |
| `NCCL_SOCKET_IFNAME=enp5s0` | force the NIC (prefix match; `^docker,lo` excludes) |
| `NCCL_IB_DISABLE=1` | force TCP sockets when there's no working InfiniBand |
| `NCCL_P2P_DISABLE=1` | debug intra-node GPU-to-GPU issues |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | fail fast instead of hanging forever on a dead peer |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | catches mismatched collectives / unused parameters |

### 7.4 Cloud option without buying hardware

The same sbatch runs on Azure CycleCloud Workspace for Slurm, AWS ParallelCluster/PCS, and Google Cluster Toolkit.
A cheap first run is 2 nodes × 1 small GPU for an hour. The goal is to see `NET/Socket` or `NET/IB`
in the NCCL log and two hostnames in the rank lines.

---

## Self-check questions

Answer these from memory. If you can't, redo the module in brackets.

1. In the torchrun jobs, why is `--ntasks-per-node=1` and not the GPU count? [3.1, 5]
2. What are RANK, LOCAL_RANK, WORLD_SIZE for rank 5 in a 2-node × 4-GPU job? [1.3]
3. Why does the batch script pick the first host from `SLURM_JOB_NODELIST` as the rendezvous host? [4.3]
4. What happens if two jobs on the same node use the same rendezvous port? How do the scripts avoid it? [3.1]
5. A job hangs right after start, with no error. List three causes. [1.4, 5, 6.1]
6. Why must the checkpoint be on shared storage? Why does only rank 0 write it, followed by a barrier? [3.3, train.py]
7. What's the difference between a job that is PENDING (Resources) and one that is rejected? [2.4, 2.5]
8. Your 2-node GPU job is slower than 1 node. What do you check first? [4.4, 7.3]
9. How would you convert `multinode-gpu.sbatch` to srun-per-rank launching? [5]
10. What does Slurm *not* enforce in this lab's config, and what would a production cluster use? (README: `proctrack/cgroup`, `task/cgroup`, `ConstrainDevices=yes`) [2.2]

## Cheat sheet

```bash
# Inspect
sinfo -N -l | sinfo -R | scontrol show node <n> | scontrol show partition
squeue -o "%.8i %.12j %.4T %.12r %.10M %D %R" | scontrol show job <id>
sacct -j <id> --format=JobID,JobName,State,Elapsed,AllocTRES%40,ExitCode,NodeList

# Submit / control
sbatch [--job-name=X --nodes=N --time=HH:MM:SS --export=ALL,VAR=v] script.sbatch [train args]
srun [-N nodes] [-n tasks] [--gpus=G] [--pty bash]
scancel <id> | scancel --name=X | scancel -u $USER
scontrol requeue|hold|release <id>
scontrol update nodename=<n> state=drain reason="..." | state=resume

# Slurm → PyTorch variables
SLURM_PROCID→RANK  SLURM_LOCALID→LOCAL_RANK  SLURM_NTASKS→WORLD_SIZE  SLURM_NODEID→node rank
first host of $(scontrol show hostnames "$SLURM_JOB_NODELIST") → MASTER_ADDR / rdzv host
```

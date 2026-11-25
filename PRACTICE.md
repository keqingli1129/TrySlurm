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
| 8 | Containers on Slurm: Enroot + Pyxis | host Slurm |
| 9 | Containers on Slurm: Apptainer | host Slurm |
| 10 | Containers without Slurm: Kubernetes (k3s) | host |
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

## Modules 8–10: running jobs in containers

So far every GPU job ran `.venv/bin/torchrun` from the shared folder, and Module 7 kept every node
identical with `uv.lock`. Production GPU clusters usually go further and run every job inside a
**container image**. The image holds the whole user-space stack (OS libraries, CUDA runtime,
cuDNN, NCCL, Python, torch), so every node runs the same bits no matter what is installed on it.

```
┌──────────── container image (same on every node) ─────────────┐
│ torch 2.11.0, NCCL, cuDNN, CUDA 12.8 runtime, Python 3.12     │
└───────────────────────────────────────────────────────────────┘
          ▲ libcuda.so + /dev/nvidia* injected when the container starts
┌──────────── host (each node) ─────────────────────────────────┐
│ NVIDIA kernel driver, slurmd, munge, network drivers          │
└───────────────────────────────────────────────────────────────┘
```

The driver is a kernel module, so it **can't** live in the image. A hook mounts the host's driver
into the container at start. Rule: the host driver must be new enough for the CUDA version in the
image (your 615 driver runs CUDA 12.8 images fine).

All three modules use **one image**, built once from `containers/Dockerfile`. It starts from
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, the same torch that `uv.lock` pins. You will
run that image with three different tool sets:

| Module | Tool set | Who uses it | How a job starts |
|---|---|---|---|
| 8 | Enroot + Pyxis | NVIDIA DGX/SuperPOD, many AI labs | `srun --container-image=...` (Slurm plugin) |
| 9 | Apptainer | university and national HPC centers | `srun apptainer exec --nv image.sif ...` |
| 10 | Kubernetes | cloud and ML-platform teams | `kubectl apply -f job.yaml` (no Slurm) |

Files used: `containers/Dockerfile`, `containers/train.def`, `jobs/gpu-1node-pyxis.sbatch`,
`jobs/gpu-1node-apptainer.sbatch`, `k8s/gpu-job.yaml`, `k8s/ddp-cpu-2pod.yaml`. Images go in
`images/` (git-ignored). On a real cluster that folder would be on the shared filesystem, so
every node reads the same image file.

Disk: about 15 GB across the three modules (the base image is 4.3 GB compressed).

---

## Module 8: Containers on Slurm with Enroot + Pyxis

**Enroot** is NVIDIA's container runtime. It turns a Docker image into a plain directory and runs
it as *your* user, without a root daemon. **Pyxis** is a Slurm plugin (SPANK) that adds
`--container-*` options to `srun`, so Slurm starts each task inside an Enroot container.

### 8.1 One-time host prep (also needed by Modules 9 and 10)

**a) NVIDIA Container Toolkit.** It's the hook that injects the driver into containers. Its apt
repo is already configured on this machine, but the package isn't installed:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L     # a plain Ubuntu image sees the 5070
```

The last line is the whole idea in one command: `ubuntu:24.04` has no NVIDIA software in it, but
`nvidia-smi` works, because the toolkit mounted the host's driver in.

**b) Let unprivileged containers use user namespaces.** Ubuntu 24.04 blocks this by default
through AppArmor, and Enroot (and Apptainer's unprivileged builds) need it:

```bash
sysctl kernel.apparmor_restrict_unprivileged_userns      # 1 = restricted (Ubuntu 24.04 default)
echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-lab-userns.conf
sudo sysctl --system
```

This loosens a hardening setting, which is fine for a lab box. Module 10's clean-up section
shows how to undo it. HPC sites handle this with an AppArmor profile per tool or a
setuid install instead.

**c) Build the lab image** (from the repo root; the first build downloads about 4.3 GB):

```bash
cd ~/Documents/TrySlurm
docker build -t slurm-lab-train:v1 -f containers/Dockerfile .
docker run --rm --gpus all -v "$PWD":/lab -w /lab slurm-lab-train:v1 python check_gpu.py
mkdir -p images logs checkpoints
```

Look for `sm_120` and `bf16 matmul OK`, as in Module 0, but this time from inside the container.
Read `containers/Dockerfile` while it builds: it has just a base image, one `pip install`, and a
copy of `train.py` for Kubernetes.

### 8.2 Install Enroot

```bash
cd /tmp
arch=$(dpkg --print-architecture)
ver=4.2.1                     # latest release: https://github.com/NVIDIA/enroot/releases
curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v$ver/enroot_${ver}-1_${arch}.deb"
curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v$ver/enroot+caps_${ver}-1_${arch}.deb"
sudo apt install -y ./enroot_${ver}-1_${arch}.deb ./enroot+caps_${ver}-1_${arch}.deb
enroot version
cd ~/Documents/TrySlurm
```

`enroot+caps` gives the image-import helpers the file capabilities they need, so you never run
Enroot as root.

### 8.3 Turn the Docker image into one file

```bash
enroot import -o images/train.sqsh dockerd://slurm-lab-train:v1
ls -lh images/
```

`dockerd://` reads from your local Docker. Without it, Enroot pulls from a registry, e.g.
`enroot import docker://pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`. The result is a single
compressed **squashfs** file. That's the format sites keep on shared storage. Copying one file is
far cheaper than having 500 nodes each pull 4 GB from a registry at the same time.

### 8.4 Run it by hand (no Slurm yet)

```bash
enroot create --name train images/train.sqsh        # unpack to ~/.local/share/enroot/train
enroot list
enroot start --mount "$PWD":/lab train bash -c \
  'whoami; cat /etc/os-release | head -2; nvidia-smi -L; python /lab/check_gpu.py'
enroot remove -f train
```

Things to notice:
- `whoami` prints **your user**, not root. No daemon, no privilege. That's why HPC admins accept it.
- `/etc/os-release` shows the *image's* Ubuntu, not your 24.04. The GPU still works, through the
  hook from 8.1a.

### 8.5 Install Pyxis (the Slurm plugin)

Pyxis must be compiled against the **exact** Slurm version that runs it (23.11.4 here), so build
it from source against the distro's Slurm headers:

```bash
sudo apt install -y libslurm-dev build-essential git      # libslurm-dev provides slurm/spank.h
git clone --depth 1 --branch v0.24.0 https://github.com/NVIDIA/pyxis.git ~/src/pyxis
cd ~/src/pyxis && make && sudo make install && cd ~/Documents/TrySlurm

# Tell Slurm to load SPANK plugins from plugstack.conf.d/, then register Pyxis there
echo 'include /etc/slurm/plugstack.conf.d/*' | sudo tee /etc/slurm/plugstack.conf
sudo ln -sf /usr/local/share/pyxis/pyxis.conf /etc/slurm/plugstack.conf.d/pyxis.conf
sudo systemctl restart slurmctld slurmd

srun --help | grep -A1 container-image     # new [pyxis] options in srun itself
```

If `srun` prints `Incompatible plugin version`, Pyxis was built against different Slurm headers
than the running Slurm. Check `dpkg -l libslurm-dev slurm-wlm` show the same version.

### 8.6 First containerized srun

```bash
srun --gpus=1 --container-image="$PWD/images/train.sqsh" \
     --container-mounts="$PWD":/lab --container-workdir=/lab \
     python check_gpu.py

srun --container-image="$PWD/images/train.sqsh" python -c 'import torch; print(torch.__version__)'
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

Both print torch 2.11.0. One comes from the image and one from `.venv`, which proves the image
matches what you've been using. Each `srun` unpacks the image again, so the first run takes a few
seconds longer.

### 8.7 The batch job

```bash
diff jobs/gpu-1node.sbatch jobs/gpu-1node-pyxis.sbatch
sbatch jobs/gpu-1node-pyxis.sbatch
tail -f logs/ddp-pyxis-<jobid>.out
ls -l checkpoints/ddp-pyxis.pt
```

The diff shows the point of this module: the `#SBATCH` lines, rendezvous and `train.py`
arguments are unchanged. Only the srun line gained three `--container-*` options and points at
the image's `torchrun`. Check:
- `backend=nccl` with the 5070, the same as Module 3.
- `checkpoints/ddp-pyxis.pt` is owned by **you**, because the container ran as your UID.
- The image provides the libraries, and the bind-mounted repo provides the code, logs and
  checkpoints. Edit `train.py` and resubmit: no rebuild needed.

### 8.8 Try it

- **Pull from a registry instead of a file:** `IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime sbatch --job-name=pyxis-pull jobs/gpu-1node-pyxis.sbatch --epochs 2`.
  It works because `train.py` needs only torch. The download goes into `~/.cache/enroot` and
  takes minutes. Picture 100 nodes doing that at once, and you have the reason sites pre-stage `.sqsh` files.
- **Keep the container between steps:** add `--container-name=train` to two `srun` lines in the
  same job. The second one reuses the unpacked container instead of unpacking again.
- **Requeue still works:** repeat Module 3.3 with `jobs/gpu-1node-pyxis.sbatch`. The checkpoint
  is on the bind mount, so the new container resumes it.

### 8.9 What multi-node adds

Nothing new in the script: set `--nodes=2` and keep `images/train.sqsh` on NFS. Pyxis containers
share the **host's network**, so NCCL sees the real NICs and `NCCL_SOCKET_IFNAME` works the same
as in Module 7.3. For InfiniBand, the image must also contain the RDMA user-space libraries.
NVIDIA's NGC images (`nvcr.io#nvidia/pytorch:<yy.mm>-py3`) include them, which is why they are
the usual choice on GPU clusters.

---

## Module 9: Containers on Slurm with Apptainer

**Apptainer** (formerly Singularity) is the HPC-center standard. An image is a single read-only
`.sif` file, and a container runs as your user with your home folder mounted. It isn't a Slurm
plugin. `srun` simply runs the `apptainer` command, so it works on any cluster where it's installed.

### 9.1 Install

Needs 8.1a and 8.1b done first.

```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer
apptainer --version
```

### 9.2 Convert the same Docker image to a .sif

```bash
apptainer build images/train.sif docker-daemon://slurm-lab-train:v1
ls -lh images/
```

Now you have one environment in two formats: `train.sqsh` for Enroot and `train.sif` for Apptainer.

### 9.3 Run it by hand

```bash
apptainer exec images/train.sif nvidia-smi -L            # fails: no GPU without --nv
apptainer exec --nv images/train.sif nvidia-smi -L       # works
apptainer exec --nv images/train.sif python check_gpu.py # no --bind, yet check_gpu.py is found
apptainer exec images/train.sif bash -c 'whoami; pwd; touch /usr/x'
```

Things to notice:
- `--nv` is Apptainer's version of the 8.1a hook. It binds the host's NVIDIA libraries in.
- `check_gpu.py` was found without a `--bind`, because Apptainer mounts `$HOME` and the current
  folder by default. Docker isolates by default, while Apptainer integrates by default. That's
  why HPC users like it.
- `touch /usr/x` fails with a read-only file system. The image can't drift between runs.

### 9.4 Build from a definition file (the HPC way)

HPC centers often don't have Docker at all. They build from a `.def` recipe instead. Read
`containers/train.def`, then:

```bash
apptainer build images/train-def.sif containers/train.def   # pulls the base from Docker Hub (~4 GB)
apptainer inspect --deffile images/train-def.sif             # the recipe is stored in the image
apptainer run --nv images/train-def.sif --standalone --nproc_per_node=1 \
  train.py --epochs 1 --ckpt /tmp/def.pt                     # %runscript = torchrun
```

The `%test` section runs at the end of the build, and the build fails if torch isn't a CUDA 12.8
build. If the build errors about fakeroot or permissions, use `sudo apptainer build ...`.

### 9.5 The batch job

```bash
diff jobs/gpu-1node.sbatch jobs/gpu-1node-apptainer.sbatch
sbatch jobs/gpu-1node-apptainer.sbatch
tail -f logs/ddp-apptainer-<jobid>.out
```

Same check as 8.7: `backend=nccl`, and the checkpoint is owned by you.

### 9.6 Pyxis or Apptainer?

| | Enroot + Pyxis | Apptainer |
|---|---|---|
| Image file | `.sqsh` (or pull at job start) | `.sif` |
| Slurm integration | plugin: `srun --container-image` | none needed: `srun apptainer exec` |
| GPU | automatic (image's `NVIDIA_VISIBLE_DEVICES`) | `--nv` flag |
| Default mounts | nothing; you list `--container-mounts` | `$HOME`, cwd, `/tmp` |
| Writable image | yes (`--container-writable`) | no (read-only) |
| Builds images | no, imports Docker images | yes, from `.def` files |
| Typical home | NVIDIA DGX / AI clusters | university / national HPC |

Both run as your user, with no daemon. That's the property that makes containers acceptable on
a shared Slurm cluster, and the reason plain Docker isn't used there.

---

## Module 10: Containers without Slurm: Kubernetes (k3s)

Cloud and platform teams often skip Slurm and run training on **Kubernetes**. The same image
runs, but the scheduler, the launch and the storage all work differently. **k3s** is a
single-binary Kubernetes that fits on one machine.

> The GPU is not shared between Slurm and Kubernetes. Neither knows about the other, so don't run
> Slurm GPU jobs while a k8s GPU job is running.

### 10.1 Map what you know

| Slurm | Kubernetes |
|---|---|
| `sbatch job.sbatch` | `kubectl apply -f job.yaml` (a `Job`) |
| `squeue` / `scontrol show job` | `kubectl get pods` / `kubectl describe pod` |
| `tail -f logs/...out` | `kubectl logs -f ...` |
| `--gpus-per-node=1` | `resources.limits: nvidia.com/gpu: 1` |
| GRES + `gres.conf` | NVIDIA device plugin (a DaemonSet) |
| `--nodes=2` + `srun` | Indexed Job, `completions: 2` |
| `SLURM_NODEID` | `JOB_COMPLETION_INDEX` |
| first host of `SLURM_JOB_NODELIST` | stable DNS name from a headless Service |
| shared NFS folder | PersistentVolumeClaim (here: a `hostPath`) |
| `--requeue` | `backoffLimit` / Job re-creates the pod |

### 10.2 Install k3s with the NVIDIA runtime

Needs 8.1a done first. k3s finds `nvidia-container-runtime` at startup, and
`--default-runtime=nvidia` makes every pod use it:

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--default-runtime=nvidia --write-kubeconfig-mode=644" sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml        # add to ~/.bashrc while you use this module
kubectl get nodes                                   # one node, Ready
sudo grep -c nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml   # > 0
```

If the grep prints `0`, the toolkit was installed after k3s. Run `sudo systemctl restart k3s`.

### 10.3 Teach Kubernetes about the GPU

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.20.2/deployments/static/nvidia-device-plugin.yml
kubectl -n kube-system get pods -l name=nvidia-device-plugin-ds    # Running
kubectl describe node | grep -A7 Allocatable                       # nvidia.com/gpu: 1
```

This is the Kubernetes counterpart of `GresTypes=gpu` + `gres.conf`. Until the device plugin
reports `nvidia.com/gpu: 1`, no pod requesting a GPU can be scheduled.

### 10.4 Load the image into k3s

k3s has its own containerd and doesn't see Docker's images. There's no registry in this lab, so
copy the image in:

```bash
docker save slurm-lab-train:v1 | sudo k3s ctr images import -
sudo k3s crictl images | grep slurm-lab
```

A real cluster pushes to a registry instead, and every node pulls from it.

### 10.5 Single-GPU training Job

Read `k8s/gpu-job.yaml`, then:

```bash
kubectl apply -f k8s/gpu-job.yaml
kubectl get pods -w                       # ContainerCreating → Running → Completed (Ctrl-C)
kubectl logs -f job/ddp-gpu
```

Look for `backend=nccl`. Note what's different from Slurm:
- The code is **in the image** (`/app/train.py`). No shared folder is mounted, so changing
  `train.py` means rebuilding and re-importing the image.
- `--standalone` replaces the rendezvous endpoint because there's only one node.
- The memory **limit** is enforced. Kubernetes always uses cgroups, unlike this lab's Slurm (Module 2.2).

**Requeue, Kubernetes style.** Delete the checkpoint, start again, and kill the pod mid-run:

```bash
kubectl delete job ddp-gpu; sudo rm -f /tmp/k8s-ckpt/ddp-gpu.pt
kubectl apply -f k8s/gpu-job.yaml
# once epochs are printing:
kubectl delete pod -l job-name=ddp-gpu
kubectl get pods -w                       # the Job controller creates a replacement pod
kubectl logs -f job/ddp-gpu               # "Resumed from /ckpt/ddp-gpu.pt ..."
```

**Pending, Kubernetes style.** Submit a second GPU job while the first runs:

```bash
sed 's/name: ddp-gpu$/name: ddp-gpu2/' k8s/gpu-job.yaml | kubectl apply -f -
kubectl get pods                          # ddp-gpu2-... is Pending
kubectl describe pod -l job-name=ddp-gpu2 | tail -3   # "Insufficient nvidia.com/gpu" = PD (Resources)
kubectl delete job ddp-gpu2
```

### 10.6 Two "nodes": DDP across two pods

Kubernetes has no `srun`, so `k8s/ddp-cpu-2pod.yaml` does srun's jobs by hand. It starts two
numbered pods (an Indexed Job) and gives pod 0 a DNS name (`ddp-cpu-0.ddp`) for the master
address. Read it, then:

```bash
kubectl apply -f k8s/ddp-cpu-2pod.yaml
kubectl get pods -o wide -l job-name=ddp-cpu
kubectl logs -f -l job-name=ddp-cpu --prefix --max-log-requests 2
```

Check:
- 4 `[rank r/4]` lines, two with `host=ddp-cpu-0` and two with `host=ddp-cpu-1`. That's the
  same layout as Module 4.3, with pods in place of `c1`/`c2`.
- It uses **static** rendezvous (Module 1.3): `--node_rank=$JOB_COMPLETION_INDEX`. In Slurm,
  c10d plus `srun` handled the numbering for you.
- Pod 1 may log a few `cannot be retrieved (gai error...)` warnings at first. Pod 0's DNS record
  didn't exist yet, and torchrun retries until it does.

Clean up: `kubectl delete -f k8s/ddp-cpu-2pod.yaml`.

**Try it: a GPU leak.** Run a pod that does *not* request a GPU:

```bash
kubectl run gpu-peek --rm -it --restart=Never --image=slurm-lab-train:v1 \
  --image-pull-policy=Never -- nvidia-smi -L
```

It will most likely list the 5070. The image sets `NVIDIA_VISIBLE_DEVICES=all`, and with
`--default-runtime=nvidia` the runtime honours it even when no GPU was requested. This is
Module 2.2's "scheduled, not enforced" lesson again. Production clusters close the gap with the
NVIDIA GPU Operator and the device plugin's `volume-mounts`/CDI device-list strategies.

### 10.7 What production adds on top

You did the launch wiring by hand on purpose, the same reason Module 5 skipped torchrun.
Real teams use:
- **Kubeflow Trainer** (`TrainJob`, earlier `PyTorchJob`): you give it the image and the replica
  count, and it creates the pods and Service and sets `MASTER_ADDR`/`RANK`/`WORLD_SIZE` for you.
- **Kueue or Volcano** for **gang scheduling**: start all pods of a job or none. Slurm does this
  by default. Plain Kubernetes doesn't, so a 2-pod job can end up half-running on a busy cluster
  while it waits forever for its other half.
- **NVIDIA GPU Operator**: installs the driver hook, the device plugin and monitoring on every GPU node.
- **PersistentVolumeClaims** on shared storage for checkpoints and datasets, instead of `hostPath`.

### 10.8 Clean up everything from Modules 8–10

```bash
kubectl delete -f k8s/ --ignore-not-found
/usr/local/bin/k3s-uninstall.sh                                  # removes k3s completely
sudo rm -rf /tmp/k8s-ckpt
rm -rf images/ ~/.cache/enroot ~/.local/share/enroot             # image files and Enroot caches
rm -f checkpoints/ddp-pyxis.pt checkpoints/ddp-apptainer.pt checkpoints/pyxis-pull.pt
docker image rm slurm-lab-train:v1
# optional: restore Ubuntu's user-namespace restriction from 8.1b
sudo rm /etc/sysctl.d/60-lab-userns.conf && sudo sysctl kernel.apparmor_restrict_unprivileged_userns=1
```

Pyxis can stay installed. It only acts when you pass `--container-*`. To remove it:
`sudo rm /etc/slurm/plugstack.conf.d/pyxis.conf && sudo systemctl restart slurmd slurmctld`.

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
11. Which part of the GPU stack can't go in a container image, and what rule links it to the image's CUDA version? [8.1]
12. Why do Slurm sites use Enroot/Pyxis or Apptainer instead of Docker? Why do they keep images as `.sqsh`/`.sif` files on shared storage? [8.3, 9.6]
13. In the Kubernetes 2-pod job, what replaced `srun`, `SLURM_NODEID` and the "first host of the node list"? What does Slurm give you that plain Kubernetes doesn't? [10.6, 10.7]

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

# Containers (Modules 8–10)
docker build -t slurm-lab-train:v1 -f containers/Dockerfile .
enroot import -o images/train.sqsh dockerd://slurm-lab-train:v1
srun --gpus=1 --container-image=$PWD/images/train.sqsh --container-mounts=$PWD:/lab --container-workdir=/lab <cmd>
apptainer build images/train.sif docker-daemon://slurm-lab-train:v1
srun --gpus=1 apptainer exec --nv images/train.sif <cmd>
docker save slurm-lab-train:v1 | sudo k3s ctr images import -
kubectl apply -f k8s/gpu-job.yaml | kubectl get pods | kubectl logs -f job/ddp-gpu | kubectl delete -f <file>
```

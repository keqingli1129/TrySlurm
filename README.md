# Slurm lab: one Linux box, one RTX 5070

This kit sets up Slurm on your machine with the GPU managed by Slurm. It also gives you a
2-node Slurm cluster in Docker, so you can practice the real multi-node workflow on one computer.

| Part | What it gives you | Where it runs |
|---|---|---|
| 1 | Slurm controller + compute node on your box, GPU as a GRES, DDP training on the 5070 | Your machine (systemd services) |
| 2 | A 2-node cluster (`c1`, `c2`) + controller in containers, real multi-node DDP over the network | Docker on the same machine |
| 3 | Checklist for real multi-node GPU training once you have a 2nd GPU machine or cloud nodes | Later |

**The one honest limitation:** with one GPU you can't do multi-node *GPU* training. Part 2 runs on CPU
(gloo backend instead of NCCL). Everything else is identical to a real GPU cluster: the job script,
`srun` + `torchrun`, rendezvous, gradient all-reduce across nodes, and checkpoint/resume. Moving to
GPUs is a backend switch that `train.py` already handles.

## Files

```
setup-host-slurm.sh        Part 1: installs and configures Slurm + munge (run with sudo)
pyproject.toml, uv.lock   Python deps managed by uv (torch from the cu128 index, pinned in uv.lock)
setup-python.sh            Part 1: `uv sync` (.venv with a Blackwell-capable PyTorch), then a GPU test
check_gpu.py               verifies torch can run kernels on the 5070
train.py                   DDP training script (synthetic data, checkpoint/resume) used by every job
jobs/gpu-1node.sbatch      Part 1 job: train on the 5070
jobs/multinode-cpu.sbatch  Part 2 job: 2 nodes x 2 workers on the Docker cluster
jobs/multinode-gpu.sbatch  Part 3 template: N nodes x M GPUs
docker-cluster/            Part 2: Dockerfile, compose file, slurm.conf, entrypoint
```

---

## Before you start

- **OS:** Ubuntu 22.04 / 24.04 or Debian 12. The scripts use `apt`. On Fedora or Rocky,
  install `slurm slurm-slurmctld slurm-slurmd munge` from EPEL/Fedora and copy the config the script
  would generate.
- **NVIDIA driver 570 or newer, open kernel modules.** RTX 50-series (Blackwell) cards only work
  with the open modules. Check with `nvidia-smi`. If it fails: `sudo ubuntu-drivers install`, then reboot.
- **For Part 2:** Docker Engine with the compose plugin (`docker compose version`), and your user
  in the `docker` group.

- **[uv](https://docs.astral.sh/uv/)** for the Python environment (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

You do **not** need the CUDA toolkit. The PyTorch wheel brings its own CUDA runtime.

---

## Part 1: Slurm + the RTX 5070

```bash
tar xzf slurm-lab.tar.gz && cd slurm-lab
sudo ./setup-host-slurm.sh      # installs Slurm, writes /etc/slurm/{slurm,gres}.conf, runs a GPU test job
./setup-python.sh               # as your normal user: uv sync (.venv + PyTorch cu128) + GPU checks
sbatch jobs/gpu-1node.sbatch    # submit training
```

The setup script ends by running `srun --gpus=1 nvidia-smi -L`. If that prints your RTX 5070,
Slurm is allocating the GPU correctly.

Watch the job:

```bash
squeue                              # queue: R = running, PD = pending (with a reason)
tail -f logs/ddp-gpu-<jobid>.out    # training output
scontrol show job <jobid>           # everything Slurm knows about the job
sacct -j <jobid>                    # after it finishes (basic accounting)
scancel <jobid>                     # kill it
```

Expected output, roughly:

```
[rank 0/1] host=yourbox local_rank=0 device=NVIDIA GeForce RTX 5070 backend=nccl
world_size=1  params=8.7M  global_batch=1024  epochs 0..9
epoch   0  loss 1.0423  acc 0.612   ...  samples/s
...
done
```

### Things worth trying (and good interview talking points)

- **Pending jobs and backfill.** Submit the job 3 times in a row. One runs, the others sit in `PD` with
  reason `Resources`, because there is only one GPU. `squeue -o "%i %j %T %r"` shows the reason.
- **Checkpoint/resume.** While a job is running: `scontrol requeue <jobid>`. It goes back to the
  queue, restarts, and prints `Resumed from checkpoints/ddp-gpu.pt`. To start fresh, delete
  `checkpoints/ddp-gpu.pt`.
- **Interactive GPU shell.** `srun --gpus=1 --cpus-per-task=4 --pty bash`, then run `nvidia-smi`
  and `echo $CUDA_VISIBLE_DEVICES`. Exit to release the GPU.
- **Resource requests.** Try `--gpus=2` and see it rejected, because the node only has one.
  Try `--mem=` and `--cpus-per-task=` values above what the node has.
- **Node state.** `sinfo -N -l`, `scontrol show node`, then
  `sudo scontrol update nodename=$(hostname -s) state=drain reason=maintenance` followed by `state=resume`.

### What the setup script configured

- `munge`: shared-secret authentication between Slurm daemons.
- `slurmctld` (scheduler) and `slurmd` (node agent), both on this machine and enabled at boot.
- `slurm.conf`: one node with the CPU/memory layout `slurmd -C` detected (minus 4 GB for the OS),
  `Gres=gpu:1`, partition `gpu`, `select/cons_tres` so GPUs, cores and memory are scheduled separately.
- `gres.conf`: maps the GPU to `/dev/nvidia0`. Slurm sets `CUDA_VISIBLE_DEVICES` for each job.

It uses `proctrack/linuxproc` rather than cgroups. That keeps setup simple, but Slurm won't
*enforce* CPU/memory/GPU limits; it only schedules them. Production clusters use
`proctrack/cgroup` + `task/cgroup` with `ConstrainDevices=yes`. That's worth knowing for interviews.

---

## Part 2: a 2-node cluster in Docker (multi-node practice)

```bash
cd docker-cluster
docker compose up -d --build                 # first build downloads ~1 GB (Ubuntu + Slurm + CPU torch)
docker compose exec slurmctld sinfo -N       # c1 and c2 should be "idle"
docker compose exec -u ubuntu slurmctld bash # a shell on the "login node"
```

Inside that shell:

```bash
cd /lab
mkdir -p logs checkpoints
sbatch jobs/multinode-cpu.sbatch
squeue
tail -f logs/ddp-cpu-2node-*.out
```

The proof that it's really multi-node is the first lines of the log. Four ranks on two hosts
talk through one rendezvous:

```
[rank 0/4] host=c1 local_rank=0 device=cpu backend=gloo
[rank 1/4] host=c1 local_rank=1 device=cpu backend=gloo
[rank 2/4] host=c2 local_rank=0 device=cpu backend=gloo
[rank 3/4] host=c2 local_rank=1 device=cpu backend=gloo
```

(The order varies.) Every `loss.backward()` all-reduces gradients between `c1` and `c2` over the
container network.

**How the pieces fit**, the same on any cluster:

```
sbatch ──► slurmctld allocates c1,c2
            └─ batch script runs on c1
                └─ srun starts 1 task per node ──► torchrun on c1    torchrun on c2
                                                    ├ rank 0           ├ rank 2
                                                    └ rank 1           └ rank 3
                   all torchruns meet at the c10d rendezvous on c1:<port>,
                   then DDP syncs gradients with gloo (CPU) / NCCL (GPU)
```

Notes:
- `/lab` inside the containers is this whole folder, standing in for a shared NFS mount. That's why
  logs and checkpoints show up on your machine.
- Jobs run as the image's `ubuntu` user (UID 1000) so files in your folder stay owned by you. If your
  own UID isn't 1000 (`id -u`), use `docker compose exec slurmctld bash` (root) instead.
- Want more nodes? Add `c3` to `docker-compose.yml`, change `NodeName=c[1-3]` and the partition in
  `docker-cluster/slurm.conf`, run `docker compose up -d --build`, and submit with `--nodes=3`.
- Stop the cluster with `docker compose down` (add `-v` and `docker image rm slurm-lab` to remove everything).

This cluster is separate from the Slurm in Part 1. Its ports aren't published, so the two don't interfere.

---

## Part 3: real multi-node GPU training (when you have a 2nd machine)

`jobs/multinode-gpu.sbatch` is ready. What has to be true across the machines:

1. **Same Slurm config everywhere.** One machine runs `slurmctld`. Every machine has the same
   `slurm.conf` listing all nodes (e.g. `NodeName=box1,box2 ... Gres=gpu:1`) and runs `slurmd`.
2. **Same munge key.** Copy `/etc/munge/munge.key` from the controller to every node (owner `munge`,
   mode `400`), then restart `munge`.
3. **Same users with the same UIDs** on every machine.
4. **A shared filesystem.** Export the lab folder over NFS and mount it at the same path everywhere,
   so the code, logs and checkpoints are identical on all nodes. For Python, either share `.venv` too
   (its interpreter is a uv-managed Python under `~/.local/share/uv/python`, so install the same one on
   every node with `uv python install 3.12`), or run `uv sync --frozen` on each node. `uv.lock` pins the
   exact torch build, so every node gets identical packages either way. In job scripts call
   `.venv/bin/torchrun` directly rather than `uv run`, so the nodes don't all try to sync the same `.venv` at once.
5. **Network.** Hostnames resolve on every node (`/etc/hosts` is fine). Open ports 6817–6819 for
   Slurm, plus the torchrun/NCCL ports. On a LAN, the simplest option is to allow all traffic
   between the nodes. If NCCL picks the wrong NIC, set `NCCL_SOCKET_IFNAME`.
6. Submit: `sbatch jobs/multinode-gpu.sbatch`. The log should show ranks on both hosts with `backend=nccl`.

Consumer cards over Ethernet will work for learning, but gradient sync over 1–10 GbE is slow
compared to NVLink/InfiniBand on real clusters. That's a good thing to be able to explain.
In the cloud, the same job script runs as-is on Azure CycleCloud Workspace for Slurm,
AWS ParallelCluster/PCS, or Google Cluster Toolkit.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sinfo` shows the node `drain` or `down` | `sinfo -R` shows the reason. Usually a memory or CPU mismatch: lower `RealMemory` in `/etc/slurm/slurm.conf`, `sudo systemctl restart slurmctld slurmd`, then `sudo scontrol update nodename=$(hostname -s) state=resume` |
| Services won't start | `sudo journalctl -u slurmctld -u slurmd -n 50`, `/var/log/slurm/*.log`, or run in the foreground: `sudo systemctl stop slurmd && sudo slurmd -Dvvv` |
| `Invalid generic resource (gres) specification` | `GresTypes=gpu` missing, or `gres.conf` doesn't match the node name. Re-run `sudo ./setup-host-slurm.sh` |
| `no kernel image is available for execution on the device` | Your torch build doesn't include sm_120. Re-run `./setup-python.sh` (it uses the cu128 index), or change the index URL in `pyproject.toml` to a newer one from pytorch.org and run `uv lock && uv sync` |
| Job stuck in `PD` with `Resources` | Something else holds the GPU. `squeue` to see what. |
| Job stuck in `PD` with `ReqNodeNotAvail` | Node is drained. See the first row. |
| Docker: `c1`/`c2` show `down*` | `docker compose logs c1`. If slurmd complains about cgroups, add `privileged: true` and `cgroup: host` under `x-node` in `docker-compose.yml`, then `docker compose up -d` |
| Docker job hangs at startup | The rendezvous can't connect. Check `GLOO_SOCKET_IFNAME` matches `docker compose exec c1 ip -br addr` (normally `eth0`) |
| `Permission denied` writing `logs/` from Docker | Your UID isn't 1000. Use the root shell (see Part 2 notes) |

To remove the host Slurm install: `sudo systemctl disable --now slurmd slurmctld munge && sudo apt remove slurm-wlm munge`.

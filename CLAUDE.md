# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Slurm learning lab for one Linux box with an RTX 5070 (Blackwell, sm_120). It is shell scripts, Slurm configs, sbatch jobs and a single PyTorch DDP script. There is no build system, test suite or linter. README.md is the user-facing guide (it calls this folder `slurm-lab`).

Three "parts" share the same `train.py`:
1. **Host Slurm**: `setup-host-slurm.sh` (sudo) installs munge + slurmctld + slurmd on this machine and *generates* `/etc/slurm/slurm.conf` and `gres.conf` from `slurmd -C` / `nvidia-smi`. Those host configs live only in `/etc/slurm` and are not in the repo, so edit the script's heredocs, not a repo file. Python deps are managed by **uv**: `pyproject.toml` pulls torch from the cu128 index (`explicit = true`, so only torch uses it) and `uv.lock` pins it. `setup-python.sh` runs `uv sync --frozen` into `./.venv`, then GPU checks.
2. **Docker cluster** (`docker-cluster/`): a controller and nodes `c1`, `c2`, all built from one image (Ubuntu 24.04, Slurm, CPU-only torch in `/opt/venv`). The munge key is baked into the image, so all nodes share it. `slurm.conf` hardcodes `CPUs=4 RealMemory=2000` with `SlurmdParameters=config_overrides`. To add a node, edit both `docker-compose.yml` and `NodeName=`/`PartitionName=` in `slurm.conf`.
3. **Multi-node GPU** template (`jobs/multinode-gpu.sbatch`) for when a second GPU machine exists.

## Commands

```bash
# Host (Part 1)
uv sync                             # .venv from uv.lock; `uv add <pkg>` to add deps, edit pyproject.toml + `uv lock` to change the torch index
sudo ./setup-host-slurm.sh          # re-runnable; backs up existing /etc/slurm configs with a timestamp
./setup-python.sh                   # as normal user; ends with check_gpu.py run directly and via srun
sbatch jobs/gpu-1node.sbatch        # submit from the repo root
squeue; sinfo -R; scontrol show job <id>; tail -f logs/<job-name>-<id>.out

# Docker cluster (Part 2)
cd docker-cluster && docker compose up -d --build
docker compose exec slurmctld sinfo -N
docker compose exec -u ubuntu slurmctld bash    # then: cd /lab && mkdir -p logs checkpoints && sbatch jobs/multinode-cpu.sbatch
docker compose down

# Quick local check of train.py without Slurm (CPU, small)
.venv/bin/torchrun --nproc_per_node=2 train.py --cpu --epochs 1 --samples 10000 --hidden 256 --ckpt /tmp/t.pt
.venv/bin/python check_gpu.py        # confirms torch can run sm_120 kernels
```

## Conventions that span files

- **sbatch scripts assume cwd = repo root.** They `cd "$SLURM_SUBMIT_DIR"` and use relative `.venv/`, `train.py`, `logs/`, `checkpoints/`. `logs/` must exist before submitting, because Slurm won't create the `--output` directory.
- **Launch pattern** (keep it the same in every job): `srun` starts one task per node (`--ntasks-per-node=1`), and each task runs `torchrun` with c10d rendezvous at `<first node>:$((29500 + SLURM_JOB_ID % 1000))`. `--nproc_per_node` must match the hardcoded `GPUS_PER_NODE`/`PROCS_PER_NODE` variable *and* the `#SBATCH --gpus-per-node` line.
- **Interpreter path differs by environment**: host jobs call `.venv/bin/torchrun` directly, not `uv run`, so multiple nodes don't race to sync a shared `.venv`. The Docker job uses `/opt/venv/bin/torchrun` and needs `--cpu` plus `GLOO_SOCKET_IFNAME=eth0`.
- **The repo root is mounted at `/lab`** in every container and stands in for NFS. Containers run jobs as `ubuntu` (UID 1000) so files on the host keep the right owner.
- **Checkpoint/resume**: jobs pass `--ckpt checkpoints/$SLURM_JOB_NAME.pt` and use `#SBATCH --requeue`. `train.py` auto-resumes when that file exists. A leftover checkpoint from a finished run makes the next job exit immediately ("already complete"), so delete it to start over.
- **`train.py` must run unchanged on 1 GPU, N GPUs × M nodes, and CPU/gloo.** The backend is NCCL when CUDA is available and gloo with `--cpu`. Only rank 0 prints metrics and saves, and a `dist.barrier()` follows each save. Checkpoints store the unwrapped `model.module.state_dict()`, loaded *before* the DDP wrap. Saves are atomic (write `.tmp`, then `os.replace`). The data is synthetic from a fixed seed, so nothing is downloaded.
- Both Slurm configs use `proctrack/linuxproc` (no cgroups). Resources are scheduled but not enforced. The host uses `CR_Core_Memory` with GPU GRES; Docker uses `CR_CPU`.
- Blackwell needs a CUDA 12.8+ torch build with open-kernel NVIDIA driver 570+. "no kernel image is available" means the torch build is wrong.

#!/bin/bash
# gpu-1node.sbatch, but the job runs inside a container started by Pyxis + Enroot (PRACTICE.md Module 8).
# Submit from the repo root after building images/train.sqsh:  sbatch jobs/gpu-1node-pyxis.sbatch
# Only the srun line changes: libraries come from the image, code/logs/checkpoints from the repo.
#SBATCH --job-name=ddp-pyxis         # names the log and checkpoint files
#SBATCH --partition=gpu              # queue to submit to (defined in /etc/slurm/slurm.conf)
#SBATCH --nodes=1                    # number of machines in the allocation
#SBATCH --ntasks-per-node=1          # one container (and one torchrun) per node
#SBATCH --gpus-per-node=1            # torchrun starts one worker per GPU
#SBATCH --cpus-per-task=4            # CPU cores for that one task (shared by its workers)
#SBATCH --time=00:30:00              # wall-clock limit; Slurm kills the job after 30 min
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job ID; logs/ must already exist
#SBATCH --requeue                    # if requeued, train.py resumes from its checkpoint
set -euo pipefail                    # stop on any error, unset variable, or failed pipe stage
cd "$SLURM_SUBMIT_DIR"               # run from the folder sbatch was called in (the repo root)

GPUS_PER_NODE=1                      # must match --gpus-per-node above
# The image file on shared storage, so every node starts the identical environment.
# Override to pull from a registry instead, e.g. IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
IMAGE=${IMAGE:-$PWD/images/train.sqsh}
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node

echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port  image=$IMAGE"

# Pyxis adds the --container-* options to srun. Each task starts inside the image:
#   --container-image     .sqsh file (or registry image) to run; Enroot unpacks it on each node
#   --container-mounts    repo -> /lab, the same "shared filesystem" idea as the Docker cluster
#   --container-workdir   start in /lab so the relative train.py/checkpoints/ paths work
#   /usr/local/bin/torchrun  the image's torch, not the host's .venv
srun --container-image="$IMAGE" \
     --container-mounts="$PWD":/lab \
     --container-workdir=/lab \
  /usr/local/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt" "$@"

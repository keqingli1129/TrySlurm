#!/bin/bash
# gpu-1node.sbatch, but the job runs inside an Apptainer image (PRACTICE.md Module 9).
# Submit from the repo root after building images/train.sif:  sbatch jobs/gpu-1node-apptainer.sbatch
# Unlike Pyxis, Apptainer is not a Slurm plugin: srun starts the apptainer command like any program.
#SBATCH --job-name=ddp-apptainer     # names the log and checkpoint files
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
IMAGE=${IMAGE:-images/train.sif}     # one read-only file on shared storage, used by every node
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node

echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port  image=$IMAGE"

# Each srun task runs `apptainer exec` on its node:
#   --nv      bind the host's NVIDIA driver libraries and /dev/nvidia* into the container
#   --bind    repo -> /lab (Apptainer also mounts $HOME and the cwd by default; this makes /lab explicit)
#   --pwd     start in /lab so the relative train.py/checkpoints/ paths work
#   /usr/local/bin/torchrun  the image's torch, not the host's .venv
srun apptainer exec --nv \
     --bind "$PWD":/lab \
     --pwd /lab \
     "$IMAGE" \
  /usr/local/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt" "$@"

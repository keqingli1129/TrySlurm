#!/bin/bash
# DDP training on the RTX 5070. Submit from the slurm-lab folder:  sbatch jobs/gpu-1node.sbatch
#SBATCH --job-name=ddp-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=1            # torchrun starts one worker per GPU
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue                    # if requeued, train.py resumes from its checkpoint
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

GPUS_PER_NODE=1
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node

echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun .venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

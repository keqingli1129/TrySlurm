#!/bin/bash
# Real multi-node GPU training. Use this once you have 2+ GPU machines in the same Slurm cluster
# (see "Part 3" in README.md). Nothing here changes except the node and GPU counts.
#SBATCH --job-name=ddp-gpu-multinode
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=1            # set to the GPUs in each machine
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"               # must be a shared folder (NFS) visible at the same path on every node

GPUS_PER_NODE=1
# Uncomment and set to the NIC that connects the machines (see `ip -br addr`) if NCCL picks the wrong one:
# export NCCL_SOCKET_IFNAME=enp5s0
export NCCL_DEBUG=WARN                # set to INFO to see which transport/NIC NCCL chose

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))
echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port"

srun --kill-on-bad-exit=1 venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

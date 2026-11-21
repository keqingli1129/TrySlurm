#!/bin/bash
# 2-node DDP training on the Docker cluster (CPU + gloo). Submit inside the slurmctld container:
#   cd /lab && sbatch jobs/multinode-cpu.sbatch
#SBATCH --job-name=ddp-cpu-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PROCS_PER_NODE=2                     # 2 workers x 2 nodes = world size 4
export OMP_NUM_THREADS=2             # 4 CPUs per node / 2 workers
export GLOO_SOCKET_IFNAME=eth0       # the container's network interface

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))
echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port"

srun --kill-on-bad-exit=1 /opt/venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$PROCS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --cpu --epochs 5 --samples 60000 --hidden 512 --batch-size 256 \
           --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

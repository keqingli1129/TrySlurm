#!/bin/bash
# DDP training on the RTX 5070. Submit from the slurm-lab folder:  sbatch jobs/gpu-1node.sbatch
# sbatch reads the #SBATCH lines below as options until it reaches the first real command.
#SBATCH --job-name=ddp-gpu           # name shown in squeue; also sets $SLURM_JOB_NAME and %x
#SBATCH --partition=gpu              # queue to submit to (defined in /etc/slurm/slurm.conf)
#SBATCH --nodes=1                    # number of machines in the allocation
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=1            # torchrun starts one worker per GPU
#SBATCH --cpus-per-task=4            # CPU cores for that one task (shared by its workers)
#SBATCH --time=00:30:00              # wall-clock limit; Slurm kills the job after 30 min
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job ID; logs/ must already exist
#SBATCH --requeue                    # if requeued, train.py resumes from its checkpoint
set -euo pipefail                    # stop on any error, unset variable, or failed pipe stage
cd "$SLURM_SUBMIT_DIR"               # run from the folder sbatch was called in (the repo root)

GPUS_PER_NODE=1                      # must match --gpus-per-node above
# Expand the compact node list (e.g. "node[1-2]") and take the first name as the rendezvous host.
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node

# One line at the top of the log so you can see where the job ran and which GPU it was given.
echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# srun starts one task per node (from --ntasks-per-node); each task runs torchrun.
#   --nnodes           total machines; torchrun waits until this many have joined
#   --nproc_per_node   worker processes per machine, one per GPU
#   --rdzv_id          shared ID so all nodes of this job find each other (job ID is unique)
#   --rdzv_backend     c10d = PyTorch's built-in TCP rendezvous, no etcd needed
#   --rdzv_endpoint    host:port everyone connects to for rendezvous
#   train.py args      10 epochs; checkpoint named after the job so a requeue resumes it
srun .venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

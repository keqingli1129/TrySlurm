#!/bin/bash
# Real multi-node GPU training. Use this once you have 2+ GPU machines in the same Slurm cluster
# (see "Part 3" in README.md). Compared with gpu-1node.sbatch: --nodes=2, more CPUs and time,
# NCCL network settings, and srun --kill-on-bad-exit. The torchrun launch itself is identical.
#SBATCH --job-name=ddp-gpu-multinode # own name, so its log and checkpoint don't collide with ddp-gpu
#SBATCH --partition=gpu              # queue to submit to (defined in /etc/slurm/slurm.conf)
#SBATCH --nodes=2                    # number of machines; becomes $SLURM_NNODES -> torchrun --nnodes
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --gpus-per-node=1            # set to the GPUs in each machine (and update GPUS_PER_NODE below)
#SBATCH --cpus-per-task=8            # CPU cores per node for its one task (shared by its workers)
#SBATCH --time=01:00:00              # wall-clock limit; extra headroom for cross-node rendezvous/comm
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job ID; logs/ must already exist
#SBATCH --requeue                    # if requeued, train.py resumes from its checkpoint
set -euo pipefail                    # stop on any error, unset variable, or failed pipe stage
cd "$SLURM_SUBMIT_DIR"               # must be a shared folder (NFS) visible at the same path on every node

GPUS_PER_NODE=1                      # must match --gpus-per-node above
# Across machines NCCL has to pick a network interface (on one node it uses shared memory/PCIe instead).
# Uncomment and set to the NIC that connects the machines (see `ip -br addr`) if NCCL picks the wrong one:
# export NCCL_SOCKET_IFNAME=enp5s0
export NCCL_DEBUG=WARN                # set to INFO to see which transport/NIC NCCL chose

# Expand the compact node list (e.g. "node[1-2]") and take the first name as the rendezvous host.
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node
# This runs only on the first node, so it shows the node list but not each node's GPU assignment.
echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port"

# srun starts one task per node (from --ntasks-per-node); each task runs torchrun.
#   --kill-on-bad-exit=1  if one node's torchrun fails, kill the others instead of letting them
#                         hang in an NCCL collective until --time runs out
#   --nnodes              total machines; torchrun waits until this many have joined
#   --nproc_per_node      worker processes per machine, one per GPU
#   --rdzv_id             shared ID so all nodes of this job find each other (job ID is unique)
#   --rdzv_backend        c10d = PyTorch's built-in TCP rendezvous, no etcd needed
#   --rdzv_endpoint       host:port everyone connects to for rendezvous
#   train.py args         10 epochs; checkpoint named after the job so a requeue resumes it
srun --kill-on-bad-exit=1 .venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --epochs 10 --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

#!/bin/bash
# 2-node DDP training on the Docker cluster (CPU + gloo). Submit inside the slurmctld container:
#   cd /lab && sbatch jobs/multinode-cpu.sbatch
# Compared with multinode-gpu.sbatch: no GPUs, so train.py gets --cpu (gloo instead of NCCL),
# 2 worker processes per node instead of 1 per GPU, and torchrun comes from the image's /opt/venv.
#SBATCH --job-name=ddp-cpu-2node     # names the log and checkpoint files
#SBATCH --nodes=2                    # number of containers (c1, c2); becomes $SLURM_NNODES -> torchrun --nnodes
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --cpus-per-task=4            # all 4 CPUs each node has (CPUs=4 in docker-cluster/slurm.conf)
#SBATCH --time=00:20:00              # wall-clock limit; Slurm kills the job after this
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job ID; logs/ must already exist
#SBATCH --requeue                    # if requeued, train.py resumes from its checkpoint
set -euo pipefail                    # stop on any error, unset variable, or failed pipe stage
cd "$SLURM_SUBMIT_DIR"               # /lab, the repo mounted in every container (stands in for NFS)

PROCS_PER_NODE=2                     # 2 workers x 2 nodes = world size 4
export OMP_NUM_THREADS=2             # 4 CPUs per node / 2 workers, so workers don't oversubscribe cores
export GLOO_SOCKET_IFNAME=eth0       # the container's network interface; gloo uses it between nodes

# Expand the compact node list (e.g. "c[1-2]") and take the first name as the rendezvous host.
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$(( 29500 + SLURM_JOB_ID % 1000 ))   # avoids clashes if two jobs share a node
# This runs only on the first node (the batch host), before srun fans out.
echo "job $SLURM_JOB_ID  nodes=$SLURM_JOB_NODELIST  head=$head_node:$port"

# srun starts one task per node (from --ntasks-per-node); each task runs torchrun.
#   --kill-on-bad-exit=1  if one node's torchrun fails, kill the others instead of letting them
#                         hang in a gloo collective until --time runs out
#   /opt/venv/bin/torchrun  the CPU-only torch baked into the image (not the host's .venv)
#   --nnodes              total machines; torchrun waits until this many have joined
#   --nproc_per_node      worker processes per machine (CPU workers, so any count works)
#   --rdzv_id             shared ID so all nodes of this job find each other (job ID is unique)
#   --rdzv_backend        c10d = PyTorch's built-in TCP rendezvous, no etcd needed
#   --rdzv_endpoint       host:port everyone connects to for rendezvous
#   train.py args         --cpu selects gloo; smaller run sized for CPUs; checkpoint named
#                         after the job so a requeue resumes it
srun --kill-on-bad-exit=1 /opt/venv/bin/torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$PROCS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node:$port" \
  train.py --cpu --epochs 5 --samples 60000 --hidden 512 --batch-size 256 \
           --ckpt "checkpoints/$SLURM_JOB_NAME.pt"

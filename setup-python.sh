#!/usr/bin/env bash
# Create ./venv with a PyTorch build that supports the RTX 50-series (Blackwell, sm_120).
# Run as your normal user (NOT sudo), from anywhere:  ./setup-python.sh
set -euo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB_DIR"

# cu128 = CUDA 12.8 runtime bundled in the wheel; needs driver >= 570. If pytorch.org lists a newer
# index (cu129, cu130, ...) that also works, as long as your driver is new enough for it.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install torch --index-url "$TORCH_INDEX"
mkdir -p logs checkpoints

echo
echo "==> GPU check (directly, not through Slurm)"
venv/bin/python check_gpu.py
echo
echo "==> Same check, but as a Slurm job step on the GPU Slurm allocates"
srun --gpus=1 --cpus-per-task=2 venv/bin/python check_gpu.py
echo
echo "Ready. Submit the training job with:  sbatch jobs/gpu-1node.sbatch"

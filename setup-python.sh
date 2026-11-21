#!/usr/bin/env bash
# Create ./.venv with uv from pyproject.toml + uv.lock (torch from the cu128 index, which supports
# the RTX 50-series / Blackwell, sm_120). Run as your normal user (NOT sudo), from anywhere:  ./setup-python.sh
set -euo pipefail
LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB_DIR"

command -v uv >/dev/null || { echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# --frozen: install exactly what uv.lock pins, so every node gets identical packages.
# To change the torch index (cu129, cu130, ...), edit pyproject.toml and run `uv lock`.
uv sync --frozen
mkdir -p logs checkpoints

echo
echo "==> GPU check (directly, not through Slurm)"
.venv/bin/python check_gpu.py
echo
echo "==> Same check, but as a Slurm job step on the GPU Slurm allocates"
srun --gpus=1 --cpus-per-task=2 .venv/bin/python check_gpu.py
echo
echo "Ready. Submit the training job with:  sbatch jobs/gpu-1node.sbatch"

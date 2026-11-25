# The lab's training environment as one image (PRACTICE.md Modules 8-10).
# Build from the repo root:  docker build -t slurm-lab-train:v1 -f containers/Dockerfile .
#
# The base already has Python 3.12 and torch 2.11.0 built for CUDA 12.8, the same torch that
# uv.lock pins for .venv, so the container and host runs use identical libraries. The NVIDIA
# driver is NOT in the image: the container runtime mounts the host's driver in at start.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# Match pyproject.toml. Add any other job dependency here, then rebuild with a new tag.
RUN pip install --no-cache-dir "numpy>=2.5.5"

# Kubernetes has no shared filesystem, so the code is baked in. Slurm jobs ignore this copy
# and run /lab/train.py from the bind-mounted repo instead, so edits don't need a rebuild.
WORKDIR /app
COPY train.py check_gpu.py /app/

ENV PYTHONUNBUFFERED=1

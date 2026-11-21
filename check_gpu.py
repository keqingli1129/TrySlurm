#!/usr/bin/env python3
"""Sanity check: can this PyTorch build actually run kernels on the GPU?

The RTX 50-series (Blackwell, compute capability 12.0 / sm_120) needs a PyTorch build
made with CUDA 12.8 or newer. Older builds import fine but fail with
"no kernel image is available for execution on the device".
"""
import sys
import time

import torch

print(f"torch {torch.__version__}  (built with CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available: check `nvidia-smi` and that you installed a CUDA build of torch.")

major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
print(f"GPU 0: {torch.cuda.get_device_name(0)}  compute capability {major}.{minor} ({arch})")
print(f"arches in this build: {' '.join(torch.cuda.get_arch_list())}")
if arch not in torch.cuda.get_arch_list():
    print(f"WARNING: {arch} is not in this build. Reinstall torch from a cu128 (or newer) wheel index.")

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):  # warm-up
    a @ a
torch.cuda.synchronize()
t0, n = time.time(), 20
for _ in range(n):
    a @ a
torch.cuda.synchronize()
tflops = 2 * 8192 ** 3 * n / (time.time() - t0) / 1e12
print(f"bf16 matmul OK: ~{tflops:.0f} TFLOPS")

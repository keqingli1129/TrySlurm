#!/usr/bin/env python3
"""Minimal DistributedDataParallel (DDP) training script.

The same file runs unchanged on:
  * 1 GPU on 1 node        (your RTX 5070, NCCL backend)
  * many GPUs on many nodes (NCCL backend)
  * CPU-only nodes          (gloo backend, e.g. the Docker cluster)

Launch it with torchrun (directly, or through `srun torchrun ...` in a Slurm job).
torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT for every process.

The data is synthetic (generated from a fixed seed), so nothing has to be downloaded and
every rank sees the same dataset; DistributedSampler gives each rank its own slice.
Checkpoints are written every epoch and picked up automatically on restart, so the job
survives `scontrol requeue`, preemption or a time limit.
"""
import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024, help="per process")
    p.add_argument("--samples", type=int, default=400_000)
    p.add_argument("--features", type=int, default=128)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt", default="checkpoints/ckpt.pt",
                   help="must be on storage every node can see (shared filesystem) for multi-node resume")
    p.add_argument("--cpu", action="store_true", help="force CPU + gloo even if a GPU is visible")
    return p.parse_args()


def make_dataset(n, d, k, seed=0):
    """A learnable synthetic classification task: labels come from a fixed random 'teacher' net."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w1 = torch.randn(d, 256, generator=g) / d ** 0.5
    w2 = torch.randn(256, k, generator=g) / 16
    y = (torch.tanh(x @ w1) @ w2).argmax(dim=1)
    return TensorDataset(x, y)


def make_model(d, h, k):
    return nn.Sequential(
        nn.Linear(d, h), nn.GELU(),
        nn.Linear(h, h), nn.GELU(),
        nn.Linear(h, h), nn.GELU(),
        nn.Linear(h, k),
    )


def save_checkpoint(path, model, opt, epoch):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, tmp)
    os.replace(tmp, path)  # atomic: a job killed mid-save never leaves a broken checkpoint


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    rank, world = dist.get_rank(), dist.get_world_size()
    dev_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    print(f"[rank {rank}/{world}] host={socket.gethostname()} local_rank={local_rank} "
          f"device={dev_name} backend={dist.get_backend()}", flush=True)
    dist.barrier()

    torch.manual_seed(1234)  # identical initial weights on every rank (DDP also broadcasts them)
    dataset = make_dataset(args.samples, args.features, args.classes)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=use_cuda)

    model = make_model(args.features, args.hidden, args.classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        if rank == 0:
            print(f"Resumed from {args.ckpt} (finished epoch {ckpt['epoch']})", flush=True)

    model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"world_size={world}  params={n_params / 1e6:.1f}M  global_batch={args.batch_size * world}  "
              f"epochs {start_epoch}..{args.epochs - 1}", flush=True)
        if start_epoch >= args.epochs:
            print("Checkpoint says training is already complete. Delete it to start over.", flush=True)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(3, device=device)  # loss_sum, correct, count
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # DDP all-reduces gradients across every rank on every node here
            opt.step()

            totals[0] += loss.detach().float() * y.size(0)
            totals[1] += (logits.argmax(dim=1) == y).sum()
            totals[2] += y.size(0)

        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        dist.all_reduce(totals)  # sum the metrics from all ranks

        if rank == 0:
            loss_avg = (totals[0] / totals[2]).item()
            acc = (totals[1] / totals[2]).item()
            print(f"epoch {epoch:3d}  loss {loss_avg:.4f}  acc {acc:.3f}  "
                  f"{elapsed:6.1f}s  {totals[2].item() / elapsed:,.0f} samples/s", flush=True)
            save_checkpoint(args.ckpt, model, opt, epoch)
        dist.barrier()  # nobody races ahead while rank 0 is writing the checkpoint

    if rank == 0:
        print("done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

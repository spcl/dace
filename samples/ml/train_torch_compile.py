# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Training a PyTorch model with ``torch.compile(backend='dace')``.

TorchDynamo captures the model and AOTAutograd derives its backward graph. DaCe compiles the forward and backward
graphs into a single SDFG with two phases: the forward call runs the first phase, and ``loss.backward()`` runs the
second phase of the same compiled library. With ``dynamic=True`` the SDFG has symbolic sizes, so batches of any size
use it without recompiling.
"""

import argparse

import torch
import torch.nn as nn

import dace.ml


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, 1))


TARGET = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))


def make_batch(size: int):
    """A synthetic regression task: y = tanh(x @ TARGET) + noise."""
    x = torch.randn(size, 8)
    y = torch.tanh(x @ TARGET) + 0.01 * torch.randn(size, 1)
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = make_model()
    backend = dace.ml.DaceBackend()  # Same as backend='dace', but exposes how many SDFGs were compiled
    compiled = torch.compile(model, backend=backend, dynamic=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        x, y = make_batch(16 + step % 3 * 8)  # Batches of different sizes share one SDFG
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(compiled(x), y)
        loss.backward()  # Runs the backward phase of the SDFG
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f'step {step:4d}  loss {loss.item():.4f}')

    print(f'SDFGs compiled: {backend.compile_count}')

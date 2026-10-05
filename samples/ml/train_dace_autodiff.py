# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Training a PyTorch model used inside a ``@dace.program`` with DaCe's automatic differentiation.

The program calls the model like any function (DaCe captures it through TorchDynamo) and computes the loss.
``dace.ml.training_step`` differentiates the whole program with ``dace.autodiff``, which works on the SDFG and also
handles control flow in the program, such as the loop below whose trip count is a runtime argument. Every call runs
one SDFG that computes the loss and the gradients of the model's parameters, and writes the gradients to ``.grad``
like ``loss.backward()`` does, so any ``torch.optim`` optimizer can update the parameters.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

import dace
import dace.ml

N = dace.symbol('N')

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.Tanh())  # Applied repeatedly: a tiny recurrent refinement
head = nn.Linear(8, 1)


@dace.program
def loss_program(x: dace.float32[N, 8], y: dace.float32[N, 1], refinements: dace.int64):
    h = np.copy(x)
    for _ in range(refinements):
        h[:] = model(h)
    difference = head(h) - y
    return np.sum(difference * difference) / N


TARGET = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))


def make_batch(size: int):
    """A synthetic regression task: y = tanh(x @ TARGET) + noise."""
    x = torch.randn(size, 8)
    y = torch.tanh(x @ TARGET) + 0.01 * torch.randn(size, 1)
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--refinements', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    step_fn = dace.ml.training_step(loss_program)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    for step in range(args.steps):
        x, y = make_batch(16 + step % 3 * 8)  # Batches of different sizes share one SDFG
        optimizer.zero_grad()
        loss = step_fn(x, y, args.refinements)  # Loss and gradients in one call
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f'step {step:4d}  loss {loss.item():.4f}')

    print(f'SDFGs compiled: {step_fn.compile_count}')

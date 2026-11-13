# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Adaptive quadrature with the Consume scope: tasks that split themselves until they are accurate enough.

The integral of ``f`` over ``[A, B]`` is computed with Simpson's rule on sub-intervals. Each task is one
sub-interval. It compares Simpson's rule on the whole interval with Simpson's rule on its two halves: if they
agree, the task adds its estimate to the result; otherwise, it pushes its two halves back onto the stream as new
tasks. Work therefore concentrates where ``f`` is hard to integrate (here, around a sharp peak), which a map
over a fixed grid cannot do.

A task names its interval by depth ``d`` and index ``i`` (the interval ``[A + i h, A + (i + 1) h]`` with
``h = (B - A) / 2**d``), packed into one integer.
"""

import argparse
import math

import numpy as np

import dace

# Integrand: a peak of width ~sqrt(EPS) at C, with a closed-form integral to compare against
C = 0.3
EPS = 1e-6
A, B = 0.0, 1.0
TOLERANCE = 1e-10
MAX_DEPTH = 40
DEPTH_SHIFT = 2**42  # A task is ``depth * DEPTH_SHIFT + index``

NUM_SEEDS = dace.symbol("NUM_SEEDS")  # The initial tasks: the intervals of depth log2(NUM_SEEDS)


@dace.program
def integrate(result: dace.float64[1], seed_depth: dace.int64):
    # The tasks in flight (on a GPU, a queue of this capacity)
    S = dace.define_stream(dace.int64, 1 << 20)

    # Start from a few equal intervals, so that every processing element has work at once
    for i in dace.map[0:NUM_SEEDS]:
        S.append(seed_depth * DEPTH_SHIFT + i)

    # Every processing element pops tasks, and a task may push more. Consumption ends once the stream is empty and no
    # task is running (that could still push).
    for task, _ in dace.consume(S, 8):
        depth = task // DEPTH_SHIFT
        index = task - depth * DEPTH_SHIFT
        width = (B - A) / (2.0**depth)
        a = A + index * width
        m = a + 0.5 * width

        # f at the ends, quarter points and middle of the interval
        f0 = 1.0 / (EPS + (a - C) * (a - C))
        f1 = 1.0 / (EPS + (a + 0.25 * width - C) * (a + 0.25 * width - C))
        f2 = 1.0 / (EPS + (m - C) * (m - C))
        f3 = 1.0 / (EPS + (a + 0.75 * width - C) * (a + 0.75 * width - C))
        f4 = 1.0 / (EPS + (a + width - C) * (a + width - C))

        whole = width / 6.0 * (f0 + 4.0 * f2 + f4)
        halved = width / 12.0 * (f0 + 4.0 * f1 + 2.0 * f2 + 4.0 * f3 + f4)
        error = abs(halved - whole)

        if error <= 15.0 * TOLERANCE * width / (B - A) or depth >= MAX_DEPTH:
            # Richardson extrapolation. Processing elements add at once: an augmented assignment is an atomic update
            result[0] += halved + (halved - whole) / 15.0
        else:  # Split into the two halves
            S.append((depth + 1) * DEPTH_SHIFT + 2 * index)
            S.append((depth + 1) * DEPTH_SHIFT + 2 * index + 1)


def exact() -> float:
    root = math.sqrt(EPS)
    return (math.atan((B - C) / root) - math.atan((A - C) / root)) / root


def to_gpu(sdfg: dace.SDFG) -> None:
    """Runs the same program on a GPU: one GPU thread per processing element."""
    for desc in sdfg.arrays.values():
        if not isinstance(desc, dace.data.Scalar):
            desc.storage = dace.StorageType.GPU_Global
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.GPU_Device
        elif isinstance(node, dace.nodes.ConsumeEntry):
            node.consume.schedule = dace.ScheduleType.GPU_Device
            node.consume.num_pes = 4096


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=64, help="initial intervals (a power of two)")
    parser.add_argument("--gpu", action="store_true", help="run on a GPU (requires CuPy)")
    args = parser.parse_args()

    seed_depth = int(math.log2(args.seeds))
    sdfg = integrate.to_sdfg(simplify=True)
    if args.gpu:
        import cupy as xp

        to_gpu(sdfg)
    else:
        xp = np
    result = xp.zeros(1, dtype=np.float64)
    sdfg(result=result, seed_depth=seed_depth, NUM_SEEDS=2**seed_depth)

    value = float(result[0])
    reference = exact()
    print(f"Integral: {value:.12f}, exact: {reference:.12f}, relative error: {abs(value - reference) / reference:.2e}")
    exit(0 if abs(value - reference) / reference < 1e-8 else 1)

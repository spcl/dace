# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k12 Dimension-Lifted Transposition (DLT) -- a 1D 3-point stencil under a blocked layout.

    y[i] = w0*x[i-1] + w1*x[i] + w2*x[i+1]

The DLT layout reshapes the flat array ``x`` to ``[N/V, V]`` (Block), so the three shifted streams
``x[i-1] / x[i] / x[i+1]`` become row-shifted views of one blocked array instead of three
overlapping unaligned streams. The sweep exercises the Block family over ``x`` (a boundary-crossing
neighbor access -- SplitDimensions blocks the offset indices ``x[i-1]``/``x[i+1]`` correctly).

Source: Henretty et al., CC'11 (data-layout transformation for SIMD stencils).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import block_candidates

N = dace.symbol("N")
W0, W1, W2 = 0.25, 0.5, 0.25


@dace.program
def stencil(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[1:N - 1] @ dace.ScheduleType.Sequential:
        y[i] = W0 * x[i - 1] + W1 * x[i] + W2 * x[i + 1]


def oracle(x):
    y = numpy.zeros_like(x)  # boundaries y[0], y[-1] are never written -> 0
    y[1:-1] = W0 * x[:-2] + W1 * x[1:-1] + W2 * x[2:]
    return {"y": y}


def make_inputs(n, seed=0):
    return {"x": numpy.random.default_rng(seed).random(n)}


def candidates():
    """The Block family over x (unblocked + [N/V, V] for V in {4, 8})."""
    return dict(block_candidates("x", 1, factors=(4, 8)))


def run_closure(inputs, n):
    """``run`` reshapes the flat logical x into each candidate's laid-out (blocked) shape."""

    def run(sdfg):
        x_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["x"].shape)
        x_in = inputs["x"].reshape(x_shape).copy()  # packed-C, fresh (not a view)
        y = numpy.zeros(n)
        sdfg(x=x_in, y=y, N=n)
        return {"y": y}

    return run

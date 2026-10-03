# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k01 STREAM triad -- the bandwidth / fragmentation instrument (McCalpin STREAM).

    a[i] = b[i] + Q * c[i]

The reference kernel is a memory-bandwidth calibration instrument -- it carries no layout primitive
of its own; it measures the fraction of a cache line that is useful under a strided read. Ported here
as a 1D elementwise triad whose read operands ``b``, ``c`` can be BLOCKED: ``SplitDimensions`` splits
the contiguous axis into ``[N/T, T]`` tiles, the tile width ``T`` being exactly the cache-line
fraction the STREAM strided variant sweeps. Every block factor is transparent (the tiled read
reproduces the contiguous one), so all candidates match the oracle -- the sweep only picks the
physical tiling.

Source: J. D. McCalpin, "Memory Bandwidth and Machine Balance in Current High Performance
Computers," IEEE TCCA Newsletter, 1995 (STREAM Triad); SC26 layout paper (fragmentation parameter).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import block_candidates

N = dace.symbol("N")
Q = 1.5


@dace.program
def triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        a[i] = b[i] + Q * c[i]


def oracle(b, c):
    return {"a": b + Q * c}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"b": rng.random(n), "c": rng.random(n)}


def pack_1d(v, shape):
    """Lay a logical ``[N]`` vector out into the candidate descriptor shape: unblocked ``[N]`` is a
    plain copy; blocked ``[N/T, T]`` is a plain C-reshape (the tile axis is already last)."""
    if len(shape) == 1:
        return v.copy()
    return v.reshape(shape[0], shape[1]).copy()


def candidates():
    """Block family over each read operand (unblocked + ``[N/T, T]`` tiles for T in {8, 16})."""
    cands = {"noblock": (lambda sdfg: None)}
    for arr in ("b", "c"):
        for name, apply in block_candidates(arr, 1, factors=(8, 16)):
            if name.startswith("noblock"):
                continue
            cands[name] = apply
    return cands


def run_closure(inputs, n):
    """``run`` lays each read operand out into its candidate descriptor shape (see :func:`pack_1d`);
    the output ``a`` stays contiguous ``[N]`` so it is allocated fresh each call."""

    def run(sdfg):
        b_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["b"].shape)
        c_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["c"].shape)
        b_in = pack_1d(inputs["b"], b_shape)
        c_in = pack_1d(inputs["c"], c_shape)
        a = numpy.zeros(n)
        sdfg(a=a, b=b_in, c=c_in, N=n)
        return {"a": a}

    return run

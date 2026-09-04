# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""atax -- the pair-nest GLOBAL-Permute witness (PolyBench atax, reduction form).

    tmp[i] = sum_j A[i,j] * x[j]     (row-streaming of A)
    y[j]   = sum_i A[i,j] * tmp[i]   (column-streaming of A)   i.e. y = A^T (A x)

The single shared matrix ``A`` (shape ``[M, N]``) is read in BOTH orientations -- row-streamed to form
``tmp = A x``, then column-streamed to form ``y = A^T tmp`` -- so no single physical layout is best for
both nests. That is the global layout decision the sweep explores by permuting ``A`` (row-major vs
column-major). The permutation is transparent (``add_permute_maps`` wraps the input and permutes every
``A`` memlet subset in both nests), so every candidate must reproduce the oracle -- the sweep only
picks the fastest physical order.

``tmp`` is a zeroed transient that must be FULLY accumulated (first WCR nest) before it is read (second
WCR nest); the two nests are kept separate so the read-after-write through ``tmp`` is respected. Both
nests are WCR reduce maps (the reduced axis lives in the ``dace.map``, ``out[...] += ...`` into a
pre-zeroed target), not a nested scalar accumulation -- exactly like k04 mvt.

Source: Pouchet & Yuki, PolyBench/C 4.2 (``atax``, ``y = A^T (A x)``); Ziogas et al., NPBench (ICS'21).
Reduction (not BLAS gemv) form so the layout is honest -- BLAS packs operands internally and would
hide it. Sibling of k04 mvt (two nests, shared ``A`` read both ways); here the two nests are chained
through the transient ``tmp`` instead of writing two independent outputs.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

M, N = dace.symbol("M"), dace.symbol("N")


@dace.program
def atax(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[N]):
    tmp = numpy.zeros((M, ), dace.float64)  # zeroed transient: fully accumulated before it is read
    for i, j in dace.map[0:M, 0:N] @ dace.ScheduleType.Sequential:
        tmp[i] += A[i, j] * x[j]
    for i, j in dace.map[0:M, 0:N] @ dace.ScheduleType.Sequential:
        y[j] += A[i, j] * tmp[i]


def oracle(A, x):
    tmp = A @ x
    return {"y": A.T @ tmp}


def make_inputs(m, n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((m, n)), "x": rng.random(n)}


def candidates():
    """The global layout candidates for this kernel: every dimension permutation of the shared ``A``
    (row-major vs column-major storage). All are transparent under ``add_permute_maps``, so every
    candidate reproduces the oracle."""
    return dict(permutation_candidates("A", 2))


def run_closure(inputs, m, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep. The permute is transparent (``A`` keeps its
    logical ``[M, N]`` shape -- the transpose happens inside the SDFG), so ``A`` is packed by a plain
    copy, ``x`` binds as-is, and a fresh zeroed ``y`` is allocated each call."""

    def run(sdfg):
        y = numpy.zeros(n)
        sdfg(A=inputs["A"].copy(), x=inputs["x"].copy(), y=y, M=m, N=n)
        return {"y": y}

    return run

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k04 mvt -- the "layouts are global" witness (PolyBench mvt, reduction form).

    x1[i] = sum_j A[i,j] * y1[j]     (row-streaming of A)
    x2[j] = sum_i A[i,j] * y2[i]     (column-streaming of A)

One array ``A`` is read in BOTH orientations, so no single physical layout is best for both nests --
the global layout decision the sweep explores by permuting ``A`` (row-major vs column-major). The
permutation is transparent (``add_permute_maps`` wraps the input), so every candidate must reproduce
the oracle.

Source: Pouchet & Yuki, PolyBench/C 4.2; Ziogas et al., NPBench (ICS'21). Reduction (not BLAS gemv)
form so the layout is honest -- BLAS packs operands internally and would hide it.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

N = dace.symbol("N")


@dace.program
def mvt(A: dace.float64[N, N], y1: dace.float64[N], y2: dace.float64[N], x1: dace.float64[N], x2: dace.float64[N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        x1[i] += A[i, j] * y1[j]
        x2[j] += A[i, j] * y2[i]


def oracle(A, y1, y2):
    return {"x1": (A * y1[None, :]).sum(axis=1), "x2": (A * y2[:, None]).sum(axis=0)}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, n)), "y1": rng.random(n), "y2": rng.random(n)}


def candidates():
    """The global layout candidates for this kernel: every dimension permutation of ``A``."""
    return dict(permutation_candidates("A", 2))


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds fresh zeroed outputs each call."""

    def run(sdfg):
        x1 = numpy.zeros(n)
        x2 = numpy.zeros(n)
        sdfg(A=inputs["A"].copy(), y1=inputs["y1"].copy(), y2=inputs["y2"].copy(), x1=x1, x2=x2, N=n)
        return {"x1": x1, "x2": x2}

    return run

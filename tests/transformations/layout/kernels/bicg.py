# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""bicg -- the pair-nest GLOBAL-Permute witness (PolyBench BiCG, reduction form).

    q[i] = sum_j A[i,j] * p[j]     (row-streaming of A)      -> q = A p
    s[j] = sum_i A[i,j] * r[i]     (column-streaming of A)    -> s = A^T r

The single (non-square) array ``A[N, M]`` is read in BOTH orientations within one fused nest: ``q``
streams it row-major (contract the M axis), ``s`` streams it column-major (contract the N axis). No
single physical layout is best for both, so the global layout decision the sweep explores is the
dimension order of ``A`` (row-major vs column-major) -- exactly the k04 mvt witness, but the two
reductions here contract into two INDEPENDENT WCR outputs (``q[N]`` and ``s[M]``) rather than the
matched-shape pair of mvt. The permutation is transparent (``add_permute_maps`` wraps the input and
permutes each memlet subset), so every candidate reproduces the oracle -- the sweep only picks the
fastest.

Source: Pouchet & Yuki, PolyBench/C 4.2 (bicg, the BiConjugate Gradient stabilized kernel); Ziogas et
al., NPBench (ICS'21). Reduction (not BLAS gemv) form so the layout is honest -- BLAS packs operands
internally and would hide the read-orientation conflict.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

N, M = dace.symbol("N"), dace.symbol("M")


@dace.program
def bicg(A: dace.float64[N, M], p: dace.float64[M], r: dace.float64[N], q: dace.float64[N], s: dace.float64[M]):
    for i, j in dace.map[0:N, 0:M] @ dace.ScheduleType.Sequential:
        q[i] += A[i, j] * p[j]
        s[j] += A[i, j] * r[i]


def oracle(A, p, r):
    return {"q": (A * p[None, :]).sum(axis=1), "s": (A * r[:, None]).sum(axis=0)}


def make_inputs(n, m, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, m)), "p": rng.random(m), "r": rng.random(n)}


def candidates():
    """The global layout candidates for this kernel: every dimension permutation of ``A`` (row-major
    vs column-major), the read-orientation conflict the pair nest exposes."""
    return dict(permutation_candidates("A", 2))


def run_closure(inputs, n, m):
    """A ``run(sdfg) -> outputs`` closure for the sweep. The permute is transparent (``A`` keeps its
    logical shape, so packing it into the candidate descriptor is a plain copy); ``p`` / ``r`` bind
    as-is and fresh zeroed ``q`` / ``s`` are allocated each call (the WCR accumulators)."""

    def run(sdfg):
        q = numpy.zeros(n)
        s = numpy.zeros(m)
        sdfg(A=inputs["A"].copy(), p=inputs["p"].copy(), r=inputs["r"].copy(), q=q, s=s, N=n, M=m)
        return {"q": q, "s": s}

    return run

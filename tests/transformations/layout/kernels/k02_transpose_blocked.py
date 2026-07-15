# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k02 blocked transpose -- the Block/Permute witness (out-of-place 2D transpose).

    B[j, i] = A[i, j]        (out-of-place transpose)

The write scatters at stride N under row-major, so no single loop order serves both the contiguous
reads of ``A`` and the contiguous writes of ``B`` -- the classic transpose memory problem. The
layout decision is how ``A`` is stored:

    row-major : addr(i, j) = i*N + j             (baseline, strided writes)
    blocked   : A reshaped [N/T, T, N] (Block)   (T x T tiles stored contiguously)
    permuted  : A stored column-major (Permute)  (transparent dim swap)

The sweep exercises the Block family (SplitDimensions splits one dimension of ``A`` into
``[N/T, T]``) and the Permute family (a transparent dimension swap of ``A``) -- both preserve the
result, so every candidate reproduces the oracle; the sweep only picks the physical layout.

Source: Springer, Su, Bientinesi, "HPTT," ARRAY@PLDI'17 (reference transposer); Chatterjee et al.,
"Nonlinear array layouts for hierarchical memory systems," ICS'99 (4D blocked layout); SC26 layout
paper SS IV-B2 (transpose microbenchmark, Block primitive).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import block_candidates, permutation_candidates

N = dace.symbol("N")


@dace.program
def transpose(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        B[j, i] = A[i, j]


def oracle(A):
    return {"B": A.T.copy()}


def make_inputs(n, seed=0):
    return {"A": numpy.random.default_rng(seed).random((n, n))}


def candidates():
    """Block family over A (unblocked + [N/T, T] tiles for T in {8, 16, 32}) plus the transparent
    Permute family (row-major vs column-major storage of A)."""
    cands = dict(block_candidates("A", 2, factors=(8, 16, 32)))
    cands.update(permutation_candidates("A", 2))
    return cands


def pack_a(A, shape, n):
    """Lay the logical [N, N] ``A`` out into candidate ``shape`` (the SDFG descriptor of ``A``).

    ``SplitDimensions`` groups the inner tile axis LAST, so blocking a non-final dimension permutes
    axes -- a plain C-reshape is NOT enough there:

      * ndim 2  -> [N, N]        : unblocked / transparent Permute, storage is logical -> plain copy.
      * [N/b,N,b]-> dim-0 block  : phys[bi, j, ti] = A[bi*b+ti, j] -> reshape [N/b,b,N] then swap
                                    the tile axis to the end (``transpose(0, 2, 1)``).
      * [N,N/b,b]-> dim-1 block  : phys[i, bj, tj] = A[i, bj*b+tj] -> plain reshape (tile axis is
                                    already last).
    """
    if len(shape) == 2:
        return A.copy()
    b = shape[-1]
    if shape[0] == n // b:  # dim 0 blocked: (N/b, N, b)
        return A.reshape(n // b, b, n).transpose(0, 2, 1).copy()
    return A.reshape(n, n // b, b).copy()  # dim 1 blocked: (N, N/b, b)


def run_closure(inputs, n):
    """``run`` lays the logical A out into each candidate's descriptor shape (see :func:`pack_a`);
    for the Permute family the descriptor stays [N, N] so it is a plain copy."""

    def run(sdfg):
        a_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["A"].shape)
        a_in = pack_a(inputs["A"], a_shape, n)  # laid out per descriptor, fresh (not a view)
        B = numpy.zeros((n, n))
        sdfg(A=a_in, B=B, N=n)
        return {"B": B}

    return run

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""E1 matrix add -- the SC26 Fig.4 running example (dense two-input elementwise add).

    C[i, j] = A[i, j] + B[i, j]        (dense elementwise add over an [N, N] domain)

This is the paper's running example (SC26 Fig.4, E1_MatrixAdd): the simplest dense kernel that still
carries a real layout decision. Both operands are read contiguously under row-major, so no layout is
forced -- which makes it the honest instrument for the *read* side of the algebra. The decision here
is on operand ``B``: how it is physically stored while ``A`` and ``C`` stay logical [N, N].

    row-major : addr(i, j) = i*N + j             (baseline, identity Permute)
    permuted  : B stored column-major (Permute)  (transparent dim swap of the read operand)
    blocked   : B reshaped into [N/T, T] tiles   (Block, tile width T)

The sweep exercises the Permute family (row- vs column-major storage of ``B``) and the Block family
(``SplitDimensions`` splits one dimension of ``B`` into ``[N/T, T]``). Every candidate is transparent
-- the permute wraps each read memlet subset, the block re-tiles the read -- so all reproduce the
oracle; the sweep only picks the physical layout of ``B``. Structurally this is k02 (blocked
transpose) minus the transposed write: a plain contiguous ``C`` output.

Source: SC26 layout-algebra paper Fig.4 (E1_MatrixAdd running example, Permute + Block primitives on
the read operand); Chatterjee et al., "Nonlinear array layouts for hierarchical memory systems,"
ICS'99 (blocked layout).
"""
import numpy
import dace

from dace.transformation.layout.brute_force import block_candidates, permutation_candidates

N = dace.symbol("N")


@dace.program
def matrix_add(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] + B[i, j]


def oracle(A, B):
    return {"C": A + B}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, n)), "B": rng.random((n, n))}


def candidates():
    """The layout candidates for the read operand ``B``: the transparent Permute family (row-major
    vs column-major storage) plus the Block family (unblocked + ``[N/T, T]`` tiles for T in
    {8, 16, 32}). ~9 candidates, all focused on ``B`` to match Fig.4."""
    cands = dict(permutation_candidates("B", 2))
    cands.update(block_candidates("B", 2, factors=(8, 16, 32)))
    return cands


def pack_b(B, shape, n):
    """Lay the logical [N, N] ``B`` out into candidate ``shape`` (the SDFG descriptor of ``B``).

    ``SplitDimensions`` groups the inner tile axis LAST, so blocking a non-final dimension permutes
    axes -- a plain C-reshape is NOT enough there (identical to k02's ``pack_a``):

      * ndim 2  -> [N, N]        : unblocked / transparent Permute, storage is logical -> plain copy.
      * [N/b,N,b]-> dim-0 block  : phys[bi, j, ti] = B[bi*b+ti, j] -> reshape [N/b,b,N] then swap
                                    the tile axis to the end (``transpose(0, 2, 1)``).
      * [N,N/b,b]-> dim-1 block  : phys[i, bj, tj] = B[i, bj*b+tj] -> plain reshape (tile axis is
                                    already last).
    """
    if len(shape) == 2:
        return B.copy()
    b = shape[-1]
    if shape[0] == n // b:  # dim 0 blocked: (N/b, N, b)
        return B.reshape(n // b, b, n).transpose(0, 2, 1).copy()
    return B.reshape(n, n // b, b).copy()  # dim 1 blocked: (N, N/b, b)


def run_closure(inputs, n):
    """``run`` lays the read operand ``B`` out into each candidate's descriptor shape (see
    :func:`pack_b`); the Permute family keeps the descriptor [N, N] (plain copy). ``A`` stays logical
    [N, N] and the output ``C`` is allocated fresh each call."""

    def run(sdfg):
        b_shape = tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in sdfg.arrays["B"].shape)
        b_in = pack_b(inputs["B"], b_shape, n)  # laid out per descriptor, fresh (not a view)
        C = numpy.zeros((n, n))
        sdfg(A=inputs["A"].copy(), B=b_in, C=C, N=n)
        return {"C": C}

    return run

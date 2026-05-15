# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Audit tests for the inter-lane-stride classification in
``Vectorize._setup_strided_nsdfg_edges_inline``.

The strided-vs-contiguous decision is made by the *inter-lane stride*
— the wide-dim displacement between consecutive lanes,
``begin(map_param + 1) - begin(map_param)``:

- stride ``1`` → contiguous, even with a wide stencil halo (``bbox >
  W``); must fall through to the normal reshape path (NOT the
  ``(W-1)*S+K`` strided handler).
- stride ``> 1`` → genuinely strided; routed to the strided handler.

These kernels iterate a non-divisible trip count (``S - 2`` / ``N``)
so P1/P2 wrap+split and the NSDFG-boundary strided classifier is
actually exercised — the regression these guard is jacobi-style
stencils mis-routed as strided (``NotImplementedError: bbox volume
... doesn't match (W-1)*S+K``).
"""
import copy

import dace
import numpy as np

from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

M = dace.symbol("M")
S = dace.symbol("S")


@dace.program
def jacobi_5point(A: dace.float64[S, S], B: dace.float64[S, S]):
    """5-point stencil: inter-lane stride 1 on the j-dim, bbox W+2."""
    for i, j in dace.map[0:S - 2, 0:S - 2]:
        B[i + 1, j + 1] = 0.2 * (A[i + 1, j + 1] + A[i, j + 1] + A[i + 2, j + 1] + A[i + 1, j] + A[i + 1, j + 2])


@dace.program
def stencil_wide_j(A: dace.float64[S], B: dace.float64[S]):
    """Wide 5-in-j stencil: inter-lane stride still 1, bbox W+4."""
    for i in dace.map[0:S - 4:1]:
        B[i + 2] = 0.1 * (A[i] + A[i + 1] + A[i + 2] + A[i + 3] + A[i + 4])


@dace.program
def strided_2i(src: dace.float64[2 * 8 * M], dst: dace.float64[8 * M], scale: dace.float64):
    """Genuine stride-2 access: inter-lane stride 2.

    Uses a provably-divisible ``8*M`` trip so this isolates the
    *classification* (stride-2 → strided handler) from the separate
    strided + P1-NSDFG-wrap numerical bug that bites only when the
    trip is not provably divisible.
    """
    for i in dace.map[0:8 * M:1]:
        dst[i] = src[2 * i] * scale


def _run(prog, arrays, params):
    """Vectorize ``prog`` and compare against the unvectorized reference.

    :param prog: ``@dace.program`` to build.
    :param arrays: name -> ndarray (mutated in place by the SDFG call).
    :param params: extra scalar symbols passed to both SDFG calls.
    :returns: max abs diff between reference and vectorized outputs.
    """
    ref = {k: v.copy() for k, v in arrays.items()}
    vec = {k: v.copy() for k, v in arrays.items()}

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = f"{prog.name}_ils_ref"
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{prog.name}_ils_v"
    VectorizeCPU(vector_width=8,
                 fail_on_unvectorizable=True,
                 use_fp_factor=True,
                 branch_normalization=False,
                 insert_copies=False,
                 remainder_strategy="scalar").apply_pass(vsdfg, {})

    sdfg(**ref, **params)
    vsdfg(**vec, **params)
    return max(np.max(np.abs(ref[k] - vec[k])) for k in arrays)


def test_jacobi_5point_contiguous_halo_not_strided():
    """5-point jacobi: bbox = W+2 > W but inter-lane stride 1 — must
    vectorize through the normal reshape path (no strided
    NotImplementedError) and be numerically exact."""
    _S = 66  # S-2 = 64 (not provably divisible — exercises remainder)
    A = np.random.rand(_S, _S)
    B = np.random.rand(_S, _S)
    diff = _run(jacobi_5point, {"A": A, "B": B}, {"S": _S})
    assert diff < 1e-12, f"max abs diff = {diff}"


def test_wide_stencil_contiguous_halo_not_strided():
    """5-in-j stencil: bbox = W+4 > W, inter-lane stride still 1.
    A wider halo must NOT flip the contiguous classification."""
    _S = 70  # S-4 = 66 (not divisible — remainder path)
    A = np.random.rand(_S)
    B = np.random.rand(_S)
    diff = _run(stencil_wide_j, {"A": A, "B": B}, {"S": _S})
    assert diff < 1e-12, f"max abs diff = {diff}"


def test_stride_2_is_strided_and_correct():
    """``src[2*i]``: inter-lane stride 2 — genuinely strided, routed to
    the strided handler, numerically exact (provably-divisible 8*M so
    no remainder, isolating the classification)."""
    _Mv = 8  # 8*M = 64
    src = np.random.rand(2 * 8 * _Mv)
    dst = np.zeros(8 * _Mv)
    diff = _run(strided_2i, {"src": src, "dst": dst}, {"M": _Mv, "scale": 1.5})
    assert diff < 1e-12, f"max abs diff = {diff}"


if __name__ == "__main__":
    test_jacobi_5point_contiguous_halo_not_strided()
    test_wide_stencil_contiguous_halo_not_strided()
    test_stride_2_is_strided_and_correct()

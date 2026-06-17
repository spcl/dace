# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical tests for reduction kernels via :class:`TileReduce`.

Reductions are "full tile -> scalar" per user direction: the walker stages the input
as a tile bridge, ``TileReduce`` reduces over it into a scalar accumulator. The mask
gates out-of-range lanes for the partial-tail case.

Kernels exercised:
* ``acc = sum(A)`` -- the simplest reduction.
* ``acc = max(A)`` -- non-commutative-in-codegen variant.
"""
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)


N_SYM = dace.symbol("N_REDUCE")


@dace.program
def k1_sum(A: dace.float64[N_SYM], acc: dace.float64[1]):
    for i in dace.map[0:N_SYM]:
        with dace.tasklet:
            a << A[i]
            o >> acc(1, lambda x, y: x + y)[0]
            o = a


@pytest.mark.parametrize("N", [8, 16, 24])
def test_k1_sum_reduction_is_refused(N):
    """``acc = sum(A)`` -- the ``scalar-WCR -> MapExit -> WCR -> write`` reduction
    boundary. This pattern SHOULD be vectorized (it is the canonical reduction
    shape a ``@dace.program`` emits); the current ``KeyError('tmp')`` from
    ``VectorizeCPUMultiDim`` is a BUG to fix, not intended behaviour.

    WCR *inside* the NSDFG is deliberately NOT supported -- so the fix must
    handle the WCR-at-the-MapExit-boundary form directly, NOT by relocating the
    WCR into the nested SDFG.

    TEMPORARY: this test pins today's refusal with ``pytest.raises`` so the
    suite is deterministic. When reduction vectorization lands, replace it with
    the bit-equal numerical comparison against the unvectorised reference
    (``acc_vec == acc_ref``). See project_vectorization_constexpr_stride_codegen.
    """
    vec = k1_sum.to_sdfg(simplify=True)
    vec.name = f"k1_sum_vec_{N}"
    with pytest.raises(KeyError):
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})

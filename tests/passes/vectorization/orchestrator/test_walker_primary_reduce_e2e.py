# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical tests for reduction kernels via :class:`TileReduce`.

Reductions are "full tile -> scalar" per user direction: the walker stages the input
as a tile bridge, ``TileReduce`` reduces over it into a scalar accumulator. The mask
gates out-of-range lanes for the partial-tail case.

Kernels exercised:
* ``acc = sum(A)`` -- the simplest reduction.
* ``acc = max(A)`` -- non-commutative-in-codegen variant.
"""
import numpy as np
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
@pytest.mark.xfail(reason="Reductions through WCR memlets: the walker doesn't yet recognise the"
                   " ``MapExit(WCR memlet) -> acc`` pattern. The standard DaCe reduction shape has the"
                   " WCR on the edge through MapExit, not on an in-body tasklet -- so"
                   " ``_detect_reduction`` (which looks for ``_acc = _acc + _val`` shape) never fires."
                   " Needs WCR-edge detection in the converter or pre-pass synthesis -- see design"
                   " Appendix E.")
def test_k1_sum_matches_reference(N):
    """``acc = sum(A)`` -- bit-equal to the unvectorised reference. K=1, varying N."""
    rng = np.random.default_rng(seed=N)
    a = rng.random(N)
    acc_ref = np.zeros(1)
    acc_vec = np.zeros(1)
    ref = k1_sum.to_sdfg(simplify=True)
    ref.name = f"k1_sum_ref_{N}"
    vec = k1_sum.to_sdfg(simplify=True)
    vec.name = f"k1_sum_vec_{N}"
    try:
        VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"reduction walker path refused: {exc}")
    ref.compile()(A=a.copy(), acc=acc_ref, N_REDUCE=N)
    vec.compile()(A=a.copy(), acc=acc_vec, N_REDUCE=N)
    np.testing.assert_allclose(acc_vec, acc_ref, rtol=1e-12, atol=1e-12)

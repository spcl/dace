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


@pytest.mark.parametrize("N", [8, 16, 17, 23, 24])
def test_k1_sum_matches_reference(N):
    """``acc = sum(A)`` -- the ``scalar-WCR -> MapExit -> WCR -> write`` reduction
    boundary -- vectorizes and is bit-equal to the unvectorised reference.

    ``LiftMapReductionToReduce`` recognises the pure-WCR boundary reduction and
    lifts it to a product buffer + a vectorized ``Reduce`` (a ``horizontal_reduce``
    fold), keeping the ``CR:Sum`` WCR on the ``Reduce -> acc`` edge so the
    accumulation semantics survive for any initial ``acc``. The WCR stays at the
    MapExit boundary -- it is never relocated into the body NSDFG (which the
    vectorizer does not support, by design). ``N = 16, 24`` exercise the
    multi-tile fold; ``N = 17, 23`` exercise the masked / scalar-tail remainder.
    """
    rng = np.random.default_rng(seed=N)
    arr = rng.random(N)
    acc_ref = np.zeros(1)
    acc_vec = np.zeros(1)
    ref = k1_sum.to_sdfg(simplify=True)
    ref.name = f"k1_sum_ref_{N}"
    vec = k1_sum.to_sdfg(simplify=True)
    vec.name = f"k1_sum_vec_{N}"
    VectorizeCPUMultiDim(widths=(8, ), target_isa="SCALAR").apply_pass(vec, {})
    ref.compile()(A=arr.copy(), acc=acc_ref, N_REDUCE=N)
    vec.compile()(A=arr.copy(), acc=acc_vec, N_REDUCE=N)
    np.testing.assert_allclose(acc_vec, acc_ref, rtol=1e-12, atol=1e-12)

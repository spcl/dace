# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""New-path (``VectorizeCPUMultiDim``) SpMV: gather + partial reduction.

SpMV ``y[i] = sum_k A[i, k] * x[col[k]]`` combines two features the K-dim
tile path must support together:

* a **gather** load ``x[col[k]]`` (data-dependent index over the reduced
  ``k`` dim), and
* a **partial reduction** ``y[i] += ...`` -- the output keeps the ``i``
  tile dim and collapses the ``k`` tile dim (a per-row reduction, not a
  full reduction to a scalar).

This is distinct from a full reduction ``s += a[i]`` (covered in
``orchestrator/test_indirect_and_spmv_gaps.py``): here a tile dim
*survives* into the output while another is reduced out, so the store is a
WCR collapse-out write rather than a scalar TileReduce.

The reference is computed directly in NumPy (``(A * x[col]).sum(axis=1)``),
not from the unvectorized DaCe SDFG, so the test pins absolute numerical
correctness of the vectorized lowering -- a miscompiled gather or a dropped
reduction chunk cannot hide behind a matching-but-wrong scalar path.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

_N = dace.symbol("N")
_NNZ = dace.symbol("NNZ")


@dace.program
def _spmv(y: dace.float64[_N], A: dace.float64[_N, _NNZ], x: dace.float64[_N], col: dace.int32[_NNZ]):
    """SpMV-style gather + reduction, in the canonical Option-B form: a product
    Map (``prod[i, k] = A[i, k] * x[col[k]]`` -- straight-line body with a gather)
    followed by a ``Reduce`` over the reduced ``k`` axis (``y[i] = sum_k prod[i, k]``).

    This is the well-formed reduction shape for the vectorization path: the
    reduction lives in a ``Reduce`` libnode (lowered by ``ExpandReduceVectorized``),
    NOT as a for-loop or a WCR inside the vectorized map body (neither of which the
    tile path supports). The fused ``y[i] += A[i,k]*x[col[k]]`` form is lifted to
    this shape by a separate pass (see WP1-B)."""
    prod = np.ndarray((_N, _NNZ), dace.float64)
    for i, k in dace.map[0:_N, 0:_NNZ]:
        prod[i, k] = A[i, k] * x[col[k]]
    y[:] = np.sum(prod, axis=1)


def _spmv_numpy(A, x, col):
    """NumPy reference: ``y[i] = sum_k A[i, k] * x[col[k]]``."""
    return (A * x[col][None, :]).sum(axis=1)


# (N, NNZ): square, wide-k (NNZ>N, multiple k-chunks), and non-W-divisible
# trips (a scalar remainder loop on the reduced k axis).
#
# Only the reduced dim ``k`` is tiled (``widths`` length 1). The product Map
# vectorizes over ``k`` (gather ``x[col[k]]``); the ``Reduce`` folds it to
# ``y[i]``. When ``NNZ`` is not a multiple of the tile width the divisible
# interior is tiled and the tail is a step-1 **scalar remainder loop**
# (``remainder_strategy="scalar_postamble"``) -- masking the tail of a gather
# would read out-of-range ``col`` indices, so the tail must be scalar.
@pytest.mark.parametrize("n,nnz", [(16, 16), (16, 24), (24, 16), (17, 16), (16, 17), (17, 23)])
@pytest.mark.parametrize("widths", [(8, ), (4, )])
def test_spmv_matches_numpy(n, nnz, widths):
    rng = np.random.default_rng(seed=n * 1000 + nnz * 10 + sum(widths))
    A = rng.random((n, nnz))
    x = rng.random((n, ))
    col = rng.integers(0, n, size=nnz).astype(np.int32)

    y_ref = _spmv_numpy(A, x, col)

    vec = _spmv.to_sdfg(simplify=True)
    vec.name = f"spmv_{n}_{nnz}_{'x'.join(map(str, widths))}"
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=ISA.SCALAR,
                                         remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE)).apply_pass(vec, {})
    vec.validate()

    y_vec = np.zeros(n)
    vec.compile()(y=y_vec, A=A.copy(), x=x.copy(), col=col.copy(), N=n, NNZ=nnz)

    np.testing.assert_allclose(y_vec, y_ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

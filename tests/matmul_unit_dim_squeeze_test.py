"""A ``subsets.Range`` entry of size 1 is ambiguous, and the MatMul library nodes read it two
incompatible ways.

``Range`` stores both a rank-reducing index and a genuine extent-1 dimension as the same triple
``(0, 0, 1)``, so ``A[r, 0:N, 0:P]`` (dimension indexed away, rank 2) and ``V[0:N, 0, 0:P]``
(full read of a ``(N, 1, P)`` view, rank 3) are indistinguishable at the subset level.
``Range.squeeze()`` drops both.

``np.reshape(x, (NQ, 1, NP))`` builds exactly the second kind: a 3D view whose middle dimension is
part of the shape. ``SpecializeMatMul`` dispatched on the squeezed sizes and saw a matrix, so it
picked ``Gemm``, whose ``validate`` re-read the unsqueezed subset and raised
"matrix-matrix product only supported on matrices". npbench's doitgen hits this on every size.

The descriptor is the authority: a subset entry of size 1 is rank-reducing only where the
descriptor's own extent is larger. Under that rule the reshape stays 3D and dispatches to
``BatchedMatMul``, while a real index into a larger dimension still squeezes to ``Gemm``.
"""
import numpy as np
import pytest

import dace
from dace.libraries.blas.nodes.batched_matmul import BatchedMatMul
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.transformation.auto.auto_optimize import auto_optimize

NR, NQ, NP = (dace.symbol(s, dtype=dace.int64) for s in ('NR', 'NQ', 'NP'))


@dace.program
def doitgen_reshape(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
    # npbench polybench/doitgen, verbatim: the (NQ, 1, NP) reshape is the point of the benchmark.
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))


@dace.program
def indexed_slice_matmul(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
    # A[r] indexes dimension 0 away: a genuine 2D operand that must still reach Gemm.
    for r in range(NR):
        A[r, :, :] = A[r] @ C4


def reference(A, C4):
    NR, NQ, NP = A.shape
    return np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


def initialize(nr, nq, np_):
    A = np.fromfunction(lambda i, j, k: ((i * j + k) % np_) / np_, (nr, nq, np_), dtype=np.float64)
    C4 = np.fromfunction(lambda i, j: (i * j % np_) / np_, (np_, np_), dtype=np.float64)
    return A, C4


def count_nodes(sdfg, nodetype):
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodetype))


def specialize_matmuls(sdfg):
    """Expand the MatMul meta-nodes one level, leaving the chosen specialization in the graph."""
    for node, state in list(sdfg.all_nodes_recursive()):
        if type(node) is MatMul:
            node.expand(state)


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('sizes', [(3, 4, 5), (1, 1, 1), (8, 10, 12), (5, 1, 7)])
def test_reshape_unit_dim_matmul(optimize, sizes):
    nr, nq, np_ = sizes
    A, C4 = initialize(nr, nq, np_)
    ref = reference(A, C4)

    sdfg = doitgen_reshape.to_sdfg(simplify=False)
    sdfg.simplify()
    if optimize:
        auto_optimize(sdfg, dace.dtypes.DeviceType.CPU, symbols=dict(NR=nr, NQ=nq, NP=np_))

    sdfg(A=A, C4=C4, NR=nr, NQ=nq, NP=np_)
    assert np.allclose(A, ref)


def test_reshape_unit_dim_dispatches_to_batched():
    """The reshaped operand keeps its unit dimension, so the matmul stays batched."""
    sdfg = doitgen_reshape.to_sdfg(simplify=False)
    sdfg.simplify()
    specialize_matmuls(sdfg)
    assert count_nodes(sdfg, BatchedMatMul) == 1
    assert count_nodes(sdfg, Gemm) == 0


def test_indexed_dim_still_squeezes_to_gemm():
    """An index into a larger dimension is rank-reducing, so the operand is a plain matrix."""
    nr, nq, np_ = 3, 4, 5
    A, C4 = initialize(nr, nq, np_)
    ref = reference(A, C4)

    sdfg = indexed_slice_matmul.to_sdfg(simplify=False)
    sdfg.simplify()
    specialize_matmuls(sdfg)
    assert count_nodes(sdfg, Gemm) == 1
    assert count_nodes(sdfg, BatchedMatMul) == 0

    sdfg(A=A, C4=C4, NR=nr, NQ=nq, NP=np_)
    assert np.allclose(A, ref)


if __name__ == '__main__':
    for opt in (False, True):
        test_reshape_unit_dim_matmul(opt, (3, 4, 5))
    test_reshape_unit_dim_dispatches_to_batched()
    test_indexed_dim_still_squeezes_to_gemm()

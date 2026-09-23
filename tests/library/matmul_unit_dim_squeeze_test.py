# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""``SpecializeMatMul`` and ``Gemm`` must read the same view of an operand.

The dispatcher matches on squeezed sizes, so ``np.reshape(x, (NQ, 1, NP)) @ C4`` routes to ``Gemm``
as ``(NQ, NP) @ (NP, NP)``. ``Gemm.validate`` and the GEMM codegen used to re-read the raw subset,
see rank 3, and reject the operand -- "matrix-matrix product only supported on matrices". npbench's
doitgen hits this. Collapsing the unit dim is exact (one GEMM), and keeping it on ``Gemm`` preserves
``alpha``, ``beta`` and the WCR that ``BatchedMatMul`` drops.
"""
import numpy as np
import pytest

import dace
from dace.library import change_default
from dace.libraries import blas
from dace.libraries.blas.blas_helpers import packed_unit_extent
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
    nr, nq, np_ = A.shape
    return np.reshape(np.reshape(A, (nr, nq, 1, np_)) @ C4, (nr, nq, np_))


def initialize(nr, nq, np_, seed=0):
    # Random, not polybench's ((i*j+k) % NP)/NP, which is all-zero at NP == 1 (a vacuous compare).
    rng = np.random.default_rng(seed)
    return rng.random((nr, nq, np_)), rng.random((np_, np_))


def count_nodes(sdfg, nodetype):
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodetype))


def specialize_matmuls(sdfg):
    """Expand the MatMul meta-nodes one level, leaving the chosen specialization in the graph."""
    for node, state in list(sdfg.all_nodes_recursive()):
        if type(node) is MatMul:
            node.expand(state)


# NQ == 1 collapses a second dim when squeezed, where a "whichever view looks 2D" heuristic breaks.
SIZES = [(3, 4, 5), (1, 1, 1), (8, 10, 12), (5, 1, 7)]


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('sizes', SIZES)
def test_reshape_unit_dim_matmul(optimize, sizes):
    nr, nq, np_ = sizes
    A, C4 = initialize(nr, nq, np_)
    ref = reference(A, C4)
    assert np.abs(ref).max() > 0.0  # guard against a vacuous comparison

    sdfg = doitgen_reshape.to_sdfg(simplify=False)
    sdfg.simplify()
    if optimize:
        auto_optimize(sdfg, dace.dtypes.DeviceType.CPU, symbols=dict(NR=nr, NQ=nq, NP=np_))

    sdfg(A=A, C4=C4, NR=nr, NQ=nq, NP=np_)
    assert np.allclose(A, ref)


def test_reshape_unit_dim_stays_one_gemm():
    """The collapse is exact, so the product must not fan out into NQ batched calls."""
    sdfg = doitgen_reshape.to_sdfg(simplify=False)
    sdfg.simplify()
    specialize_matmuls(sdfg)
    assert count_nodes(sdfg, Gemm) == 1
    assert count_nodes(sdfg, BatchedMatMul) == 0


def test_indexed_dim_still_reaches_gemm():
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


def test_unit_batch_keeps_alpha():
    """Routing a unit-batch product away from Gemm would silently drop alpha, beta and WCR."""
    from dace import memlet as mm

    sdfg = dace.SDFG('unit_batch_alpha')
    b, m, k, n = 1, 8, 6, 5
    sdfg.add_array('A', [b, m, k], dace.float64)
    sdfg.add_array('B', [k, n], dace.float64)
    sdfg.add_array('C', [b, m, n], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    node = MatMul('mm', alpha=2.0)
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', mm.Memlet('A[0:1, 0:8, 0:6]'))
    state.add_edge(state.add_read('B'), None, node, '_b', mm.Memlet('B[0:6, 0:5]'))
    state.add_edge(node, '_c', state.add_write('C'), None, mm.Memlet('C[0:1, 0:8, 0:5]'))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    a, bmat, c = rng.random((b, m, k)), rng.random((k, n)), np.zeros((b, m, n))
    sdfg(A=a, B=bmat, C=c)
    assert np.allclose(c[0], 2.0 * (a[0] @ bmat))


A_, B_, I_, J_ = (dace.symbol(s, dtype=dace.int64) for s in ('A_', 'B_', 'I_', 'J_'))


@dace.program
def contract_rows(qr: dace.complex128[A_, B_, I_, J_], vc: dace.complex128[A_, B_], out: dace.complex128[A_, I_, J_]):
    out[:] = np.einsum('abij,ab->aij', qr, vc)


@dace.program
def contract_columns(bp: dace.complex128[A_, J_], aux: dace.complex128[A_, I_, J_], out: dace.complex128[A_, I_]):
    out[:] = np.einsum('aj,aij->ai', bp, aux)


def test_a_unit_matrix_extent_gets_the_stride_of_a_packed_matrix():
    """BLAS checks a leading dimension against the other extent even when one row is all there is."""
    assert packed_unit_extent([1, 5], [1, 1]) == [5, 1]
    assert packed_unit_extent([3, 1], [1, 1]) == [1, 3]
    assert packed_unit_extent([2, 3, 5], [15, 5, 1]) == [15, 5, 1]
    # A (M, K) operand of the einsum batch GEMM with both strides 1 and K = 1 only at run time.
    m, k = dace.symbol('M'), dace.symbol('K')
    assert packed_unit_extent([7, m, k], [m, 1, 1]) == [m, k, 1]


VENDOR_BLAS = [
    pytest.param('OpenBLAS',
                 marks=[
                     pytest.mark.openblas,
                     pytest.mark.skipif(not blas.environments.OpenBLAS.is_installed(), reason='OpenBLAS not installed')
                 ]),
    pytest.param('MKL',
                 marks=[
                     pytest.mark.mkl,
                     pytest.mark.skipif(not blas.environments.IntelMKL.is_installed(), reason='MKL not installed')
                 ]),
]


@pytest.mark.parametrize('impl', VENDOR_BLAS)
@pytest.mark.parametrize('program', [contract_rows, contract_columns])
def test_a_batched_product_of_a_unit_row_block_keeps_its_batch(impl, program):
    """``abij,ab->aij`` lowers to a batched product of an (A, 1, B) row block. The vendor expansions read
    it through the squeezed matrix view, as an (A, B) matrix every batch multiplied, and with
    ``lda = 1`` on the one row, which OpenBLAS refused without computing: npbench vexx_k's real-space
    augmentation was wrong on every canonicalize column."""
    a, b, i, j = 2, 27, 2, 3
    rng = np.random.default_rng(0)
    c = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if program is contract_rows:
        args = dict(qr=c(a, b, i, j), vc=c(a, b), out=np.zeros((a, i, j), np.complex128))
        expected = np.einsum('abij,ab->aij', args['qr'], args['vc'])
    else:
        args = dict(bp=c(a, j), aux=c(a, i, j), out=np.zeros((a, i), np.complex128))
        expected = np.einsum('aj,aij->ai', args['bp'], args['aux'])
    sdfg = program.to_sdfg()
    with change_default(blas, impl):
        sdfg.expand_library_nodes()
    sdfg(**args, A_=a, B_=b, I_=i, J_=j)
    assert np.allclose(args['out'], expected)


if __name__ == '__main__':
    for opt in (False, True):
        for sz in SIZES:
            test_reshape_unit_dim_matmul(opt, sz)
    test_reshape_unit_dim_stays_one_gemm()
    test_indexed_dim_still_reaches_gemm()
    test_unit_batch_keeps_alpha()

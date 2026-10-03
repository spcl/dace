# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GEMM whose operand BLAS cannot address takes the expansion that indexes it directly.

BLAS names a matrix by a pointer and one leading dimension, so it needs a unit stride on one of the
two matrix axes. cegterg's canonicalized ``Gemm`` read a view strided on both, the canonicalize perf
tail still chose OpenBLAS, and codegen raised ``sAM or sAK should be 1``.
"""
import numpy as np

import dace
from dace import dtypes
from dace.libraries.blas.nodes.gemm import Gemm
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations

N = dace.symbol('N')


def gemm(a_strides) -> tuple:
    """``C = A @ B`` with ``A`` laid out by ``a_strides``."""
    sdfg = dace.SDFG('gemm_strided_a')
    sdfg.add_array('A', [N, N], dace.float64, strides=a_strides)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_array('C', [N, N], dace.float64)
    state = sdfg.add_state()
    node = Gemm('gemm')
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet('A[0:N, 0:N]'))
    state.add_edge(state.add_read('B'), None, node, '_b', dace.Memlet('B[0:N, 0:N]'))
    state.add_edge(node, '_c', state.add_write('C'), None, dace.Memlet('C[0:N, 0:N]'))
    sdfg.validate()
    return sdfg, node


def test_an_operand_strided_on_both_axes_does_not_get_blas():
    sdfg, node = gemm([2 * N, 2])
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation in ('rowwise', 'pure'), node.implementation


def test_a_contiguous_operand_keeps_the_blas_call():
    sdfg, node = gemm([N, 1])
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation not in ('rowwise', 'pure'), node.implementation


def test_the_strided_gemm_computes_what_numpy_computes():
    sdfg, _ = gemm([2 * N, 2])
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    rng = np.random.default_rng(0)
    n = 16
    wide = rng.random((n, 2 * n))
    a = wide[:, ::2]
    b = rng.random((n, n))
    c = np.zeros((n, n))
    # ``a`` is the strided view the descriptor declares, which is the point of the test.
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        sdfg(A=a, B=b, C=c, N=n)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12, atol=0)


def nested_gemm(a_strides) -> tuple:
    """The same GEMM, one nesting level down, reached through a NestedSDFG."""
    inner = dace.SDFG('inner_gemm')
    inner.add_array('a', [N, N], dace.float64, strides=a_strides)
    inner.add_array('b', [N, N], dace.float64)
    inner.add_array('c', [N, N], dace.float64)
    inner_state = inner.add_state()
    node = Gemm('gemm')
    inner_state.add_node(node)
    inner_state.add_edge(inner_state.add_read('a'), None, node, '_a', dace.Memlet('a[0:N, 0:N]'))
    inner_state.add_edge(inner_state.add_read('b'), None, node, '_b', dace.Memlet('b[0:N, 0:N]'))
    inner_state.add_edge(node, '_c', inner_state.add_write('c'), None, dace.Memlet('c[0:N, 0:N]'))

    outer = dace.SDFG('outer_gemm')
    outer.add_array('A', [N, N], dace.float64, strides=a_strides)
    outer.add_array('B', [N, N], dace.float64)
    outer.add_array('C', [N, N], dace.float64)
    state = outer.add_state()
    nested = state.add_nested_sdfg(inner, {'a': None, 'b': None}, {'c': None})
    state.add_edge(state.add_read('A'), None, nested, 'a', dace.Memlet('A[0:N, 0:N]'))
    state.add_edge(state.add_read('B'), None, nested, 'b', dace.Memlet('B[0:N, 0:N]'))
    state.add_edge(nested, 'c', state.add_write('C'), None, dace.Memlet('C[0:N, 0:N]'))
    outer.validate()
    return outer, node


def test_a_nested_gemm_reads_its_own_arrays():
    """The operand names live in the nested SDFG that holds the ``Gemm``.

    Looking them up in the top-level SDFG raised ``KeyError: Data descriptor with name "__inl4_ps"
    not found in SDFG`` and took down the whole canonicalize run (cegterg and
    warpx_esirkepov_deposition, on both the CPU and the GPU column).
    """
    sdfg, node = nested_gemm([2 * N, 2])
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation in ('rowwise', 'pure'), node.implementation


def test_a_nested_contiguous_gemm_still_gets_blas():
    sdfg, node = nested_gemm([N, 1])
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    assert node.implementation not in ('rowwise', 'pure'), node.implementation

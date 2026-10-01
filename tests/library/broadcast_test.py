# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the :class:`Broadcast` library node: Fortran ``SPREAD`` and the right-aligned NumPy rule."""
import itertools

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import Broadcast

N = dace.symbol('N', dace.int64, nonnegative=True)

#: One compiled library per built SDFG: a reused name would load the previous case's library.
BUILD_IDS = itertools.count()


def build(src_shape, dst_shape, dim, src_dtype=dace.float64, dst_dtype=dace.float64, src_memlet=None, **desc):
    """One Broadcast from ``src`` to ``dst``; the memlets cover the whole arrays unless ``src_memlet`` is given."""
    sdfg = dace.SDFG(f'broadcast_{next(BUILD_IDS)}')
    sdfg.add_array('src', list(src_shape), src_dtype, **desc.get('src', {}))
    sdfg.add_array('dst', list(dst_shape), dst_dtype, **desc.get('dst', {}))
    state = sdfg.add_state()
    node = Broadcast('broadcast', dim=dim)
    state.add_node(node)
    state.add_edge(state.add_read('src'), None, node, '_src', src_memlet
                   or dace.Memlet.from_array('src', sdfg.arrays['src']))
    state.add_edge(node, '_dst', state.add_write('dst'), None, dace.Memlet.from_array('dst', sdfg.arrays['dst']))
    return sdfg


def spread(src, dim, ncopies):
    return np.repeat(np.expand_dims(src, dim - 1), ncopies, axis=dim - 1)


@pytest.mark.parametrize('src_shape, dim, ncopies', [((3, ), 1, 2), ((3, ), 2, 4), ((2, 3), 2, 5), ((1, ), 1, 4)])
def test_spread_inserts_the_axis_at_dim(src_shape, dim, ncopies):
    src = np.arange(1.0, np.prod(src_shape) + 1).reshape(src_shape).copy()
    expected = spread(src, dim, ncopies)
    dst = np.zeros(expected.shape)
    build(src_shape, expected.shape, dim)(src=src, dst=dst)
    np.testing.assert_array_equal(dst, expected)


def test_spread_of_a_fortran_scalar_fills_a_vector():
    """A Fortran scalar arrives as an array of one element; ``SPREAD(s, 1, n)`` has rank 1."""
    dst = np.zeros(5)
    build((1, ), (5, ), 1)(src=np.array([2.5]), dst=dst)
    np.testing.assert_array_equal(dst, np.full(5, 2.5))


@pytest.mark.parametrize('src_shape, dst_shape', [
    ((3, ), (2, 3)),
    ((3, 1), (3, 4)),
    ((1, 4), (3, 4)),
    ((1, ), (2, 5)),
    ((2, 3), (4, 2, 3)),
])
def test_numpy_rule_agrees_with_numpy(src_shape, dst_shape):
    src = np.arange(np.prod(src_shape), dtype=np.float64).reshape(src_shape).copy()
    dst = np.zeros(dst_shape)
    build(src_shape, dst_shape, None)(src=src, dst=dst)
    np.testing.assert_array_equal(dst, np.broadcast_to(src, dst_shape))


@pytest.mark.parametrize('src_shape, dst_shape, dim, message', [
    ((3, ), (2, 5), None, 'cannot broadcast'),
    ((3, 2), (4, 3, 2, 2), 1, 'adds one axis'),
    ((3, ), (3, 3), 3, 'out of range'),
])
def test_a_shape_that_cannot_broadcast_is_refused_before_expansion(src_shape, dst_shape, dim, message):
    with pytest.raises(ValueError, match=message):
        sdfg = build(src_shape, dst_shape, dim)
        (state, ) = sdfg.states()
        next(n for n in state.nodes() if isinstance(n, Broadcast)).validate(sdfg, state)


def test_symbolic_extents_expand_and_run():
    sdfg = build((N, ), (3, N), 1)
    sdfg.expand_library_nodes()
    src = np.arange(6.0)
    dst = np.zeros((3, 6))
    sdfg(src=src, dst=dst, N=6)
    np.testing.assert_array_equal(dst, spread(src, 1, 3))


def test_operands_are_read_and_written_by_their_own_strides():
    """Column-major operands: the expanded connectors keep the strides, so no packed C layout is assumed."""
    sdfg = build((3, 2), (4, 3, 2), None, src={'strides': (1, 3)}, dst={'strides': (1, 4, 12)})
    sdfg.expand_library_nodes()
    inner = next(n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert tuple(inner.arrays['_src'].strides) == (1, 3)
    assert tuple(inner.arrays['_dst'].strides) == (1, 4, 12)
    src = np.asfortranarray(np.arange(6.0).reshape(3, 2).copy())
    dst = np.zeros((4, 3, 2), order='F')
    sdfg(src=src, dst=dst)
    np.testing.assert_array_equal(dst, np.broadcast_to(src, (4, 3, 2)))


def test_a_sliced_source_with_a_lower_bound_offset():
    """A descriptor offset of -1 puts the elements the memlet ``src[1:4]`` selects at ``src[0:3]``."""
    sdfg = build((6, ), (2, 3), 1, src={'offset': [-1]}, src_memlet=dace.Memlet('src[1:4]'))
    src = np.arange(6.0)
    dst = np.zeros((2, 3))
    sdfg(src=src, dst=dst)
    np.testing.assert_array_equal(dst, spread(src[0:3], 1, 2))


def test_the_destination_type_converts():
    dst = np.zeros((2, 3), dtype=np.float32)
    build((3, ), (2, 3), 1, src_dtype=dace.int32, dst_dtype=dace.float32)(src=np.arange(3, dtype=np.int32), dst=dst)
    np.testing.assert_array_equal(dst, spread(np.arange(3, dtype=np.float32), 1, 2))


def test_broadcast_to_in_a_program():

    @dace.program
    def program(a: dace.float64[3, 1], out: dace.float64[3, 4]):
        out[:] = np.broadcast_to(a, (3, 4))

    assert len([n for n, _ in program.to_sdfg(simplify=False).all_nodes_recursive() if isinstance(n, Broadcast)]) == 1
    a = np.random.randn(3, 1)
    out = np.zeros((3, 4))
    program(a=a, out=out)
    np.testing.assert_array_equal(out, np.broadcast_to(a, (3, 4)))


def test_the_library_call_in_a_program_keeps_its_dim():

    @dace.program
    def program(a: dace.float64[3], out: dace.float64[3, 4]):
        dace.libraries.standard.broadcast(a, out, dim=2)

    assert [n.dim for n, _ in program.to_sdfg(simplify=False).all_nodes_recursive() if isinstance(n, Broadcast)] == [2]
    a = np.arange(3.0)
    out = np.zeros((3, 4))
    program(a=a, out=out)
    np.testing.assert_array_equal(out, spread(a, 2, 4))

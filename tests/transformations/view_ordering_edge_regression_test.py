# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regressions for passes that treated a View as storage or an ordering edge as dataflow."""
import warnings

import numpy as np
import pytest

import dace
from dace import Memlet, data
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation.dataflow import (MapFission, RedundantArrayCopying, RedundantArrayCopyingIn, RemoveSliceView)


def count_empty_edges(state: dace.SDFGState) -> int:
    """Number of ordering (empty-memlet) edges in a state."""
    return sum(1 for e in state.edges() if e.data.is_empty())


def scale_row_through_view() -> dace.SDFG:
    """``v = A[1, :]; v[:] = v * 2.0`` -- a write-through view chain, unsimplified."""

    @dace.program
    def scale_row(A: dace.float64[2, 2]):
        v = A[1, :]
        v[:] = v * 2.0

    sdfg = scale_row.to_sdfg(simplify=False)
    sdfg.validate()
    return sdfg


def test_get_view_edge_ignores_ordering_out_edges() -> None:
    """An empty out-edge must not make a well-formed view look ambiguous (dace/sdfg/utils.py)."""
    sdfg = dace.SDFG('view_with_ordering_out_edge')
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_view('V', [2], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    view = state.add_access('V')
    write = state.add_access('A')
    state.add_edge(view, 'views', write, None, Memlet('A[1, 0:2]'))
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', view, None, Memlet('V[0]'))
    other = state.add_access('B')
    state.add_edge(view, None, other, None, Memlet())  # ordering edge, carries no binding

    edge = sdutil.get_view_edge(state, view)
    assert edge is not None
    assert edge.src is view and edge.dst is write
    sdfg.validate()


def test_partial_view_write_survives_simplify() -> None:
    """The ordering out-edges of a partially written view used to make simplify() reject it."""

    @dace.program
    def partial_view_write(A: dace.float64[2, 2], B: dace.float64[2]):
        v = A[1, :]
        v[0] = B[0]
        v[1] = B[1]

    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    b = np.array([100.0, 200.0])
    expected = a.copy()
    expected[1, :] = b
    got = a.copy()
    partial_view_write(got, b)
    assert np.allclose(got, expected), got


def test_redundant_array_copying_in_declines_a_view_chain() -> None:
    """The A -> B -> C fold deleted the binding edge of a View, losing the write-through."""
    sdfg = scale_row_through_view()
    assert sdfg.apply_transformations_repeated(RedundantArrayCopyingIn, validate=False, validate_all=False) == 0
    sdfg.validate()

    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    expected = a.copy()
    expected[1, :] *= 2.0
    got = a.copy()
    sdfg(A=got)
    assert np.allclose(got, expected), got


def test_redundant_array_copying_declines_a_view_chain() -> None:
    """Folding the chain moved the 'views' connector onto a node that views nothing."""
    sdfg = scale_row_through_view()
    assert sdfg.apply_transformations_repeated(RedundantArrayCopying, validate=False, validate_all=False) == 0
    for state in sdfg.states():
        for edge in state.edges():
            if edge.src_conn == 'views':
                assert isinstance(sdfg.arrays[edge.src.data], data.View), edge.src.data
            if edge.dst_conn == 'views':
                assert isinstance(sdfg.arrays[edge.dst.data], data.View), edge.dst.data

    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    expected = a.copy()
    expected[1, :] *= 2.0
    got = a.copy()
    sdfg(A=got)
    assert np.allclose(got, expected), got


def build_write_view_with_ordering_out_edge() -> dace.SDFG:
    """Write view ``V -> A[0:1]``, plus an ordering edge putting the read of ``A`` after it."""
    sdfg = dace.SDFG('slice_view_write_ordering')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_view('V', [1], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    # The reader is inserted first so it wins the insertion-order tie-break once nothing orders it
    reader = state.add_access('A')
    copy_tasklet = state.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    state.add_edge(reader, None, copy_tasklet, 'a', Memlet('A[0]'))
    state.add_edge(copy_tasklet, 'o', state.add_access('B'), None, Memlet('B[0]'))
    init = state.add_tasklet('init', {}, {'o'}, 'o = 5.0')
    view = state.add_access('V')
    state.add_edge(init, 'o', view, None, Memlet('V[0]'))
    state.add_edge(view, 'views', state.add_access('A'), None, Memlet('A[0:1]'))
    state.add_edge(view, None, reader, None, Memlet())
    sdfg.validate()
    return sdfg


def build_read_view_with_ordering_in_edge() -> dace.SDFG:
    """Read view ``A[0:1] -> V``, plus an ordering edge putting the write to ``A`` before it."""
    sdfg = dace.SDFG('slice_view_read_ordering')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [2], dace.float64)
    sdfg.add_view('V', [1], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    view = state.add_access('V')
    state.add_edge(state.add_access('A'), None, view, 'views', Memlet('A[0:1]'))
    copy_tasklet = state.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    state.add_edge(view, None, copy_tasklet, 'a', Memlet('V[0]'))
    state.add_edge(copy_tasklet, 'o', state.add_access('B'), None, Memlet('B[0]'))
    init = state.add_tasklet('init', {}, {'o'}, 'o = 5.0')
    write = state.add_access('A')
    state.add_edge(init, 'o', write, None, Memlet('A[0]'))
    state.add_edge(write, None, view, None, Memlet())
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('builder', [build_write_view_with_ordering_out_edge, build_read_view_with_ordering_in_edge])
def test_remove_slice_view_keeps_the_ordering_edge(builder) -> None:
    """RemoveSliceView rewired one side only, so the ordering edge died with the node."""
    sdfg = builder()
    state = sdfg.states()[0]
    assert count_empty_edges(state) == 1
    assert sdfg.apply_transformations(RemoveSliceView, validate=False, validate_all=False) == 1
    assert not any(isinstance(n, nodes.AccessNode) and n.data == 'V' for n in state.nodes())
    assert count_empty_edges(state) >= 1
    sdfg.validate()

    a = np.zeros(2)
    b = np.zeros(2)
    sdfg(A=a, B=b)
    assert a[0] == 5.0, a
    assert b[0] == 5.0, b


def build_fission_border_view(viewed_is_border_transient: bool) -> dace.SDFG:
    """Outer map ``i``; component 1 fills ``T``, a View ``V`` of ``T`` feeds component 2."""
    sdfg = dace.SDFG('fission_border_view_' + ('t' if viewed_is_border_transient else 'nt'))
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_array('B', [2, 2], dace.float64)
    if viewed_is_border_transient:
        sdfg.add_transient('T', [2], dace.float64)
    else:
        sdfg.add_array('T', [2], dace.float64)
    sdfg.add_view('V', [2], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = state.add_map('outer', dict(i='0:2'))
    viewed = state.add_access('T')
    view = state.add_access('V')

    first_entry, first_exit = state.add_map('c1', dict(j='0:2'))
    add_one = state.add_tasklet('add_one', {'a'}, {'o'}, 'o = a + 1.0')
    state.add_memlet_path(state.add_access('A'), entry, first_entry, add_one, memlet=Memlet('A[i, j]'), dst_conn='a')
    state.add_memlet_path(add_one, first_exit, viewed, memlet=Memlet('T[j]'), src_conn='o')
    state.add_edge(viewed, None, view, 'views', Memlet('T[0:2]'))

    second_entry, second_exit = state.add_map('c2', dict(k='0:2'))
    double = state.add_tasklet('double', {'x'}, {'o'}, 'o = x * 2.0')
    state.add_memlet_path(view, second_entry, double, memlet=Memlet('V[k]'), dst_conn='x')
    state.add_memlet_path(double, second_exit, exit_node, state.add_access('B'), memlet=Memlet('B[i, k]'), src_conn='o')
    sdfg.validate()
    return sdfg


def test_map_fission_declines_a_view_of_non_border_data() -> None:
    """Nothing may widen data the scope does not own, and one descriptor holds one binding."""
    sdfg = build_fission_border_view(viewed_is_border_transient=False)
    assert sdfg.apply_transformations_repeated(MapFission, validate=False, validate_all=False) == 0
    assert list(sdfg.arrays['V'].shape) == [2]
    sdfg.validate()

    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    b = np.zeros((2, 2))
    sdfg(A=a.copy(), B=b, T=np.zeros(2))
    assert np.allclose(b, (a + 1.0) * 2.0), b


def test_map_fission_does_not_push_a_view_into_a_nested_sdfg() -> None:
    """The nested SDFG connector took the outer View descriptor, which has no binding inside."""

    @dace.program
    def accumulate_into_view(A: dace.float64[2, 2], B: dace.float64[2]):
        v = A[1, :]
        for i in dace.map[0:2]:
            v[0] += B[i]

    sdfg = accumulate_into_view.to_sdfg(simplify=False)
    assert sdfg.apply_transformations(MapFission, validate=False, validate_all=False) == 1
    sdfg.validate()
    for nsdfg in sdfg.all_sdfgs_recursive():
        for name, desc in nsdfg.arrays.items():
            assert desc.transient or not isinstance(desc, data.View), f'{nsdfg.label}.{name}'

    a = np.arange(4, dtype=np.float64).reshape(2, 2)
    b = np.arange(1, 3, dtype=np.float64)
    expected = a.copy()
    expected[1, 0] += b.sum()
    got = a.copy()
    sdfg(A=got, B=b)
    assert np.allclose(got, expected), got


def test_redundant_array_race_guard_skips_ordering_edges() -> None:
    """The data-race guards fed ordering memlets to _validate_subsets, which warned and refused."""

    @dace.program
    def partial_view_write(A: dace.float64[2, 2], B: dace.float64[2]):
        v = A[1, :]
        v[0] = B[0]
        v[1] = B[1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sdfg = partial_view_write.to_sdfg(simplify=True)
    offenders = [str(w.message) for w in caught if 'validate_subsets failed' in str(w.message)]
    assert not offenders, offenders
    sdfg.validate()


if __name__ == '__main__':
    pytest.main([__file__])

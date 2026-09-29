# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``dace.sdfg.utils.convert_to_view``, which turns a data container into a view of another one."""
import numpy as np

import dace
from dace import data as dt, subsets
from dace.sdfg import nodes, utils as sdutil

N = 16


def _byte_buffer(sdfg: dace.SDFG, name: str = 'buf', size: int = 2 * N * 8) -> str:
    """Adds a transient byte buffer that other containers can be turned into views of."""
    sdfg.add_array(name, [size], dace.uint8, transient=True)
    return name


def _assert_views_resolve(sdfg: dace.SDFG, name: str, viewed: str) -> None:
    """Asserts that every access node of ``name`` has an unambiguous view edge to ``viewed``."""
    count = 0
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and node.data == name:
            edge = sdutil.get_view_edge(state, node)
            assert edge is not None, f'No view edge for {node} in {state}'
            other = edge.src if edge.dst is node else edge.dst
            assert isinstance(other, nodes.AccessNode) and other.data == viewed
            count += 1
    assert count > 0


def _two_state_sdfg() -> dace.SDFG:
    """``tmp = 2 * A`` in one state, ``B = tmp + 1`` in the next: ``tmp`` is written in one and read in the other."""
    sdfg = dace.SDFG('convert_to_view_two_states')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('tmp', [N], dace.float64, transient=True)
    first = sdfg.add_state()
    first.add_mapped_tasklet('double', {'i': f'0:{N}'}, {'a': dace.Memlet('A[i]')},
                             't = 2 * a', {'t': dace.Memlet('tmp[i]')},
                             external_edges=True)
    second = sdfg.add_state_after(first)
    second.add_mapped_tasklet('increment', {'i': f'0:{N}'}, {'t': dace.Memlet('tmp[i]')},
                              'b = t + 1', {'b': dace.Memlet('B[i]')},
                              external_edges=True)
    return sdfg


def _run(sdfg: dace.SDFG) -> np.ndarray:
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    sdfg(A=A, B=B)
    return B


def test_read_only_and_write_only_nodes():
    sdfg = _two_state_sdfg()
    expected = _run(sdfg)

    buf = _byte_buffer(sdfg)
    view = sdutil.convert_to_view(sdfg, 'tmp', buf, subsets.Range.from_string(f'{N * 8}:{2 * N * 8}'))
    assert isinstance(view, dt.View) and sdfg.arrays['tmp'] is view
    assert view.dtype == dace.float64 and tuple(view.shape) == (N, )
    sdfg.validate()
    _assert_views_resolve(sdfg, 'tmp', buf)

    assert np.array_equal(_run(sdfg), expected)


def test_node_both_written_and_read_is_split():
    sdfg = dace.SDFG('convert_to_view_split')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('tmp', [N], dace.float64, transient=True)
    state = sdfg.add_state()
    tmp = state.add_access('tmp')
    state.add_mapped_tasklet('double', {'i': f'0:{N}'}, {'a': dace.Memlet('A[i]')},
                             't = 2 * a', {'t': dace.Memlet('tmp[i]')},
                             external_edges=True,
                             output_nodes={'tmp': tmp})
    state.add_mapped_tasklet('increment', {'i': f'0:{N}'}, {'t': dace.Memlet('tmp[i]')},
                             'b = t + 1', {'b': dace.Memlet('B[i]')},
                             external_edges=True,
                             input_nodes={'tmp': tmp})
    expected = _run(sdfg)

    buf = _byte_buffer(sdfg)
    sdutil.convert_to_view(sdfg, 'tmp', buf, subsets.Range.from_string(f'0:{N * 8}'))
    sdfg.validate()
    _assert_views_resolve(sdfg, 'tmp', buf)
    # The written view precedes the buffer, which precedes the read view
    tmp_nodes = [n for n in state.data_nodes() if n.data == 'tmp']
    assert len(tmp_nodes) == 2
    written = next(n for n in tmp_nodes if any(e.src_conn == 'views' for e in state.out_edges(n)))
    read = next(n for n in tmp_nodes if any(e.dst_conn == 'views' for e in state.in_edges(n)))
    assert written is not read
    assert state.out_edges(written)[0].dst.data == buf
    assert state.in_edges(read)[0].src.data == buf

    assert np.array_equal(_run(sdfg), expected)


def test_nodes_inside_a_map_scope():
    """A per-iteration temporary inside a sequential map, viewed through a buffer inside the same scope."""
    sdfg = dace.SDFG('convert_to_view_in_scope')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('tmp', [1], dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': f'0:{N}'}, schedule=dace.ScheduleType.Sequential)
    double = state.add_tasklet('double', {'a'}, {'t'}, 't = 2 * a')
    increment = state.add_tasklet('increment', {'t'}, {'b'}, 'b = t + 1')
    tmp_write, tmp_read = state.add_access('tmp'), state.add_access('tmp')
    state.add_memlet_path(state.add_read('A'), me, double, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_edge(double, 't', tmp_write, None, dace.Memlet('tmp[0]'))
    state.add_nedge(tmp_write, tmp_read, dace.Memlet())
    state.add_edge(tmp_read, None, increment, 't', dace.Memlet('tmp[0]'))
    state.add_memlet_path(increment, mx, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    expected = _run(sdfg)

    buf = _byte_buffer(sdfg, size=8)
    sdutil.convert_to_view(sdfg, 'tmp', buf, subsets.Range.from_string('0:8'))
    sdfg.validate()
    _assert_views_resolve(sdfg, 'tmp', buf)
    scope = state.scope_dict()
    assert all(scope[n] is me for n in state.data_nodes() if n.data in ('tmp', buf))

    assert np.array_equal(_run(sdfg), expected)


def test_two_views_of_one_buffer():
    sdfg = _two_state_sdfg()
    sdfg.add_array('tmp2', [N], dace.float64, transient=True)
    # B = tmp + 1 becomes tmp2 = tmp + 1; B = tmp2 * 3
    second = next(s for s in sdfg.states() if any(isinstance(n, nodes.AccessNode) and n.data == 'B' for n in s.nodes()))
    for node in second.data_nodes():
        if node.data == 'B':
            node.data = 'tmp2'
    for e in second.edges():
        if e.data.data == 'B':
            e.data.data = 'tmp2'
    third = sdfg.add_state_after(second)
    third.add_mapped_tasklet('triple', {'i': f'0:{N}'}, {'t': dace.Memlet('tmp2[i]')},
                             'b = 3 * t', {'b': dace.Memlet('B[i]')},
                             external_edges=True)
    expected = _run(sdfg)

    buf = _byte_buffer(sdfg)
    sdutil.convert_to_view(sdfg, 'tmp', buf, subsets.Range.from_string(f'0:{N * 8}'))
    sdutil.convert_to_view(sdfg, 'tmp2', buf, subsets.Range.from_string(f'{N * 8}:{2 * N * 8}'))
    sdfg.validate()
    _assert_views_resolve(sdfg, 'tmp', buf)
    _assert_views_resolve(sdfg, 'tmp2', buf)

    assert np.array_equal(_run(sdfg), expected)


if __name__ == '__main__':
    test_read_only_and_write_only_nodes()
    test_node_both_written_and_read_is_split()
    test_nodes_inside_a_map_scope()
    test_two_views_of_one_buffer()

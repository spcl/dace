# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the schedule-tree map-to-loop conversion pass."""
import numpy as np

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import convert_map_to_loop

H = dace.symbol('H')
W = dace.symbol('W')
nnz = dace.symbol('nnz')


def _convert(stree: tn.ScheduleTreeRoot) -> int:
    count = convert_map_to_loop(stree)
    tn.validate_children_and_parents_align(stree, root=True)
    return count


def _nodes(stree: tn.ScheduleTreeRoot, kind: type) -> list:
    return [n for n in stree.preorder_traversal() if isinstance(n, kind)]


def _headers(stree: tn.ScheduleTreeRoot) -> list:
    return [(f.loop.init_statement.as_string, f.loop.loop_condition.as_string, f.loop.update_statement.as_string)
            for f in _nodes(stree, tn.ForScope)]


def _run(stree: tn.ScheduleTreeRoot, **kwargs):
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))
    return sdfg(**kwargs)


def test_one_dimensional_map():

    @dace.program
    def scale(A: dace.float64[20], B: dace.float64[20]):
        for i in dace.map[0:20]:
            B[i] = A[i] * 2.0

    stree = scale.to_sdfg().as_schedule_tree()
    assert _convert(stree) == 1
    assert not _nodes(stree, tn.MapScope)
    assert _headers(stree) == [('i = 0', '(i < 20)', 'i = (i + 1)')]

    a = np.random.rand(20)
    b = np.zeros(20)
    _run(stree, A=a, B=b)
    assert np.allclose(b, a * 2.0)


def test_multidimensional_map():

    @dace.program
    def fill(A: dace.float64[10, 20]):
        for i, j in dace.map[0:10, 1:20:2]:
            A[i, j] = i * 100 + j

    stree = fill.to_sdfg().as_schedule_tree()
    assert _convert(stree) == 1

    # One loop per dimension, nested in map parameter order
    outer, inner = _nodes(stree, tn.ForScope)
    assert inner.parent is outer
    assert _headers(stree) == [('i = 0', '(i < 10)', 'i = (i + 1)'), ('j = 1', '(j < 20)', 'j = (j + 2)')]

    a = np.zeros((10, 20))
    _run(stree, A=a)
    expected = np.zeros((10, 20))
    for i in range(10):
        for j in range(1, 20, 2):
            expected[i, j] = i * 100 + j
    assert np.allclose(a, expected)


def test_negative_step():
    sdfg = dace.SDFG('negative_step_map')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)
    state = sdfg.add_state()
    _, map_entry, _ = state.add_mapped_tasklet('reverse', {'i': '0:10'}, {'a': dace.Memlet('A[i]')},
                                               'b = a + 1', {'b': dace.Memlet('B[i]')},
                                               external_edges=True)
    map_entry.map.range = dace.subsets.Range([(9, 0, -1)])

    stree = sdfg.as_schedule_tree()
    assert _convert(stree) == 1
    assert _headers(stree) == [('i = 9', '(i > (- 1))', 'i = (i - 1)')]

    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b)
    assert np.allclose(b, a + 1)


def test_nested_maps_and_write_conflict_resolution():

    @dace.program
    def rowsum(A: dace.float64[H, W], out: dace.float64[H]):
        for i in dace.map[0:H]:
            for j in dace.map[0:W]:
                out[i] += A[i, j]

    stree = rowsum.to_sdfg().as_schedule_tree()
    assert _convert(stree) == 2
    assert not _nodes(stree, tn.MapScope)
    outer, inner = _nodes(stree, tn.ForScope)
    assert inner.parent is outer

    a = np.random.rand(5, 7)
    out = np.zeros(5)
    _run(stree, A=a, out=out, H=5, W=7)
    assert np.allclose(out, a.sum(axis=1))


@dace.program
def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
         b: dace.float32[H]):
    for i in dace.map[0:H]:
        for j in dace.map[A_row[i]:A_row[i + 1]]:
            b[i] += A_val[j] * x[A_col[j]]


def _spmv_inputs():
    rng = np.random.default_rng(42)
    dense = rng.random((8, 6), dtype=np.float32) * (rng.random((8, 6)) < 0.4)
    rows, cols = np.nonzero(dense)
    a_row = np.searchsorted(rows, np.arange(9)).astype(np.uint32)
    a_col = cols.astype(np.uint32)
    a_val = dense[rows, cols].astype(np.float32)
    x = rng.random(6, dtype=np.float32)
    return dense, dict(A_row=a_row, A_col=a_col, A_val=a_val, x=x, H=8, W=6, nnz=len(a_val))


def test_dynamic_map_range():
    stree = spmv.to_sdfg().as_schedule_tree()
    assert len(_nodes(stree, tn.DynScopeCopyNode)) == 2
    assert _convert(stree) == 2
    assert not _nodes(stree, tn.MapScope)
    assert not _nodes(stree, tn.DynScopeCopyNode)

    # The dynamic inputs are read once, right before the loop they bound
    outer, inner = _nodes(stree, tn.ForScope)
    start, end, loop = outer.children[:3]
    assert isinstance(start, tn.AssignNode) and start.value.as_string == 'A_row[i]'
    assert isinstance(end, tn.AssignNode) and end.value.as_string == 'A_row[(i + 1)]'
    assert loop is inner
    assert inner.loop.init_statement.as_string == f'j = {start.name}'
    assert inner.loop.loop_condition.as_string == f'(j < {end.name})'
    assert stree.symbols[start.name] == dace.uint32
    assert stree.symbols[end.name] == dace.uint32

    dense, args = _spmv_inputs()
    b = np.zeros(8, dtype=np.float32)
    _run(stree, b=b, **args)
    assert np.allclose(b, dense @ args['x'], rtol=1e-5)


def test_dynamic_map_range_from_scalar():
    sdfg = dace.SDFG('dynamic_scalar_range')
    sdfg.add_scalar('size', dace.int32)
    sdfg.add_array('A', [10], dace.float64)
    state = sdfg.add_state()
    map_entry, map_exit = state.add_map('fill', {'i': '0:ub'})
    map_entry.add_in_connector('ub')
    state.add_edge(state.add_read('size'), None, map_entry, 'ub', dace.Memlet('size', dynamic=True))
    tasklet = state.add_tasklet('one', {}, {'o'}, 'o = 1')
    state.add_nedge(map_entry, tasklet, dace.Memlet())
    state.add_memlet_path(tasklet, map_exit, state.add_write('A'), src_conn='o', memlet=dace.Memlet('A[i]'))

    stree = sdfg.as_schedule_tree()
    assert _convert(stree) == 1
    assign, loop = stree.children
    assert isinstance(assign, tn.AssignNode) and assign.name == 'ub'
    assert assign.value.as_string == 'size'  # Scalars are read without an index
    assert isinstance(loop, tn.ForScope)

    a = np.zeros(10)
    _run(stree, size=np.int32(6), A=a)
    assert np.allclose(a, [1] * 6 + [0] * 4)


def test_dynamic_input_name_clash_keeps_map():
    stree = spmv.to_sdfg().as_schedule_tree()
    dscopy = _nodes(stree, tn.DynScopeCopyNode)[0]
    stree.symbols[dscopy.target] = dace.int32  # An existing symbol would be overwritten by the assignment

    # The outer map is converted, the one with dynamic inputs is kept together with its inputs
    assert _convert(stree) == 1
    outer, = _nodes(stree, tn.ForScope)
    assert [type(c) for c in outer.children] == [tn.DynScopeCopyNode, tn.DynScopeCopyNode, tn.MapScope]


if __name__ == '__main__':
    test_one_dimensional_map()
    test_multidimensional_map()
    test_negative_step()
    test_nested_maps_and_write_conflict_resolution()
    test_dynamic_map_range()
    test_dynamic_map_range_from_scalar()
    test_dynamic_input_name_clash_keeps_map()

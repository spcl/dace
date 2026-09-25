# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for fusing coarse loops for data reuse in schedule trees."""
import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import fuse_loops_for_reuse

NI, NJ, NK = 4, 4, 16


def _tasklet(code: str, inputs: dict, outputs: dict) -> tn.TaskletNode:
    tasklet = dace.nodes.Tasklet('t', {c: None for c in inputs}, {c: None for c in outputs}, code)
    return tn.TaskletNode(node=tasklet,
                          in_memlets={
                              c: dace.Memlet(m)
                              for c, m in inputs.items()
                          },
                          out_memlets={
                              c: dace.Memlet(m)
                              for c, m in outputs.items()
                          })


def _plane(body: list, name: str) -> tn.ForScope:
    """``for j in range(NJ): for i in range(NI): <body>``."""
    inner = dace.sdfg.state.LoopRegion(f'{name}_i', f'i < {NI}', 'i', 'i = 0', 'i = i + 1')
    outer = dace.sdfg.state.LoopRegion(f'{name}_j', f'j < {NJ}', 'j', 'j = 0', 'j = j + 1')
    return tn.ForScope(loop=outer, children=[tn.ForScope(loop=inner, children=body)])


def _tree(loops: list) -> tn.ScheduleTreeRoot:
    """Vertical loops ``for k in range(first, last + 1): <plane>``, one per ``(first, last, body)``, over arrays
    ``A`` to ``D`` of shape (NI, NJ, NK), a view ``Bv`` of ``B`` and a transient scalar ``t``."""
    sdfg = dace.SDFG('fusion')
    for name in 'ABCD':
        sdfg.add_array(name, [NI, NJ, NK], dace.float64)
    sdfg.add_scalar('t', dace.float64, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    children = []
    for n, (first, last, body, var) in enumerate(loops):
        loop = dace.sdfg.state.LoopRegion(f'vertical{n}', f'{var} < {last + 1}', var, f'{var} = {first}',
                                          f'{var} = {var} + 1')
        children.append(tn.ForScope(loop=loop, children=[_plane(body, f'plane{n}')]))
    stree.children = []
    stree.add_children(children)
    return stree


def _check(make, fused: int, cache_bytes: int = 1024) -> list:
    stree, reference = make(), make()
    log = []
    assert fuse_loops_for_reuse(stree, cache_bytes=cache_bytes, log=log) == fused, log
    tn.validate_children_and_parents_align(stree, root=True)
    rng = np.random.default_rng(0)
    inputs = {name: rng.random((NI, NJ, NK)) for name in 'ABCD'}
    results = []
    for tree in (reference, stree):
        arrays = {name: value.copy() for name, value in inputs.items()}
        tree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))(**arrays)
        results.append(arrays)
    for name in 'ABCD':
        assert np.array_equal(results[0][name], results[1][name]), name
    return log


def _producer_consumer(read: str, var: str = 'k', first: int = 1, last: int = NK - 1):
    return lambda: _tree([(0, NK - 1, [_tasklet('b = 2 * a', {'a': 'A[i, j, k]'}, {'b': 'B[i, j, k]'})], 'k'),
                          (first, last, [_tasklet('c = b + 1', {'b': read}, {'c': f'C[i, j, {var}]'})], var)])


@pytest.mark.parametrize('read', ['B[i, j, k]', 'B[i, j, k - 1]'])
def test_fuse_producer_consumer(read):
    _check(_producer_consumer(read), fused=1)


def test_fuse_renames_loop_variable_and_guards_range():
    make = _producer_consumer('B[i, j, kk - 1]', var='kk', first=2)
    stree = make()
    assert fuse_loops_for_reuse(stree, cache_bytes=1024) == 1
    assert len([n for n in stree.preorder_traversal() if isinstance(n, tn.IfScope)]) == 1
    _check(make, fused=1)


def test_fuse_not_forward_read():
    """The second loop reads ``B[k + 1]``, which the first computes only in its next iteration."""
    log = _check(_producer_consumer('B[i, j, k + 1]', first=0, last=NK - 2), fused=0)
    assert 'B' in log[0]


def test_fuse_not_anti_dependence():
    """The first loop reads ``A[k - 1]``, which the second overwrites in the previous iteration."""
    make = lambda: _tree([(1, NK - 1, [_tasklet('b = a', {'a': 'A[i, j, k - 1]'}, {'b': 'B[i, j, k]'})], 'k'),
                          (0, NK - 1, [_tasklet('o = b + 1', {'b': 'B[i, j, k]'}, {'o': 'A[i, j, k]'})], 'k')])
    log = _check(make, fused=0)
    assert 'A' in log[0]


def _scalar_bodies(second_writes_first: bool):

    def make():
        first = [
            _tasklet('o = 2 * a', {'a': 'A[i, j, k]'}, {'o': 't[0]'}),
            _tasklet('b = x', {'x': 't[0]'}, {'b': 'B[i, j, k]'})
        ]
        second = [_tasklet('c = x + b', {'x': 't[0]', 'b': 'B[i, j, k]'}, {'c': 'C[i, j, k]'})]
        if second_writes_first:
            second.insert(0, _tasklet('o = 3 * b', {'b': 'B[i, j, k]'}, {'o': 't[0]'}))
        return _tree([(0, NK - 1, first, 'k'), (0, NK - 1, second, 'k')])

    return make


def test_fuse_with_shared_scalar_temporary():
    _check(_scalar_bodies(second_writes_first=True), fused=1)


def test_fuse_not_scalar_carried_between_loops():
    """The second loop reads the value of ``t`` the first loop left."""
    log = _check(_scalar_bodies(second_writes_first=False), fused=0)
    assert 't' in log[0]


def test_fuse_not_without_shared_data():
    make = lambda: _tree([(0, NK - 1, [_tasklet('b = a', {'a': 'A[i, j, k]'}, {'b': 'B[i, j, k]'})], 'k'),
                          (0, NK - 1, [_tasklet('d = c', {'c': 'C[i, j, k]'}, {'d': 'D[i, j, k]'})], 'k')])
    log = _check(make, fused=0)
    assert 'no shared data' in log[0]


def test_fuse_not_when_data_stays_in_cache():
    log = _check(_producer_consumer('B[i, j, k]'), fused=0, cache_bytes=1 << 20)
    assert 'stays in cache' in log[0]


def test_fuse_not_when_iteration_overflows_cache():
    log = _check(_producer_consumer('B[i, j, k]'), fused=0, cache_bytes=256)
    assert 'exceeds' in log[0]


def test_fuse_not_through_view():
    """The second loop reads what the first writes through a view, whose indices are not analyzed."""

    def make():
        stree = _producer_consumer('Bv[i, j, k]', first=0)()
        stree.containers['Bv'] = dace.data.ArrayView(dace.float64, [NI, NJ, NK], transient=True)
        view = tn.ViewNode(target='Bv',
                           source='B',
                           memlet=dace.Memlet.from_array('B', stree.containers['B']),
                           src_desc=stree.containers['B'],
                           view_desc=stree.containers['Bv'])
        children = list(stree.children)
        stree.children = []
        stree.add_children([view] + children)
        return stree

    log = _check(make, fused=0)
    assert 'B' in log[0]


def test_fuse_not_innermost_loops():
    """Loops whose bodies contain no loops are left to the vectorizer."""
    sdfg = dace.SDFG('innermost')
    sdfg.add_array('A', [NK], dace.float64)
    sdfg.add_array('B', [NK], dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    loops = []
    for n, (inp, out) in enumerate([('A', 'B'), ('B', 'A')]):
        loop = dace.sdfg.state.LoopRegion(f'loop{n}', f'k < {NK}', 'k', 'k = 0', 'k = k + 1')
        loops.append(tn.ForScope(loop=loop, children=[_tasklet('b = a + 1', {'a': f'{inp}[k]'}, {'b': f'{out}[k]'})]))
    stree.children = []
    stree.add_children(loops)
    assert fuse_loops_for_reuse(stree, cache_bytes=16) == 0


if __name__ == '__main__':
    test_fuse_producer_consumer('B[i, j, k - 1]')
    test_fuse_renames_loop_variable_and_guards_range()
    test_fuse_not_forward_read()
    test_fuse_not_anti_dependence()
    test_fuse_with_shared_scalar_temporary()
    test_fuse_not_scalar_carried_between_loops()
    test_fuse_not_without_shared_data()
    test_fuse_not_when_data_stays_in_cache()
    test_fuse_not_when_iteration_overflows_cache()
    test_fuse_not_through_view()
    test_fuse_not_innermost_loops()

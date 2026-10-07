# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for reusing the memory of transients and moving small transients to the stack in schedule trees."""
import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import move_small_transients_to_stack, reuse_transients

NI, NJ, NK = 4, 4, 8


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


def _nest(tasklet: tn.TaskletNode, name: str, rows: int = NJ) -> tn.ForScope:
    """``for j in range(rows): for i in range(NI): <tasklet>``."""
    inner = dace.sdfg.state.LoopRegion(f'{name}_i', f'i < {NI}', 'i', 'i = 0', 'i = i + 1')
    outer = dace.sdfg.state.LoopRegion(f'{name}_j', f'j < {rows}', 'j', 'j = 0', 'j = j + 1')
    return tn.ForScope(loop=outer, children=[tn.ForScope(loop=inner, children=[tasklet])])


def _tree(nests: list, shapes: dict = None) -> tn.ScheduleTreeRoot:
    """``for k in range(NK): <nests>`` over arrays ``A`` to ``D`` of shape (NI, NJ, NK) and planes ``P`` and ``Q``
    (shapes ``shapes``, by default (NI, NJ, 1))."""
    sdfg = dace.SDFG('reuse')
    for name in 'ABCD':
        sdfg.add_array(name, [NI, NJ, NK], dace.float64)
    for name in 'PQ':
        shape = (shapes or {}).get(name, (NI, NJ, 1))
        sdfg.add_array(name, shape, dace.float64, transient=True, strides=[1, NI, NI * shape[1]])
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    loop = dace.sdfg.state.LoopRegion('vertical', f'k < {NK}', 'k', 'k = 0', 'k = k + 1')
    stree.children = []
    stree.add_children([tn.ForScope(loop=loop, children=nests)])
    return stree


def _write(plane: str, source: str, rows: int = NJ, name: str = 'w') -> tn.ForScope:
    return _nest(_tasklet('o = 2 * a', {'a': f'{source}[i, j, k]'}, {'o': f'{plane}[i, j, 0]'}), name + plane, rows)


def _read(plane: str, target: str, rows: int = NJ, name: str = 'r') -> tn.ForScope:
    return _nest(_tasklet('b = x + 1', {'x': f'{plane}[i, j, 0]'}, {'b': f'{target}[i, j, k]'}), name + plane, rows)


def _check(make, shared: int, **kwargs):
    stree, reference = make(), make()
    assert reuse_transients(stree, **kwargs) == shared
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
    return stree


def test_reuse_consecutive_planes():
    stree = _check(lambda: _tree([_write('P', 'A'), _read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')]), shared=2)
    assert 'P' not in stree.containers and 'Q' not in stree.containers


def test_reuse_not_overlapping_lives():
    _check(lambda: _tree([_write('P', 'A'), _write('Q', 'C'), _read('P', 'B'), _read('Q', 'D')]), shared=0)


def test_reuse_not_value_carried_between_iterations():
    """``P`` is read at the start of an iteration, before being written: its value comes from the previous one."""
    stree = _tree([_read('P', 'B'), _write('P', 'A'), _write('Q', 'C'), _read('Q', 'D')])
    assert reuse_transients(stree) == 0  # (Not run: the first iteration reads uninitialized values)


def test_reuse_plane_written_and_read_on_some_rows():
    make = lambda: _tree(
        [_write('P', 'A', rows=NJ - 1),
         _read('P', 'B', rows=NJ - 1),
         _write('Q', 'C'),
         _read('Q', 'D')])
    _check(make, shared=2)


@pytest.mark.parametrize('trust_reads', [False, True])
def test_reuse_partially_written_plane(trust_reads):
    """``P`` is written on three rows and read on four: that the last row is written in the iteration cannot be proven
    (only assumed, when trusting reads)."""
    stree = _tree([_write('P', 'A', rows=NJ - 1), _read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')])
    assert reuse_transients(stree, trust_reads=trust_reads) == (2 if trust_reads else 0)


def _branch_write(plane: str, both: bool) -> list:
    """``if C[0, 0, k] > 0.5: <plane> = 2 * A else: <plane> = 3 * A`` (the ``else`` only if ``both``)."""
    condition = dace.properties.CodeBlock('C[0, 0, k] > 0.5')
    result = [tn.IfScope(condition=condition, children=[_write(plane, 'A', name='t')])]
    if both:
        result.append(
            tn.ElseScope(
                children=[_nest(_tasklet('o = 3 * a', {'a': 'A[i, j, k]'}, {'o': f'{plane}[i, j, 0]'}), 'e' + plane)]))
    return result


def test_reuse_plane_written_in_both_branches():
    make = lambda: _tree(_branch_write('P', both=True) + [_read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')])
    _check(make, shared=2)


def test_reuse_not_plane_written_in_one_branch():
    stree = _tree(_branch_write('P', both=False) + [_read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')])
    assert reuse_transients(stree) == 0  # (Not run: reads uninitialized values when the condition does not hold)


def test_reuse_smaller_plane_in_larger_slot():
    make = lambda: _tree([
        _write('P', 'A'),
        _read('P', 'B'), _write('Q', 'C', rows=NJ - 1),
        _read('Q', 'D', rows=NJ - 1)
    ],
                         shapes={'Q': (NI, NJ - 1, 1)})
    stree = _check(make, shared=2)
    slot, = [d for name, d in stree.containers.items() if name.startswith('__reused')]
    assert tuple(slot.shape) == (NI, NJ, 1)


def test_move_small_transients_to_stack():
    stree = _tree([_write('P', 'A'), _read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')])
    plane = NI * NJ * 8
    assert move_small_transients_to_stack(stree, max_array_bytes=plane, max_total_bytes=plane) == 1
    storages = sorted(stree.containers[n].storage.name for n in 'PQ')
    assert storages == ['CPU_Heap', 'Register'] or storages == ['Default', 'Register']
    reference = _tree([_write('P', 'A'), _read('P', 'B'), _write('Q', 'C'), _read('Q', 'D')])
    rng = np.random.default_rng(0)
    inputs = {name: rng.random((NI, NJ, NK)) for name in 'ABCD'}
    results = []
    for tree in (reference, stree):
        arrays = {name: value.copy() for name, value in inputs.items()}
        tree.as_sdfg(simplify=False)(**arrays)
        results.append(arrays)
    for name in 'ABCD':
        assert np.array_equal(results[0][name], results[1][name]), name


def test_move_to_stack_not_read_before_written():
    """``P`` is read before it is written: moving it to the stack would change the values those reads see."""
    stree = _tree([_read('P', 'B'), _write('P', 'A'), _write('Q', 'C'), _read('Q', 'D')])
    assert move_small_transients_to_stack(stree) == 1
    assert stree.containers['Q'].storage == dace.StorageType.Register
    assert stree.containers['P'].storage != dace.StorageType.Register


def test_move_to_stack_zero_initialized():
    """With ``zero_read_before_written``, ``P`` (read before written) is moved and zeroed once, before the loop: the
    program then behaves as one that zeroes ``P`` explicitly first."""
    nests = lambda: [_read('P', 'B'), _write('P', 'A'), _write('Q', 'C'), _read('Q', 'D')]
    stree = _tree(nests())
    assert move_small_transients_to_stack(stree, zero_read_before_written=True) == 2
    assert stree.zero_initialized == {'P'}
    sdfg = stree.as_sdfg(simplify=False)
    code = sdfg.generate_code()[0].clean_code
    declarations = [line for line in code.splitlines() if ' P[' in line and 'DACE_ALIGN' in line]
    assert len(declarations) == 1 and '= {0}' in declarations[0]
    assert code.index(declarations[0]) < code.index('for (k = 0')

    reference = _tree(nests())
    zero = _nest(_tasklet('o = 0', {}, {'o': 'P[i, j, 0]'}), 'zero')
    children = list(reference.children)
    reference.children = []
    reference.add_children([zero] + children)
    rng = np.random.default_rng(0)
    inputs = {name: rng.random((NI, NJ, NK)) for name in 'ABCD'}
    results = []
    for program in (reference.as_sdfg(simplify=False), sdfg):
        arrays = {name: value.copy() for name, value in inputs.items()}
        program(**arrays)
        results.append(arrays)
    for name in 'ABCD':
        assert np.array_equal(results[0][name], results[1][name]), name


def test_move_to_stack_respects_limits():
    stree = _tree([_write('P', 'A'), _read('P', 'B')])
    assert move_small_transients_to_stack(stree, max_array_bytes=NI * NJ * 8 - 1) == 0


if __name__ == '__main__':
    test_reuse_consecutive_planes()
    test_reuse_not_overlapping_lives()
    test_reuse_not_value_carried_between_iterations()
    test_reuse_plane_written_and_read_on_some_rows()
    test_reuse_partially_written_plane(False)
    test_reuse_partially_written_plane(True)
    test_reuse_plane_written_in_both_branches()
    test_reuse_not_plane_written_in_one_branch()
    test_reuse_smaller_plane_in_larger_slot()
    test_move_small_transients_to_stack()
    test_move_to_stack_not_read_before_written()
    test_move_to_stack_zero_initialized()
    test_move_to_stack_respects_limits()

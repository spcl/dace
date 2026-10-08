# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for sinking statements into guards and coarsening guards in schedule trees."""
import copy
from typing import Callable, Dict, List

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import coarsen_guards, sink_into_guards
from dace.sdfg.state import LoopRegion


def _root(name: str, arrays: Dict[str, tuple], transients: Dict[str, tuple] = None) -> tn.ScheduleTreeRoot:
    """An empty tree with the given arrays (name -> (shape, dtype)) and transients (scalars if the shape is empty)."""
    sdfg = dace.SDFG(name)
    for array, (shape, dtype) in arrays.items():
        sdfg.add_array(array, shape, dtype)
    for array, (shape, dtype) in (transients or {}).items():
        if shape:
            sdfg.add_array(array, shape, dtype, transient=True)
        else:
            sdfg.add_scalar(array, dtype, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = sdfg.as_schedule_tree()
    stree.children = []
    return stree


def _loop(var: str, start: int, end: int, body: list) -> tn.ForScope:
    loop = LoopRegion(f'loop_{var}', f'{var} < {end}', var, f'{var} = {start}', f'{var} = {var} + 1')
    return tn.ForScope(loop=loop, children=body)


def _tasklet(code: str, inputs: Dict[str, str], outputs: Dict[str, str]) -> tn.TaskletNode:
    """A tasklet with connectors -> memlet strings."""
    tasklet = dace.nodes.Tasklet('compute', set(inputs), set(outputs), code)
    return tn.TaskletNode(node=tasklet,
                          in_memlets={
                              c: dace.Memlet(m)
                              for c, m in inputs.items()
                          },
                          out_memlets={
                              c: dace.Memlet(m)
                              for c, m in outputs.items()
                          })


def _if(condition: str, body: list) -> tn.IfScope:
    return tn.IfScope(condition=CodeBlock(condition), children=body)


def _nodes(stree: tn.ScheduleTreeRoot, kind: type) -> list:
    return [n for n in stree.preorder_traversal() if type(n) is kind]


def _run(stree: tn.ScheduleTreeRoot, arguments: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arguments = {k: v.copy() for k, v in arguments.items()}
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))
    sdfg(**arguments)
    return arguments


def _same_results(build: Callable[[], tn.ScheduleTreeRoot], transform: Callable[[tn.ScheduleTreeRoot], None],
                  inputs: List[Dict[str, np.ndarray]]) -> tn.ScheduleTreeRoot:
    """Transforms a tree and checks that it computes what the original does for each set of inputs."""
    transformed = build()
    transform(transformed)
    tn.validate_children_and_parents_align(transformed, root=True)
    for arguments in inputs:
        expected = _run(build(), arguments)
        result = _run(copy.deepcopy(transformed), arguments)
        for name in expected:
            assert np.array_equal(expected[name], result[name]), name
    return transformed


# ----------------------------------------------------------------------------------------------------------------------
# Coarsening guards
# ----------------------------------------------------------------------------------------------------------------------


def _borrowing_tree() -> tn.ScheduleTreeRoot:
    """``for k: for i: if L[k-1, i] != 0: A[k, i] -= L[k-1, i]; if A[k, i] < 0: L[k, i] = A[k, i]; A[k, i] = 0``, the
    shape of the interior sweep that fixes negative values."""
    stree = _root('coarsen_borrowing', {'A': ([8, 16], dace.float64), 'L': ([8, 16], dace.float64)})
    carry = _if('L[k - 1, i] != 0', [_tasklet('b = a - l', {'a': 'A[k, i]', 'l': 'L[k - 1, i]'}, {'b': 'A[k, i]'})])
    fix = _if('A[k, i] < 0', [_tasklet('l = a; b = 0', {'a': 'A[k, i]'}, {'l': 'L[k, i]', 'b': 'A[k, i]'})])
    stree.add_child(_loop('k', 1, 8, [_loop('i', 0, 16, [carry, fix])]))
    return stree


def _inputs_with_negatives() -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    positive = rng.random((8, 16))
    some_negative = positive.copy()
    some_negative[3, 5] = -0.5
    some_negative[6, 0] = -0.25
    return [{'A': positive, 'L': np.zeros((8, 16))}, {'A': some_negative, 'L': np.zeros((8, 16))}]


def test_coarsen_nest_and_rows():
    stree = _same_results(_borrowing_tree, lambda t: coarsen_guards(t) == 1 or None, _inputs_with_negatives())
    # The tests read the guards' elements through memlets, not by name in their code
    for test in _nodes(stree, tn.TaskletNode):
        if test.node.label == 'any_guard':
            assert 'A[' not in test.node.code.as_string and 'L[' not in test.node.code.as_string
            assert {m.data for m in test.in_memlets.values()} >= {'A', 'L'}
    # The guards read both loop variables: only row tests, inside the k loop over i
    conditions = [n.condition.as_string for n in _nodes(stree, tn.IfScope)]
    assert any(c.startswith('__any_row') for c in conditions)
    assert not any(c.startswith('__any') and not c.startswith('__any_row') for c in conditions)
    assert len(stree.children) == 1 and stree.children[0].loop.loop_variable == 'k'
    # Without row tests, one test over both loops before the nest
    stree = _same_results(_borrowing_tree, lambda t: coarsen_guards(t, levels=('nest', )) == 1 or None,
                          _inputs_with_negatives())
    assert type(stree.children[-1]) is tn.IfScope and stree.children[-2].loop.loop_variable == 'k'


def test_coarsen_counts():
    stree = _borrowing_tree()
    assert coarsen_guards(stree) == 1
    assert coarsen_guards(_borrowing_tree(), levels=('nest', )) == 1
    assert coarsen_guards(_borrowing_tree(), min_iterations=1000) == 0


def test_coarsen_test_only_over_guard_variables():
    """A guard on ``Z[j]`` in ``for k: for j:`` is tested once per ``j``, before the ``k`` loop."""

    def build():
        stree = _root('coarsen_invariant', {'A': ([8, 16], dace.float64), 'Z': ([16], dace.int32)})
        body = _if('Z[j] > 0', [_tasklet('b = a * 2', {'a': 'A[k, j]'}, {'b': 'A[k, j]'})])
        stree.add_child(_loop('k', 0, 8, [_loop('j', 0, 16, [body])]))
        return stree

    rng = np.random.default_rng(1)
    zeros, some = np.zeros(16, dtype=np.int32), np.zeros(16, dtype=np.int32)
    some[4] = 1
    stree = _same_results(build, lambda t: coarsen_guards(t, levels=('nest', )), [{
        'A': rng.random((8, 16)),
        'Z': zeros
    }, {
        'A': rng.random((8, 16)),
        'Z': some
    }])
    test_loop = stree.children[1]
    assert type(test_loop) is tn.ForScope and test_loop.loop.loop_variable == 'j'
    assert type(test_loop.children[0]) is tn.TaskletNode


def test_coarsen_requires_all_statements_guarded():
    stree = _borrowing_tree()
    inner = stree.children[0].children[0]
    inner.add_child(_tasklet('b = a', {'a': 'A[k, i]'}, {'b': 'L[k, i]'}))
    assert coarsen_guards(stree) == 0


def test_coarsen_skippable_access_must_be_in_bounds():
    """The test reads every access of a guard, also those ``and`` skips: ``A[i + 1]`` must exist for all ``i``."""
    for size, expected in ((16, 0), (17, 1)):
        stree = _root(f'coarsen_bounds_{size}', {'A': ([size], dace.float64), 'B': ([16], dace.float64)})
        body = _if('A[i] < 0 and A[i + 1] > 0', [_tasklet('b = 1', {}, {'b': 'B[i]'})])
        stree.add_child(_loop('k', 0, 4, [_loop('i', 0, 16, [body])]))
        assert coarsen_guards(stree) == expected


def test_coarsen_not_if_loop_variable_used_after():
    stree = _borrowing_tree()
    stree.add_child(_tasklet('b = k', {}, {'b': 'A[0, 0]'}))
    assert coarsen_guards(stree) == 0


# ----------------------------------------------------------------------------------------------------------------------
# Sinking statements into guards
# ----------------------------------------------------------------------------------------------------------------------


def _column_arrays() -> Dict[str, tuple]:
    return {'dm': ([8, 4, 4], dace.float64), 'q': ([8, 4, 4], dace.float64), 'zfix': ([4, 4], dace.int32)}


def _column_inputs() -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(2)
    none, some = np.zeros((4, 4), dtype=np.int32), np.zeros((4, 4), dtype=np.int32)
    some[1, 2] = 1
    some[3, 0] = 2
    return [{'dm': rng.random((8, 4, 4)), 'q': rng.random((8, 4, 4)), 'zfix': z} for z in (none, some)]


def _reset() -> tn.ForScope:
    return _loop('j', 0, 4, [_loop('i', 0, 4, [_tasklet('s = 0', {}, {'s': 'sum0[i, j]'})])])


def _fused_tree() -> tn.ScheduleTreeRoot:
    """A column sum and a correction that uses it in one loop nest, under a per-column guard."""
    stree = _root('sink_fused', _column_arrays(), {'sum0': ([4, 4], dace.float64), 'fac': ([], dace.float64)})
    accumulate = _tasklet('t = s + d', {'s': 'sum0[i, j]', 'd': 'dm[k, i, j]'}, {'t': 'sum0[i, j]'})
    factor = _tasklet('f = s * 2', {'s': 'sum0[i, j]'}, {'f': 'fac[0]'})
    correct = _if('zfix[i, j] > 0 and fac > 1', [_tasklet('o = f', {'f': 'fac[0]'}, {'o': 'q[k, i, j]'})])
    stree.add_children(
        [_reset(), _loop('k', 0, 8, [_loop('j', 0, 4, [_loop('i', 0, 4, [accumulate, factor, correct])])])])
    return stree


def test_sink_fused_sum_and_correction():

    def transform(stree):
        # The correction factor, the sum and the reset of the sum (all of its reads are under the guard)
        assert sink_into_guards(stree) == 3
        assert coarsen_guards(stree, levels=('nest', )) == 1  # The reset nest is too small to be worth a test

    stree = _same_results(_fused_tree, transform, _column_inputs())
    body = stree.children[-1].children[0]  # The fused nest under its test
    innermost = body.children[0].children[0]
    assert all(type(c) is tn.IfScope for c in innermost.children)


def _split_tree(between: list = None, after: list = None) -> tn.ScheduleTreeRoot:
    """The column sum and the correction in separate nests, the second with other loop variable names."""
    stree = _root('sink_split', {**_column_arrays(), 'out': ([1], dace.float64)}, {'sum0': ([4, 4], dace.float64)})
    accumulate = _tasklet('t = s + d', {'s': 'sum0[i, j]', 'd': 'dm[k, i, j]'}, {'t': 'sum0[i, j]'})
    correct = _if('zfix[a, b] > 0',
                  [_tasklet('o = s * x', {
                      's': 'sum0[a, b]',
                      'x': 'q[c, a, b]'
                  }, {'o': 'q[c, a, b]'})])
    stree.add_children([
        _reset(),
        _loop('k', 0, 8, [_loop('j', 0, 4, [_loop('i', 0, 4, [accumulate])])]), *(between or []),
        _loop('c', 0, 8, [_loop('b', 0, 4, [_loop('a', 0, 4, [correct])])]), *(after or [])
    ])
    return stree


def test_sink_across_nests_maps_guard_to_element():

    def transform(stree):
        assert sink_into_guards(stree) == 2  # The sum and its reset
        sunk = [
            g for g in _nodes(stree, tn.IfScope) if any(m.data == 'sum0' for m in g.children[0].out_memlets.values())
        ]
        assert len(sunk) == 2
        for guard in sunk:
            assert guard.condition.as_string.translate(str.maketrans('', '', ' ()')) == 'zfix[i,j]>0'
        assert coarsen_guards(stree, levels=('nest', )) == 2  # Not the small reset nest

    _same_results(lambda: _split_tree(), transform, [{**i, 'out': np.zeros(1)} for i in _column_inputs()])


def test_sink_not_if_guard_input_written_in_between():
    bump = _loop('j', 0, 4, [_loop('i', 0, 4, [_tasklet('o = z + 1', {'z': 'zfix[i, j]'}, {'o': 'zfix[i, j]'})])])
    assert sink_into_guards(_split_tree(between=[bump])) == 0


def test_sink_not_if_read_unguarded():
    read = _tasklet('o = s', {'s': 'sum0[0, 0]'}, {'o': 'out[0]'})
    assert sink_into_guards(_split_tree(after=[read])) == 0


def test_sink_not_if_guard_changes_while_value_accumulates():
    """The sum is reset once and accumulated in every iteration of ``t``, while the guard changes with ``t``."""
    stree = _root('sink_accumulate', _column_arrays(), {'sum0': ([4, 4], dace.float64)})
    accumulate = _tasklet('t = s + d', {'s': 'sum0[i, j]', 'd': 'dm[k, i, j]'}, {'t': 'sum0[i, j]'})
    bump = _loop('j', 0, 4, [_loop('i', 0, 4, [_tasklet('o = z + 1', {'z': 'zfix[i, j]'}, {'o': 'zfix[i, j]'})])])
    correct = _if('zfix[i, j] > 3', [_tasklet('o = s', {'s': 'sum0[i, j]'}, {'o': 'q[k, i, j]'})])
    stree.add_children([
        _reset(),
        _loop('t', 0, 3, [
            bump,
            _loop('k', 0, 8, [_loop('j', 0, 4, [_loop('i', 0, 4, [accumulate])])]),
            _loop('k', 0, 8, [_loop('j', 0, 4, [_loop('i', 0, 4, [correct])])]),
        ])
    ])
    assert sink_into_guards(stree) == 0


if __name__ == '__main__':
    test_coarsen_nest_and_rows()
    test_coarsen_counts()
    test_coarsen_test_only_over_guard_variables()
    test_coarsen_requires_all_statements_guarded()
    test_coarsen_skippable_access_must_be_in_bounds()
    test_coarsen_not_if_loop_variable_used_after()
    test_sink_fused_sum_and_correction()
    test_sink_across_nests_maps_guard_to_element()
    test_sink_not_if_guard_input_written_in_between()
    test_sink_not_if_read_unguarded()
    test_sink_not_if_guard_changes_while_value_accumulates()

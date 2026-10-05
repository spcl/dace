# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``WhileToForLoop``: while loops that step a counter by a constant become for loops."""
import copy

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.while_to_for_loop import WhileToForLoop

N = dace.symbol('N')


def _add_tasklet(state: dace.SDFGState, code: str, reads: dict, writes: dict):
    tasklet = state.add_tasklet('compute', set(reads), set(writes), code)
    for connector, subset in reads.items():
        data = subset.split('[')[0]
        state.add_edge(state.add_read(data), None, tasklet, connector, dace.Memlet(subset))
    for connector, subset in writes.items():
        data = subset.split('[')[0]
        state.add_edge(tasklet, connector, state.add_write(data), None, dace.Memlet(subset))


def _while_loop(condition: str, start: str, step: str, step_is_last: bool = False):
    """
    ``i = start; while condition: B[i] = A[i] * 2; i = step; B[i - 1] += 1`` (or without the last statement), with the
    step on an edge in the middle of the body.
    """
    sdfg = dace.SDFG('while_loop')
    sdfg.add_array('A', ['N'], dace.float64)
    sdfg.add_array('B', ['N + 1'], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', condition_expr=condition)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'i': start}))
    first = loop.add_state('first', is_start_block=True)
    _add_tasklet(first, 'b = a * 2', {'a': 'A[i]'}, {'b': 'B[i]'})
    second = loop.add_state('second')
    loop.add_edge(first, second, dace.InterstateEdge(assignments={'i': step}))
    if not step_is_last:
        _add_tasklet(second, 'b = c + 1', {'c': 'B[i - 1]'}, {'b': 'B[i - 1]'})
    sdfg.add_edge(loop, sdfg.add_state('after'), dace.InterstateEdge())
    return sdfg, loop


def _assert_for_loop(loop: LoopRegion, start, end, stride):
    """The loop is a for loop that DaCe's loop analyses understand (``end`` is the last value of the counter)."""
    assert loop.loop_variable == 'i'
    assert loop_analysis.get_init_assignment(loop) == start
    assert loop_analysis.get_loop_end(loop) == end
    assert loop_analysis.get_loop_stride(loop) == stride


def _run(sdfg: dace.SDFG, n: int) -> np.ndarray:
    a = np.arange(n, dtype=np.float64) + 1
    b = np.zeros(n + 1)
    sdfg(A=a, B=b, N=n)
    return b


def test_step_in_the_middle_of_the_body():
    """The block after the step reads the stepped counter: it is rewritten to ``i + 1`` of the for loop."""
    sdfg, loop = _while_loop('i < N', '0', 'i + 1')
    reference = _run(copy.deepcopy(sdfg), 5)
    assert WhileToForLoop().apply_pass(sdfg, {}) == 1
    _assert_for_loop(loop, start=0, end=N - 1, stride=1)
    assert all('i' not in edge.data.assignments for edge in loop.edges())
    sdfg.validate()
    np.testing.assert_array_equal(_run(sdfg, 5), reference)


def test_counting_down_with_non_strict_comparison():
    """``while i >= 0: ...; i = i - 2`` becomes ``for (i = N - 1; i > -1; i = i - 2)``."""
    sdfg, loop = _while_loop('i >= 0', 'N - 1', 'i - 2', step_is_last=True)
    reference = _run(copy.deepcopy(sdfg), 7)
    assert WhileToForLoop().apply_pass(sdfg, {}) == 1
    _assert_for_loop(loop, start=N - 1, end=0, stride=-2)
    np.testing.assert_array_equal(_run(sdfg, 7), reference)


def test_truth_test_of_a_comparison():
    """Frontends may emit conditions as ``(i < N) != 0``."""
    sdfg, loop = _while_loop('(i < N) != 0', '1', 'i + 1')
    reference = _run(copy.deepcopy(sdfg), 4)
    assert WhileToForLoop().apply_pass(sdfg, {}) == 1
    _assert_for_loop(loop, start=1, end=N - 1, stride=1)
    np.testing.assert_array_equal(_run(sdfg, 4), reference)


def test_conditional_step_is_not_converted():
    """A step that some iterations skip is not a for loop."""
    sdfg, loop = _while_loop('i < N', '0', 'i + 1', step_is_last=True)
    first = loop.start_block
    other = loop.add_state('other')
    loop.add_edge(first, other, dace.InterstateEdge(condition='N > 100'))
    loop.out_edges(first)[0].data.condition = dace.properties.CodeBlock('N <= 100')
    assert WhileToForLoop().apply_pass(sdfg, {}) is None
    assert loop.update_statement is None


def test_bound_assigned_in_the_loop_is_not_converted():
    sdfg, loop = _while_loop('i < M', '0', 'i + 1', step_is_last=True)
    loop.edges()[0].data.assignments['M'] = 'M - 1'
    assert WhileToForLoop().apply_pass(sdfg, {}) is None


def test_start_value_on_an_earlier_edge():
    """The value of the counter when the loop is entered is found on the chain of edges before the loop."""
    sdfg, loop = _while_loop('i < N', '0', 'i + 1')
    init = sdfg.start_block
    edge = sdfg.in_edges(loop)[0]
    sdfg.remove_edge(edge)
    middle = sdfg.add_state('middle')
    sdfg.add_edge(init, middle, dace.InterstateEdge(assignments={'i': '2'}))
    sdfg.add_edge(middle, loop, dace.InterstateEdge(assignments={'j': 'i + 1'}))
    reference = _run(copy.deepcopy(sdfg), 6)
    assert WhileToForLoop().apply_pass(sdfg, {}) == 1
    _assert_for_loop(loop, start=2, end=N - 1, stride=1)
    np.testing.assert_array_equal(_run(sdfg, 6), reference)


if __name__ == '__main__':
    test_step_in_the_middle_of_the_body()
    test_counting_down_with_non_strict_comparison()
    test_truth_test_of_a_comparison()
    test_conditional_step_is_not_converted()
    test_bound_assigned_in_the_loop_is_not_converted()
    test_start_value_on_an_earlier_edge()

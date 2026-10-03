# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The guarded-fallback predicate must see the same SDFGs the loop counter does.

``count`` counts LoopRegions across nested SDFGs. When ``guarded_fallback_loops`` only
looked at the top-level SDFG, a guarded parallelization ``if cond: <Map> else: <seq loop>``
that ended up inside a nested SDFG was charged as a residual sequential loop while its Map
still counted -- the kernel WAS parallelized, under a predicate, and the metric said it was
not. Both SDFGs here are built in process; nothing goes through the frontend.
"""
import dace
import pytest
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

from tests.corpus.measure_parallelization import count, guarded_fallback_loop_set, guarded_fallback_loops

N = dace.symbol('N')


def fill_loop_body(loop: LoopRegion, array: str) -> None:
    """A body that writes one element -- enough for a countable loop."""
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('t', {}, {'b'}, 'b = 1.0')
    state.add_edge(tasklet, 'b', state.add_access(array), None, dace.Memlet(f'{array}[i]'))


def guarded_sdfg(name: str, array: str) -> dace.SDFG:
    """``if N > 0: <Map> else: <seq loop>`` over ``array`` -- one guarded parallelization."""
    sdfg = dace.SDFG(name)
    sdfg.add_array(array, [N], dace.float64)
    guard = ConditionalBlock('guard', sdfg=sdfg)
    sdfg.add_node(guard, is_start_block=True)
    parallel = ControlFlowRegion('par', sdfg=sdfg)
    guard.add_branch(CodeBlock('N > 0'), parallel)
    state = parallel.add_state('ps', is_start_block=True)
    state.add_mapped_tasklet('m', {'i': '0:N'}, {}, 'b = 1.0', {'b': dace.Memlet(f'{array}[i]')}, external_edges=True)
    sequential = ControlFlowRegion('seq', sdfg=sdfg)
    guard.add_branch(None, sequential)
    loop = LoopRegion('fallback', 'i < N', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sequential.add_node(loop, is_start_block=True)
    fill_loop_body(loop, array)
    return sdfg


def nested_guarded_sdfg() -> dace.SDFG:
    """The same guard, one nesting level down: an outer SDFG whose only payload is a
    NestedSDFG holding it."""
    inner = guarded_sdfg('inner', 'a')
    outer = dace.SDFG('outer')
    outer.add_array('A', [N], dace.float64)
    state = outer.add_state('s')
    nested = state.add_nested_sdfg(inner, {}, {'a'})
    state.add_edge(nested, 'a', state.add_access('A'), None, dace.Memlet('A[0:N]'))
    return outer


def test_top_level_guard_counts():
    """The case that already worked, pinned so the widening did not lose it."""
    sdfg = guarded_sdfg('flat', 'A')
    assert guarded_fallback_loops(sdfg) == 1
    assert count(sdfg)[0] == 1  # one LoopRegion, and it is the fallback


def test_nested_guard_counts():
    """A guarded fallback inside a nested SDFG is a parallelization, not a residual loop."""
    sdfg = nested_guarded_sdfg()
    # The counter sees the nested loop and the nested Map...
    loops, _, maps = count(sdfg)[0], count(sdfg)[1], count(sdfg)[2]
    assert (loops, maps) == (1, 1)
    # ...and nothing at the top level is a ConditionalBlock, so a top-level-only walk finds none.
    assert not [c for c in sdfg.all_control_flow_regions() if isinstance(c, ConditionalBlock)]
    assert guarded_fallback_loops(sdfg) == 1


def test_set_form_returns_the_loops_the_count_counts():
    """The count derives from the set, so a consumer can bucket the loops themselves."""
    sdfg = nested_guarded_sdfg()
    loops = guarded_fallback_loop_set(sdfg)
    assert [lp.label for lp in loops] == ['fallback']
    assert len(loops) == guarded_fallback_loops(sdfg)


def test_unguarded_nested_loop_is_not_a_fallback():
    """A nested sequential loop with no parallel sibling branch stays residual."""
    inner = dace.SDFG('inner_plain')
    inner.add_array('a', [N], dace.float64)
    loop = LoopRegion('plain', 'i < N', 'i', 'i = 0', 'i = i + 1', sdfg=inner)
    inner.add_node(loop, is_start_block=True)
    fill_loop_body(loop, 'a')
    outer = dace.SDFG('outer_plain')
    outer.add_array('A', [N], dace.float64)
    state = outer.add_state('s')
    nested = state.add_nested_sdfg(inner, {}, {'a'})
    state.add_edge(nested, 'a', state.add_access('A'), None, dace.Memlet('A[0:N]'))
    assert count(outer)[0] == 1
    assert guarded_fallback_loops(outer) == 0


def test_guard_with_no_parallel_branch_is_not_a_fallback():
    """``if cond: <seq loop> else: <seq loop>`` was never parallelized -- neither loop counts."""
    sdfg = dace.SDFG('no_map_guard')
    sdfg.add_array('A', [N], dace.float64)
    guard = ConditionalBlock('guard', sdfg=sdfg)
    sdfg.add_node(guard, is_start_block=True)
    for label, cond in (('then', CodeBlock('N > 0')), ('else', None)):
        branch = ControlFlowRegion(label, sdfg=sdfg)
        guard.add_branch(cond, branch)
        loop = LoopRegion(f'{label}_loop', 'i < N', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
        branch.add_node(loop, is_start_block=True)
        fill_loop_body(loop, 'A')
    assert count(sdfg)[0] == 2
    assert guarded_fallback_loops(sdfg) == 0
    assert not any(isinstance(n, nd.MapEntry) for n, _ in sdfg.all_nodes_recursive())


if __name__ == '__main__':
    pytest.main([__file__])

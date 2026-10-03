# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``PruneAndInlineNestedSDFGs``, the traversal half of ``PruneConnectors`` + ``InlineSDFG``.

The pass replaced ``PatternApplyOnceEverywhere([PruneConnectors(), InlineSDFG()])`` and skips refusals it
proves unchanged, so what it owes is the wrapper's application sequence: the same graph, node order included.
"""
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG
from dace.transformation.passes.canonicalize.prune_and_inline_nested_sdfgs import PruneAndInlineNestedSDFGs
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere

N = 8


def ordered_signature(sdfg: dace.SDFG) -> list:
    """Per-SDFG arrays and per-state nodes and edges, in graph order: application order moves node order."""
    sig = []
    for sd in sdfg.all_sdfgs_recursive():
        sig.append(('arrays', sd.label, list(sd.arrays.keys())))
        for state in sd.all_states():
            index = {node: i for i, node in enumerate(state.nodes())}
            sig.append(('nodes', state.label, [(type(n).__name__, str(n)) for n in state.nodes()]))
            sig.append(('edges', state.label, [(index[e.src], e.src_conn, index[e.dst], e.dst_conn, str(e.data.data),
                                                str(e.data.subset)) for e in state.edges()]))
    return sig


def run_both(sdfg: dace.SDFG):
    by_pass, by_wrapper = copy.deepcopy(sdfg), copy.deepcopy(sdfg)
    count = PruneAndInlineNestedSDFGs().apply_pass(by_pass, {}) or 0
    wrapper = PatternApplyOnceEverywhere([PruneConnectors(), InlineSDFG()])
    wrapper.progress = False
    applied = wrapper.apply_pass(by_wrapper, {}) or {}
    return by_pass, by_wrapper, count, sum(len(v) for v in applied.values())


def nested_sdfg_nodes(sdfg: dace.SDFG) -> list:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]


def leaf_with_an_unused_input() -> dace.SDFG:
    """Reads ``c`` into ``b``; its ``a`` input is connected but never read."""
    leaf = dace.SDFG('leaf')
    for name in ('a', 'b', 'c'):
        leaf.add_array(name, [N], dace.float64)
    state = leaf.add_state('leaf_body')
    tasklet = state.add_tasklet('copy', {'inp': None}, {'out': None}, 'out = inp')
    state.add_edge(state.add_read('c'), None, tasklet, 'inp', dace.Memlet('c[0]'))
    state.add_edge(tasklet, 'out', state.add_write('b'), None, dace.Memlet('b[0]'))
    return leaf


def two_level_nest() -> dace.SDFG:
    """``outer -> middle (two states, never inlined) -> leaf``, with ``a`` threaded through both boundaries."""
    middle = dace.SDFG('middle')
    for name in ('a', 'b', 'c'):
        middle.add_array(name, [N], dace.float64)
    middle.add_state('middle_entry', is_start_block=True)
    body = middle.add_state_after(middle.start_block, 'middle_body')
    leaf_node = body.add_nested_sdfg(leaf_with_an_unused_input(), {'a': None, 'c': None}, {'b': None})
    body.add_edge(body.add_read('a'), None, leaf_node, 'a', dace.Memlet(f'a[0:{N}]'))
    body.add_edge(body.add_read('c'), None, leaf_node, 'c', dace.Memlet(f'c[0:{N}]'))
    body.add_edge(leaf_node, 'b', body.add_write('b'), None, dace.Memlet(f'b[0:{N}]'))

    outer = dace.SDFG('outer')
    for name in ('a', 'b', 'c'):
        outer.add_array(name, [N], dace.float64)
    state = outer.add_state('outer_body')
    middle_node = state.add_nested_sdfg(middle, {'a': None, 'c': None}, {'b': None})
    state.add_edge(state.add_read('a'), None, middle_node, 'a', dace.Memlet(f'a[0:{N}]'))
    state.add_edge(state.add_read('c'), None, middle_node, 'c', dace.Memlet(f'c[0:{N}]'))
    state.add_edge(middle_node, 'b', state.add_write('b'), None, dace.Memlet(f'b[0:{N}]'))
    outer.validate()
    return outer


def test_a_prune_inside_a_body_reopens_the_refused_parent():
    """The parent reads ``a`` only to hand it to the leaf, so it is refused until the leaf's prune drops that
    read. A pass that trusted the parent's first refusal would leave ``a`` connected at the outer boundary."""
    by_pass, by_wrapper, count, wrapper_count = run_both(two_level_nest())
    middle = next(n for n in nested_sdfg_nodes(by_pass) if n.sdfg.label == 'middle')
    assert 'a' not in middle.in_connectors, f'the parent kept its dead input: {sorted(middle.in_connectors)}'
    assert count == wrapper_count == 3, f'prune leaf, prune parent, inline leaf; pass {count}, wrapper {wrapper_count}'
    assert ordered_signature(by_pass) == ordered_signature(by_wrapper), 'the resulting graphs differ'


def scalar_body(factor: float) -> dace.SDFG:
    body = dace.SDFG(f'times_{int(factor)}')
    body.add_scalar('x', dace.float64)
    body.add_scalar('y', dace.float64)
    state = body.add_state('body_state')
    tasklet = state.add_tasklet('scale', {'inp': None}, {'out': None}, f'out = inp * {factor}')
    state.add_edge(state.add_read('x'), None, tasklet, 'inp', dace.Memlet('x'))
    state.add_edge(tasklet, 'out', state.add_write('y'), None, dace.Memlet('y'))
    return body


def two_mapped_bodies_in_one_state() -> dace.SDFG:
    """``b[i] = 2 * a[i]`` then ``c[i] = 3 * b[i]``, each map body a single-state nested SDFG, one state."""
    sdfg = dace.SDFG('mapped_bodies')
    for name in ('a', 'b', 'c'):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state()
    middle = state.add_access('b')
    for src, dst, factor in ((state.add_read('a'), middle, 2.0), (middle, state.add_write('c'), 3.0)):
        entry, exit_node = state.add_map(f'map_{int(factor)}', dict(i=f'0:{N}'))
        body = state.add_nested_sdfg(scalar_body(factor), {'x': None}, {'y': None})
        state.add_memlet_path(src, entry, body, dst_conn='x', memlet=dace.Memlet(f'{src.data}[i]'))
        state.add_memlet_path(body, exit_node, dst, src_conn='y', memlet=dace.Memlet(f'{dst.data}[i]'))
    sdfg.validate()
    return sdfg


def test_every_mapped_body_in_a_state_is_inlined_as_the_wrapper_does():
    """Two accepted bodies share a state, so the second is found only by re-probing the state the first
    inline rewrote."""
    by_pass, by_wrapper, count, wrapper_count = run_both(two_mapped_bodies_in_one_state())
    assert not nested_sdfg_nodes(by_pass), 'both single-state bodies must be inlined'
    assert count == wrapper_count == 2, f'application counts differ: pass {count}, wrapper {wrapper_count}'
    assert ordered_signature(by_pass) == ordered_signature(by_wrapper), 'the resulting graphs differ'

    a, b, c = np.random.default_rng(3).random(N), np.zeros(N), np.zeros(N)
    by_pass(a=a, b=b, c=c)
    assert np.allclose(b, 2.0 * a) and np.allclose(c, 6.0 * a)


def test_nothing_accepted_returns_none_and_leaves_the_graph_alone():
    sdfg = dace.SDFG('flat')
    sdfg.add_array('a', [N], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {}, {'out': None}, 'out = 1.0')
    state.add_edge(tasklet, 'out', state.add_write('a'), None, dace.Memlet('a[0]'))
    before = ordered_signature(sdfg)
    assert PruneAndInlineNestedSDFGs().apply_pass(sdfg, {}) is None
    assert ordered_signature(sdfg) == before

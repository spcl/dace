# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :mod:`dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination`."""
from typing import Dict, Optional, Tuple

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.redundant_ordering_edge_elimination import (
    RedundantOrderingEdgeElimination)


def scope_ids(state: SDFGState) -> Dict[int, Optional[int]]:
    """Identity view of the state's scope dict, so the comparison is by object and not by value."""
    return {id(n): (None if s is None else id(s)) for n, s in state.scope_dict().items()}


def tasklet_named(state: SDFGState, label: str) -> nodes.Tasklet:
    """The tasklet of ``state`` carrying ``label``."""
    return next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == label)


def build_chain(name: str) -> Tuple[dace.SDFG, SDFGState]:
    """``A -> t1 -> B -> t2 -> C`` in one state, all scalars."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'B', 'C'):
        sdfg.add_array(arr, [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    a, b, c = state.add_access('A'), state.add_access('B'), state.add_access('C')
    t1 = state.add_tasklet('t1', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    t2 = state.add_tasklet('t2', {'inp': None}, {'out': None}, 'out = inp * 2.0')
    state.add_edge(a, None, t1, 'inp', dace.Memlet('A[0]'))
    state.add_edge(t1, 'out', b, None, dace.Memlet('B[0]'))
    state.add_edge(b, None, t2, 'inp', dace.Memlet('B[0]'))
    state.add_edge(t2, 'out', c, None, dace.Memlet('C[0]'))
    return sdfg, state


def build_two_chains(name: str) -> Tuple[dace.SDFG, SDFGState]:
    """Two independent chains, ``A -> t1 -> B`` and ``C -> t2 -> D``, in one state."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'B', 'C', 'D'):
        sdfg.add_array(arr, [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    t1 = state.add_tasklet('t1', {'inp': None}, {'out': None}, 'out = inp + 1.0')
    t2 = state.add_tasklet('t2', {'inp': None}, {'out': None}, 'out = inp * 2.0')
    state.add_edge(state.add_access('A'), None, t1, 'inp', dace.Memlet('A[0]'))
    state.add_edge(t1, 'out', state.add_access('B'), None, dace.Memlet('B[0]'))
    state.add_edge(state.add_access('C'), None, t2, 'inp', dace.Memlet('C[0]'))
    state.add_edge(t2, 'out', state.add_access('D'), None, dace.Memlet('D[0]'))
    return sdfg, state


def build_map_state(name: str) -> Tuple[dace.SDFG, SDFGState]:
    """A map whose body holds a no-input ``seed`` tasklet attached to the entry by ordering only."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_transient('tmp', [1], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    entry, exit_node = state.add_map('m', {'i': '0:8'})
    seed = state.add_tasklet('seed', {}, {'out': None}, 'out = 1.0')
    body = state.add_tasklet('body', {'inp': None, 'off': None}, {'out': None}, 'out = inp + off')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_access('A'), entry, body, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_nedge(entry, seed, dace.Memlet())
    state.add_edge(seed, 'out', tmp, None, dace.Memlet('tmp[0]'))
    state.add_edge(tmp, None, body, 'off', dace.Memlet('tmp[0]'))
    state.add_memlet_path(body, exit_node, state.add_access('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    return sdfg, state


def test_redundant_ordering_edge_is_removed():
    sdfg, state = build_chain('roee_redundant')
    ordering = state.add_nedge(tasklet_named(state, 't1'), tasklet_named(state, 't2'), dace.Memlet())
    sdfg.validate()

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) == 1
    assert all(e is not ordering for e in state.edges())
    assert state.number_of_edges() == 4
    sdfg.validate()


def test_only_path_ordering_edge_is_kept():
    sdfg, state = build_two_chains('roee_only_path')
    ordering = state.add_nedge(tasklet_named(state, 't1'), tasklet_named(state, 't2'), dace.Memlet())
    sdfg.validate()

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) is None
    assert any(e is ordering for e in state.edges())
    sdfg.validate()


def test_parallel_ordering_edges_keep_the_first_inserted():
    sdfg, state = build_two_chains('roee_parallel')
    t1, t2 = tasklet_named(state, 't1'), tasklet_named(state, 't2')
    first = state.add_nedge(t1, t2, dace.Memlet())
    second = state.add_nedge(t1, t2, dace.Memlet())

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) == 1
    survivors = list(state.edges_between(t1, t2))
    assert len(survivors) == 1
    assert survivors[0] is first
    assert all(e is not second for e in state.edges())
    sdfg.validate()


def test_scope_edge_is_kept_and_scopes_are_unchanged():
    sdfg, state = build_map_state('roee_scope')
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    seed, body = tasklet_named(state, 'seed'), tasklet_named(state, 'body')
    scope_edge = list(state.edges_between(entry, seed))[0]
    implied = state.add_nedge(seed, body, dace.Memlet())  # already implied by seed -> tmp -> body
    sdfg.validate()
    before = scope_ids(state)

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) == 1
    assert all(e is not implied for e in state.edges())
    assert any(e is scope_edge for e in state.edges())
    assert scope_ids(state) == before
    assert state.scope_dict()[seed] is entry
    sdfg.validate()


def test_ordering_edge_that_is_the_scopes_only_link_is_kept():
    sdfg, state = build_map_state('roee_scope_only_link')
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    seed = tasklet_named(state, 'seed')
    sdfg.validate()
    before = scope_ids(state)

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) is None
    assert len(list(state.edges_between(entry, seed))) == 1
    assert scope_ids(state) == before
    sdfg.validate()


def test_idempotence_leaves_the_sdfg_bit_identical():
    sdfg, state = build_chain('roee_idempotent')
    state.add_nedge(tasklet_named(state, 't1'), tasklet_named(state, 't2'), dace.Memlet())

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) == 1
    after_first = sdfg.to_json()
    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == after_first


def test_non_applying_pass_leaves_the_sdfg_bit_identical():
    sdfg, state = build_two_chains('roee_no_apply')
    state.add_nedge(tasklet_named(state, 't1'), tasklet_named(state, 't2'), dace.Memlet())
    before = sdfg.to_json()

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


def test_implied_data_edge_is_never_removed():
    sdfg, state = build_chain('roee_data_edge')
    a = next(n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == 'A')
    t2 = tasklet_named(state, 't2')
    t2.add_in_connector('inp2')
    t2.code.as_string = 'out = inp * 2.0 + inp2'
    # Ordering-wise implied by A -> t1 -> B -> t2, but it also moves data, so it must stay.
    state.add_edge(a, None, t2, 'inp2', dace.Memlet('A[0]'))
    before = state.number_of_edges()
    sdfg.validate()

    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) is None
    assert state.number_of_edges() == before


def test_nested_sdfg_states_are_reduced():
    inner, inner_state = build_chain('roee_inner')
    inner.arrays['B'].transient = True
    ordering = inner_state.add_nedge(tasklet_named(inner_state, 't1'), tasklet_named(inner_state, 't2'), dace.Memlet())

    outer = dace.SDFG('roee_outer')
    outer.add_array('A', [1], dace.float64)
    outer.add_array('C', [1], dace.float64)
    outer_state = outer.add_state('main', is_start_block=True)
    nsdfg = outer_state.add_nested_sdfg(inner, {'A'}, {'C'}, {})
    outer_state.add_edge(outer_state.add_access('A'), None, nsdfg, 'A', dace.Memlet('A[0]'))
    outer_state.add_edge(nsdfg, 'C', outer_state.add_access('C'), None, dace.Memlet('C[0]'))
    outer.validate()

    assert RedundantOrderingEdgeElimination().apply_pass(outer, {}) == 1
    assert all(e is not ordering for e in inner_state.edges())
    outer.validate()


def test_end_to_end_numbers_are_unchanged():
    sdfg, state = build_chain('roee_e2e')
    state.add_nedge(tasklet_named(state, 't1'), tasklet_named(state, 't2'), dace.Memlet())
    assert RedundantOrderingEdgeElimination().apply_pass(sdfg, {}) == 1
    sdfg.validate()

    a = np.array([3.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    c = np.zeros(1, dtype=np.float64)
    sdfg(A=a, B=b, C=c)
    assert np.allclose(b, [4.0])
    assert np.allclose(c, [8.0])


def test_pipeline_kernel_is_still_correct():
    N = dace.symbol('N', dtype=dace.int64)

    @dace.program
    def roee_kernel(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
        for i in range(N):
            b[i] = a[i] * 2.0
        for i in range(N):
            out[i] = b[i] + a[i]

    sdfg = roee_kernel.to_sdfg(simplify=False)
    canonicalize(sdfg)
    sdfg.validate()

    a = np.random.rand(24)
    b = np.zeros(24)
    out = np.zeros(24)
    sdfg(a=a, b=b, out=out, N=24)
    assert np.allclose(b, a * 2.0)
    assert np.allclose(out, a * 3.0)


if __name__ == '__main__':
    test_redundant_ordering_edge_is_removed()
    test_only_path_ordering_edge_is_kept()
    test_parallel_ordering_edges_keep_the_first_inserted()
    test_scope_edge_is_kept_and_scopes_are_unchanged()
    test_ordering_edge_that_is_the_scopes_only_link_is_kept()
    test_idempotence_leaves_the_sdfg_bit_identical()
    test_non_applying_pass_leaves_the_sdfg_bit_identical()
    test_implied_data_edge_is_never_removed()
    test_nested_sdfg_states_are_reduced()
    test_end_to_end_numbers_are_unchanged()
    test_pipeline_kernel_is_still_correct()

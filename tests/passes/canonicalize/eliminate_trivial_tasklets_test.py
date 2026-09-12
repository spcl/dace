# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``EliminateTrivialTasklets``, the traversal half of ``TrivialTaskletElimination``.

The pass replaced ``PatternApplyOnceEverywhere([TrivialTaskletElimination()])`` in the
canonicalization recipe, so what it owes is that walking the tasklets of a state finds the SAME
sites the subgraph matcher found, for all three patterns, and refuses exactly what the
transformation refuses.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import TrivialTaskletElimination
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import (EliminateTrivialTasklets,
                                                                                trivial_tasklet_candidates)
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere

N = 8


def _tasklets(sdfg):
    """Every tasklet body left in ``sdfg``, sorted."""
    return sorted(n.code.as_string for sd in sdfg.all_sdfgs_recursive() for s in sd.states() for n in s.nodes()
                  if isinstance(n, nodes.Tasklet))


def _signature(sdfg):
    """Order-independent fingerprint of the graph shape, its memlets and its tasklet bodies."""
    sig = []
    for sd in sdfg.all_sdfgs_recursive():
        for s in sd.states():
            sig.append(('nodes', s.label, len(s.nodes())))
            for e in s.edges():
                sig.append(('edge', s.label, type(e.src).__name__, type(e.dst).__name__, str(e.data.data),
                            str(e.data.subset), str(e.data.other_subset), str(e.data.wcr)))
            for n in s.nodes():
                if isinstance(n, nodes.Tasklet):
                    sig.append(('tasklet', s.label, n.code.as_string))
    return sorted(sig)


def _access_to_access():
    """expr 0: ``AccessNode -> Tasklet -> AccessNode``, a plain elementwise copy."""
    sdfg = dace.SDFG('expr0')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    state.add_edge(state.add_read('b'), None, t, 'inp', dace.Memlet('b[0]'))
    state.add_edge(t, 'out', state.add_write('a'), None, dace.Memlet('a[1]'))
    sdfg.validate()
    return sdfg


def _mapentry_to_access():
    """expr 1: ``MapEntry -> Tasklet -> AccessNode``, the copy staged inside a map."""
    sdfg = dace.SDFG('expr1')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_read('b'), me, t, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_edge(t, 'out', tmp, None, dace.Memlet('tmp[0]'))
    state.add_memlet_path(tmp, mx, state.add_write('a'), memlet=dace.Memlet('a[i]'))
    sdfg.validate()
    return sdfg


def _access_to_mapexit():
    """expr 2: ``AccessNode -> Tasklet -> MapExit``, the copy at the stage-out boundary.

    Kept, not eliminated, whenever the stage-out memlet names the outer array rather than the
    read access node: splicing then yields an invalid ``<scalar> -> MapExit`` edge. This is the
    shape ``InsertAssignTaskletsAtMapBoundary`` re-creates, so the refusal is the normal outcome.
    """
    sdfg = dace.SDFG('expr2')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_scalar('tmp', dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    inner = state.add_tasklet('inner', {'inp': None}, {'out': None}, 'out = inp * 2')
    copy_t = state.add_tasklet('copy', {'inp': None}, {'out': None}, 'out = inp')
    tmp = state.add_access('tmp')
    state.add_memlet_path(state.add_read('b'), me, inner, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_edge(inner, 'out', tmp, None, dace.Memlet('tmp[0]'))
    state.add_edge(tmp, None, copy_t, 'inp', dace.Memlet('tmp[0]'))
    mx.add_in_connector('IN_a')
    mx.add_out_connector('OUT_a')
    state.add_edge(copy_t, 'out', mx, 'IN_a', dace.Memlet('a[i]'))
    state.add_edge(mx, 'OUT_a', state.add_write('a'), None, dace.Memlet(f'a[0:{N}]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('builder,label,expected', [(_access_to_access, 't', 0), (_mapentry_to_access, 't', 1),
                                                    (_access_to_mapexit, 'copy', 2)])
def test_each_pattern_shape_is_enumerated_under_its_own_index(builder, label, expected):
    """A tasklet's endpoint types must select the SAME ``expressions()`` entry the subgraph matcher
    would have bound it to. Getting the index wrong is silent: ``can_be_applied`` reads the wrong
    ``read``/``write`` pattern node, refuses, and the site is skipped instead of eliminated."""
    sdfg = builder()
    state = sdfg.states()[0]
    found = {
        expr_index
        for expr_index, binding in trivial_tasklet_candidates(state)
        if binding[TrivialTaskletElimination.tasklet].label == label
    }
    assert found == {expected}, f'{builder.__name__}/{label} must enumerate expr {expected}, got {sorted(found)}'


def test_a_copy_between_access_nodes_is_eliminated():
    sdfg = _access_to_access()
    assert EliminateTrivialTasklets().apply_pass(sdfg, {}) == 1, 'the copy tasklet must go'
    sdfg.validate()
    assert _tasklets(sdfg) == [], 'no tasklet survives'


def test_a_copy_out_of_a_map_entry_is_eliminated():
    sdfg = _mapentry_to_access()
    assert EliminateTrivialTasklets().apply_pass(sdfg, {}) == 1, 'the staged copy must go'
    sdfg.validate()
    assert _tasklets(sdfg) == [], 'no tasklet survives'


def test_a_copy_at_a_map_exit_boundary_is_kept():
    """The transformation's refusals must stay refusals: the pass owns traversal only."""
    sdfg = _access_to_mapexit()
    assert EliminateTrivialTasklets().apply_pass(sdfg, {}) is None, 'the boundary copy must be kept'
    assert 'out = inp' in _tasklets(sdfg), 'the boundary copy tasklet is still there'


def test_a_tasklet_that_casts_is_kept():
    """A copy between differently typed endpoints performs an implicit cast a memlet does not, so
    it is not trivial. Enumerated as a candidate, refused by the transformation."""
    sdfg = dace.SDFG('cast')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    state.add_edge(state.add_read('b'), None, t, 'inp', dace.Memlet('b[0]'))
    state.add_edge(t, 'out', state.add_write('a'), None, dace.Memlet('a[1]'))
    sdfg.validate()

    assert {i for i, _ in trivial_tasklet_candidates(state)} == {0}, 'the cast copy is still a candidate'
    assert EliminateTrivialTasklets().apply_pass(sdfg, {}) is None, 'the cast must survive'
    assert _tasklets(sdfg) == ['out = inp'], 'the casting tasklet is still there'


def test_a_many_edged_tasklet_is_never_built_into_a_candidate():
    """The arity gate is the transformation's first check; applying it during enumeration keeps a
    compute tasklet out of the candidate set entirely."""
    sdfg = dace.SDFG('binop')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {'x': None, 'y': None}, {'out': None}, 'out = x + y')
    state.add_edge(state.add_read('b'), None, t, 'x', dace.Memlet('b[0]'))
    state.add_edge(state.add_read('b'), None, t, 'y', dace.Memlet('b[1]'))
    state.add_edge(t, 'out', state.add_write('a'), None, dace.Memlet('a[0]'))
    sdfg.validate()
    assert list(trivial_tasklet_candidates(state)) == [], 'a two-input tasklet is not a candidate'


def test_every_site_in_one_state_is_eliminated_in_a_single_run():
    """The wrapper this pass replaced re-derived every match after each application; the pass must
    reach the same fixpoint without being run repeatedly."""
    sdfg = dace.SDFG('many_copies')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    for k in range(5):
        t = state.add_tasklet(f't{k}', {'inp': None}, {'out': None}, 'out = inp')
        state.add_edge(state.add_read('b'), None, t, 'inp', dace.Memlet(f'b[{k}]'))
        state.add_edge(t, 'out', state.add_write('a'), None, dace.Memlet(f'a[{k}]'))
    sdfg.validate()
    assert EliminateTrivialTasklets().apply_pass(sdfg, {}) == 5, 'all five copies go in one call'
    assert _tasklets(sdfg) == [], 'no tasklet survives'


def test_the_pass_and_the_wrapper_produce_the_same_graph():
    """The equivalence the replacement rests on: same count, same resulting graph -- down to
    memlets and tasklet bodies -- on a program carrying several shapes at once."""

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = b[i]
        c[:] = a[:] + b[:]

    base = prog.to_sdfg(simplify=False)
    by_pass, by_wrapper = copy.deepcopy(base), copy.deepcopy(base)

    n_pass = EliminateTrivialTasklets().apply_pass(by_pass, {}) or 0
    wrapper = PatternApplyOnceEverywhere([TrivialTaskletElimination()])
    wrapper.progress = False
    wrapper.validate = False
    applied = wrapper.apply_pass(by_wrapper, {}) or {}
    n_wrapper = sum(len(v) for v in applied.values())

    assert n_pass == n_wrapper, f'apply counts differ: pass {n_pass}, wrapper {n_wrapper}'
    assert n_pass > 0, 'the fixture must contain an eliminable copy'
    assert _signature(by_pass) == _signature(by_wrapper), 'the resulting graphs differ'


def test_nested_sdfg_states_are_visited():
    """``all_sdfgs_recursive`` is the traversal, not the top-level state list."""
    inner = dace.SDFG('inner')
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    istate.add_edge(istate.add_read('b'), None, t, 'inp', dace.Memlet('b[0]'))
    istate.add_edge(t, 'out', istate.add_write('a'), None, dace.Memlet('a[1]'))

    outer = dace.SDFG('outer')
    outer.add_array('a', [N], dace.float64)
    outer.add_array('b', [N], dace.float64)
    ostate = outer.add_state()
    nsdfg = ostate.add_nested_sdfg(inner, {'b': None}, {'a': None})
    ostate.add_edge(ostate.add_read('b'), None, nsdfg, 'b', dace.Memlet(f'b[0:{N}]'))
    ostate.add_edge(nsdfg, 'a', ostate.add_write('a'), None, dace.Memlet(f'a[0:{N}]'))
    outer.validate()

    assert EliminateTrivialTasklets().apply_pass(outer, {}) == 1, 'the nested copy must go'
    assert _tasklets(outer) == [], 'no tasklet survives inside the nested SDFG'


def test_the_elimination_preserves_the_computed_values():
    """Value equivalence, not just shape: the spliced memlet must still move the same data."""

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = b[i]

    sdfg = prog.to_sdfg(simplify=False)
    EliminateTrivialTasklets().apply_pass(sdfg, {})
    sdfg.validate()

    rng = np.random.default_rng(5)
    b = rng.random(N)
    got = np.zeros(N)
    sdfg(a=got, b=b)
    assert np.allclose(got, b), f'a[i] = b[i]; got {got}, ref {b}'


if __name__ == '__main__':
    pytest.main([__file__])

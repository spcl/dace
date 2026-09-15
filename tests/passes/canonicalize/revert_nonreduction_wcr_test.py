# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests ``RevertNonReductionWCR``, the traversal half of ``WCRToAugAssign``.

The pass replaced ``PatternApplyOnceEverywhere([WCRToAugAssign()])`` at every canonicalization
site, so what it owes is not "reverts a WCR" -- the transformation does that and is tested next
door -- but that walking WCR edges finds the SAME sites the subgraph matcher found, for every one
of the six patterns, and that nothing changes about which ones are refused.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import WCRToAugAssign
from dace.transformation.passes.canonicalize.revert_nonreduction_wcr import RevertNonReductionWCR, wcr_candidates
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere

N = 8


def _wcr_edges(sdfg):
    """Every surviving WCR edge in ``sdfg``, as ``(state, data, str(wcr))`` triples."""
    return sorted((s.label, str(e.data.data), str(e.data.wcr)) for sd in sdfg.all_sdfgs_recursive()
                  for s in sd.states() for e in s.edges() if e.data is not None and e.data.wcr is not None)


def _signature(sdfg):
    """Order-independent fingerprint of the graph shape, its memlets and its tasklet bodies."""
    sig = []
    for sd in sdfg.all_sdfgs_recursive():
        for s in sd.states():
            sig.append(('nodes', s.label, len(s.nodes())))
            for e in s.edges():
                sig.append(('edge', s.label, type(e.src).__name__, type(e.dst).__name__, str(e.data.data),
                            str(e.data.subset), str(e.data.wcr)))
            for n in s.nodes():
                if isinstance(n, nodes.Tasklet):
                    sig.append(('tasklet', s.label, n.code.as_string))
    return sorted(sig)


def _tasklet_to_access():
    """expr 0: ``Tasklet -[wcr]-> AccessNode``, an in-place fold with no enclosing map."""
    sdfg = dace.SDFG('expr0')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    state.add_edge(state.add_read('b'), None, t, 'inp', dace.Memlet('b[0]'))
    state.add_edge(t, 'out', state.add_write('a'), None, dace.Memlet(data='a', subset='1', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def _tasklet_through_exit():
    """expr 1: ``Tasklet -[wcr]-> MapExit -> AccessNode``, injective over the map param."""
    sdfg = dace.SDFG('expr1')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    state.add_memlet_path(state.add_read('b'), me, t, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_memlet_path(t,
                          mx,
                          state.add_write('a'),
                          src_conn='out',
                          memlet=dace.Memlet(data='a', subset='i', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def _access_to_access():
    """expr 2: ``AccessNode -[wcr]-> AccessNode``, the WCR copy canon leaves for ``a[:] += b[:]``."""
    sdfg = dace.SDFG('expr2')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    state.add_edge(state.add_read('b'), None, state.add_write('a'), None,
                   dace.Memlet(data='a', subset='1', other_subset='0', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def _access_through_exit():
    """expr 3: ``AccessNode -[wcr]-> MapExit -> AccessNode``, the privatized-source variant."""
    sdfg = dace.SDFG('expr3')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_scalar('priv', dace.float64, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    priv = state.add_access('priv')
    state.add_memlet_path(state.add_read('b'), me, t, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_edge(t, 'out', priv, None, dace.Memlet('priv[0]'))
    mx.add_in_connector('IN_a')
    mx.add_out_connector('OUT_a')
    state.add_edge(priv, None, mx, 'IN_a', dace.Memlet(data='a', subset='i', wcr='lambda x, y: x + y'))
    state.add_edge(mx, 'OUT_a', state.add_write('a'), None, dace.Memlet(data='a', subset=f'0:{N}'))
    sdfg.validate()
    return sdfg


def _exit_stranded_wcr():
    """expr 4: the WCR stranded on the OUTER ``map_exit -> output`` edge, inner write WCR-free."""
    sdfg = dace.SDFG('expr4')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_tasklet('t', {'__in1': None, '__in2': None}, {'__out': None}, '__out = (__in1 + __in2)')
    state.add_memlet_path(state.add_read('a'), me, t, dst_conn='__in1', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(state.add_read('b'), me, t, dst_conn='__in2', memlet=dace.Memlet('b[i]'))
    mx.add_in_connector('IN_a')
    mx.add_out_connector('OUT_a')
    state.add_edge(t, '__out', mx, 'IN_a', dace.Memlet('a[i]'))
    state.add_edge(mx, 'OUT_a', state.add_write('a'), None,
                   dace.Memlet(data='a', subset=f'0:{N}', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


def _cross_lane_reduction():
    """A GENUINE reduction: every lane folds into ``acc[0]``, so the WCR must be kept."""
    sdfg = dace.SDFG('reduction')
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    t = state.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    state.add_memlet_path(state.add_read('b'), me, t, dst_conn='inp', memlet=dace.Memlet('b[i]'))
    state.add_memlet_path(t,
                          mx,
                          state.add_write('acc'),
                          src_conn='out',
                          memlet=dace.Memlet(data='acc', subset='0', wcr='lambda x, y: x + y'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('builder,expected', [(_tasklet_to_access, 0),
                                              (_tasklet_through_exit, 1), (_access_to_access, 2),
                                              (_access_through_exit, 3), (_exit_stranded_wcr, 4)])
def test_each_pattern_shape_is_enumerated_under_its_own_index(builder, expected):
    """A WCR edge's endpoint types must select the SAME ``expressions()`` entry the subgraph
    matcher would have bound it to. Getting the index wrong is silent: ``can_be_applied`` simply
    refuses, and the site is skipped instead of reverted."""
    sdfg = builder()
    state = sdfg.states()[0]
    found = {expr_index for expr_index, _ in wcr_candidates(state)}
    assert expected in found, f'{builder.__name__} must enumerate expr {expected}, got {sorted(found)}'


@pytest.mark.parametrize(
    'builder', [_tasklet_to_access, _tasklet_through_exit, _access_to_access, _access_through_exit, _exit_stranded_wcr])
def test_every_pattern_shape_reverts_and_leaves_no_wcr(builder):
    """Enumeration is only useful if the site actually converts: each shape must lose its WCR."""
    sdfg = builder()
    assert RevertNonReductionWCR().apply_pass(sdfg, {}) == 1, f'{builder.__name__} must revert exactly one site'
    sdfg.validate()
    assert not _wcr_edges(sdfg), f'{builder.__name__} left a WCR behind'


def test_a_cross_lane_reduction_keeps_its_wcr():
    """The injectivity gate lives on the transformation and the pass must not widen it: reverting a
    real reduction to a plain store is a data race."""
    sdfg = _cross_lane_reduction()
    before = _wcr_edges(sdfg)
    assert RevertNonReductionWCR().apply_pass(sdfg, {}) is None, 'a cross-lane fold must not revert'
    assert _wcr_edges(sdfg) == before, 'the reduction keeps every WCR it started with'


def test_a_map_exit_writing_several_arrays_binds_the_one_the_edge_writes():
    """One exit, two output arrays: nothing in the path pattern ties ``output`` to the array the WCR
    edge writes, and the mismatched binding pairs this edge's memlet with the other array's access
    node. Both sites must revert, each against its own array (CloudSC's flux band writes four
    arrays through one exit)."""
    sdfg = dace.SDFG('two_outputs')
    for name in ('a', 'c', 'b'):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'))
    read = state.add_read('b')
    for out in ('a', 'c'):
        t = state.add_tasklet(f't_{out}', {'inp': None}, {'out': None}, 'out = inp')
        state.add_memlet_path(read, me, t, dst_conn='inp', memlet=dace.Memlet('b[i]'))
        state.add_memlet_path(t,
                              mx,
                              state.add_write(out),
                              src_conn='out',
                              memlet=dace.Memlet(data=out, subset='i', wcr='lambda x, y: x + y'))
    sdfg.validate()

    # ``add_memlet_path`` stamps the WCR on both the inner and the outer edge, so the same topology
    # also offers an expr-4 candidate per array -- which ``can_be_applied`` refuses because the inner
    # write is not WCR-free. What matters here is that neither index ever crosses the arrays.
    bound = {(expr_index, binding[WCRToAugAssign.output].data) for expr_index, binding in wcr_candidates(state)}
    assert {b for b in bound if b[0] == 1} == {(1, 'a'), (1, 'c')}, f'expr-1 bindings crossed arrays: {sorted(bound)}'
    assert RevertNonReductionWCR().apply_pass(sdfg, {}) == 2, 'both sites revert'
    assert not _wcr_edges(sdfg), 'no WCR survives'


def test_every_site_in_one_state_reverts_in_a_single_run():
    """The wrapper this pass replaced re-derived every match after each application; the pass must
    reach the same fixpoint without being run repeatedly."""
    sdfg = dace.SDFG('many_sites')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    for k in range(5):
        t = state.add_tasklet(f't{k}', {'inp': None}, {'out': None}, 'out = inp')
        state.add_edge(state.add_read('b'), None, t, 'inp', dace.Memlet(f'b[{k}]'))
        state.add_edge(t, 'out', state.add_write('a'), None,
                       dace.Memlet(data='a', subset=str(k), wcr='lambda x, y: x + y'))
    sdfg.validate()
    assert RevertNonReductionWCR().apply_pass(sdfg, {}) == 5, 'all five sites revert in one call'
    assert not _wcr_edges(sdfg), 'no WCR survives'


def test_the_pass_and_the_wrapper_produce_the_same_graph():
    """The equivalence the replacement rests on: same sites, same count, same resulting graph --
    down to memlets and tasklet bodies -- on a program carrying several shapes at once."""

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N], acc: dace.float64[1]):
        for i in dace.map[0:N]:
            a[i] += b[i]
        for i in dace.map[0:N]:
            acc[0] += b[i]

    base = prog.to_sdfg(simplify=False)
    by_pass, by_wrapper = copy.deepcopy(base), copy.deepcopy(base)

    n_pass = RevertNonReductionWCR().apply_pass(by_pass, {}) or 0
    wrapper = PatternApplyOnceEverywhere([WCRToAugAssign()])
    wrapper.progress = False
    wrapper.validate = False
    applied = wrapper.apply_pass(by_wrapper, {}) or {}
    n_wrapper = sum(len(v) for v in applied.values())

    assert n_pass == n_wrapper, f'apply counts differ: pass {n_pass}, wrapper {n_wrapper}'
    assert n_pass > 0, 'the fixture must contain a revertible site'
    assert _signature(by_pass) == _signature(by_wrapper), 'the resulting graphs differ'
    assert _wcr_edges(by_pass) == _wcr_edges(by_wrapper), 'a different set of WCRs survived'


def test_nested_sdfg_states_are_visited():
    """``all_sdfgs_recursive`` is the traversal, not the top-level state list: a WCR inside a body
    NestedSDFG is as revertible as one at the top, and the matcher this replaced recursed."""
    inner = dace.SDFG('inner')
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet('t', {'inp': None}, {'out': None}, 'out = inp')
    istate.add_edge(istate.add_read('b'), None, t, 'inp', dace.Memlet('b[0]'))
    istate.add_edge(t, 'out', istate.add_write('a'), None, dace.Memlet(data='a', subset='1', wcr='lambda x, y: x + y'))

    outer = dace.SDFG('outer')
    outer.add_array('a', [N], dace.float64)
    outer.add_array('b', [N], dace.float64)
    ostate = outer.add_state()
    nsdfg = ostate.add_nested_sdfg(inner, {'b': None}, {'a': None})
    ostate.add_edge(ostate.add_read('b'), None, nsdfg, 'b', dace.Memlet(f'b[0:{N}]'))
    ostate.add_edge(nsdfg, 'a', ostate.add_write('a'), None, dace.Memlet(f'a[0:{N}]'))
    outer.validate()

    assert RevertNonReductionWCR().apply_pass(outer, {}) == 1, 'the nested site must revert'
    assert not _wcr_edges(outer), 'no WCR survives inside the nested SDFG'


def test_the_revert_preserves_the_computed_values():
    """Value equivalence, not just shape: the reverted graph must still accumulate."""

    @dace.program
    def prog(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] += b[i]

    sdfg = prog.to_sdfg(simplify=False)
    assert RevertNonReductionWCR().apply_pass(sdfg, {}), 'the injective accumulate must revert'
    sdfg.validate()

    rng = np.random.default_rng(11)
    a0, b = rng.random(N), rng.random(N)
    got = a0.copy()
    sdfg(a=got, b=b)
    assert np.allclose(got, a0 + b), f'a[i] += b[i]; got {got}, ref {a0 + b}'


if __name__ == '__main__':
    pytest.main([__file__])

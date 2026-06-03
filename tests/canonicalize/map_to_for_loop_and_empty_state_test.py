# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end + structural tests for two canonicalization building blocks.

PART A -- ``MapToForLoop`` (``dace/transformation/dataflow/map_for_loop.py``):
sequentializing a parallel ``dace.map`` into a ``LoopRegion``-backed nested
SDFG. Each case is a python-frontend ``@dace.program`` over ``dace.symbol``
shapes; correctness is checked numerically against a pure-numpy oracle, and
the rewrite is checked structurally (``MapEntry`` count drops, ``LoopRegion``
count rises -- counted via ``all_nodes_recursive`` /
``all_control_flow_regions``). Multidimensional maps are expanded with
``MapExpansion`` first (``MapToForLoop`` only accepts a single map
parameter), so every map dimension becomes its own nested ``LoopRegion``.
The start-block edge case is exercised both ways: the map state as the
parent's first block, and with a real predecessor state running before it.

PART B -- ``EmptyStateElimination``
(``dace/transformation/passes/canonicalize/empty_state_elimination.py``):
splicing out empty, trivially-connected boundary states. Covers the
start-block splice (``start_block`` must follow the surviving successor),
the dead-tail sink drop, the not-start-block interior splice, the
fixpoint over a chain of empties, and the conservative guards (a state
holding dataflow, or reached by a conditional/assigning edge, must
survive). Every case is value-preserved end to end.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow import MapExpansion, MapToForLoop
from dace.transformation.passes.canonicalize.empty_state_elimination import EmptyStateElimination

N, L, A, B, C, D, E = (dace.symbol('N'), dace.symbol('L'), dace.symbol('A'), dace.symbol('B'), dace.symbol('C'),
                       dace.symbol('D'), dace.symbol('E'))


# --------------------------------------------------------------------------- #
# Structural helpers                                                           #
# --------------------------------------------------------------------------- #
def _map_entries(sdfg: dace.SDFG):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _loop_regions(sdfg: dace.SDFG):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def _apply_map_to_for(sdfg: dace.SDFG, min_dims: int):
    """Expand every multi-dim map then sequentialize every 1-D map, and
    assert the rewrite is *structurally* complete and consistent.

    The number of applications equals the number of 1-D maps present after
    expansion (one per map parameter across every map the frontend emitted --
    a whole-array assignment like ``pre[:] = a + 1`` is itself a map), so we
    derive the expectation from the post-expansion structure rather than
    hard-coding it: every ``MapEntry`` must become a ``LoopRegion`` and the
    loop count must rise by exactly the number of applications.

    :param sdfg: SDFG to rewrite in place.
    :param min_dims: Lower bound on the parameter count of the map under
                     test (sanity floor; the kernel may emit extra maps).
    :returns: The number of ``MapToForLoop`` applications.
    """
    assert len(_map_entries(sdfg)) >= 1, "fixture has no map"
    loops_before = len(_loop_regions(sdfg))
    # MapToForLoop only accepts a single map parameter; flatten N-D maps.
    sdfg.apply_transformations_repeated([MapExpansion])
    n_maps = len(_map_entries(sdfg))
    assert n_maps >= min_dims, f"expected >= {min_dims} 1-D maps post-expansion, got {n_maps}"
    applied = sdfg.apply_transformations_repeated([MapToForLoop])
    assert applied == n_maps, f"every map must be sequentialized: {n_maps} maps, {applied} applications"
    assert not _map_entries(sdfg), "a MapEntry survived sequentialization"
    assert len(_loop_regions(sdfg)) == loops_before + applied, "LoopRegion count did not rise one per application"
    sdfg.validate()
    return applied


# --------------------------------------------------------------------------- #
# PART A -- MapToForLoop                                                        #
# --------------------------------------------------------------------------- #
@dace.program
def simple_1d(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + 1.0


def test_a1_simple_1d():
    rng = np.random.default_rng(1)
    n = 16
    a = rng.random(n)
    oracle = a * 2.0 + 1.0

    sdfg = simple_1d.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    b_ref = np.full(n, -7.0)
    ref(a=a.copy(), b=b_ref, N=n)
    assert np.allclose(b_ref, oracle)

    _apply_map_to_for(sdfg, 1)

    b_post = np.full(n, -7.0)
    sdfg(a=a.copy(), b=b_post, N=n)
    assert np.allclose(b_post, oracle)


@dace.program
def indirect_stencil(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L]):
    for i, k in dace.map[0:N, 0:L]:
        out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


def test_a2_indirect_stencil():
    rng = np.random.default_rng(2)
    n, l = 12, 5
    w = rng.random((n, l))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    oracle = np.full((n, l), -7.0)
    for i in range(n):
        for k in range(l):
            oracle[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]

    sdfg = indirect_stencil.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    o_ref = np.full((n, l), -7.0)
    ref(w=w.copy(), cidx=cidx.copy(), out=o_ref, N=n, L=l)
    assert np.allclose(o_ref, oracle)

    _apply_map_to_for(sdfg, 2)

    o_post = np.full((n, l), -7.0)
    sdfg(w=w.copy(), cidx=cidx.copy(), out=o_post, N=n, L=l)
    assert np.allclose(o_post, oracle)


@dace.program
def very_multidim(src: dace.float64[A, B, C, D, E], dst: dace.float64[A, B, C, D, E]):
    for a, b, c, d, e in dace.map[0:A, 0:B, 0:C, 0:D, 0:E]:
        dst[a, b, c, d, e] = src[a, b, c, d, e] + 1.0


def test_a3_very_multidim_5d():
    rng = np.random.default_rng(3)
    sa, sb, sc, sd, se = 2, 3, 2, 3, 2
    src = rng.random((sa, sb, sc, sd, se))
    oracle = src + 1.0

    sdfg = very_multidim.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    d_ref = np.full((sa, sb, sc, sd, se), -7.0)
    ref(src=src.copy(), dst=d_ref, A=sa, B=sb, C=sc, D=sd, E=se)
    assert np.allclose(d_ref, oracle)

    # All 5 map dims must become nested LoopRegions.
    _apply_map_to_for(sdfg, 5)

    d_post = np.full((sa, sb, sc, sd, se), -7.0)
    sdfg(src=src.copy(), dst=d_post, A=sa, B=sb, C=sc, D=sd, E=se)
    assert np.allclose(d_post, oracle)


@dace.program
def very_multidim_gather(src: dace.float64[A, B, C, D, E], gidx: dace.int32[A, 2], dst: dace.float64[A, B, C, D, E]):
    for a, b, c, d, e in dace.map[0:A, 0:B, 0:C, 0:D, 0:E]:
        dst[a, b, c, d, e] = src[gidx[a, 0], b, c, d, e] - src[gidx[a, 1], b, c, d, e]


def test_a3b_very_multidim_5d_gather():
    rng = np.random.default_rng(13)
    sa, sb, sc, sd, se = 4, 2, 2, 2, 2
    src = rng.random((sa, sb, sc, sd, se))
    gidx = rng.integers(0, sa, size=(sa, 2), dtype=np.int32)
    oracle = np.full((sa, sb, sc, sd, se), -7.0)
    for a in range(sa):
        for b in range(sb):
            for c in range(sc):
                for d in range(sd):
                    for e in range(se):
                        oracle[a, b, c, d, e] = src[gidx[a, 0], b, c, d, e] - src[gidx[a, 1], b, c, d, e]

    sdfg = very_multidim_gather.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    d_ref = np.full((sa, sb, sc, sd, se), -7.0)
    ref(src=src.copy(), gidx=gidx.copy(), dst=d_ref, A=sa, B=sb, C=sc, D=sd, E=se)
    assert np.allclose(d_ref, oracle)

    _apply_map_to_for(sdfg, 5)

    d_post = np.full((sa, sb, sc, sd, se), -7.0)
    sdfg(src=src.copy(), gidx=gidx.copy(), dst=d_post, A=sa, B=sb, C=sc, D=sd, E=se)
    assert np.allclose(d_post, oracle)


@dace.program
def memset_map(out: dace.float64[N, L]):
    for i, k in dace.map[0:N, 0:L]:
        out[i, k] = 0.0


def test_a4_memset_map():
    n, l = 9, 4
    sdfg = memset_map.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    o_ref = np.full((n, l), 3.0)
    ref(out=o_ref, N=n, L=l)
    assert np.allclose(o_ref, 0.0)

    _apply_map_to_for(sdfg, 2)

    o_post = np.full((n, l), 3.0)
    sdfg(out=o_post, N=n, L=l)
    assert np.allclose(o_post, 0.0)


@dace.program
def map_is_start(a: dace.float64[N], b: dace.float64[N]):
    # The map is the FIRST (and only) block of the parent region.
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + 1.0


def _map_not_start_sdfg(n: int):
    """Built imperatively: a *plain* (non-map) predecessor state, an
    interstate edge, then a separate state holding the map. The Python
    frontend + ``simplify`` fuses a scalar predecessor into the map's own
    dataflow state, so the only way to genuinely put the map in a non-start
    block is to construct the multi-state CFG directly. Computes
    ``b[i] = a[i]*2 + (a[0] + 100)``."""
    sdfg = dace.SDFG('map_not_start')
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('b', [n], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)

    pred = sdfg.add_state('pred', is_start_block=True)  # no map -> plain state
    ra = pred.add_read('a')
    ws = pred.add_write('s')
    tc = pred.add_tasklet('addc', {'x'}, {'y'}, 'y = x + 100.0')
    pred.add_edge(ra, None, tc, 'x', dace.Memlet('a[0]'))
    pred.add_edge(tc, 'y', ws, None, dace.Memlet('s[0]'))

    mapst = sdfg.add_state('mapst')
    sdfg.add_edge(pred, mapst, dace.InterstateEdge())
    ra2 = mapst.add_read('a')
    rs = mapst.add_read('s')
    wb = mapst.add_write('b')
    me, mx = mapst.add_map('m', dict(i='0:%d' % n))
    tk = mapst.add_tasklet('body', {'xa', 'xs'}, {'yb'}, 'yb = xa * 2.0 + xs')
    mapst.add_memlet_path(ra2, me, tk, dst_conn='xa', memlet=dace.Memlet('a[i]'))
    mapst.add_memlet_path(rs, me, tk, dst_conn='xs', memlet=dace.Memlet('s[0]'))
    mapst.add_memlet_path(tk, mx, wb, src_conn='yb', memlet=dace.Memlet('b[i]'))
    return sdfg, pred, mapst


def test_a5_start_block_edge_cases():
    rng = np.random.default_rng(5)
    n = 10
    a = rng.random(n)

    # --- Variant 1: map IS the parent's start block ---------------------- #
    s1 = map_is_start.to_sdfg(simplify=True)
    start_before = s1.start_block
    oracle1 = a * 2.0 + 1.0
    _apply_map_to_for(s1, 1)
    s1.validate()
    # MapToForLoop nests inside the map's own state; the parent's start
    # block is the same state object and must still be the start block.
    assert s1.start_block is start_before, "start block changed for the start-block map"
    b1 = np.full(n, -7.0)
    s1(a=a.copy(), b=b1, N=n)
    assert np.allclose(b1, oracle1)

    # --- Variant 2: the map is NOT the parent's start block ------------- #
    s2, pred, mapst = _map_not_start_sdfg(n)
    assert s2.start_block is pred and any(isinstance(x, nodes.MapEntry) for x in mapst.nodes())
    oracle2 = a * 2.0 + (a[0] + 100.0)
    ref2 = copy.deepcopy(s2)
    b_ref = np.full(n, -7.0)
    ref2(a=a.copy(), b=b_ref, N=n)
    assert np.allclose(b_ref, oracle2)

    applied = s2.apply_transformations_repeated([MapToForLoop])
    assert applied == 1, f"MapToForLoop must sequentialize the one map, got {applied}"
    s2.validate()
    # The predecessor must still be the start block and still reach the map
    # state (execution order preserved -- the predecessor is not skipped).
    assert s2.start_block is pred, "start block changed -- predecessor would be skipped"
    assert [e.dst for e in s2.out_edges(pred)] == [mapst], "predecessor lost its edge into the map state"
    assert not [m for m, _ in s2.all_nodes_recursive() if isinstance(m, nodes.MapEntry)]
    assert any(isinstance(r, LoopRegion) for r in s2.all_control_flow_regions(recursive=True))
    b_post = np.full(n, -7.0)
    s2(a=a.copy(), b=b_post, N=n)
    assert np.allclose(b_post, oracle2), "not-start-block map produced wrong result"


def _map_with_two_successor_branches_sdfg(n: int):
    """Build a state machine where the map state has TWO unconditional
    successor states (a malformed-ish but legal CFG shape that
    ``simplify``'s ``control_flow_raising`` can produce mid-pipeline):
    ``pred -> mapst -> tail_a`` and ``mapst -> tail_b``, both
    interstate edges unconditional. With ``inline_after=True`` the
    naive migration of ``mapst`` 's dataflow into a fresh successor
    leaves the placeholder ``mapst`` with three out-edges (the new
    ``mapst -> target`` plus the original two), giving the placeholder
    multiple unconditional edges. ``control_flow_raising`` then lifts
    these into a ``ConditionalBlock`` with an ``else`` branch that
    isn't last and ``DeadStateElimination`` aborts. The fix reparents
    the original successor edges onto ``target_state`` so the
    placeholder has exactly one out-edge.
    """
    sdfg = dace.SDFG('map_two_successors')
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('b', [n], dace.float64)
    sdfg.add_array('c', [n], dace.float64)

    pred = sdfg.add_state('pred', is_start_block=True)
    mapst = sdfg.add_state('mapst')
    tail_a = sdfg.add_state('tail_a')
    tail_b = sdfg.add_state('tail_b')

    sdfg.add_edge(pred, mapst, dace.InterstateEdge())
    sdfg.add_edge(mapst, tail_a, dace.InterstateEdge())
    sdfg.add_edge(mapst, tail_b, dace.InterstateEdge())

    ra = mapst.add_read('a')
    wb = mapst.add_write('b')
    me, mx = mapst.add_map('m', dict(i='0:%d' % n))
    tk = mapst.add_tasklet('dbl', {'x'}, {'y'}, 'y = x * 2.0')
    mapst.add_memlet_path(ra, me, tk, dst_conn='x', memlet=dace.Memlet('a[i]'))
    mapst.add_memlet_path(tk, mx, wb, src_conn='y', memlet=dace.Memlet('b[i]'))

    # Tail states stay map-less so ``apply_transformations_repeated``
    # only fires on the one map under test. Each tail just copies ``b``
    # to ``c`` via an interstate-style direct edge between data nodes.
    rb_a = tail_a.add_read('b')
    wc_a = tail_a.add_write('c')
    tail_a.add_nedge(rb_a, wc_a, dace.Memlet('b[0:%d]->c[0:%d]' % (n, n)))
    rb_b = tail_b.add_read('b')
    wc_b = tail_b.add_write('c')
    tail_b.add_nedge(rb_b, wc_b, dace.Memlet('b[0:%d]->c[0:%d]' % (n, n)))
    return sdfg, pred, mapst, tail_a, tail_b


def test_a6_multi_successor_reparent_on_inline():
    """Regression: ``inline_after=True`` must reparent the map state's
    pre-existing successor edges onto the new ``target_state``, leaving
    the original block with exactly one out-edge. Without the
    reparent, the placeholder ends up with multiple unconditional
    out-edges and the downstream ``control_flow_raising`` /
    ``DeadStateElimination`` chain aborts with
    ``InvalidSDFGNodeError(Conditional block ... else branch is not
    the last branch)``."""
    n = 8
    sdfg, pred, mapst, tail_a, tail_b = _map_with_two_successor_branches_sdfg(n)

    applied = sdfg.apply_transformations_repeated([MapToForLoop])
    assert applied == 1
    sdfg.validate()

    # The placeholder map state must still be present (start-block
    # invariant is unaffected here) and must have exactly ONE
    # out-edge -- the freshly added migration edge. The pre-existing
    # ``mapst -> tail_a`` and ``mapst -> tail_b`` edges must have been
    # reparented to ``target_state``.
    out_from_mapst = sdfg.out_edges(mapst)
    assert len(out_from_mapst) == 1, (f'placeholder block must have one out-edge after migration, '
                                      f'got {len(out_from_mapst)}: '
                                      f'{[(e.dst.label, e.data.is_unconditional()) for e in out_from_mapst]}')
    target = out_from_mapst[0].dst

    # The reparented edges may pass through ``isolate_nested_sdfg`` 's
    # pre/middle/post split before reaching the tails -- the contract
    # is reachability, not edge identity. Confirm both tails are still
    # reachable from the migration target via the state-machine
    # forward closure (execution order preserved, no successor
    # dropped).
    reachable = set()
    queue = [target]
    while queue:
        node = queue.pop()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(e.dst for e in sdfg.out_edges(node) if e.dst not in reachable)
    assert tail_a in reachable, 'tail_a unreachable from the migration target after inline'
    assert tail_b in reachable, 'tail_b unreachable from the migration target after inline'

    # No ConditionalBlock with a misplaced else branch must exist
    # anywhere in the SDFG.
    from dace.sdfg.state import ConditionalBlock
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(cfg, ConditionalBlock):
            for i, (cond, _) in enumerate(cfg.branches):
                assert cond is not None or i == len(cfg.branches) - 1, \
                    f'ConditionalBlock {cfg.label!r} has else at index {i}/{len(cfg.branches)}'


# --------------------------------------------------------------------------- #
# PART B -- EmptyStateElimination                                              #
# --------------------------------------------------------------------------- #
def _add_double_tasklet(state: dace.SDFGState, in_name: str, out_name: str, n: int):
    """Wire ``out[0:n] = in[0:n] * 2`` into ``state`` via an elementwise map."""
    r = state.add_read(in_name)
    w = state.add_write(out_name)
    me, mx = state.add_map('m', dict(i='0:%d' % n))
    t = state.add_tasklet('dbl', {'x'}, {'y'}, 'y = x * 2.0')
    state.add_memlet_path(r, me, t, dst_conn='x', memlet=dace.Memlet('%s[i]' % in_name))
    state.add_memlet_path(t, mx, w, src_conn='y', memlet=dace.Memlet('%s[i]' % out_name))


def _base_sdfg(name: str, n: int):
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('b', [n], dace.float64)
    return sdfg


def test_b1_start_block_splice():
    """Empty start state, unconditional edge to the worker -> spliced; the
    surviving worker becomes the new start block, value preserved."""
    n = 8
    sdfg = _base_sdfg('b1', n)
    empty = sdfg.add_state('empty_pre', is_start_block=True)
    work = sdfg.add_state('work')
    sdfg.add_edge(empty, work, dace.InterstateEdge())
    _add_double_tasklet(work, 'a', 'b', n)

    assert sdfg.start_block is empty
    removed = EmptyStateElimination().apply_pass(sdfg, {})
    assert removed == 1, "the empty start state was not removed"
    assert sdfg.start_block is work, "start block did not follow the surviving successor"
    sdfg.validate()

    a = np.arange(n, dtype=np.float64) + 1.0
    b = np.zeros(n)
    sdfg(a=a.copy(), b=b)
    assert np.allclose(b, a * 2.0)


def test_b2_dead_tail_sink_drop():
    """An empty sink with predecessors but no successor is a dead tail and
    is dropped; the real work still runs."""
    n = 6
    sdfg = _base_sdfg('b2', n)
    work = sdfg.add_state('work', is_start_block=True)
    tail = sdfg.add_state('empty_post')
    sdfg.add_edge(work, tail, dace.InterstateEdge())
    _add_double_tasklet(work, 'a', 'b', n)

    removed = EmptyStateElimination().apply_pass(sdfg, {})
    assert removed == 1, "the dead empty sink was not dropped"
    assert tail not in sdfg.nodes()
    assert sdfg.start_block is work
    sdfg.validate()

    a = np.arange(n, dtype=np.float64) + 1.0
    b = np.zeros(n)
    sdfg(a=a.copy(), b=b)
    assert np.allclose(b, a * 2.0)


def test_b3_interior_not_start_splice_and_fixpoint():
    """A chain ``work -> empty1 -> empty2 -> work2`` collapses to
    ``work -> work2`` (fixpoint over multiple empties); neither empty is the
    start block. Both worker states still execute in order."""
    n = 7
    sdfg = dace.SDFG('b3')
    sdfg.add_array('a', [n], dace.float64)
    sdfg.add_array('t', [n], dace.float64, transient=True)
    sdfg.add_array('b', [n], dace.float64)
    work = sdfg.add_state('work', is_start_block=True)
    e1 = sdfg.add_state('empty1')
    e2 = sdfg.add_state('empty2')
    work2 = sdfg.add_state('work2')
    sdfg.add_edge(work, e1, dace.InterstateEdge())
    sdfg.add_edge(e1, e2, dace.InterstateEdge())
    sdfg.add_edge(e2, work2, dace.InterstateEdge())
    _add_double_tasklet(work, 'a', 't', n)
    _add_double_tasklet(work2, 't', 'b', n)

    removed = EmptyStateElimination().apply_pass(sdfg, {})
    assert removed == 2, f"expected both empties spliced, got {removed}"
    assert e1 not in sdfg.nodes() and e2 not in sdfg.nodes()
    assert sdfg.start_block is work, "start block must remain the first worker"
    succ = [e.dst for e in sdfg.out_edges(work)]
    assert succ == [work2], "predecessor was not rewired straight to the successor"
    sdfg.validate()

    a = np.arange(n, dtype=np.float64) + 1.0
    b = np.zeros(n)
    sdfg(a=a.copy(), b=b)
    assert np.allclose(b, a * 4.0), "execution order across the spliced chain was not preserved"


def test_b4_conservative_guards():
    """The pass must NOT touch a state that holds dataflow, nor splice
    across a conditional or assigning interstate edge."""
    n = 5

    # (a) A non-empty state is never removed. (An empty trivially-connected
    # start state IS correctly spliced -- that is the pass's job -- so the
    # guarantee under test is specifically that the state holding dataflow
    # survives, not that the pass is a no-op here.)
    sdfg = _base_sdfg('b4a', n)
    pre = sdfg.add_state('pre', is_start_block=True)
    work = sdfg.add_state('work')
    sdfg.add_edge(pre, work, dace.InterstateEdge())
    _add_double_tasklet(work, 'a', 'b', n)  # 'work' has nodes
    EmptyStateElimination().apply_pass(sdfg, {})
    assert work in sdfg.nodes(), "a non-empty state must never be removed"
    assert pre not in sdfg.nodes(), "the empty, trivially-connected start state should be spliced out"
    assert sdfg.start_block is work, "start block must follow the spliced-out empty state"

    # (b) Empty state reached by a conditional edge must survive.
    sdfg = _base_sdfg('b4b', n)
    sdfg.add_symbol('c', dace.int32)
    head = sdfg.add_state('head', is_start_block=True)
    empty = sdfg.add_state('empty')
    work = sdfg.add_state('work')
    sdfg.add_edge(head, empty, dace.InterstateEdge(condition='c > 0'))
    sdfg.add_edge(empty, work, dace.InterstateEdge())
    _add_double_tasklet(work, 'a', 'b', n)
    assert EmptyStateElimination().apply_pass(sdfg, {}) is None, "spliced across a conditional edge"
    assert empty in sdfg.nodes()

    # (c) Empty state whose out-edge carries an assignment must survive.
    sdfg = _base_sdfg('b4c', n)
    sdfg.add_symbol('k', dace.int32)
    head = sdfg.add_state('head', is_start_block=True)
    empty = sdfg.add_state('empty')
    work = sdfg.add_state('work')
    sdfg.add_edge(head, empty, dace.InterstateEdge())
    sdfg.add_edge(empty, work, dace.InterstateEdge(assignments={'k': '1'}))
    _add_double_tasklet(work, 'a', 'b', n)
    # The empty ``head`` start state has a trivial out-edge and is correctly
    # spliced; the guarantee under test is that ``empty`` -- whose out-edge
    # carries an assignment -- is NOT spliced (the assignment would be lost).
    EmptyStateElimination().apply_pass(sdfg, {})
    assert empty in sdfg.nodes(), "must not splice across an assigning edge"
    assert work in sdfg.nodes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

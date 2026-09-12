# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What ``canonicalize`` + ``finalize_for_target`` leaves behind on three focus40 kernels.

A code-check sweep over 40 loop-level-reasoning kernels found three where the canonicalized form
still had a ``Sequential`` map: ``ext_war_unit``, ``fuse_move_ifs``, ``tsvc_2_s1232``. A map count
alone cannot say whether that is correct (a real cross-iteration dependence) or a missed
parallelization (a pass that declined for no dependence reason). Each case here is pinned to
which one it is, by NAME, not by count:

* ``ext_war_unit`` -- ``a[i] = a[i+1] + b[i]`` carries a genuine distance-1 write-after-read
  anti-dependence. ``chunk_anti_dependence.py`` (cpu_specialization) resolves it by running
  chunks in parallel and each chunk's interior in order; the two Sequential MAPS left over are
  its single-iteration seam/prologue points -- there is no loop to extract from a domain of one.
* ``tsvc_2_s1232`` -- a data-parallel inner loop (no aliasing between iterations at all) that
  stays ``Sequential`` only because ``sequentialize_unprofitable_parallel_scopes.py``'s rule 1
  forbids a SECOND ``CPU_Multicore`` region nested inside one that is already parallel (OpenMP
  gives one team level; nesting would oversubscribe, not speed anything up). The outer loop
  already carries the parallelism; the inner one is correctly ``Sequential`` by POLICY, not by
  dependence. This is the kernel that witnesses the policy.
* ``fuse_move_ifs`` used to witness the same policy, and since ``56c2a2ec3`` no longer does: its
  two nests fuse into ONE flat map over both dimensions, so no second parallel region is left to
  sequentialize. Its own policy test said in as many words that collapsing the two loops into one
  map is a permitted outcome, so the case stays here pinning the OTHER half of that sentence --
  one parallel region, no map nested in another map, and the data-dependent guard still standing
  as a branch predicate rather than quietly dropped by the fusion.

Each kernel gets a numeric test (canonicalize must stay value-preserving) and a structural test
that would fail if the map inventory changes shape, if a future edit turns a genuinely serial map
parallel (a race) or leaves a genuinely parallel map serial (a silent regression), or if a fusion
loses a guard.
"""
import typing

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.dtypes import ScheduleType
from dace.ordered import OrderedSet
from dace.properties import CodeBlock
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, SDFGState
from dace.transformation.helpers import get_parent_map_and_loop_scopes
from dace.transformation.passes.canonicalize import finalize, pipeline as cp

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES


def canonicalized_for_cpu(sdfg: dace.SDFG) -> dace.SDFG:
    """The production CPU recipe: ``canonicalize`` then ``finalize_for_target``, in place."""
    cp.canonicalize(sdfg, target='cpu')
    finalize.finalize_for_target(sdfg, 'cpu')
    return sdfg


def maps_with_schedule(sdfg: dace.SDFG, schedule: dace.ScheduleType) -> typing.List[typing.Tuple[nd.MapEntry, object]]:
    """``[(MapEntry, owning SDFGState)]`` for every map of ``schedule`` anywhere in ``sdfg``."""
    return [(n, state) for n, state in sdfg.all_nodes_recursive()
            if isinstance(n, nd.MapEntry) and n.map.schedule == schedule]


def root_name(sdfg: dace.SDFG, name: str) -> str:
    """``name`` as the ROOT SDFG spells it, following nested-SDFG connector renames outward.

    Writes are collected per nested SDFG, where a connector rename can give the same array a
    different local name; resolving outward lets an assertion name the array the kernel signature
    names instead of whatever the current nesting happens to call it.
    """
    while sdfg.parent_nsdfg_node is not None:
        state, node = sdfg.parent, sdfg.parent_nsdfg_node
        edges = [e for e in state.in_edges(node) if e.dst_conn == name]
        edges += [e for e in state.out_edges(node) if e.src_conn == name]
        if not edges or edges[0].data.data is None:
            return name
        name, sdfg = edges[0].data.data, state.sdfg
    return name


def tasklets_writing(sdfg: dace.SDFG, data: str) -> typing.List[typing.Tuple[nd.Tasklet, SDFGState]]:
    """``[(Tasklet, owning SDFGState)]`` for every tasklet whose output lands in ``data``.

    The tasklet is what the guard has to sit above. An ``AccessNode`` for ``data`` in an enclosing
    state is only the outward end of a memlet path, so counting access nodes would report the
    write once more at every nesting level it passes through, outside the guard each time.
    """
    found: typing.List[typing.Tuple[nd.Tasklet, SDFGState]] = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.nodes():
                if not isinstance(node, nd.Tasklet):
                    continue
                for edge in state.out_edges(node):
                    end = state.memlet_path(edge)[-1].dst
                    if isinstance(end, nd.AccessNode) and root_name(sd, end.data) == data:
                        found.append((node, state))
                        break
    return found


def enclosing_branch_conditions(block: ControlFlowBlock) -> typing.List[str]:
    """Predicate text of every ``ConditionalBlock`` branch ``block`` sits inside, innermost first."""
    conditions: typing.List[str] = []
    child, parent = block, block.parent_graph
    while parent is not None:
        if isinstance(parent, ConditionalBlock):
            conditions += [c.as_string for c, region in parent.branches if region is child and c is not None]
        child, parent = parent, parent.parent_graph
    return conditions


def names_behind(sdfg: dace.SDFG, condition: str) -> OrderedSet[str]:
    """Names ``condition`` reads once staged interstate assignments are substituted back in.

    Canonicalize stages an array read into a symbol on an interstate edge (``cond_index =
    cond[i]``) and writes the SYMBOL into the branch predicate, so the predicate's own free
    symbols no longer mention the array the guard actually tests.
    """
    assigns = {
        lhs: rhs
        for sd in sdfg.all_sdfgs_recursive()
        for cfg in sd.all_control_flow_regions(recursive=False)
        for e in cfg.edges()
        for lhs, rhs in e.data.assignments.items()
    }
    names: OrderedSet[str] = OrderedSet(str(s) for s in CodeBlock(condition).get_free_symbols())
    for _ in range(len(assigns) + 1):
        grown = OrderedSet(names)
        for name in names:
            if name in assigns:
                grown |= OrderedSet(str(s) for s in CodeBlock(assigns[name]).get_free_symbols())
        if len(grown) == len(names):
            return grown
        names = grown
    return names


# --------------------------------------------------------------------------------------------- #
# ext_war_unit: a[i] = a[i+1] + b[i] -- a real distance-1 WAR, resolved by chunking.             #
# --------------------------------------------------------------------------------------------- #

WAR_LEN_1D = dace.symbol('LEN_1D', dtype=dace.int64, positive=True)


@dace.program
def ext_war_unit(a: dace.float64[WAR_LEN_1D], b: dace.float64[WAR_LEN_1D]):
    for i in range(WAR_LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


def reference_ext_war_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = a.copy()
    for i in range(a.shape[0] - 1):
        out[i] = out[i + 1] + b[i]
    return out


def test_ext_war_unit_matches_reference_across_a_chunk_boundary():
    """``N`` spans several 4096-wide chunks plus a partial one, so both the chunk-parallel body
    and the boundary seams the chunking scheme adds actually execute, not just the size-1 cases.
    """
    n = 4096 * 3 + 137
    rng = np.random.default_rng(10)
    a, b = rng.random(n), rng.random(n)
    expected = reference_ext_war_unit(a, b)

    sdfg = canonicalized_for_cpu(ext_war_unit.to_sdfg(simplify=False))
    got = a.copy()
    csdfg = sdfg.compile()
    csdfg(a=got, b=b.copy(), LEN_1D=n)
    assert np.allclose(got, expected), 'the WAR chunk/seam rewrite must reproduce in-order semantics'


def test_ext_war_unit_sequential_maps_are_single_iteration_seams():
    """The WAR is real, so something here must be Sequential -- but not a MAP: the chunk interior
    that actually carries the order (iteration i reads a[i+1], which only i+1 overwrites) is a
    ``LoopRegion`` inside the parallel per-chunk map, not a Map, so it never appears in this
    inventory at all. The two Sequential MAPS that remain are ``chunk_anti_dependence.py``'s
    documented boundary points (dace/transformation/passes/cpu_specialization/
    chunk_anti_dependence.py:23-26): the prologue ``map i in [lo, lo]`` and each chunk's last
    element ``map i in [e(t), e(t)]``. Both ranges are PROVABLY exactly one iteration for every
    N this loop ever runs with (N >= 2, since ``range(N - 1)`` must be non-empty to reach the
    body at all) -- a Sequential schedule over a domain of one point is correct by construction,
    not a missed parallelization. Widening either range without keeping it singleton would put a
    real anti-dependence under a parallel schedule, i.e. a race.
    """
    sdfg = canonicalized_for_cpu(ext_war_unit.to_sdfg(simplify=False))
    seq_maps = maps_with_schedule(sdfg, ScheduleType.Sequential)
    par_maps = maps_with_schedule(sdfg, ScheduleType.CPU_Multicore)
    assert len(seq_maps) == 2, f'expected exactly 2 boundary seam maps, got {len(seq_maps)}'
    assert len(par_maps) == 2, f'expected exactly 2 chunk-parallel maps, got {len(par_maps)}'

    for map_entry, _state in seq_maps:
        size = map_entry.map.range.num_elements()
        for n in (2, 3, 4096, 4097, 8192, 4096 * 3 + 137):
            assert symbolic.evaluate(size, {'LEN_1D': n}) == 1, \
                f'seam map {map_entry.map.label} must be exactly one iteration at LEN_1D={n}, got {size}'

    # One seam (the per-chunk finisher) is nested inside a chunk-parallel map; the other (the
    # i=0 prologue) runs once before the parallel region, at top level -- both are singleton
    # regardless of nesting, which is the property this test protects.
    nested = [get_parent_map_and_loop_scopes(sdfg, n, s) for n, s in seq_maps]
    assert sum(1 for scopes in nested if scopes) == 1, \
        'exactly one seam map must be nested inside the chunk-parallel map'


# --------------------------------------------------------------------------------------------- #
# fuse_move_ifs: two nests that fuse into ONE flat parallel map, guard kept as a predicate.      #
# --------------------------------------------------------------------------------------------- #

FMI_LEN_2D = dace.symbol('LEN_2D', dtype=dace.int64, positive=True)
FMI_K = dace.symbol('K', dtype=dace.int64, positive=True)


@dace.program
def fuse_move_ifs(a: dace.float64[FMI_LEN_2D, FMI_LEN_2D], b: dace.float64[FMI_LEN_2D, FMI_LEN_2D],
                  src: dace.float64[FMI_LEN_2D, FMI_LEN_2D], cond: dace.float64[FMI_LEN_2D]):
    for i in range(FMI_LEN_2D):
        if cond[i] > 0.0:
            for j in range(FMI_LEN_2D):
                a[i, j] = src[i, j] * 2.0
    if FMI_K > 0:
        for i in range(FMI_LEN_2D):
            for j in range(FMI_LEN_2D):
                b[i, j] = src[i, j] + 1.0


def reference_fuse_move_ifs(a: np.ndarray, b: np.ndarray, src: np.ndarray, cond: np.ndarray,
                            k: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    a, b = a.copy(), b.copy()
    n = a.shape[0]
    for i in range(n):
        if cond[i] > 0.0:
            for j in range(n):
                a[i, j] = src[i, j] * 2.0
    if k > 0:
        for i in range(n):
            for j in range(n):
                b[i, j] = src[i, j] + 1.0
    return a, b


def test_fuse_move_ifs_matches_reference_in_every_guard_regime():
    """One build, three ``cond`` regimes. The all-negative one is the regime that separates a
    surviving guard from a dropped one: with no ``cond[i]`` positive the reference leaves ``a``
    exactly as it came in, so a fusion that folded the guarded write into the unconditional map
    body would show up as ``a`` overwritten everywhere, while the mixed regime alone could still
    pass on the rows that happen to be positive.
    """
    n = 6
    rng = np.random.default_rng(3)
    a, b, src = rng.random((n, n)), rng.random((n, n)), rng.random((n, n))
    conds = {
        'mixed': rng.random(n) - 0.5,
        'all_negative': -rng.random(n) - 0.5,
        'all_positive': rng.random(n) + 0.5,
    }

    sdfg = canonicalized_for_cpu(fuse_move_ifs.to_sdfg(simplify=False))
    csdfg = sdfg.compile()
    for regime, cond in conds.items():
        expected_a, expected_b = reference_fuse_move_ifs(a, b, src, cond, k=1)
        got_a, got_b = a.copy(), b.copy()
        csdfg(a=got_a, b=got_b, src=src.copy(), cond=cond.copy(), LEN_2D=n, K=1)
        assert np.array_equal(got_a, expected_a), f'{regime}: the guarded a-write must be preserved'
        assert np.array_equal(got_b, expected_b), f'{regime}: the unconditional b-write must be preserved'


def test_fuse_move_ifs_fuses_to_one_parallel_region_and_keeps_the_guard_as_a_predicate():
    """``a[i, j] = src[i, j] * 2.0`` under ``if cond[i] > 0.0``, and an unconditional
    ``b[i, j] = src[i, j] + 1.0``, have no cross-iteration dependence at all -- every ``(i, j)``
    writes a distinct cell and reads only its own ``src`` element. Until ``56c2a2ec3`` the two
    nests stayed apart and the guarded inner j-loop came out ``Sequential`` under a parallel
    per-i map, by rule 1 of ``SequentializeUnprofitableParallelScopes.decide_map``
    (dace/transformation/passes/cpu_specialization/sequentialize_unprofitable_parallel_scopes.py
    :157-161): no second ``CPU_Multicore`` region inside one that is already parallel. Sinking the
    guard's condition prep with the guard made both nests perfect, they fuse, and ONE flat map
    over both dimensions is what is left -- an outcome the previous pin named as permitted, and
    the case ``tsvc_2_s1232`` below still witnesses the nested-parallelism rule itself.

    What must hold for the fused form is pinned here instead, and neither half is weaker than the
    map count it replaces. Nothing may be nested inside a parallel map that is parallel again --
    that is the ``#pragma omp parallel for`` inside ``#pragma omp parallel for`` the previous pin
    guarded against, asserted directly rather than inferred from a count of two. And the fusion
    must not have bought its perfect nest by dropping the guard: the ``a`` write has to stay
    behind a branch predicate that reads ``cond``, since a guard folded away would make the fused
    map write ``a`` on every row.

    ``K`` is declared ``positive=True``, so ``K > 0`` is proved at trace time: the b-nest is
    already unguarded before canonicalize sees it, and this program has exactly one guard.
    """
    sdfg = canonicalized_for_cpu(fuse_move_ifs.to_sdfg(simplify=False))
    seq_maps = maps_with_schedule(sdfg, ScheduleType.Sequential)
    par_maps = maps_with_schedule(sdfg, ScheduleType.CPU_Multicore)
    assert len(par_maps) == 1, f'the two nests must fuse into exactly 1 parallel map, got {len(par_maps)}'
    assert not seq_maps, f'nothing is left to sequentialize once the nests fuse, got {len(seq_maps)}'

    (fused, _fused_state), = par_maps
    assert len(fused.map.params) == 2, f'the fused map must cover both dimensions, got {fused.map.params}'
    for begin, end, _step in fused.map.range:
        assert (begin, str(end)) == (0, 'LEN_2D - 1'), \
            f'the fused map must span the full 2D domain, got {fused.map.range}'

    # Rule 1's actual subject, stated as itself: no map re-enters a parallel scope.
    for map_entry, state in maps_with_schedule(sdfg, ScheduleType.CPU_Multicore):
        enclosing = get_parent_map_and_loop_scopes(sdfg, map_entry, state)
        assert not [s for s in enclosing if isinstance(s, nd.MapEntry)], \
            f'{map_entry.map.label} is parallel inside another map -- nested OpenMP regions'

    a_writers = tasklets_writing(sdfg, 'a')
    assert a_writers, 'the a-write must survive canonicalization'
    for tasklet, state in a_writers:
        conditions = enclosing_branch_conditions(state)
        assert conditions, f'{tasklet.label} writes a unguarded -- the fusion dropped the branch'
        assert any('cond' in names_behind(sdfg, c) for c in conditions), \
            f'{tasklet.label} is guarded by {conditions}, which no longer tests cond'


# --------------------------------------------------------------------------------------------- #
# tsvc_2_s1232: same no-nested-parallelism policy, on the untiled TSVC corpus kernel.            #
# --------------------------------------------------------------------------------------------- #


def canonicalized_s1232(tag: str) -> typing.Tuple[object, dace.SDFG]:
    kernel = tsvc.collect(name='s1232_d_single')[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    return kernel, canonicalized_for_cpu(sdfg)


def test_tsvc_2_s1232_matches_reference():
    kernel, sdfg = canonicalized_s1232('sequential_maps_survive_finalize_numeric')
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {name: arr.copy() for name, arr in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {name: arr.copy() for name, arr in arrays.items()}
    csdfg = sdfg.compile()
    csdfg(**got, **call_kwargs)
    for name, arr in arrays.items():
        assert np.allclose(ref[name], got[name]), f'{kernel.name}: value mismatch on {name}'


def test_tsvc_2_s1232_inner_loop_is_sequential_by_nested_parallelism_policy_not_a_dependence():
    """``for j in range(LEN_2D): for i in range(j*VLEN, LEN_2D): aa[i,j] = bb[i,j] + cc[i,j]``
    has no dependence either: every ``(i, j)`` pair writes a distinct ``aa`` cell from distinct
    ``bb``/``cc`` cells, triangular iteration domain or not. Canonicalize UNTILES the ragged bound
    into an outer map over ``i`` (``LEN_2D`` independent values, made ``CPU_Multicore``) and an
    inner map over the valid ``j <= floor(i / VLEN)`` (made ``Sequential``) -- the SAME rule 1
    nested-parallelism decision as ``fuse_move_ifs``, at the same file:line
    (dace/transformation/passes/cpu_specialization/sequentialize_unprofitable_parallel_scopes.py
    :157-161). The outer loop already gives every thread independent work; parallelizing the
    (much smaller, per-i) inner loop too would only add a second, nested OpenMP region.
    """
    _kernel, sdfg = canonicalized_s1232('sequential_maps_survive_finalize_structural')
    seq_maps = maps_with_schedule(sdfg, ScheduleType.Sequential)
    par_maps = maps_with_schedule(sdfg, ScheduleType.CPU_Multicore)
    assert len(seq_maps) == 1, f'expected exactly 1 sequential (untiled inner) map, got {len(seq_maps)}'
    assert len(par_maps) == 1, f'expected exactly 1 parallel (untiled outer) map, got {len(par_maps)}'

    (seq_entry, seq_state), = seq_maps
    (par_entry, _par_state), = par_maps
    enclosing = get_parent_map_and_loop_scopes(sdfg, seq_entry, seq_state)
    assert enclosing == [par_entry], 'the sequential inner map must be nested directly in the one parallel outer map'


if __name__ == '__main__':
    pytest.main([__file__])

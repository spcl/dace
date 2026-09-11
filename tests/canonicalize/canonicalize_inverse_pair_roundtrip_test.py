# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ping-pong / inverse-pair round-trip tests.

Several canonicalize transformations are exact inverses; composing them
(down-then-up or up-then-down) must land the guard / loop / map exactly
where intended and stay value-preserving. These tests prove the inverse
property directly -- the basis for the pipeline being free to choose a
direction without losing information.

Pairs covered:

* **MoveIfIntoLoop  <->  MoveLoopInvariantIfUp** (the primary pair):
  push an invariant guard INTO a loop, then hoist it back OUT -- exercised
  with BOTH ``require_full_hoist=True`` and ``False`` on the up-pass.
* **LoopToMap  <->  MapToForLoop**: a parallel loop becomes a map and
  back, round-tripping to a parallel map.
* **MapFission  <->  MapFusion** (vertical): a fused multi-statement map
  fissions then re-fuses to a single map.

Each test checks value preservation against the original (non-transformed)
SDFG, the INTERMEDIATE structure after the first direction, and the structure
after the second. Pinning the intermediate is what makes a round-trip test
mean anything: when both directions no-op the input shape comes back
unchanged, so the end state alone cannot tell a round trip from a pair of
refusals.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp

N = dace.symbol('N')
M = dace.symbol('M')


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def _top_conds(sdfg):
    return [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]


def conds_inside_loops(sdfg):
    """ConditionalBlocks nested anywhere inside a LoopRegion."""
    return sum(
        sum(1 for b in r.all_control_flow_blocks() if isinstance(b, ConditionalBlock))
        for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


# ----------------------------------------------------------------------
# MoveIfIntoLoop <-> MoveLoopInvariantIfUp
# ----------------------------------------------------------------------


@dace.program
def guard_over_loop(a: dace.float64[N], b: dace.float64[N], act: dace.int32[1]):
    """``if act: for i: b[i] = a[i] * 2`` -- an invariant guard wrapping a
    loop. MoveIfIntoLoop pushes it in; MoveLoopInvariantIfUp hoists it out."""
    if act[0] > 0:
        for i in range(N):
            b[i] = a[i] * 2.0


def _guard_over_loop_oracle(a, act):
    out = np.zeros_like(a)
    if act > 0:
        out[:] = a * 2.0
    return out


@pytest.mark.parametrize('require_full_hoist', [True, False])
def test_moveif_into_then_up_roundtrip(require_full_hoist):
    """Push the guard into the loop, then hoist it back out: the guard
    ends up wrapping the loop again (1 top-level ConditionalBlock), and
    the result is value-preserving for both guard-taken and not-taken.
    Exercised with both require_full_hoist modes on the up-pass."""
    n = 10
    rng = np.random.default_rng(80)
    a = rng.standard_normal(n)

    sdfg = guard_over_loop.to_sdfg(simplify=True)
    assert len(_top_conds(sdfg)) == 1, 'the fixture must start with the guard above the loop'

    assert MoveIfIntoLoop().apply_pass(sdfg, {}) == 1, 'the down direction refused'
    sdfg.validate()
    assert not _top_conds(sdfg), 'the guard did not leave the top level'
    assert conds_inside_loops(sdfg) == 1, 'the guard did not land inside the loop'

    assert MoveLoopInvariantIfUp(require_full_hoist=require_full_hoist).apply_pass(sdfg, {}) == 1, \
        'the up direction refused'
    sdfg.validate()
    assert len(_top_conds(sdfg)) == 1, 'guard did not return to the top level after the round-trip'
    assert conds_inside_loops(sdfg) == 0, 'a copy of the guard was left inside the loop'

    for act in (1, 0):
        exp = _guard_over_loop_oracle(a, act)
        got = np.zeros(n)
        sdfg(a=a, b=got, act=np.array([act], np.int32), N=n)
        assert np.allclose(got, exp), f'value mismatch act={act}'
        # Original (non-transformed) reference agrees.
        ref = np.zeros(n)
        base_run = guard_over_loop.to_sdfg(simplify=True)
        base_run(a=a, b=ref, act=np.array([act], np.int32), N=n)
        assert np.allclose(got, ref), f'round-trip diverged from original act={act}'


@dace.program
def loop_over_guard(a: dace.float64[N], b: dace.float64[N], act: dace.int32[1]):
    """``for i: if act: b[i] = a[i] + 1`` -- the guard starts inside the loop."""
    for i in range(N):
        if act[0] > 0:
            b[i] = a[i] + 1.0


def test_moveif_up_hoists_the_guard_out_of_the_loop():
    """Hoisting the invariant guard out of the loop is value-preserving for both guard values."""
    n = 9
    rng = np.random.default_rng(81)
    a = rng.standard_normal(n)

    sdfg = loop_over_guard.to_sdfg(simplify=True)
    assert conds_inside_loops(sdfg) == 1, 'the fixture must start with the guard inside the loop'
    assert MoveLoopInvariantIfUp(require_full_hoist=True).apply_pass(sdfg, {}) == 1, 'the up direction refused'
    sdfg.validate()
    assert len(_top_conds(sdfg)) == 1, 'the guard did not reach the top level'
    assert conds_inside_loops(sdfg) == 0, 'a copy of the guard was left inside the loop'

    for act in (1, 0):
        got = np.zeros(n)
        sdfg(a=a, b=got, act=np.array([act], np.int32), N=n)
        ref = np.zeros(n)
        base = loop_over_guard.to_sdfg(simplify=True)
        base(a=a, b=ref, act=np.array([act], np.int32), N=n)
        assert np.allclose(got, ref), f'hoist diverged act={act}'


def test_moveif_up_then_into_completes_the_roundtrip():
    """After the hoist, pushing the same guard back into the loop must fire again."""
    sdfg = loop_over_guard.to_sdfg(simplify=True)
    assert MoveLoopInvariantIfUp(require_full_hoist=True).apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert MoveIfIntoLoop().apply_pass(sdfg, {}) == 1, 'the down direction refused on the hoisted form'
    sdfg.validate()
    assert not _top_conds(sdfg)
    assert conds_inside_loops(sdfg) == 1


# ----------------------------------------------------------------------
# LoopToMap <-> MapToForLoop
# ----------------------------------------------------------------------


@dace.program
def parallel_loop(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N):
        b[i] = a[i] * 3.0 + 1.0


def test_looptomap_then_maptoloop_then_looptomap():
    """A parallel loop -> map -> loop -> map round-trips back to a parallel
    Map, value-preserving (extends the existing iter-index-through-NSDFG
    round-trip coverage)."""
    from dace.transformation.interstate import LoopToMap
    from dace.transformation.dataflow import MapToForLoop

    n = 12
    rng = np.random.default_rng(82)
    a = rng.standard_normal(n)
    exp = a * 3.0 + 1.0

    sdfg = parallel_loop.to_sdfg(simplify=True)
    assert _nmaps(sdfg) == 0 and _nloops(sdfg) == 1, 'the fixture must start as one sequential loop'
    assert sdfg.apply_transformations_repeated(LoopToMap) == 1, 'loop did not become a map'
    sdfg.validate()
    assert _nmaps(sdfg) == 1 and _nloops(sdfg) == 0

    assert sdfg.apply_transformations_repeated(MapToForLoop) == 1, 'map did not become a loop'
    sdfg.validate()
    assert _nmaps(sdfg) == 0, 'the intermediate must carry no map at all'

    assert sdfg.apply_transformations_repeated(LoopToMap) == 1, 'the re-lift refused'
    sdfg.validate()
    assert _nmaps(sdfg) == 1, 'round-trip did not recover a parallel map'
    got = np.zeros(n)
    sdfg(a=a, b=got, N=n)
    assert np.allclose(got, exp), 'LoopToMap<->MapToForLoop round-trip changed values'


# ----------------------------------------------------------------------
# MapFission <-> MapFusion (vertical)
# ----------------------------------------------------------------------


@dace.program
def two_stmt_map(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
        c[i] = b[i] + 1.0


def test_mapfission_then_mapfusion_roundtrip():
    """A two-statement map fissions into two maps, then vertical fusion
    recombines them into a single map -- value-preserving."""
    from dace.transformation.dataflow import MapFission, MapFusionVertical

    n = 10
    rng = np.random.default_rng(83)
    a = rng.standard_normal(n)
    exp_b = a * 2.0
    exp_c = exp_b + 1.0

    sdfg = two_stmt_map.to_sdfg(simplify=True)
    assert _nmaps(sdfg) == 1, 'the fixture must start as a single two-statement map'
    nfis = sdfg.apply_transformations_repeated(MapFission)
    sdfg.validate()
    assert nfis >= 1, 'fission refused'
    assert _nmaps(sdfg) >= 2, 'fission did not split the map'

    assert sdfg.apply_transformations_repeated(MapFusionVertical) >= 1, 'fusion refused'
    sdfg.validate()
    assert _nmaps(sdfg) == 1, f'fusion did not recombine to a single map, got {_nmaps(sdfg)}'
    got_b = np.zeros(n)
    got_c = np.zeros(n)
    sdfg(a=a, b=got_b, c=got_c, N=n)
    assert np.allclose(got_b, exp_b) and np.allclose(got_c, exp_c), 'fission<->fusion round-trip changed values'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

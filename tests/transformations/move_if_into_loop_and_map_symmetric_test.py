# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Symmetric loop/map tests for pushing a scope-invariant guard inside.

    Each scenario is written twice, differing *only* in loop vs map:

    * loop variant (``for k in range(N)``): the generalized ``MoveIfIntoLoop``
      pushes the invariant guard into the loop directly.
    * map variant (``for ... in dace.map[...]``): the same effect is obtained
      by *reusing existing transformations* -- lower the map to loops
      (``MapExpansion`` + ``MapToForLoop``), flatten the lowering's
      NestedSDFG (``InlineMultistateSDFG``), push the guard in
      (``MoveIfIntoLoop``), then recover parallelism (``LoopToMap``, which
      MUST re-apply). This is exactly what the canonicalize pipeline does
      internally; no transformation is modified.

    Both variants assert the structural effect (no top-level
    ``ConditionalBlock`` survives; the guard is duplicated inside, not
    dropped) and end-to-end numerics vs a pure-numpy oracle for the guard
    taken and not-taken.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation.dataflow import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate import LoopToMap
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')
M = dace.symbol('M')


def _top_conds(sdfg):
    return [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)]


def _any_cond(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _n_maps(sdfg):
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _push_guard_into_loop(sdfg):
    """Loop variant: MoveIfIntoLoop pushes the invariant guard in directly."""
    applied = MoveIfIntoLoop().apply_pass(sdfg, {})
    assert applied is not None, "MoveIfIntoLoop must fire on the loop variant"
    sdfg.validate()


def _push_guard_into_map(sdfg):
    """Map variant via *existing-transformation composition* (the pipeline's
    own mechanism): lower -> inline -> MoveIfIntoLoop -> LoopToMap. LoopToMap
    MUST re-apply so parallelism (maps) is recovered."""
    sdfg.apply_transformations_repeated([MapExpansion])
    nl = sdfg.apply_transformations_repeated([MapToForLoop])
    assert nl >= 1, "MapToForLoop must lower the guarded map(s) to loops"
    PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {})
    moved = MoveIfIntoLoop().apply_pass(sdfg, {})
    assert moved is not None, "MoveIfIntoLoop must push the guard into the lowered loop"
    recovered = sdfg.apply_transformations_repeated(LoopToMap)
    assert recovered >= 1, "LoopToMap must recover parallelism after the guard moved in"
    sdfg.validate()
    assert _n_maps(sdfg) >= 1, "the guarded body must end up as a parallel map again"


def _assert_guard_moved_inside(sdfg):
    assert not _top_conds(sdfg), "no ConditionalBlock may survive at SDFG top level"
    assert _any_cond(sdfg), "the guard must be duplicated inside, not dropped"


# --------------------------------------------------------------------------- #
# Scenario 1: invariant scalar/symbolic guard over an elementwise body         #
# --------------------------------------------------------------------------- #


@dace.program
def s1_loop(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    if active[0] > 0:
        for k in range(N):
            b[k] = a[k] + 1.0


@dace.program
def s1_map(a: dace.float64[N], b: dace.float64[N], active: dace.int32[1]):
    if active[0] > 0:
        for k in dace.map[0:N]:
            b[k] = a[k] + 1.0


def _s1_oracle(a, av, n):
    return a + 1.0 if av > 0 else np.full(n, 9.0)


def test_s1_invariant_guard_loop():
    n = 16
    a = np.random.rand(n)
    for av in (1, 0):
        sdfg = s1_loop.to_sdfg(simplify=True)
        _push_guard_into_loop(sdfg)
        _assert_guard_moved_inside(sdfg)
        out = np.full(n, 9.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, _s1_oracle(a, av, n)), f"loop av={av}"


def test_s1_invariant_guard_map():
    n = 16
    a = np.random.rand(n)
    for av in (1, 0):
        sdfg = s1_map.to_sdfg(simplify=True)
        _push_guard_into_map(sdfg)
        _assert_guard_moved_inside(sdfg)
        out = np.full(n, 9.0)
        sdfg(a=a.copy(), b=out, active=np.array([av], np.int32), N=n)
        assert np.allclose(out, _s1_oracle(a, av, n)), f"map av={av}"


# --------------------------------------------------------------------------- #
# Scenario 2: invariant guard over an ICON-style neighbour-gather 2-D body     #
# --------------------------------------------------------------------------- #


@dace.program
def s2_loop(w: dace.float64[N, M], cidx: dace.int32[N, 2], out: dace.float64[N, M], active: dace.int32[1]):
    if active[0] > 0:
        for i in range(N):
            for k in range(M):
                out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


@dace.program
def s2_map(w: dace.float64[N, M], cidx: dace.int32[N, 2], out: dace.float64[N, M], active: dace.int32[1]):
    if active[0] > 0:
        for i, k in dace.map[0:N, 0:M]:
            out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


def _s2_oracle(w, cidx, av, n, m):
    if av > 0:
        return 2.0 * w[cidx[:, 0], :] - w[cidx[:, 1], :]
    return np.full((n, m), 7.0)


def test_s2_gather_guard_loop():
    n, m = 8, 5
    w = np.random.rand(n, m)
    cidx = np.random.default_rng(1).integers(0, n, (n, 2)).astype(np.int32)
    for av in (1, 0):
        sdfg = s2_loop.to_sdfg(simplify=True)
        _push_guard_into_loop(sdfg)
        _assert_guard_moved_inside(sdfg)
        out = np.full((n, m), 7.0)
        sdfg(w=w.copy(), cidx=cidx.copy(), out=out, active=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(out, _s2_oracle(w, cidx, av, n, m)), f"loop av={av}"


def test_s2_gather_guard_map():
    n, m = 8, 5
    w = np.random.rand(n, m)
    cidx = np.random.default_rng(1).integers(0, n, (n, 2)).astype(np.int32)
    for av in (1, 0):
        sdfg = s2_map.to_sdfg(simplify=True)
        _push_guard_into_map(sdfg)
        _assert_guard_moved_inside(sdfg)
        out = np.full((n, m), 7.0)
        sdfg(w=w.copy(), cidx=cidx.copy(), out=out, active=np.array([av], np.int32), N=n, M=m)
        assert np.allclose(out, _s2_oracle(w, cidx, av, n, m)), f"map av={av}"


if __name__ == '__main__':
    test_s1_invariant_guard_loop()
    test_s1_invariant_guard_map()
    test_s2_gather_guard_loop()
    test_s2_gather_guard_map()

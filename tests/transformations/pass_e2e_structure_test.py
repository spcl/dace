# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Per-pass end-to-end + structure tests built purely from the Python
    frontend, using stencils with **indirection in a subset of dimensions**
    (the ICON velocity-advection neighbour-gather shape:
    ``out[i, k] = c1*w[cidx[i,0], k] - c2*w[cidx[i,1], k]`` -- dim 0 is
    gathered through a neighbour-index table, the level dim ``k`` is
    structured; cf. ``icon_loopnest_1/4`` and cloudsc column kernels).

    Each test applies exactly one core transformation to a frontend SDFG,
    asserts numerical identity to a pure-numpy oracle (e2e), and asserts the
    structural outcome (map / loop / conditional counts). No canonicalization
    pipeline code is used -- only core transformations -- so these run on
    ``main`` as-is.
"""
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')  # number of edges (structured horizontal index)
L = dace.symbol('L')  # number of levels (structured vertical index)


def _n_maps(sdfg):
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _n_loops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)])


def _top_conds(sdfg):
    return [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)]


def _any_cond(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _neighbours(n, seed):
    """Two random in-range neighbour-index columns ``cidx[n, 2]`` (the ICON
    ``icidx`` neighbour table: a gather index per edge)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n, 2)).astype(np.int32)


# --------------------------------------------------------------------------- #
# MoveIfIntoLoop -- a guard over a neighbour-gather nest moves inside          #
# --------------------------------------------------------------------------- #


@dace.program
def guarded_gather_nest(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L], active: dace.int32[1]):
    if active[0] > 0:
        for i in range(N):
            for k in range(L):
                # dim 0 gathered through the neighbour table, dim 1 (level) structured
                out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


def test_move_if_into_loop_gather_nest_inside_and_e2e():
    """``if c: for i: for k: out = 2*w[cidx[i,0],k] - w[cidx[i,1],k]`` -- the
    loop-invariant guard is pushed inside; no top-level ConditionalBlock
    survives; value-preserving for c taken and not-taken."""
    n, l = 9, 5
    w = np.random.rand(n, l)
    cidx = _neighbours(n, 1)
    for av in (1, 0):
        sdfg = guarded_gather_nest.to_sdfg(simplify=True)
        assert _top_conds(sdfg), "frontend starts with a top-level guard"
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None, "pass must fire"
        sdfg.validate()
        assert not _top_conds(sdfg), "guard must move off SDFG top level"
        assert _any_cond(sdfg), "guard duplicated inside, not dropped"
        assert _n_loops(sdfg) >= 1

        exp = np.full((n, l), 7.0)
        if av > 0:
            for i in range(n):
                for k in range(l):
                    exp[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
        out = np.full((n, l), 7.0)
        sdfg(w=w.copy(), cidx=cidx.copy(), out=out, active=np.array([av], np.int32), N=n, L=l)
        assert np.allclose(out, exp), f"mismatch active={av}"


# --------------------------------------------------------------------------- #
# LoopFission -- two independent gather statements -> separate loops           #
# --------------------------------------------------------------------------- #


@dace.program
def two_independent_gathers(w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2],
                            vidx: dace.int32[N, 2], b: dace.float64[N, L], d: dace.float64[N, L]):
    for i in range(N):
        for k in range(L):
            b[i, k] = w[cidx[i, 0], k] + 1.0
            d[i, k] = v[vidx[i, 1], k] * 3.0


def test_loop_fission_splits_independent_gathers_and_e2e():
    """Two data-independent neighbour-gather statements in one loop nest
    fission into separate loops; numerically identical to the pre-pass run."""
    n, l = 11, 4
    w, v = np.random.rand(n, l), np.random.rand(n, l)
    cidx, vidx = _neighbours(n, 2), _neighbours(n, 3)
    sdfg = two_independent_gathers.to_sdfg(simplify=True)
    pre = _n_loops(sdfg)

    rb, rd = np.zeros((n, l)), np.zeros((n, l))
    copy.deepcopy(sdfg)(w=w.copy(), v=v.copy(), cidx=cidx.copy(), vidx=vidx.copy(), b=rb, d=rd, N=n, L=l)

    assert LoopFission().apply_pass(sdfg, {}) is not None, "LoopFission must fire"
    sdfg.validate()
    assert _n_loops(sdfg) > pre, f"fission must increase loop count ({pre} -> {_n_loops(sdfg)})"

    ob, od = np.zeros((n, l)), np.zeros((n, l))
    sdfg(w=w.copy(), v=v.copy(), cidx=cidx.copy(), vidx=vidx.copy(), b=ob, d=od, N=n, L=l)
    assert np.allclose(ob, rb) and np.allclose(od, rd)
    exp_b = w[cidx[:, 0], :] + 1.0
    exp_d = v[vidx[:, 1], :] * 3.0
    assert np.allclose(ob, exp_b) and np.allclose(od, exp_d)


# --------------------------------------------------------------------------- #
# MapFusionVertical -- producer/consumer gather through a transient -> 1 map   #
# --------------------------------------------------------------------------- #


@dace.program
def producer_consumer_gather(w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2],
                             vidx: dace.int32[N, 2], b: dace.float64[N, L]):
    t = np.empty_like(w)
    for i, k in dace.map[0:N, 0:L]:
        t[i, k] = w[cidx[i, 0], k] * 2.0
    for i, k in dace.map[0:N, 0:L]:
        b[i, k] = t[i, k] + v[vidx[i, 1], k]


def test_map_fusion_vertical_gather_merges_and_e2e():
    """A neighbour-gather producer and its consumer (communicating through a
    transient) fuse into a single map; numerically identical."""
    n, l = 13, 6
    w, v = np.random.rand(n, l), np.random.rand(n, l)
    cidx, vidx = _neighbours(n, 4), _neighbours(n, 5)
    sdfg = producer_consumer_gather.to_sdfg(simplify=True)
    assert _n_maps(sdfg) == 2, f"frontend should emit two maps, got {_n_maps(sdfg)}"

    ref = np.zeros((n, l))
    copy.deepcopy(sdfg)(w=w.copy(), v=v.copy(), cidx=cidx.copy(), vidx=vidx.copy(), b=ref, N=n, L=l)

    assert sdfg.apply_transformations_repeated(MapFusionVertical) >= 1, "MapFusionVertical must fire"
    sdfg.validate()
    assert _n_maps(sdfg) == 1, f"producer+consumer must fuse to one map, got {_n_maps(sdfg)}"

    out = np.zeros((n, l))
    sdfg(w=w.copy(), v=v.copy(), cidx=cidx.copy(), vidx=vidx.copy(), b=out, N=n, L=l)
    exp = w[cidx[:, 0], :] * 2.0 + v[vidx[:, 1], :]
    assert np.allclose(out, ref) and np.allclose(out, exp)


# --------------------------------------------------------------------------- #
# LoopToMap -- a sequential gather nest is provably parallel -> maps           #
# --------------------------------------------------------------------------- #


@dace.program
def sequential_gather_nest(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L]):
    for i in range(N):
        for k in range(L):
            out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


def test_loop_to_map_gather_nest_parallelizes_and_e2e():
    """Each ``(i, k)`` writes the distinct ``out[i, k]`` (the gather is only on
    reads), so the sequential nest is provably parallel: LoopToMap converts
    the loops to maps; numerically identical."""
    n, l = 10, 7
    w = np.random.rand(n, l)
    cidx = _neighbours(n, 6)
    sdfg = sequential_gather_nest.to_sdfg(simplify=True)
    assert _n_loops(sdfg) >= 1 and _n_maps(sdfg) == 0

    ref = np.zeros((n, l))
    copy.deepcopy(sdfg)(w=w.copy(), cidx=cidx.copy(), out=ref, N=n, L=l)

    assert sdfg.apply_transformations_repeated(LoopToMap) >= 1, "LoopToMap must fire"
    sdfg.validate()
    assert _n_maps(sdfg) >= 1, "the gather nest must become a parallel map"

    out = np.zeros((n, l))
    sdfg(w=w.copy(), cidx=cidx.copy(), out=out, N=n, L=l)
    exp = 2.0 * w[cidx[:, 0], :] - w[cidx[:, 1], :]
    assert np.allclose(out, ref) and np.allclose(out, exp)


if __name__ == '__main__':
    test_move_if_into_loop_gather_nest_inside_and_e2e()
    test_loop_fission_splits_independent_gathers_and_e2e()
    test_map_fusion_vertical_gather_merges_and_e2e()
    test_loop_to_map_gather_nest_parallelizes_and_e2e()

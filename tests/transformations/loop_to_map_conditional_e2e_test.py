# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" End-to-end + structure tests for ``LoopToMap`` on sequential loop nests
    that carry an ``if`` / ``if-else`` *inside the loop body*, modeled on the
    cloudsc / ICON loopnest patterns:

    * ICON neighbour-gather (``icon_loopnest_1/4``): the horizontal dim is
      gathered through a neighbour-index table
      (``out[i, k] = c1*w[cidx[i,0], k] - c2*w[cidx[i,1], k]``) while the
      level dim ``k`` is structured.
    * cloudsc column physics (``cloudsc_autoconversion_snow`` etc.): a
      per-column ``IF (qx > thresh) THEN ... ELSE ...`` threshold branch
      inside the loop.

    Each test builds a Python-frontend SDFG, runs the *pre-pass* SDFG (or a
    pure-numpy oracle) for the reference, applies ``LoopToMap`` repeatedly,
    validates, asserts the structural change (maps appear, ``LoopRegion`` s
    gone) and asserts ``np.allclose`` against the reference for both the
    condition-taken and not-taken data.  Only core transformations are used
    (no canonicalization pipeline), so these run on ``main`` as-is.
"""
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

N = dace.symbol('N')  # number of edges (structured horizontal index)
L = dace.symbol('L')  # number of levels (structured vertical index)


def _n_maps(sdfg):
    return len([n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])


def _n_loops(sdfg):
    return len([r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)])


def _neighbours(n, seed):
    """Two random in-range neighbour-index columns ``cidx[n, 2]`` (the ICON
    ``icidx`` neighbour table: a gather index per edge)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n, 2)).astype(np.int32)


# --------------------------------------------------------------------------- #
# 1. Loop-invariant ``if`` over a neighbour-gather nest                        #
# --------------------------------------------------------------------------- #


@dace.program
def cond_invariant_gather(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L],
                          active: dace.int32[1]):
    for i in range(N):
        for k in range(L):
            if active[0] > 0:
                # dim 0 gathered through the neighbour table, level dim structured
                out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
            else:
                out[i, k] = w[cidx[i, 0], k]


def test_loop_to_map_invariant_if_over_gather_parallelizes_and_e2e():
    """A loop-invariant guard (``active[0] > 0``) inside ``for i: for k:`` over
    a neighbour-gather body.  Each ``(i, k)`` writes the distinct ``out[i, k]``
    (the gather is only on reads), so the nest is provably parallel: LoopToMap
    converts the loops to maps.  Checked for the guard taken and not-taken."""
    n, l = 10, 7
    w = np.random.default_rng(101).random((n, l))
    cidx = _neighbours(n, 1)
    for av in (1, 0):
        sdfg = cond_invariant_gather.to_sdfg(simplify=True)
        assert _n_loops(sdfg) >= 1 and _n_maps(sdfg) == 0

        ref = np.zeros((n, l))
        copy.deepcopy(sdfg)(w=w.copy(), cidx=cidx.copy(), out=ref, active=np.array([av], np.int32), N=n, L=l)

        applied = sdfg.apply_transformations_repeated(LoopToMap)
        assert applied >= 1, "LoopToMap must fire"
        sdfg.validate()
        assert _n_maps(sdfg) >= 1, "the gather nest must become a parallel map"
        assert _n_loops(sdfg) == 0, "no sequential LoopRegion may survive"

        out = np.zeros((n, l))
        sdfg(w=w.copy(), cidx=cidx.copy(), out=out, active=np.array([av], np.int32), N=n, L=l)
        if av > 0:
            exp = 2.0 * w[cidx[:, 0], :] - w[cidx[:, 1], :]
        else:
            exp = w[cidx[:, 0], :]
        assert np.allclose(out, ref), f"vs pre-pass run mismatch active={av}"
        assert np.allclose(out, exp), f"vs numpy oracle mismatch active={av}"


# --------------------------------------------------------------------------- #
# 2. ``if-else`` writing to DIFFERENT output subsets per branch                #
# --------------------------------------------------------------------------- #


@dace.program
def cond_disjoint_outputs(w: dace.float64[N, L], cidx: dace.int32[N, 2], a: dace.float64[N, L], b: dace.float64[N, L],
                          active: dace.int32[1]):
    for i in range(N):
        for k in range(L):
            if active[0] > 0:
                a[i, k] = 2.0 * w[cidx[i, 0], k]
            else:
                b[i, k] = w[cidx[i, 1], k] - 1.0


def test_loop_to_map_if_else_disjoint_outputs_parallelizes_and_e2e():
    """An ``if-else`` whose two branches write *different* output arrays
    (``a`` vs ``b``).  Each iteration still touches only its own ``[i, k]``
    cell, so the nest is parallel: LoopToMap fires.  Both branch directions
    are checked against the pre-pass run and a numpy oracle."""
    n, l = 12, 5
    w = np.random.default_rng(202).random((n, l))
    cidx = _neighbours(n, 2)
    for av in (1, 0):
        sdfg = cond_disjoint_outputs.to_sdfg(simplify=True)
        assert _n_loops(sdfg) >= 1 and _n_maps(sdfg) == 0

        ra, rb = np.full((n, l), 9.0), np.full((n, l), 9.0)
        copy.deepcopy(sdfg)(w=w.copy(), cidx=cidx.copy(), a=ra, b=rb, active=np.array([av], np.int32), N=n, L=l)

        applied = sdfg.apply_transformations_repeated(LoopToMap)
        assert applied >= 1, "LoopToMap must fire"
        sdfg.validate()
        assert _n_maps(sdfg) >= 1, "the nest must become a parallel map"
        assert _n_loops(sdfg) == 0, "no sequential LoopRegion may survive"

        oa, ob = np.full((n, l), 9.0), np.full((n, l), 9.0)
        sdfg(w=w.copy(), cidx=cidx.copy(), a=oa, b=ob, active=np.array([av], np.int32), N=n, L=l)
        ea, eb = np.full((n, l), 9.0), np.full((n, l), 9.0)
        if av > 0:
            ea = 2.0 * w[cidx[:, 0], :]
        else:
            eb = w[cidx[:, 1], :] - 1.0
        assert np.allclose(oa, ra) and np.allclose(ob, rb), f"vs pre-pass run mismatch active={av}"
        assert np.allclose(oa, ea) and np.allclose(ob, eb), f"vs numpy oracle mismatch active={av}"


# --------------------------------------------------------------------------- #
# 3. ``if-else`` writing DIFFERENT VALUES to the SAME subset                   #
#    (cloudsc threshold branch: condition reads gathered, per-cell data)       #
# --------------------------------------------------------------------------- #


@dace.program
def cond_same_output_threshold(w: dace.float64[N, L], cidx: dace.int32[N, 2], x: dace.float64[N, L],
                               y: dace.float64[N, L], out: dace.float64[N, L]):
    for i in range(N):
        for k in range(L):
            # cloudsc-style per-column threshold on a neighbour-gathered value
            if w[cidx[i, 0], k] > 0.5:
                out[i, k] = x[i, k] * 2.0
            else:
                out[i, k] = y[i, k] + 1.0


def test_loop_to_map_if_else_same_output_threshold_parallelizes_and_e2e():
    """A cloudsc-style ``IF (qx > thresh) THEN ... ELSE ...`` where both
    branches write the *same* ``out[i, k]`` distinct cell with different
    values, and the condition itself reads neighbour-gathered data.  Still
    parallel (one write per cell): LoopToMap fires; numeric == numpy oracle."""
    n, l = 14, 6
    rng = np.random.default_rng(303)
    w = rng.random((n, l))
    x, y = rng.random((n, l)), rng.random((n, l))
    cidx = _neighbours(n, 3)
    sdfg = cond_same_output_threshold.to_sdfg(simplify=True)
    assert _n_loops(sdfg) >= 1 and _n_maps(sdfg) == 0

    ref = np.zeros((n, l))
    copy.deepcopy(sdfg)(w=w.copy(), cidx=cidx.copy(), x=x.copy(), y=y.copy(), out=ref, N=n, L=l)

    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied >= 1, "LoopToMap must fire"
    sdfg.validate()
    assert _n_maps(sdfg) >= 1, "the threshold nest must become a parallel map"
    assert _n_loops(sdfg) == 0, "no sequential LoopRegion may survive"

    out = np.zeros((n, l))
    sdfg(w=w.copy(), cidx=cidx.copy(), x=x.copy(), y=y.copy(), out=out, N=n, L=l)
    gathered = w[cidx[:, 0], :]
    exp = np.where(gathered > 0.5, x * 2.0, y + 1.0)
    assert np.allclose(out, ref), "vs pre-pass run mismatch"
    assert np.allclose(out, exp), "vs numpy oracle mismatch"


# --------------------------------------------------------------------------- #
# 4. Condition depends on the loop variable / per-iteration data               #
# --------------------------------------------------------------------------- #


@dace.program
def cond_loopvar_dependent(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L]):
    for i in range(N):
        for k in range(L):
            # condition varies per iteration (loop var + gathered per-cell data)
            if (i + k) % 2 == 0 and w[cidx[i, 1], k] > 0.4:
                out[i, k] = w[cidx[i, 0], k] * 3.0
            else:
                out[i, k] = w[cidx[i, 1], k] - 0.25


def test_loop_to_map_loopvar_dependent_if_parallelizes_and_e2e():
    """The branch condition depends on the loop variables ``(i + k) % 2`` and
    on per-iteration gathered data, yet each iteration still writes its own
    distinct ``out[i, k]``.  The data dependence is purely intra-iteration, so
    LoopToMap still parallelizes; numeric == numpy oracle."""
    n, l = 11, 8
    w = np.random.default_rng(404).random((n, l))
    cidx = _neighbours(n, 4)
    sdfg = cond_loopvar_dependent.to_sdfg(simplify=True)
    assert _n_loops(sdfg) >= 1 and _n_maps(sdfg) == 0

    ref = np.zeros((n, l))
    copy.deepcopy(sdfg)(w=w.copy(), cidx=cidx.copy(), out=ref, N=n, L=l)

    applied = sdfg.apply_transformations_repeated(LoopToMap)
    assert applied >= 1, "LoopToMap must fire"
    sdfg.validate()
    assert _n_maps(sdfg) >= 1, "the loop-var-conditional nest must become a parallel map"
    assert _n_loops(sdfg) == 0, "no sequential LoopRegion may survive"

    out = np.zeros((n, l))
    sdfg(w=w.copy(), cidx=cidx.copy(), out=out, N=n, L=l)
    exp = np.zeros((n, l))
    for i in range(n):
        for k in range(l):
            if (i + k) % 2 == 0 and w[cidx[i, 1], k] > 0.4:
                exp[i, k] = w[cidx[i, 0], k] * 3.0
            else:
                exp[i, k] = w[cidx[i, 1], k] - 0.25
    assert np.allclose(out, ref), "vs pre-pass run mismatch"
    assert np.allclose(out, exp), "vs numpy oracle mismatch"


if __name__ == '__main__':
    test_loop_to_map_invariant_if_over_gather_parallelizes_and_e2e()
    test_loop_to_map_if_else_disjoint_outputs_parallelizes_and_e2e()
    test_loop_to_map_if_else_same_output_threshold_parallelizes_and_e2e()
    test_loop_to_map_loopvar_dependent_if_parallelizes_and_e2e()

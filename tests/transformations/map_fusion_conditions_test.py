# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Map fusion across guarded computations.

    Branch-replicated fission produces several identical-condition guarded
    maps. ``ConditionFusion`` merges the ConditionalBlocks but emits the full
    cartesian product of branch combinations; ``LiftTrivialIf`` drops the
    provably-unsatisfiable combinations (``c and not c``), after which the
    co-located maps fuse. Each test pins the post-recipe map and conditional
    counts and checks numerical equivalence (guard taken and not-taken)
    against a deep-copied pre-pass run.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.sdfg import nodes
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.dataflow.map_fusion_horizontal import MapFusionHorizontal

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def two_guarded(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    if M > 0:
        for i in dace.map[0:N]:
            A[i] = a[i] + 1.0
    if M > 0:
        for i in dace.map[0:N]:
            B[i] = a[i] * 2.0


@dace.program
def if_two_maps_inside(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    if M > 0:
        for i in dace.map[0:N]:
            A[i] = a[i] + 1.0
        for i in dace.map[0:N]:
            B[i] = a[i] * 2.0


@dace.program
def two_guarded_diff(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N]):
    if M > 0:
        for i in dace.map[0:N]:
            A[i] = a[i] + 1.0
    if M > 5:
        for i in dace.map[0:N]:
            B[i] = a[i] * 2.0


@dace.program
def three_guarded(a: dace.float64[N], A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    if M > 0:
        for i in dace.map[0:N]:
            A[i] = a[i] + 1.0
    if M > 0:
        for i in dace.map[0:N]:
            B[i] = a[i] * 2.0
    if M > 0:
        for i in dace.map[0:N]:
            C[i] = a[i] - 1.0


def _with_M(sdfg):
    """A condition-only symbol is not auto-registered in ``sdfg.symbols``
    (frontend quirk); add it so the SDFG is callable."""
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    return sdfg


def _conds(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]


def _maps(sdfg):
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, nodes.MapEntry)]


def _fuse_recipe(sdfg):
    """The fuse-stage recipe for guarded maps: merge guards, drop the
    cartesian product's unsatisfiable branches, structural-clean, fuse."""
    PatternMatchAndApplyRepeated([ConditionFusion()]).apply_pass(sdfg, {})
    LiftTrivialIf().apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([StateFusionExtended()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([InlineSDFG()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([MapFusionVertical(), MapFusionHorizontal()]).apply_pass(sdfg, {})
    sdfg.validate()


def test_two_identical_guards_fuse_to_one_map():
    """`if c: A` ; `if c: B` (the branch-replicated-fission shape) recombine
    into one guard whose two maps fuse: 1 map, 1 conditional."""
    n = 16
    a = np.random.rand(n)
    base = _with_M(two_guarded.to_sdfg(simplify=True))
    assert len(_conds(base)) == 2 and len(_maps(base)) == 2

    for mval in (1, 0):
        ref_A, ref_B = np.full(n, 9.0), np.full(n, 9.0)
        copy.deepcopy(base)(a=a.copy(), A=ref_A, B=ref_B, N=n, M=mval)

        sdfg = _with_M(two_guarded.to_sdfg(simplify=True))
        _fuse_recipe(sdfg)
        assert len(_maps(sdfg)) == 1, f"maps not fused: {len(_maps(sdfg))}"
        assert len(_conds(sdfg)) == 1, f"guards not merged: {len(_conds(sdfg))}"

        out_A, out_B = np.full(n, 9.0), np.full(n, 9.0)
        sdfg(a=a.copy(), A=out_A, B=out_B, N=n, M=mval)
        assert np.allclose(out_A, ref_A) and np.allclose(out_B, ref_B), f"mismatch M={mval}"
        if mval > 0:
            assert np.allclose(out_A, a + 1.0) and np.allclose(out_B, a * 2.0)
        else:
            assert np.allclose(out_A, 9.0) and np.allclose(out_B, 9.0)


def test_two_maps_in_one_guard_fuse_to_one_map():
    """`if c: { A ; B }` -> the two guarded maps fuse: 1 map, 1 conditional."""
    n = 16
    a = np.random.rand(n)
    base = _with_M(if_two_maps_inside.to_sdfg(simplify=True))

    for mval in (1, 0):
        ref_A, ref_B = np.full(n, 9.0), np.full(n, 9.0)
        copy.deepcopy(base)(a=a.copy(), A=ref_A, B=ref_B, N=n, M=mval)

        sdfg = _with_M(if_two_maps_inside.to_sdfg(simplify=True))
        _fuse_recipe(sdfg)
        assert len(_maps(sdfg)) == 1 and len(_conds(sdfg)) == 1

        out_A, out_B = np.full(n, 9.0), np.full(n, 9.0)
        sdfg(a=a.copy(), A=out_A, B=out_B, N=n, M=mval)
        assert np.allclose(out_A, ref_A) and np.allclose(out_B, ref_B), f"mismatch M={mval}"


def test_distinct_guards_merge_to_one_conditional_three_maps():
    """Distinct guards (`if c1: A` ; `if c2: B`) merge into one
    ConditionalBlock but the maps cannot share a guard: the feasible
    cartesian combinations leave 3 maps, 1 conditional. Value-preserving
    across all (c1, c2) truth combinations."""
    n = 16
    a = np.random.rand(n)
    base = _with_M(two_guarded_diff.to_sdfg(simplify=True))

    for mval in (6, 1, 0):  # (c1&c2), (c1&!c2), (!c1&!c2)
        ref_A, ref_B = np.full(n, 9.0), np.full(n, 9.0)
        copy.deepcopy(base)(a=a.copy(), A=ref_A, B=ref_B, N=n, M=mval)

        sdfg = _with_M(two_guarded_diff.to_sdfg(simplify=True))
        _fuse_recipe(sdfg)
        assert len(_conds(sdfg)) == 1, f"guards not merged: {len(_conds(sdfg))}"
        assert len(_maps(sdfg)) == 3, f"unexpected map count: {len(_maps(sdfg))}"

        out_A, out_B = np.full(n, 9.0), np.full(n, 9.0)
        sdfg(a=a.copy(), A=out_A, B=out_B, N=n, M=mval)
        assert np.allclose(out_A, ref_A) and np.allclose(out_B, ref_B), f"mismatch M={mval}"
        assert np.allclose(out_A, a + 1.0 if mval > 0 else 9.0)
        assert np.allclose(out_B, a * 2.0 if mval > 5 else 9.0)


@pytest.mark.xfail(strict=True,
                   reason="3+ identical guards: ConditionFusion applies pairwise, so chained "
                   "merges build nested conjunctions the c-and-not-c LiftTrivialIf prune does "
                   "not fully collapse (ends at 7 maps, ideal is 1).")
def test_three_identical_guards_fuse_to_one_map():
    """Ideal: three `if c: <map>` blocks collapse to 1 map, 1 conditional.
    Currently blows up to 7 maps -- documented limitation."""
    n = 16
    a = np.random.rand(n)
    sdfg = _with_M(three_guarded.to_sdfg(simplify=True))
    _fuse_recipe(sdfg)
    assert len(_maps(sdfg)) == 1 and len(_conds(sdfg)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

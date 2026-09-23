# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for MoveIfIntoLoop: push a loop-invariant guarding conditional into
    the loop body (`if c: for i: body` -> `for i: if c: body`). Conservative:
    single branch, no else, branch region is exactly one LoopRegion, and the
    condition is loop-invariant. Kernels use the dace Python frontend; every
    test checks numerical equivalence against a deep-copied pre-pass run for
    both the taken and not-taken condition.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion, ConditionalBlock
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from tests.sdfg.cfg_list_in_place_test import assert_tree_consistent


def assert_cfg_list_matches_reset(sdfg: dace.SDFG) -> None:
    """The kept CFG list and ids equal a fresh copy's after ``reset_cfg_list``, and every parent pointer holds."""
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]
    assert_tree_consistent(sdfg)


N = dace.symbol('N')


@dace.program
def guarded_loop(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1]):
    if c[0] > 0:
        for i in range(N):
            b[i] = a[i] + 1.0


@dace.program
def loop_var_in_cond(a: dace.float64[N], b: dace.float64[N]):
    # Condition depends on the loop variable -> NOT loop-invariant: refuse.
    for i in range(N):
        if i < 5:
            b[i] = a[i] + 1.0


@dace.program
def nested_guards(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1], d: dace.float64[1]):
    if c[0] > 0:
        if d[0] * d[0] + 1.0 < 100.0:
            for i in range(N):
                b[i] = a[i] * 3.0 - 2.0


def _loops(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def _conds(sdfg):
    return [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, ConditionalBlock)]


def test_move_if_into_loop_basic():
    """`if c: for i: b=a+1` -> `for i: if c: b=a+1`; valid + numerically
    identical for c taken and not-taken."""
    n = 12
    a = np.random.rand(n)
    base = guarded_loop.to_sdfg(simplify=True)
    for cval in (1, 0):
        ref = np.full(n, 7.0)
        copy.deepcopy(base)(a=a.copy(), b=ref, c=np.array([cval], np.int32), N=n)

        sdfg = guarded_loop.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        assert_cfg_list_matches_reset(sdfg)
        # The loop is now hoisted to top level with the conditional inside it.
        loops = _loops(sdfg)
        assert len(loops) == 1
        assert any(isinstance(n_, ConditionalBlock) for n_ in loops[0].all_control_flow_regions(recursive=True))

        out = np.full(n, 7.0)
        sdfg(a=a.copy(), b=out, c=np.array([cval], np.int32), N=n)
        assert np.allclose(out, ref), f"mismatch for c={cval}: {out} vs {ref}"
        if cval > 0:
            assert np.allclose(out, a + 1.0)
        else:
            assert np.allclose(out, 7.0)


def test_move_if_into_loop_refuses_loop_variant_condition():
    """A condition depending on the loop variable is not loop-invariant:
    the pass must not move it (no-op), staying valid + correct."""
    n = 10
    a = np.random.rand(n)
    sdfg = loop_var_in_cond.to_sdfg(simplify=True)
    assert MoveIfIntoLoop().apply_pass(sdfg, {}) is None
    sdfg.validate()
    b = np.zeros(n)
    sdfg(a=a.copy(), b=b, N=n)
    exp = np.zeros(n)
    exp[:5] = a[:5] + 1.0
    assert np.allclose(b, exp)


def test_move_if_into_loop_nested_guards_cascade():
    """`if c: if d-expr: for i: body` -> `for i: if c: if d-expr: body`:
    the fixpoint pushes both guards in (innermost first, then outer),
    valid + numerically identical for every condition combination."""
    n = 9
    a = np.random.rand(n)
    base = nested_guards.to_sdfg(simplify=True)
    for cval, dval in ((1, 2.0), (0, 2.0), (1, 50.0), (0, 50.0)):
        ref = np.full(n, 4.0)
        copy.deepcopy(base)(a=a.copy(), b=ref, c=np.array([cval], np.int32), d=np.array([dval], np.float64), N=n)

        sdfg = nested_guards.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        loops = _loops(sdfg)
        assert len(loops) == 1
        # Both conditionals ended up inside the (now top-level) loop.
        inner_conds = [r for r in loops[0].all_control_flow_regions(recursive=True) if isinstance(r, ConditionalBlock)]
        assert len(inner_conds) >= 2
        assert not any(isinstance(r, ConditionalBlock) for r in sdfg.nodes() if isinstance(r, ConditionalBlock))

        out = np.full(n, 4.0)
        sdfg(a=a.copy(), b=out, c=np.array([cval], np.int32), d=np.array([dval], np.float64), N=n)
        assert np.allclose(out, ref), f"mismatch c={cval} d={dval}"
        if cval > 0 and dval * dval + 1.0 < 100.0:
            assert np.allclose(out, a * 3.0 - 2.0)
        else:
            assert np.allclose(out, 4.0)


_MIL_N = dace.symbol("_MIL_N")


@dace.program
def _guard_over_bound_dependent_loop(a: dace.float64[_MIL_N], b: dace.float64[_MIL_N], c: dace.float64[_MIL_N],
                                     kk: dace.int32):
    if kk > 0:
        for i in range(_MIL_N - kk):
            a[i] = a[i + kk] + b[i] * c[i]


@pytest.mark.parametrize("kk", [3, 0])
def test_refuses_when_prep_produces_loop_bound(kk):
    """Reproducer (TSVC s162): the guarded loop's trip count ``_MIL_N - kk`` is a
    prep value the loop condition consumes. ``_move`` sinks the prep into the body,
    so pushing the guard in would leave the first condition check reading an
    uninitialized bound. MoveIfIntoLoop must refuse this shape; either way the
    transformed SDFG must match the un-transformed reference (taken and not-taken)."""
    n = 64
    rng = np.random.default_rng(0)
    base = {name: rng.random(n) for name in "abc"}

    ref = _guard_over_bound_dependent_loop.to_sdfg(simplify=True)
    cand = copy.deepcopy(ref)
    MoveIfIntoLoop().apply_pass(cand, {})
    cand.validate()

    ra = {name: arr.copy() for name, arr in base.items()}
    ref(**ra, kk=kk, _MIL_N=n)
    ca = {name: arr.copy() for name, arr in base.items()}
    cand(**ca, kk=kk, _MIL_N=n)
    for name in "abc":
        assert np.allclose(ra[name], ca[name]), f"{name} (kk={kk}): MoveIfIntoLoop changed the result"


@dace.program
def carried_prep(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1], acc: dace.float64[1]):
    # The guarded prep updates ``acc`` FROM ITS OWN previous value, so running it once and running
    # it N times differ. Sinking it into the loop body runs it once per iteration.
    if c[0] > 0:
        acc[0] = acc[0] * 2.0
        for i in range(N):
            b[i] = a[i] + acc[0]


def test_refuses_a_prep_that_carries_its_own_value():
    """``_move`` re-runs the prep every iteration, so the chain must be idempotent.

    The existing guard compared the prep's reads against the LOOP's writes only. A prep that reads
    what the PREP updates passes that check and is then applied once per iteration: ``acc`` is
    doubled N times instead of once, and every ``b[i]`` reads a different multiplier.
    """
    n = 7
    rng = np.random.default_rng(0)
    a = rng.random(n)

    sdfg = carried_prep.to_sdfg(simplify=True)
    assert MoveIfIntoLoop().apply_pass(sdfg, {}) is None, 'a prep carrying its own value must not sink'
    sdfg.validate()

    for cval in (1, 0):
        acc = np.array([3.0], np.float64)
        out = np.zeros(n)
        sdfg(a=a.copy(), b=out, c=np.array([cval], np.int32), acc=acc, N=n)
        if cval > 0:
            assert np.allclose(acc, 6.0), f'the prep must run exactly once, acc={acc[0]}'
            assert np.allclose(out, a + 6.0), f'got {out[:3]}'
        else:
            assert np.allclose(acc, 3.0), 'the guard was false: the prep must not run'


def test_staged_prep_still_sinks():
    """Guard against over-reach: a prep that reads what an EARLIER prep state wrote recomputes the
    same value from the same loop-invariant inputs, so it stays idempotent and must still move.
    ``nested_guards`` is exactly that shape -- its condition is materialised in stages."""
    sdfg = nested_guards.to_sdfg(simplify=True)
    assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None, 'a staged, idempotent prep must still sink'
    sdfg.validate()


@dace.program
def many_guarded_loops(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1]):
    if c[0] > 0:
        for i in range(N):
            b[i] = a[i] + 1.0
    if c[0] > 1:
        for i in range(N):
            a[i] = b[i] * 2.0
    if c[0] > 2:
        for i in range(N):
            b[i] = a[i] - 3.0


def test_the_cfg_list_is_never_rebuilt_by_the_pass(monkeypatch):
    """Every block a move adds used to rebuild the CFG list of the whole tree -- 11808 rebuilds, 83% of
    the pass on warpx_field_gather. The graph operations keep it exact in place."""
    sdfg = many_guarded_loops.to_sdfg(simplify=True)
    lists = [sdfg.cfg_list]
    original = dace.sdfg.state.AbstractControlFlowRegion.reset_cfg_list

    def recorded(self):
        result = original(self)
        if sdfg.cfg_list is not lists[-1]:
            lists.append(sdfg.cfg_list)
        return result

    monkeypatch.setattr(dace.sdfg.state.AbstractControlFlowRegion, 'reset_cfg_list', recorded)
    assert MoveIfIntoLoop().apply_pass(sdfg, {}) == 3
    assert len(lists) == 1, len(lists)
    assert sdfg.cfg_list == list(sdfg.all_control_flow_regions(recursive=True))
    sdfg.validate()


@dace.program
def guarded_imperfect_nest(a: dace.float64[N], b: dace.float64[N], c: dace.int32[1], d: dace.float64[1]):
    if c[0] > 0:
        for i in range(N):
            b[i] = a[i] + 1.0
        d[0] = 3.0


def test_imperfect_nest_keeps_the_cfg_list_of_a_fresh_reset():
    """Each sibling of an imperfect nest gets its own guard; the tree stays exact and the values hold."""
    n = 8
    a = np.random.rand(n)
    base = guarded_imperfect_nest.to_sdfg(simplify=True)
    for cval in (1, 0):
        ref_b, ref_d = np.full(n, 7.0), np.zeros(1)
        copy.deepcopy(base)(a=a.copy(), b=ref_b, c=np.array([cval], np.int32), d=ref_d, N=n)

        sdfg = guarded_imperfect_nest.to_sdfg(simplify=True)
        assert MoveIfIntoLoop().apply_pass(sdfg, {}) is not None
        sdfg.validate()
        assert_cfg_list_matches_reset(sdfg)
        assert not any(isinstance(b, ConditionalBlock) for b in sdfg.nodes())

        out_b, out_d = np.full(n, 7.0), np.zeros(1)
        sdfg(a=a.copy(), b=out_b, c=np.array([cval], np.int32), d=out_d, N=n)
        assert np.allclose(out_b, ref_b) and np.allclose(out_d, ref_d), cval


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

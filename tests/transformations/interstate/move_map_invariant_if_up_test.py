# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``MoveMapInvariantIfUp`` -- hoisting a map-invariant guard out of
its enclosing map (the inverse of ``MoveIfIntoMap`` / the map analogue of
``MoveLoopInvariantIfUp``).

Each test builds the collapsed-map input form by running the canonicalize
pipeline up to (but excluding) the terminal ``hoist_guards`` stage, applies the
pass directly, and checks the structural contract plus value-preservation
against the un-transformed reference.
"""
import numpy as np
import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.pipeline import CANONICALIZE_STAGES
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp

N = dace.symbol('N')


@dace.program
def invariant_if_else(a: dace.float64[N, N], b: dace.float64[N, N], lim: dace.int32):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            if lim < N:
                b[i, j] = a[i, j] * 2.0
            else:
                b[i, j] = a[i, j]


@dace.program
def invariant_single_branch(a: dace.float64[N, N], b: dace.float64[N, N], lim: dace.int32):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            if lim < N:
                b[i, j] = a[i, j] * 2.0


@dace.program
def data_dependent_mask(a: dace.float64[N], b: dace.float64[N], thr: dace.float64):
    for i in dace.map[0:N]:
        if a[i] > thr:
            b[i] = a[i] * 2.0


def _collapsed_form(prog):
    """Run canonicalize stages up to (excluding) ``hoist_guards`` so the
    fully-parallel nest is a collapsed map still carrying its guard inside."""
    sdfg = prog.to_sdfg(simplify=True)
    for label, factory in CANONICALIZE_STAGES:
        if label == 'hoist_guards':
            break
        for unit in factory():
            unit.apply_pass(sdfg, {})
    return sdfg


def _top_conds(sdfg):
    return [c for c in sdfg.nodes() if isinstance(c, ConditionalBlock)]


def test_hoists_invariant_if_else():
    """``if lim < N`` (invariant) is hoisted out of the collapsed ``map[i, j]``
    to a single top-level conditional, one map copy per branch."""
    n = 8
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n, n))

    for lim in (5, 99):
        sdfg = _collapsed_form(invariant_if_else)
        applied = MoveMapInvariantIfUp().apply_pass(sdfg, {})
        sdfg.validate()
        assert applied == 1, f'expected one hoist, got {applied}'
        assert len(_top_conds(sdfg)) == 1, 'guard not hoisted to a single top-level conditional'

        got = np.zeros((n, n))
        sdfg(a=a, b=got, lim=np.int32(lim), N=n)
        assert np.allclose(got, a * (2.0 if lim < n else 1.0)), f'lim={lim}'


def test_hoists_single_branch_guard():
    """A single-branch invariant guard (no ``else``) is hoisted just the same;
    the else path leaves ``b`` untouched."""
    n = 8
    rng = np.random.default_rng(4)
    a = rng.standard_normal((n, n))

    for lim in (5, 99):
        sdfg = _collapsed_form(invariant_single_branch)
        applied = MoveMapInvariantIfUp().apply_pass(sdfg, {})
        sdfg.validate()
        assert applied == 1, f'expected one hoist, got {applied}'
        assert len(_top_conds(sdfg)) == 1

        got = np.full((n, n), -1.0)
        sdfg(a=a, b=got, lim=np.int32(lim), N=n)
        exp = a * 2.0 if lim < n else np.full((n, n), -1.0)
        assert np.allclose(got, exp), f'lim={lim}'


def test_refuses_data_dependent_guard():
    """A per-element data-dependent guard (``if a[i] > thr``, where ``a[i]`` is
    a per-iteration value) is NOT map-invariant and must stay inside the map."""
    n = 16
    rng = np.random.default_rng(5)
    a = rng.standard_normal((n))
    thr = 0.0

    sdfg = _collapsed_form(data_dependent_mask)
    applied = MoveMapInvariantIfUp().apply_pass(sdfg, {})
    sdfg.validate()
    assert applied is None, 'a data-dependent per-element mask must not be hoisted'

    got = np.zeros((n, ))
    sdfg(a=a.copy(), b=got, thr=np.float64(thr), N=n)
    exp = np.where(a > thr, a * 2.0, 0.0)
    assert np.allclose(got, exp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

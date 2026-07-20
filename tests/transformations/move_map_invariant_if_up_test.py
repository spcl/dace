# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MoveMapInvariantIfUp``: hoisting a map-invariant guard out of its map.

Covers the single-map case, the map-chain case (the guard lands between the
maps), and the negative cases where the guard is genuinely per-element and must
stay put.
"""
import copy

import numpy as np
import pytest

import dace
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate.move_map_invariant_if_up import MoveMapInvariantIfUp

N = 8


def _conditionals(sdfg: dace.SDFG):
    """Every ConditionalBlock at any nesting depth."""
    return [
        b for sd in sdfg.all_sdfgs_recursive() for b in sd.all_control_flow_blocks() if isinstance(b, ConditionalBlock)
    ]


def _guard_depth(sdfg: dace.SDFG) -> int:
    """How many map entries enclose the guard (0 = outside every map)."""
    for sd in sdfg.all_sdfgs_recursive():
        for b in sd.all_control_flow_blocks():
            if not isinstance(b, ConditionalBlock):
                continue
            depth, node = 0, sd.parent_nsdfg_node
            state = sd.parent
            while node is not None and state is not None:
                scope = state.entry_node(node)
                while scope is not None:
                    depth += 1
                    scope = state.entry_node(scope)
                sd2 = state.sdfg
                node, state = sd2.parent_nsdfg_node, sd2.parent
            return depth
    return -1


def _run(sdfg, **kwargs):
    """Compile and run a copy of the SDFG, returning the output array."""
    out = np.zeros((N, N))
    csdfg = copy.deepcopy(sdfg)
    csdfg(b=out, **kwargs)
    return out


def test_chain_guard_lands_between_maps():
    """``map i: map j: if a[i]`` -- invariant w.r.t. j, not i, so the guard is
    hoisted out of j only and comes to rest between the two maps."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a[i] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    a = np.array([1.0, -1.0] * (N // 2))
    before = _run(sdfg, a=a)

    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    # Still inside the outer map (it reads a[i]), no longer inside the inner one.
    assert _guard_depth(sdfg) == 1
    assert np.allclose(_run(sdfg, a=a), before, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_chain_guard_value_preserving_both_branches():
    """A guard with an else branch keeps both branches' values."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a[i] > 0.0:
                    b[i, j] = 2.0
                else:
                    b[i, j] = -3.0

    sdfg = kern.to_sdfg(simplify=True)
    a = np.array([1.0, -1.0] * (N // 2))
    before = _run(sdfg, a=a)

    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    after = _run(sdfg, a=a)
    assert np.allclose(after, before, rtol=1e-9, atol=1e-9, equal_nan=True)
    ref = np.where((a > 0.0)[:, None], 2.0, -3.0)
    assert np.allclose(after, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_symbolic_condition_leaves_every_map():
    """A guard reading only a free symbol is invariant w.r.t. both maps, so
    repeated application walks it clear of the whole chain -- one level per
    hoist, ending outside every map."""
    K = dace.symbol('K')

    @dace.program
    def kern(b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if K > 0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 2, "one hoist per map level"
    sdfg.validate()
    assert _guard_depth(sdfg) == 0, "a fully invariant guard must clear the whole chain"

    out = np.zeros((N, N))
    copy.deepcopy(sdfg)(b=out, K=1)
    assert np.allclose(out, np.ones((N, N)), rtol=1e-9, atol=1e-9, equal_nan=True)

    out0 = np.zeros((N, N))
    copy.deepcopy(sdfg)(b=out0, K=0)
    assert np.allclose(out0, np.zeros((N, N)), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_inner_param_dependent_guard_is_not_hoisted():
    """``if a[j]`` varies with the inner map's own parameter -- a genuine
    per-element mask that must stay inside."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a[j] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    before = copy.deepcopy(sdfg)
    a = np.array([1.0, -1.0] * (N // 2))
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None
    assert np.allclose(_run(sdfg, a=a), _run(before, a=a), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_two_dimensional_guard_is_not_hoisted():
    """``if a2[i, j]`` depends on both parameters; neither level accepts it."""

    @dace.program
    def kern(a2: dace.float64[N, N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a2[i, j] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None


def test_collapsed_map_checks_every_parameter():
    """A single N-dimensional map is invariant only if the condition avoids
    *all* of its parameters, not just the first."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            if a[j] > 0.0:
                b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None, "j is a parameter of the collapsed map"


def test_collapsed_map_invariant_condition_hoists():
    """The same collapsed map with a genuinely invariant condition hoists."""
    K = dace.symbol('K')

    @dace.program
    def kern(b: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            if K > 0:
                b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is not None
    sdfg.validate()
    assert _guard_depth(sdfg) == 0, "a single collapsed map has no chain to stall in"


def test_no_match_leaves_graph_untouched():
    """A pass that does not apply must not mutate the graph."""

    @dace.program
    def kern(a2: dace.float64[N, N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a2[i, j] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    before = sdfg.to_json()
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before, "no-apply must be a no-op on the graph"


def test_guard_whose_branch_holds_a_map_is_value_preserving():
    """A branch containing a further map does not block the hoist -- the
    condition is evaluated outside that map and cannot depend on it -- but the
    result must still compute the same values."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            if a[i] > 0.0:
                for j in dace.map[0:N]:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    a = np.array([1.0, -1.0] * (N // 2))
    before = _run(sdfg, a=a)
    MoveMapInvariantIfUp().apply_pass(sdfg, {})
    sdfg.validate()
    after = _run(sdfg, a=a)
    assert np.allclose(after, before, rtol=1e-9, atol=1e-9, equal_nan=True)
    ref = np.where((a > 0.0)[:, None], 1.0, 0.0)
    assert np.allclose(after, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_body_defined_condition_symbol_is_never_stranded():
    """Regression: hoisting past the outer map must not leave the condition's
    defining assignment behind.

    After ``if a[i]`` clears the inner map, its condition is a symbol assigned
    on an interstate edge one level in. The plain hoist can only re-express a
    condition through ``symbol_mapping``, so accepting a body-defined symbol
    there would move the guard out while its definition stayed put, leaving the
    symbol undefined at the outer scope (a KeyError at code generation).
    """

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a[i] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1, "must stop at the map its condition reads"
    sdfg.validate()
    # Code generation resolves every free symbol -- this is what regressed.
    sdfg.generate_code()

    a = np.array([1.0, -1.0] * (N // 2))
    ref = np.where((a > 0.0)[:, None], 1.0, 0.0)
    assert np.allclose(_run(sdfg, a=a), ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_idempotent():
    """Re-running finds nothing new once the guard has settled."""

    @dace.program
    def kern(a: dace.float64[N], b: dace.float64[N, N]):
        for i in dace.map[0:N]:
            for j in dace.map[0:N]:
                if a[i] > 0.0:
                    b[i, j] = 1.0

    sdfg = kern.to_sdfg(simplify=True)
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) == 1
    assert MoveMapInvariantIfUp().apply_pass(sdfg, {}) is None
    sdfg.validate()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

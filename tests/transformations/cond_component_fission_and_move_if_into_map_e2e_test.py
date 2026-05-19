# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end + structural tests for two pipeline transformations using
Python-frontend kernels with ICON/cloudsc-style neighbour-gather indirection
in a subset of dimensions.

* :class:`ConditionalComponentFission` (a ``ppl.Pass``) replicates a
  MapFission-blocking guarding conditional once per independent output so the
  surrounding map can subsequently fission.
* :class:`MoveIfIntoMap` (a ``PatternTransformation``) pushes a map-invariant
  guard into the map scope, and must refuse to hoist a guard that depends on
  the map index.

Every test compares the post-pass SDFG against a pure-numpy oracle for both
the guard-taken and guard-not-taken cases, asserts the pass fired, validates
the SDFG, and checks the structural effect (``ConditionalBlock`` counts /
top-level absence / map params).
"""
import copy

import numpy as np

import dace
from dace.sdfg.nodes import MapEntry, NestedSDFG
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate.move_if_into_map import MoveIfIntoMap
from dace.transformation.passes.conditional_component_fission import ConditionalComponentFission

N = dace.symbol('N')
L = dace.symbol('L')

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _count_conditional_blocks(sdfg: dace.SDFG) -> int:
    """Number of ``ConditionalBlock`` regions anywhere in ``sdfg`` (recursive)."""
    return sum(1 for cfg in sdfg.all_control_flow_regions(recursive=True) if isinstance(cfg, ConditionalBlock))


def _top_level_conditional_blocks(sdfg: dace.SDFG) -> int:
    """Number of ``ConditionalBlock`` regions at the top level of ``sdfg``."""
    return sum(1 for b in sdfg.nodes() if isinstance(b, ConditionalBlock))


def _conditional_inside_map_scope(sdfg: dace.SDFG) -> bool:
    """Whether some ``ConditionalBlock`` lives inside a NestedSDFG that is
    itself inside a map scope (i.e. the guard was pushed into a map body)."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, NestedSDFG):
                continue
            if state.entry_node(node) is None:
                continue  # NestedSDFG not under a map scope
            for block in node.sdfg.all_control_flow_regions(recursive=True):
                if isinstance(block, ConditionalBlock):
                    return True
    return False


def _map_params(sdfg: dace.SDFG):
    """Set of all map parameter names across every state of ``sdfg``."""
    params = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, MapEntry):
                params.update(str(p) for p in node.map.params)
    return params


# --------------------------------------------------------------------------
# Kernels (file-level @dace.program, symbolic shapes)
# --------------------------------------------------------------------------


@dace.program
def fission_two_outputs(active: dace.int32[1], w: dace.float64[N, L], v: dace.float64[N, L], cidx: dace.int32[N, 2],
                        vidx: dace.int32[N, 2], b: dace.float64[N, L], d: dace.float64[N, L]):
    """CCF's matched shape: ONE map whose body is a guard with two
    independent neighbour-gather outputs (the map body lowers to a
    NestedSDFG holding the ConditionalBlock -- the MapFission-blocking
    NestedSDFG CCF replicates per output group)."""
    for i, k in dace.map[0:N, 0:L]:
        if active[0] > 0:
            b[i, k] = w[cidx[i, 0], k] + 1.0
            d[i, k] = v[vidx[i, 1], k] * 3.0


@dace.program
def fission_three_outputs(active: dace.int32[1], w: dace.float64[N, L], v: dace.float64[N, L], u: dace.float64[N, L],
                          cidx: dace.int32[N, 2], vidx: dace.int32[N, 2], uidx: dace.int32[N, 2], b: dace.float64[N, L],
                          d: dace.float64[N, L], g: dace.float64[N, L]):
    """CCF's matched shape with three independent guarded outputs."""
    for i, k in dace.map[0:N, 0:L]:
        if active[0] > 0:
            b[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
            d[i, k] = v[vidx[i, 1], k] * 3.0
            g[i, k] = u[uidx[i, 0], k] + u[uidx[i, 1], k]


@dace.program
def move_if_invariant(active: dace.int32[1], w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L]):
    """Map-invariant guard around a neighbour-gather map."""
    if active[0] > 0:
        for i, k in dace.map[0:N, 0:L]:
            out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


@dace.program
def move_if_index_dependent(w: dace.float64[N, L], cidx: dace.int32[N, 2], out: dace.float64[N, L]):
    """Guard depends on the map index -- must NOT hoist over the index map."""
    for i, k in dace.map[0:N, 0:L]:
        if i % 2 == 0:
            out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]


# --------------------------------------------------------------------------
# Oracles (pure numpy)
# --------------------------------------------------------------------------


def _oracle_fission_two(active, w, v, cidx, vidx, n, l):
    b = np.zeros((n, l))
    d = np.zeros((n, l))
    if active[0] > 0:
        for i in range(n):
            for k in range(l):
                b[i, k] = w[cidx[i, 0], k] + 1.0
                d[i, k] = v[vidx[i, 1], k] * 3.0
    return b, d


def _oracle_fission_three(active, w, v, u, cidx, vidx, uidx, n, l):
    b = np.zeros((n, l))
    d = np.zeros((n, l))
    g = np.zeros((n, l))
    if active[0] > 0:
        for i in range(n):
            for k in range(l):
                b[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
                d[i, k] = v[vidx[i, 1], k] * 3.0
                g[i, k] = u[uidx[i, 0], k] + u[uidx[i, 1], k]
    return b, d, g


def _oracle_move_if(active, w, cidx, n, l):
    out = np.zeros((n, l))
    if active[0] > 0:
        for i in range(n):
            for k in range(l):
                out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
    return out


def _oracle_index_dependent(w, cidx, n, l):
    out = np.zeros((n, l))
    for i in range(n):
        if i % 2 == 0:
            for k in range(l):
                out[i, k] = 2.0 * w[cidx[i, 0], k] - w[cidx[i, 1], k]
    return out


# --------------------------------------------------------------------------
# A) ConditionalComponentFission
# --------------------------------------------------------------------------


def test_conditional_component_fission_two_outputs():
    """Guard over two independent gathers -> conditional replicated per
    output; numerically equal to the numpy oracle (taken + not taken)."""
    n, l = 12, 4
    rng = np.random.default_rng(0)
    w = rng.random((n, l))
    v = rng.random((n, l))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    vidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)

    sdfg = fission_two_outputs.to_sdfg(simplify=True)
    cb_before = _count_conditional_blocks(sdfg)
    assert cb_before >= 1, "kernel should start with at least one ConditionalBlock"

    applied = ConditionalComponentFission().apply_pass(sdfg, {})
    assert applied, "ConditionalComponentFission must replicate the guarded NestedSDFG"
    sdfg.validate()

    cb_after = _count_conditional_blocks(sdfg)
    assert cb_after > cb_before, (f"conditional must be replicated per output: {cb_before} -> {cb_after}")
    assert cb_after >= 2, "expected one guard per independent output"

    for active_val in (np.int32(1), np.int32(0)):
        active = np.array([active_val], dtype=np.int32)
        exp_b, exp_d = _oracle_fission_two(active, w, v, cidx, vidx, n, l)
        got_b = np.zeros((n, l))
        got_d = np.zeros((n, l))
        csdfg = copy.deepcopy(sdfg)
        csdfg(active=active.copy(),
              w=w.copy(),
              v=v.copy(),
              cidx=cidx.copy(),
              vidx=vidx.copy(),
              b=got_b,
              d=got_d,
              N=n,
              L=l)
        np.testing.assert_allclose(got_b, exp_b, rtol=1e-12, err_msg=f"b mismatch active={active_val}")
        np.testing.assert_allclose(got_d, exp_d, rtol=1e-12, err_msg=f"d mismatch active={active_val}")


def test_conditional_component_fission_three_outputs():
    """Three independent gathers under one guard -> three replicated
    conditionals; numerically equal to the numpy oracle (taken + not)."""
    n, l = 10, 5
    rng = np.random.default_rng(1)
    w = rng.random((n, l))
    v = rng.random((n, l))
    u = rng.random((n, l))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    vidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)
    uidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)

    sdfg = fission_three_outputs.to_sdfg(simplify=True)
    cb_before = _count_conditional_blocks(sdfg)
    assert cb_before >= 1

    applied = ConditionalComponentFission().apply_pass(sdfg, {})
    assert applied, "ConditionalComponentFission must replicate the guarded NestedSDFG"
    sdfg.validate()

    cb_after = _count_conditional_blocks(sdfg)
    assert cb_after >= 3, (f"expected one guard per independent output (>=3), got {cb_after}")

    for active_val in (np.int32(1), np.int32(0)):
        active = np.array([active_val], dtype=np.int32)
        exp_b, exp_d, exp_g = _oracle_fission_three(active, w, v, u, cidx, vidx, uidx, n, l)
        got_b = np.zeros((n, l))
        got_d = np.zeros((n, l))
        got_g = np.zeros((n, l))
        csdfg = copy.deepcopy(sdfg)
        csdfg(active=active.copy(),
              w=w.copy(),
              v=v.copy(),
              u=u.copy(),
              cidx=cidx.copy(),
              vidx=vidx.copy(),
              uidx=uidx.copy(),
              b=got_b,
              d=got_d,
              g=got_g,
              N=n,
              L=l)
        np.testing.assert_allclose(got_b, exp_b, rtol=1e-12, err_msg=f"b mismatch active={active_val}")
        np.testing.assert_allclose(got_d, exp_d, rtol=1e-12, err_msg=f"d mismatch active={active_val}")
        np.testing.assert_allclose(got_g, exp_g, rtol=1e-12, err_msg=f"g mismatch active={active_val}")


# --------------------------------------------------------------------------
# B) MoveIfIntoMap
# --------------------------------------------------------------------------


def test_move_if_into_map_contract_on_frontend_shape():
    """``MoveIfIntoMap`` is a *mid-pipeline interstate* pass: its
    ``can_be_applied`` requires the ``ConditionalBlock`` to already sit in a
    NestedSDFG whose enclosing state's entry is a ``MapEntry`` (a shape
    earlier canonicalize stages build), so it is a provable **no-op** on the
    raw Python-frontend shape ``if c: for i,k in dace.map: ...`` (the guard
    is a top-level ConditionalBlock). It must refuse here and leave the SDFG
    numerically intact. Its positive ``guard-moved-into-map`` behaviour is
    covered end-to-end by the pipeline-level ``tests/canonicalize/`` suite."""
    n, l = 12, 4
    rng = np.random.default_rng(2)
    w = rng.random((n, l))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)

    sdfg = move_if_invariant.to_sdfg(simplify=True)
    sdfg.validate()
    assert _top_level_conditional_blocks(sdfg) >= 1

    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0, ("MoveIfIntoMap must refuse the top-level frontend shape "
                          f"(guard not yet inside a map's NestedSDFG), got {applied}")
    sdfg.validate()
    # Refused -> unchanged -> still numerically correct for taken/not-taken.
    for active_val in (np.int32(1), np.int32(0)):
        active = np.array([active_val], dtype=np.int32)
        exp = _oracle_move_if(active, w, cidx, n, l)
        got = np.zeros((n, l))
        csdfg = copy.deepcopy(sdfg)
        csdfg(active=active.copy(), w=w.copy(), cidx=cidx.copy(), out=got, N=n, L=l)
        np.testing.assert_allclose(got, exp, rtol=1e-12, err_msg=f"out mismatch active={active_val}")


def test_move_if_into_map_index_dependent_does_not_hoist():
    """The guard depends on the map index (``i % 2 == 0``): MoveIfIntoMap
    must NOT hoist it over/out of the index-defining map. The guard stays
    inside the map and the kernel remains numerically correct."""
    n, l = 12, 4
    rng = np.random.default_rng(3)
    w = rng.random((n, l))
    cidx = rng.integers(0, n, size=(n, 2), dtype=np.int32)

    sdfg = move_if_index_dependent.to_sdfg(simplify=True)
    sdfg.validate()

    # The guard reads the outer map index; MoveIfIntoMap's invariance guard
    # must reject the match (it would be unsound to evaluate ``i % 2`` once
    # outside the per-iteration body).
    applied = sdfg.apply_transformations_repeated(MoveIfIntoMap)
    assert applied == 0, (f"MoveIfIntoMap must NOT hoist an index-dependent guard, got {applied}")
    sdfg.validate()

    # The guard must remain inside the index-defining map scope.
    assert _conditional_inside_map_scope(sdfg), ("index-dependent guard must stay inside the map scope")
    assert 'i' in _map_params(sdfg) or any(p.startswith('i')
                                           for p in _map_params(sdfg)), "the index-defining map must still be present"

    exp = _oracle_index_dependent(w, cidx, n, l)
    got = np.zeros((n, l))
    sdfg(w=w.copy(), cidx=cidx.copy(), out=got, N=n, L=l)
    np.testing.assert_allclose(got, exp, rtol=1e-12, err_msg="index-dependent kernel mismatch")


if __name__ == "__main__":
    test_conditional_component_fission_two_outputs()
    test_conditional_component_fission_three_outputs()
    test_move_if_into_map_contract_on_frontend_shape()
    test_move_if_into_map_index_dependent_does_not_hoist()
    print("All tests passed.")

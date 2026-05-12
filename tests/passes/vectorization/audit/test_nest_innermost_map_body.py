# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for ``NestInnermostMapBodyIntoNSDFG`` (P1 vectorization-prep).

After the pass:
- Every innermost map whose body had bare tasklets is wrapped in a
  single NestedSDFG.
- Innermost maps that already contained one NestedSDFG are left alone.
- Outer maps are not touched.

End-to-end numerical correctness is verified against the pre-pass SDFG
output, the rewrite is a pure SDFG-shape transform.
"""
import numpy as np

import dace
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.vectorization_utils import (
    get_single_nsdfg_inside_map,
    is_innermost_map,
)


N = dace.symbol("N")


@dace.program
def bare_tasklet_body(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


@dace.program
def nested_map_program(a: dace.float64[N, N], b: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            b[i, j] = a[i, j] + 1.0


def _innermost_maps(sdfg: dace.SDFG):
    return [(n, g) for n, g in sdfg.all_nodes_recursive()
            if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and is_innermost_map(g, n)]


def test_bare_tasklet_body_gets_nested():
    sdfg = bare_tasklet_body.to_sdfg(simplify=True)
    inner = _innermost_maps(sdfg)
    assert inner, "test setup: expected at least one innermost map"
    # Pre-pass: none of the innermost maps already wrap a single nested SDFG.
    assert all(get_single_nsdfg_inside_map(g, n) is None for n, g in inner)

    n_applied = NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    assert n_applied is not None and n_applied >= 1

    inner_post = _innermost_maps(sdfg)
    assert all(get_single_nsdfg_inside_map(g, n) is not None for n, g in inner_post), \
        "every innermost map should now wrap a single NestedSDFG"


def test_nested_pass_is_idempotent_on_already_wrapped_body():
    sdfg = bare_tasklet_body.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    # A second invocation must be a no-op.
    second = NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    assert second is None


def test_pass_does_not_touch_outer_maps_in_nested_map_program():
    sdfg = nested_map_program.to_sdfg(simplify=True)
    # Snapshot which maps are outer vs inner.
    outers_before = [(n, g) for n, g in sdfg.all_nodes_recursive()
                     if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)
                     and not is_innermost_map(g, n)]
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    outers_after = [(n, g) for n, g in sdfg.all_nodes_recursive()
                    if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState)
                    and not is_innermost_map(g, n)]
    # The outer map count is preserved (the pass only touches innermost maps).
    assert len(outers_before) == len(outers_after)


def test_numerical_correctness_bare_tasklet_body():
    sdfg = bare_tasklet_body.to_sdfg(simplify=True)

    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal(32).astype(np.float64)
    b_ref = np.zeros_like(a)
    b_post = np.zeros_like(a)

    # Reference output (pre-pass).
    sdfg(a=a, b=b_ref, N=32)

    sdfg_pass = bare_tasklet_body.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg_pass, {})
    sdfg_pass(a=a, b=b_post, N=32)

    np.testing.assert_allclose(b_post, b_ref)


def test_numerical_correctness_nested_map_program():
    sdfg = nested_map_program.to_sdfg(simplify=True)
    rng = np.random.default_rng(seed=1)
    a = rng.standard_normal((16, 16)).astype(np.float64)
    b_ref = np.zeros_like(a)
    b_post = np.zeros_like(a)
    sdfg(a=a, b=b_ref, N=16)

    sdfg_pass = nested_map_program.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg_pass, {})
    sdfg_pass(a=a, b=b_post, N=16)

    np.testing.assert_allclose(b_post, b_ref)

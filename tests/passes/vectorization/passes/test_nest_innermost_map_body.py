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
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.nest_innermost_map_body import (
    NestInnermostMapBodyIntoNSDFG, )
from dace.transformation.passes.vectorization.utils.map_predicates import (
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
    outers_before = [
        (n, g) for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and not is_innermost_map(g, n)
    ]
    NestInnermostMapBodyIntoNSDFG().apply_pass(sdfg, {})
    outers_after = [
        (n, g) for n, g in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and isinstance(g, dace.SDFGState) and not is_innermost_map(g, n)
    ]
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


NK = dace.symbol("NK")
NJ = dace.symbol("NJ")


@dace.program
def _k2_col_broadcast(a: dace.float64[NK], c: dace.float64[NK, NJ]):
    """K=2 cross-dim broadcast write: ``a[jk // 2]`` (1-D source) broadcast across
    the ``jc`` lane into the 2-D tile ``c[jk, jc]``. The broadcast is staged through
    a frontend ``c_slice`` transient."""
    for jk in range(NK):
        for jc in range(NJ):
            c[jk, jc] = a[jk // 2]


def test_k2_broadcast_tile_k1_tail_stays_valid_through_nest():
    """Regression for the #1d descent gap (symbolic-dim K=2 broadcast + tile_k1 tail).

    Root cause (caught by the orchestrator's per-subpass validate gate): with symbolic
    dims the tile dims are not provably divisible, so ``SplitMapForTileRemainder`` peels a
    ``__tile_k1_tail`` and, while replicating the broadcast scope, emits boundary edges
    whose SOURCE AccessNode (``a``) disagrees with the memlet array (the renamed
    ``c_slice_0`` tail copy). DaCe ``validate()`` tolerates that as long as ``c_slice_0``
    still exists, so it only becomes a hard failure once
    ``NestInnermostMapBodyIntoNSDFG`` moves ``c_slice_0`` inside the body NSDFG, leaving
    the outer edge dangling (``Array "c_slice_0" not found in SDFG``).

    The orchestrator must drive this kernel through preprocessing without leaving a
    malformed SDFG (the final validate must pass).
    """
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)
    sdfg = _k2_col_broadcast.to_sdfg(simplify=True)
    # tile_k1 tail is the configuration that exposes the bug. expand_tile_nodes=False keeps
    # it a transform-only check (no compile -> no UCX flake); the per-subpass validate gate
    # inside the orchestrator is what asserts each preprocessing pass left the SDFG valid.
    VectorizeCPUMultiDim(widths=(8, 8),
                         target_isa="SCALAR",
                         remainder_strategy="scalar_postamble",
                         scalar_remainder_emit="tile_k1",
                         expand_tile_nodes=False).apply_pass(sdfg, {})
    sdfg.validate()


def _build_k2_scalar_staged_map_body() -> dace.SDFG:
    """Build a K=2 map whose body stages a 2-D global read through a Scalar.

    Mirrors the post-split shape the orchestrator hands :class:`NestInnermostMapBodyIntoNSDFG`:
    a ``jk, jc`` map whose body copies ``a[jk, jc]`` (subset rank 2) into a
    frontend ``c_slice``-style transient :class:`~dace.data.Scalar` (the
    ``other_subset`` ``[0]`` is rank 1) before the compute tasklet, then stores
    back to the global ``c``. This is the exact element that, before the fix, left
    the body-NSDFG boundary edge carrying a stale rank-1 ``other_subset`` against
    the rank-2 source array, so ``validate()`` rejected it with
    ``Memlet other_subset does not match node dimension``.
    """
    NK, NJ = dace.symbol("NK"), dace.symbol("NJ")
    sdfg = dace.SDFG("k2_scalar_staged")
    sdfg.add_array("a", (NK, NJ), dace.float64)
    sdfg.add_array("c", (NK, NJ), dace.float64)
    sdfg.add_scalar("c_slice", dace.float64, transient=True)
    state = sdfg.add_state("main")
    a = state.add_access("a")
    c = state.add_access("c")
    cs = state.add_access("c_slice")
    me, mx = state.add_map("m", dict(jk="0:NK", jc="0:NJ"))
    tlet = state.add_tasklet("assign", {"_inp"}, {"_out"}, "_out = _inp")
    me.add_in_connector("IN_a")
    me.add_out_connector("OUT_a")
    mx.add_in_connector("IN_c")
    mx.add_out_connector("OUT_c")
    # Outer full-array read into the map.
    state.add_edge(a, None, me, "IN_a", Memlet(data="a", subset="0:NK, 0:NJ"))
    # Scalar-staging copy: a[jk, jc] -> c_slice[0] (subset rank 2, other_subset rank 1).
    state.add_edge(me, "OUT_a", cs, None, Memlet(data="a", subset="jk, jc", other_subset="0"))
    state.add_edge(cs, None, tlet, "_inp", Memlet("c_slice[0]"))
    # Compute store back to the global.
    state.add_edge(tlet, "_out", mx, "IN_c", Memlet(data="c", subset="jk, jc"))
    state.add_edge(mx, "OUT_c", c, None, Memlet(data="c", subset="0:NK, 0:NJ"))
    return sdfg


def test_k2_scalar_staged_body_nest_alone_is_valid():
    """Nesting a K=2 scalar-staged body must leave a VALID SDFG on its own.

    Regression for the transiently-invalid boundary the pass used to leave: with a
    scalar-staging element in the body, ``nest_state_subgraph`` deep-copied the
    ``a[jk, jc] -> c_slice[0]`` memlet onto the reconnected boundary edge, so the
    body-NSDFG connector edge carried a stale rank-1 ``other_subset`` against the
    rank-2 source array ``a`` and ``sdfg.validate()`` failed with
    ``Memlet other_subset does not match node dimension``. The pass now clears that
    stale ``other_subset`` on the boundary, so validate passes IMMEDIATELY after the
    pass -- without relying on the downstream ``ExpandNestedSDFGInputs`` repair.
    """
    sdfg = _build_k2_scalar_staged_map_body()
    sdfg.validate()  # the input is well-formed

    n_applied = NestInnermostMapBodyIntoNSDFG(nest_provably_divisible=True).apply_pass(sdfg, {})
    assert n_applied == 1, "expected the single innermost map body to be nested"

    # The boundary edges into/out of the body NSDFG must not carry a stale
    # other_subset (the connector descriptor defines the inner shape).
    for _state in sdfg.states():
        for node in _state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for edge in (*_state.in_edges(node), *_state.out_edges(node)):
                    if edge.data is not None and edge.data.data is not None:
                        assert edge.data.other_subset is None, \
                            f"boundary edge {edge.data.data} still carries other_subset {edge.data.other_subset}"

    # Nest alone must produce a valid SDFG (no ExpandNestedSDFGInputs needed).
    sdfg.validate()


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

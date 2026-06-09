# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`InferBodyTransientShapes` (forward-analysis pre-shape).

Per user direction 2026-06-09: widening of body-NSDFG transients is done
proactively (a single pre-pass) BEFORE the walker / converter run, by
forward-propagating non-transient AN access classifications.
"""
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.infer_body_transient_shapes import (InferBodyTransientShapes)


def _build_inner_body(tasklet_body, mid_t_shape=(1, ), has_b=True, ii_in_a="ii", ii_in_b="ii"):
    """Build a body-NSDFG fixture: one inner tasklet writing to mid_t."""
    sdfg = dace.SDFG("infer_fixture")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    if has_b:
        sdfg.add_array("B", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    if has_b:
        inner.add_array("B", (8, ), dace.float64, transient=False)
    inner.add_array("mid_t", mid_t_shape, dace.float64, transient=True)
    instate = inner.add_state("body")
    a_in = instate.add_access("A")
    mid = instate.add_access("mid_t")
    in_conns = {"_a", "_b"} if has_b else {"_a"}
    tasklet = instate.add_tasklet("body_t", in_conns, {"_o"}, tasklet_body)
    instate.add_edge(a_in, None, tasklet, "_a", Memlet(f"A[{ii_in_a}]"))
    if has_b:
        b_in = instate.add_access("B")
        instate.add_edge(b_in, None, tasklet, "_b", Memlet(f"B[{ii_in_b}]"))
    mid_subset = ", ".join(f"0:{s}" for s in mid_t_shape)
    instate.add_edge(tasklet, "_o", mid, None, Memlet(f"mid_t[{mid_subset}]"))
    inputs = {"A", "B"} if has_b else {"A"}
    nsdfg = state.add_nested_sdfg(inner, inputs, set(), symbol_mapping={"ii": "ii"})
    state.add_memlet_path(state.add_access("A"), me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    if has_b:
        state.add_memlet_path(state.add_access("B"), me, nsdfg, dst_conn="B", memlet=Memlet("B[0:8]"))
    state.add_nedge(nsdfg, mx, Memlet())
    return sdfg, inner


def test_pre_pass_widens_length1_transient_when_producer_reads_tile():
    """``mid_t = A[ii] + B[ii]`` (both LINEAR) -> mid_t widened to (8,)."""
    sdfg, inner = _build_inner_body("_o = _a + _b")
    InferBodyTransientShapes(widths=(8, )).apply_pass(sdfg, {})
    assert tuple(inner.arrays["mid_t"].shape) == (8, )


def test_pre_pass_widens_unop_chain():
    """``mid_t = abs(A[ii])`` -> mid_t widened to (8,)."""
    sdfg, inner = _build_inner_body("_o = abs(_a)", has_b=False)
    InferBodyTransientShapes(widths=(8, )).apply_pass(sdfg, {})
    assert tuple(inner.arrays["mid_t"].shape) == (8, )


def test_pre_pass_leaves_transient_unchanged_for_all_constant_chain():
    """Producer reads only ``A[0]`` (CONSTANT-only) -> mid_t stays length-1."""
    sdfg, inner = _build_inner_body("_o = _a + _b", ii_in_a="0", ii_in_b="0")
    InferBodyTransientShapes(widths=(8, )).apply_pass(sdfg, {})
    assert tuple(inner.arrays["mid_t"].shape) == (1, )


def test_pre_pass_propagates_through_chained_tasklets():
    """Two-tasklet chain: t1 reads A[ii] -> mid1 (transient); t2 reads mid1 -> mid2.
    Both mid1 and mid2 should be widened (mid1 directly via A[ii] tile classification;
    mid2 transitively via mid1 already being tile)."""
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", (8, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    inner = dace.SDFG("body")
    inner.add_array("A", (8, ), dace.float64, transient=False)
    inner.add_array("mid1", (1, ), dace.float64, transient=True)
    inner.add_array("mid2", (1, ), dace.float64, transient=True)
    instate = inner.add_state("body")
    a_in = instate.add_access("A")
    m1 = instate.add_access("mid1")
    m2 = instate.add_access("mid2")
    t1 = instate.add_tasklet("t1", {"_a"}, {"_o"}, "_o = _a")
    t2 = instate.add_tasklet("t2", {"_a"}, {"_o"}, "_o = abs(_a)")
    instate.add_edge(a_in, None, t1, "_a", Memlet("A[ii]"))
    instate.add_edge(t1, "_o", m1, None, Memlet("mid1[0]"))
    instate.add_edge(m1, None, t2, "_a", Memlet("mid1[0]"))
    instate.add_edge(t2, "_o", m2, None, Memlet("mid2[0]"))
    nsdfg = state.add_nested_sdfg(inner, {"A"}, set(), symbol_mapping={"ii": "ii"})
    state.add_memlet_path(state.add_access("A"), me, nsdfg, dst_conn="A", memlet=Memlet("A[0:8]"))
    state.add_nedge(nsdfg, mx, Memlet())
    InferBodyTransientShapes(widths=(8, )).apply_pass(sdfg, {})
    assert tuple(inner.arrays["mid1"].shape) == (8, ), \
        f"mid1 (directly produced from A[ii]) should be tile-widened, got {tuple(inner.arrays['mid1'].shape)}"
    assert tuple(inner.arrays["mid2"].shape) == (8, ), \
        f"mid2 (transitively via mid1) should be tile-widened, got {tuple(inner.arrays['mid2'].shape)}"


def test_pre_pass_empty_sdfg_returns_none():
    """SDFG with no tile-tagged body NSDFG -> no rewrites -> None."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    assert InferBodyTransientShapes(widths=(8, )).apply_pass(sdfg, {}) is None


def test_pre_pass_refuses_invalid_widths():
    """Constructor refuses widths outside {1, 2, 3}."""
    with pytest.raises(ValueError, match=r"widths length"):
        InferBodyTransientShapes(widths=())
    with pytest.raises(ValueError, match=r"widths length"):
        InferBodyTransientShapes(widths=(8, 8, 8, 8))

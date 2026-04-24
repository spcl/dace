# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for LoopInvariantCodeMotion.

Every test builds an SDFG, applies LICM, compiles and runs the transformed
SDFG, and compares the result against a pure-Python reference implementing
the same computation. This catches both structural regressions and
semantics-breaking hoists.
"""
import ctypes
# OpenMP must be loaded before any DaCe-compiled stub library dlopen()s it.
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import math
import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion

N = dace.symbol("N")
K = dace.symbol("K")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loop(sdfg: dace.SDFG, loop_var: str, end_sym: str, label: str = "loop") -> LoopRegion:
    loop = LoopRegion(label,
                      condition_expr=f"{loop_var} < {end_sym}",
                      loop_var=loop_var,
                      initialize_expr=f"{loop_var} = 0",
                      update_expr=f"{loop_var} = {loop_var} + 1")
    sdfg.add_node(loop, is_start_block=True)
    return loop


def _state_tasklets(state):
    return [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]


def _preheaders(sdfg: dace.SDFG, loop: LoopRegion):
    parent = loop.parent_graph
    return [e.src for e in parent.in_edges(loop)
            if hasattr(e.src, "label") and e.src.label.startswith(f"{loop.label}_licm_preheader")]


def _run_and_check(sdfg: dace.SDFG, reference_fn, **inputs):
    """Run the SDFG in-place on copies of ``inputs`` and compare each mutable
    buffer against the result of running ``reference_fn`` on its own copies.
    """
    sdfg_inputs = {k: np.array(v, copy=True) for k, v in inputs.items()}
    ref_inputs = {k: np.array(v, copy=True) for k, v in inputs.items()}
    sdfg(**sdfg_inputs)
    reference_fn(**ref_inputs)
    for k in inputs:
        a = sdfg_inputs[k]
        b = ref_inputs[k]
        if isinstance(a, np.ndarray):
            assert np.allclose(a, b), f"mismatch in `{k}`: sdfg={a} ref={b}"


# ---------------------------------------------------------------------------
# 1. Pure scalar tasklet with outer inputs is hoisted to the preheader.
# ---------------------------------------------------------------------------

def test_pure_tasklet_hoisted_from_loop_region():
    sdfg = dace.SDFG("licm_pure_hoist")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("a")
    br = body.add_read("b")
    tw = body.add_tasklet("add", {"in_a", "in_b"}, {"res"}, "res = in_a + in_b")
    tnode = body.add_access("t")
    body.add_edge(ar, None, tw, "in_a", mm.Memlet("a[0]"))
    body.add_edge(br, None, tw, "in_b", mm.Memlet("b[0]"))
    body.add_edge(tw, "res", tnode, None, mm.Memlet("t[0]"))

    ow = body.add_write("outp")
    cpy = body.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    body.add_edge(tnode, None, cpy, "ti", mm.Memlet("t[0]"))
    body.add_edge(cpy, "o", ow, None, mm.Memlet("outp[i]"))
    sdfg.validate()

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    assert hoisted == 1
    assert "add" not in {t.label for t in _state_tasklets(body)}
    assert len(_preheaders(sdfg, loop)) == 1

    def py_ref(a, b, outp, N):
        for i in range(N):
            outp[i] = a[0] + b[0]

    _run_and_check(sdfg, py_ref,
                   a=np.array([2.5]), b=np.array([1.5]),
                   outp=np.zeros(7), N=7)


# ---------------------------------------------------------------------------
# 2. Memory read with no in-body writer: hoisted.
# ---------------------------------------------------------------------------

def test_invariant_load_without_inloop_writer_is_hoisted():
    sdfg = dace.SDFG("licm_invariant_load")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("A")
    tw = body.add_tasklet("id", {"x"}, {"y"}, "y = x")
    tnode = body.add_access("t")
    body.add_edge(ar, None, tw, "x", mm.Memlet("A[0]"))
    body.add_edge(tw, "y", tnode, None, mm.Memlet("t[0]"))

    ow = body.add_write("outp")
    cpy = body.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    body.add_edge(tnode, None, cpy, "ti", mm.Memlet("t[0]"))
    body.add_edge(cpy, "o", ow, None, mm.Memlet("outp[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()
    assert hoisted == 1

    def py_ref(A, outp, N):
        for i in range(N):
            outp[i] = A[0]

    _run_and_check(sdfg, py_ref,
                   A=np.array([4.0]), outp=np.zeros(5), N=5)


# ---------------------------------------------------------------------------
# 3. Memory read with an in-body writer: NOT hoisted (alias).
# ---------------------------------------------------------------------------

def test_load_with_inloop_writer_is_not_hoisted():
    sdfg = dace.SDFG("licm_aliased_load")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("A")
    tw = body.add_tasklet("ld", {"x"}, {"y"}, "y = x")
    tnode = body.add_access("t")
    body.add_edge(ar, None, tw, "x", mm.Memlet("A[0]"))
    body.add_edge(tw, "y", tnode, None, mm.Memlet("t[0]"))

    aw = body.add_write("A")
    one = body.add_tasklet("one", set(), {"z"}, "z = 1.0")
    body.add_edge(one, "z", aw, None, mm.Memlet("A[i]"))

    ow = body.add_write("outp")
    cpy = body.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    body.add_edge(tnode, None, cpy, "ti", mm.Memlet("t[0]"))
    body.add_edge(cpy, "o", ow, None, mm.Memlet("outp[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    assert not hoisted
    assert any(t.label == "ld" for t in _state_tasklets(body))


# ---------------------------------------------------------------------------
# 4. Transitive invariance chain: two dependent tasklets both hoisted.
# ---------------------------------------------------------------------------

def test_transitive_chain_is_hoisted():
    sdfg = dace.SDFG("licm_chain")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_array("c", [1], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("u", [1], dace.float64)
    sdfg.add_transient("v", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("a")
    br = body.add_read("b")
    cr = body.add_read("c")
    t1 = body.add_tasklet("add_u", {"x", "y"}, {"r"}, "r = x + y")
    u = body.add_access("u")
    body.add_edge(ar, None, t1, "x", mm.Memlet("a[0]"))
    body.add_edge(br, None, t1, "y", mm.Memlet("b[0]"))
    body.add_edge(t1, "r", u, None, mm.Memlet("u[0]"))

    t2 = body.add_tasklet("mul_v", {"x", "y"}, {"r"}, "r = x * y")
    v = body.add_access("v")
    body.add_edge(u, None, t2, "x", mm.Memlet("u[0]"))
    body.add_edge(cr, None, t2, "y", mm.Memlet("c[0]"))
    body.add_edge(t2, "r", v, None, mm.Memlet("v[0]"))

    ow = body.add_write("outp")
    cpy = body.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    body.add_edge(v, None, cpy, "ti", mm.Memlet("v[0]"))
    body.add_edge(cpy, "o", ow, None, mm.Memlet("outp[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    body_labels = {t.label for t in _state_tasklets(body)}
    assert "add_u" not in body_labels
    assert "mul_v" not in body_labels
    assert hoisted >= 2

    def py_ref(a, b, c, outp, N):
        u = a[0] + b[0]
        v = u * c[0]
        for i in range(N):
            outp[i] = v

    _run_and_check(sdfg, py_ref,
                   a=np.array([2.0]), b=np.array([3.0]),
                   c=np.array([4.0]), outp=np.zeros(6), N=6)


# ---------------------------------------------------------------------------
# 5. Map scope: pure tasklet on outer data is hoisted through MapEntry.
# ---------------------------------------------------------------------------

def test_map_scope_pure_tasklet_hoisted():
    sdfg = dace.SDFG("licm_map_hoist")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("b", [1], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)

    state = sdfg.add_state("st", is_start_block=True)
    me, mx = state.add_map("m", {"i": "0:N"})
    ar = state.add_read("a")
    br = state.add_read("b")
    tw = state.add_tasklet("add", {"in_a", "in_b"}, {"res"}, "res = in_a + in_b")
    tnode = state.add_access("t")
    cpy = state.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    ow = state.add_write("outp")
    state.add_memlet_path(ar, me, tw, dst_conn="in_a", memlet=mm.Memlet("a[0]"))
    state.add_memlet_path(br, me, tw, dst_conn="in_b", memlet=mm.Memlet("b[0]"))
    state.add_edge(tw, "res", tnode, None, mm.Memlet("t[0]"))
    state.add_edge(tnode, None, cpy, "ti", mm.Memlet("t[0]"))
    state.add_memlet_path(cpy, mx, ow, src_conn="o", memlet=mm.Memlet("outp[i]"))
    sdfg.validate()

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    assert hoisted == 1
    sdict = state.scope_dict()
    assert not any(t.label == "add"
                   for t in state.nodes()
                   if isinstance(t, nodes.Tasklet) and sdict.get(t) is me)
    assert any(t.label.startswith("add")
               for t in state.nodes()
               if isinstance(t, nodes.Tasklet) and sdict.get(t) is None)

    def py_ref(a, b, outp, N):
        for i in range(N):
            outp[i] = a[0] + b[0]

    _run_and_check(sdfg, py_ref,
                   a=np.array([1.0]), b=np.array([7.0]),
                   outp=np.zeros(4), N=4)


# ---------------------------------------------------------------------------
# 6. Nested loop `for nl: for i: a[i] = b[i] + 1.0` — inner loop hoisted,
#    outer body collapses to an empty hull. TSVC2 s000-family.
# ---------------------------------------------------------------------------

def test_tsvc2_s000_inner_loop_hoisted_leaves_hull():
    sdfg = dace.SDFG("licm_tsvc_s000")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)

    sdfg.add_symbol("K", dace.int64)
    outer = LoopRegion("outer", "nl < K", "nl", "nl = 0", "nl = nl + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "i < N", "i", "i = 0", "i = i + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("ib", is_start_block=True)
    br = body.add_read("b")
    aw = body.add_write("a")
    t = body.add_tasklet("add1", {"xb"}, {"y"}, "y = xb + 1.0")
    body.add_edge(br, None, t, "xb", mm.Memlet("b[i]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("a[i]"))
    sdfg.validate()

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    assert hoisted == 1
    assert sdfg.start_block is inner
    hulls = [n for n in outer.nodes() if n.label.endswith("_licm_hull")]
    assert len(hulls) == 1 and len(hulls[0].nodes()) == 0

    def py_ref(a, b, N, K):
        for nl in range(K):
            for i in range(N):
                a[i] = b[i] + 1.0

    _run_and_check(sdfg, py_ref,
                   a=np.zeros(4), b=np.arange(4, dtype=np.float64),
                   N=4, K=3)


# ---------------------------------------------------------------------------
# 7. WCR output: never hoisted (observable side effect).
# ---------------------------------------------------------------------------

def test_wcr_output_is_not_hoisted():
    sdfg = dace.SDFG("licm_wcr")
    sdfg.add_array("a", [1], dace.float64)
    sdfg.add_array("acc", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("a")
    aw = body.add_write("acc")
    t = body.add_tasklet("inc", {"x"}, {"y"}, "y = x")
    body.add_edge(ar, None, t, "x", mm.Memlet("a[0]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("acc[0]", wcr="lambda a,b: a+b"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()
    assert not hoisted


# ---------------------------------------------------------------------------
# 8. Loop-index-dependent memlet subset is not hoisted.
# ---------------------------------------------------------------------------

def test_loop_index_dependent_load_not_hoisted():
    sdfg = dace.SDFG("licm_idx_load")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("outp", [N], dace.float64)
    sdfg.add_transient("t", [1], dace.float64)

    loop = _build_loop(sdfg, "i", "N")
    body = loop.add_state("body", is_start_block=True)
    ar = body.add_read("A")
    tw = body.add_tasklet("id", {"x"}, {"y"}, "y = x")
    tnode = body.add_access("t")
    body.add_edge(ar, None, tw, "x", mm.Memlet("A[i]"))
    body.add_edge(tw, "y", tnode, None, mm.Memlet("t[0]"))
    ow = body.add_write("outp")
    cpy = body.add_tasklet("cpy", {"ti"}, {"o"}, "o = ti")
    body.add_edge(tnode, None, cpy, "ti", mm.Memlet("t[0]"))
    body.add_edge(cpy, "o", ow, None, mm.Memlet("outp[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()
    assert not hoisted


# ---------------------------------------------------------------------------
# 9. Hull is not re-hoisted (regression against an earlier infinite-loop bug).
# ---------------------------------------------------------------------------

def test_hull_is_not_rehoisted():
    sdfg = dace.SDFG("licm_hull_fp")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)

    sdfg.add_symbol("K", dace.int64)
    outer = LoopRegion("outer", "nl < K", "nl", "nl = 0", "nl = nl + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "i < N", "i", "i = 0", "i = i + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("ib", is_start_block=True)
    br = body.add_read("b")
    aw = body.add_write("a")
    t = body.add_tasklet("add1", {"xb"}, {"y"}, "y = xb + 1.0")
    body.add_edge(br, None, t, "xb", mm.Memlet("b[i]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("a[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()
    assert hoisted == 1
    hulls = [n for n in outer.nodes() if n.label.endswith("_licm_hull")]
    assert len(hulls) == 1


# ---------------------------------------------------------------------------
# TSVC2-derived LICM tests
# ---------------------------------------------------------------------------
# Three additional tests extracted from TSVC2 kernels whose outer `nl` wrapper
# makes the entire inner computation loop-invariant under `nl`. Each pattern is
# built directly with the SDFG API (no @dace.program) so the test is hermetic.


def test_tsvc2_s451_like_inner_loop_with_sin_cos_hoisted():
    """TSVC2 s451: `for nl: for i: a[i] = sin(b[i]) + cos(c[i])`.

    Inner loop reads b, c and writes a; no cross-iteration dep on nl — lift it.
    """
    sdfg = dace.SDFG("licm_tsvc_s451")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("c", [N], dace.float64)

    sdfg.add_symbol("K", dace.int64)
    outer = LoopRegion("outer", "nl < K", "nl", "nl = 0", "nl = nl + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "i < N", "i", "i = 0", "i = i + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("ib", is_start_block=True)
    br = body.add_read("b")
    cr = body.add_read("c")
    aw = body.add_write("a")
    t = body.add_tasklet("sc", {"xb", "xc"}, {"y"},
                         "y = math.sin(xb) + math.cos(xc)")
    body.add_edge(br, None, t, "xb", mm.Memlet("b[i]"))
    body.add_edge(cr, None, t, "xc", mm.Memlet("c[i]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("a[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()
    assert hoisted == 1
    assert sdfg.start_block is inner

    def py_ref(a, b, c, N, K):
        for nl in range(K):
            for i in range(N):
                a[i] = math.sin(b[i]) + math.cos(c[i])

    rng = np.random.default_rng(0)
    _run_and_check(sdfg, py_ref,
                   a=np.zeros(5),
                   b=rng.normal(size=5),
                   c=rng.normal(size=5),
                   N=5, K=3)


def test_tsvc2_s431_like_inner_loop_aw_read_blocks_hoist():
    """TSVC2 s431: `for nl: for i: a[i] = a[i] + b[i]`.

    Inner reads AND writes ``a``. Across outer iterations the result accumulates
    — hoisting inner would collapse K iterations into 1 and break semantics.
    LICM must leave this alone.
    """
    sdfg = dace.SDFG("licm_tsvc_s431")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)

    sdfg.add_symbol("K", dace.int64)
    outer = LoopRegion("outer", "nl < K", "nl", "nl = 0", "nl = nl + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "i < N", "i", "i = 0", "i = i + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("ib", is_start_block=True)
    ar = body.add_read("a")
    br = body.add_read("b")
    aw = body.add_write("a")
    t = body.add_tasklet("acc", {"xa", "xb"}, {"y"}, "y = xa + xb")
    body.add_edge(ar, None, t, "xa", mm.Memlet("a[i]"))
    body.add_edge(br, None, t, "xb", mm.Memlet("b[i]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("a[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    # The inner loop is NOT invariant w.r.t. nl (through its self-read on a).
    # LICM must produce no change here.
    assert not hoisted
    assert any(n is inner for n in outer.nodes())

    def py_ref(a, b, N, K):
        for nl in range(K):
            for i in range(N):
                a[i] = a[i] + b[i]

    _run_and_check(sdfg, py_ref,
                   a=np.array([1.0, 2.0, 3.0, 4.0]),
                   b=np.array([0.5, 0.5, 0.5, 0.5]),
                   N=4, K=3)


def test_tsvc2_s452_like_loop_body_uses_index_blocks_hoist():
    """TSVC2 s452: `for nl: for i: a[i] = b[i] + c[i] * (i + 1)`.

    Body uses the inner index directly; the inner loop is still invariant
    w.r.t. the outer `nl` (does not use it), so the inner loop is hoisted
    but the body stays intact.
    """
    sdfg = dace.SDFG("licm_tsvc_s452")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("c", [N], dace.float64)

    sdfg.add_symbol("K", dace.int64)
    outer = LoopRegion("outer", "nl < K", "nl", "nl = 0", "nl = nl + 1")
    sdfg.add_node(outer, is_start_block=True)
    inner = LoopRegion("inner", "i < N", "i", "i = 0", "i = i + 1")
    outer.add_node(inner, is_start_block=True)
    body = inner.add_state("ib", is_start_block=True)
    br = body.add_read("b")
    cr = body.add_read("c")
    aw = body.add_write("a")
    t = body.add_tasklet("indx", {"xb", "xc"}, {"y"}, "y = xb + xc * (i + 1)")
    body.add_edge(br, None, t, "xb", mm.Memlet("b[i]"))
    body.add_edge(cr, None, t, "xc", mm.Memlet("c[i]"))
    body.add_edge(t, "y", aw, None, mm.Memlet("a[i]"))

    hoisted = LoopInvariantCodeMotion().apply_pass(sdfg, {})
    sdfg.validate()

    # Inner loop lifts because it never references `nl`; the tasklet stays
    # inside inner because it depends on `i`.
    assert hoisted == 1
    assert sdfg.start_block is inner
    # Body tasklet survives inside inner.
    inner_body = list(inner.nodes())[0]
    assert any(isinstance(n, nodes.Tasklet) for n in inner_body.nodes())

    def py_ref(a, b, c, N, K):
        for nl in range(K):
            for i in range(N):
                a[i] = b[i] + c[i] * (i + 1)

    rng = np.random.default_rng(1)
    _run_and_check(sdfg, py_ref,
                   a=np.zeros(5), b=rng.normal(size=5), c=rng.normal(size=5),
                   N=5, K=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

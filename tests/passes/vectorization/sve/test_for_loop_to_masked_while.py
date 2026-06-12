# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the S-SVE5a building blocks.

Two new pieces compose into the SVE-style finalize:

- ``GenerateIterationMask(mode='global', global_ub=...)`` — keys the
  ``_iter_mask`` fill to the *running loop variable* against the
  *global* trip (``mask[l] = (iter_var + l < global_ub)``) instead of
  the map's static ``lb``/``ub``.
- ``ForLoopToMaskedWhile`` — the last pass: clamps a ``_iter_mask``-gated
  per-core ``LoopRegion`` condition to ``Min(global_ub, block_end)`` and
  normalizes the update to stride by ``W``. Min lands only in the
  ``LoopRegion`` condition CodeBlock, never a map range.

The structural tests pin the rewrite shape / guards / idempotency. The
e2e test hand-assembles a correct global-mask-gated per-core loop (no
vectorizer needed) and asserts the post-``ForLoopToMaskedWhile`` SDFG is
numerically identical to a plain NumPy reference — the ragged last core
(``core_i=64``, ``N=70``) is the Min-swap + mask correctness gate.
"""
import ast

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.generate_iteration_mask import GenerateIterationMask
from dace.transformation.passes.vectorization.nest_innermost_map_body import NestInnermostMapBodyIntoNSDFG
from dace.transformation.passes.vectorization.for_loop_to_masked_while import ForLoopToMaskedWhile

N = dace.symbol("N")

@dace.program
def axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """``c = a + 2 * b`` -- inlined fixture (previously imported from the
    deleted ``test_tile_map_by_num_cores.py``)."""
    for i in dace.map[0:N]:
        c[i] = a[i] + 2.0 * b[i]


def _mask_fill_bodies(sdfg):
    return [
        t.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)
        for st in n.sdfg.all_states() for t in st.nodes()
        if isinstance(t, dace.nodes.Tasklet) and t.label == "_iter_mask_fill"
    ]


def test_global_mask_formula_keyed_to_iter_var_and_global_ub():
    """``mode='global'`` fills ``mask[l] = (i + l < N)`` — keyed to the
    loop variable and the global exclusive bound, not the map's lb/ub."""
    sd = axpy.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG(vector_width=8).apply_pass(sd, {})
    applied = GenerateIterationMask(vector_width=8, mode="global", global_ub="N").apply_pass(sd, {})
    assert applied == 1
    bodies = _mask_fill_bodies(sd)
    assert len(bodies) == 1
    lines = bodies[0].splitlines()
    assert len(lines) == 8
    # Lane 0 and lane 7, keyed to the map param against N with '<'.
    assert lines[0].replace(" ", "") == "_o[0]=((i)+0<(N));"
    assert lines[7].replace(" ", "") == "_o[7]=((i)+7<(N));"


def test_global_mode_requires_global_ub_or_core_map():
    """No explicit global_ub AND no enclosing ``core`` map (un-tiled
    kernel) → loud ValueError, not a silent wrong mask."""
    sd = axpy.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG(vector_width=8).apply_pass(sd, {})
    try:
        GenerateIterationMask(vector_width=8, mode="global").apply_pass(sd, {})
        assert False, "expected ValueError for mode='global' without global_ub/core map"
    except ValueError as e:
        assert "global_ub" in str(e)


def test_global_mode_auto_derives_from_core_map():
    """Tiled by a ``core`` map, ``mode='global'`` with no explicit
    global_ub derives the bound from the core map (range ``0:N:B`` →
    exclusive end ``N``) and fills ``mask[l] = (i + l < N)``."""
    from dace.transformation.dataflow.tiling import MapTiling
    sd = axpy.to_sdfg(simplify=True)
    me = [n for n, _ in sd.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)][0]
    MapTiling.apply_to(sd,
                       options={
                           "tile_sizes": (16, ),
                           "prefix": "core",
                           "divides_evenly": True,
                           "tile_trivial": True
                       },
                       verify=True,
                       save=False,
                       map_entry=me)
    NestInnermostMapBodyIntoNSDFG(vector_width=8, nest_provably_divisible=True).apply_pass(sd, {})
    applied = GenerateIterationMask(vector_width=8, mode="global").apply_pass(sd, {})
    assert applied == 1
    bodies = _mask_fill_bodies(sd)
    assert len(bodies) == 1
    line0 = bodies[0].splitlines()[0].replace(" ", "")
    # Auto-derived global bound is N (core map keeps original end N-1, +1).
    assert line0 == "_o[0]=((i)+0<(N));", line0


def test_default_mask_form_unchanged():
    """``step_w_only`` still fills the legacy ``(lb + l <= ub)`` form —
    the global branch must not regress the default path."""
    sd = axpy.to_sdfg(simplify=True)
    NestInnermostMapBodyIntoNSDFG(vector_width=8).apply_pass(sd, {})
    # A W-strided map so step_w_only fires.
    for n, _ in sd.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            lb, ub, _ = n.map.range[0]
            n.map.range[0] = (lb, ub, 8)
    GenerateIterationMask(vector_width=8, mode="step_w_only").apply_pass(sd, {})
    bodies = _mask_fill_bodies(sd)
    assert bodies and "<=" in bodies[0] and "Min" not in bodies[0]


def _synthetic_core_loop_sdfg():
    """A ``core``-tiled SDFG whose per-core body is a ``_iter_mask``-gated
    ``LoopRegion`` of the canonical ``MapToForLoop`` shape (step 1).

    Mirrors the post-MapToForLoop structure the SVE chain produces so
    ``ForLoopToMaskedWhile`` can be exercised in isolation.
    """
    sdfg = dace.SDFG("syn_core")
    st = sdfg.add_state(is_start_block=True)
    sdfg.add_array("o", [64], dace.float64)
    me, mx = st.add_map("core_map", {"core_i": "0:64:16"})

    body = dace.SDFG("body")
    body.add_array("o", [64], dace.float64)
    body.add_array("_iter_mask", [8], dace.bool_, transient=True)
    lr = LoopRegion("loop_inner", "i < (core_i + 16)", "i", "i = core_i", "i = i + 1")
    body.add_node(lr, is_start_block=True)
    bstate = lr.add_state("b", is_start_block=True)
    bstate.add_access("_iter_mask")  # marks this as the SVE masked loop

    nn = st.add_nested_sdfg(body, {"o_in"}, {"o_out"}, {"core_i": "core_i"})
    st.add_memlet_path(me, nn, dst_conn="o_in", memlet=dace.Memlet("o[0:64]"))
    st.add_memlet_path(nn, mx, src_conn="o_out", memlet=dace.Memlet("o[0:64]"))
    return sdfg, lr


def test_forloop_to_masked_while_rewrite_shape():
    """Condition gets ``Min(global_ub, block_end)``; update normalized to
    ``i = i + W``; only the loop condition CodeBlock is touched."""
    sdfg, lr = _synthetic_core_loop_sdfg()
    rc = ForLoopToMaskedWhile(vector_width=8, global_ub="64").apply_pass(sdfg, {})
    assert rc == 1
    cond = lr.loop_condition.code[0].value
    assert isinstance(cond, ast.Compare) and isinstance(cond.ops[0], ast.Lt)
    rhs = cond.comparators[0]
    assert isinstance(rhs, ast.Call) and rhs.func.id == "Min"
    assert {ast.unparse(a).replace(" ", "") for a in rhs.args} == {"64", "core_i+16"}
    # CodeBlock re-stringifies "i = i + 8" as "i = (i + 8)"; the contract
    # is the stride, not the parenthesisation.
    upd = lr.update_statement.as_string.replace(" ", "").replace("(", "").replace(")", "")
    assert upd == "i=i+8"


def test_forloop_to_masked_while_idempotent():
    sdfg, lr = _synthetic_core_loop_sdfg()
    assert ForLoopToMaskedWhile(vector_width=8, global_ub="64").apply_pass(sdfg, {}) == 1
    first = lr.loop_condition.as_string
    assert ForLoopToMaskedWhile(vector_width=8, global_ub="64").apply_pass(sdfg, {}) is None
    assert lr.loop_condition.as_string == first


def test_forloop_to_masked_while_auto_derives_global_ub_from_core_map():
    """With ``global_ub`` unset, the bound is auto-derived from the
    enclosing ``core`` map (range ``0:64:16`` → exclusive end 64)."""
    sdfg, lr = _synthetic_core_loop_sdfg()
    rc = ForLoopToMaskedWhile(vector_width=8).apply_pass(sdfg, {})
    assert rc == 1
    rhs = lr.loop_condition.code[0].value.comparators[0]
    assert isinstance(rhs, ast.Call) and rhs.func.id == "Min"
    assert {ast.unparse(a).replace(" ", "") for a in rhs.args} == {"64", "core_i+16"}


def test_forloop_to_masked_while_skips_unguarded_loop():
    """A LoopRegion with no ``_iter_mask`` in its body (or no enclosing
    ``core`` map) must NOT be clamped — clamping it would silently drop
    real iterations."""
    sdfg = dace.SDFG("noguard")
    st = sdfg.add_state(is_start_block=True)
    sdfg.add_array("o", [32], dace.float64)
    me, mx = st.add_map("core_map", {"core_i": "0:32:16"})
    body = dace.SDFG("body")
    body.add_array("o", [32], dace.float64)  # no _iter_mask
    lr = LoopRegion("loop_inner", "i < (core_i + 16)", "i", "i = core_i", "i = i + 1")
    body.add_node(lr, is_start_block=True)
    lr.add_state("b", is_start_block=True)
    nn = st.add_nested_sdfg(body, {"o_in"}, {"o_out"}, {"core_i": "core_i"})
    st.add_memlet_path(me, nn, dst_conn="o_in", memlet=dace.Memlet("o[0:32]"))
    st.add_memlet_path(nn, mx, src_conn="o_out", memlet=dace.Memlet("o[0:32]"))
    assert ForLoopToMaskedWhile(vector_width=8, global_ub="64").apply_pass(sdfg, {}) is None
    assert "Min" not in lr.loop_condition.as_string


def _build_masked_axpy_sve_sdfg(NV, NC, W):
    """Hand-assemble a correct SVE-style ``c = a + 2b`` over ``NV``:
    outer ``core`` map of ``NC`` clean blocks (B = roundup(ceil(NV/NC),W)),
    a per-core step-1 ``LoopRegion`` whose body fills the global-keyed
    ``_iter_mask`` and applies it per lane. ``ForLoopToMaskedWhile`` then
    W-strides the loop and Min-clamps it; correctness hinges on the mask
    gating the ragged last block. No vectorizer involved.
    """
    B = -(-(-(-NV // NC)) // W) * W  # roundup(ceil(NV/NC), W)
    sdfg = dace.SDFG("masked_axpy_sve")
    st = sdfg.add_state(is_start_block=True)
    for nm in ("a", "b", "c"):
        sdfg.add_array(nm, [NV], dace.float64)
    me, mx = st.add_map("core_map", {"core_i": f"0:{NV}:{B}"})

    body = dace.SDFG("body")
    for nm in ("a", "b", "c"):
        body.add_array(nm, [NV], dace.float64)
    body.add_array("_iter_mask", [W], dace.bool_, transient=True)
    body.add_symbol("core_i", dace.int64)

    lr = LoopRegion("loop_inner", f"i < (core_i + {B})", "i", "i = core_i", "i = i + 1")
    body.add_node(lr, is_start_block=True)
    fill = lr.add_state("fill", is_start_block=True)
    bs = lr.add_state("apply", is_start_block=False)
    lr.add_edge(fill, bs, dace.InterstateEdge())

    mfill = fill.add_access("_iter_mask")
    tf = fill.add_tasklet("mf",
                          set(), {"_o"},
                          "\n".join(f"_o[{l}] = ((i) + {l} < ({NV}));" for l in range(W)),
                          language=dace.dtypes.Language.CPP)
    fill.add_edge(tf, "_o", mfill, None, dace.Memlet(f"_iter_mask[0:{W}]"))

    an_a, an_b, an_c = bs.add_access("a"), bs.add_access("b"), bs.add_access("c")
    an_m = bs.add_access("_iter_mask")
    code = "\n".join(f"if (_m[{l}]) _c[i + {l}] = _a[i + {l}] + 2.0 * _b[i + {l}];" for l in range(W))
    tb = bs.add_tasklet("apply", {"_a", "_b", "_m"}, {"_c"}, code, language=dace.dtypes.Language.CPP)
    bs.add_edge(an_a, None, tb, "_a", dace.Memlet(f"a[0:{NV}]"))
    bs.add_edge(an_b, None, tb, "_b", dace.Memlet(f"b[0:{NV}]"))
    bs.add_edge(an_m, None, tb, "_m", dace.Memlet(f"_iter_mask[0:{W}]"))
    bs.add_edge(tb, "_c", an_c, None, dace.Memlet(f"c[0:{NV}]"))

    nn = st.add_nested_sdfg(body, {"a", "b"}, {"c"}, {"core_i": "core_i"})
    oa, ob, oc = st.add_access("a"), st.add_access("b"), st.add_access("c")
    st.add_memlet_path(oa, me, nn, dst_conn="a", memlet=dace.Memlet(f"a[0:{NV}]"))
    st.add_memlet_path(ob, me, nn, dst_conn="b", memlet=dace.Memlet(f"b[0:{NV}]"))
    st.add_memlet_path(nn, mx, oc, src_conn="c", memlet=dace.Memlet(f"c[0:{NV}]"))
    sdfg.validate()
    return sdfg


def test_forloop_to_masked_while_e2e_ragged_last_block():
    """End-to-end numeric: NV=70, NC=8, W=8 → B=16, last core (core_i=64)
    is ragged. After ForLoopToMaskedWhile the SDFG must equal a plain
    NumPy ``a + 2b`` with zero OOB — proving the Min-swap + global mask."""
    NV, NC, W = 70, 8, 8
    sdfg = _build_masked_axpy_sve_sdfg(NV, NC, W)
    rc = ForLoopToMaskedWhile(vector_width=W, global_ub=str(NV)).apply_pass(sdfg, {})
    assert rc == 1
    sdfg.validate()

    a = np.random.rand(NV)
    b = np.random.rand(NV)
    c = np.zeros(NV)
    sdfg.compile()(a=a.copy(), b=b.copy(), c=c)
    assert np.allclose(c, a + 2.0 * b, rtol=0, atol=0), f"max|d|={np.max(np.abs(c - (a + 2*b)))}"

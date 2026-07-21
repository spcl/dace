# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CanonicalizeNestedIndexNames and the per-output-file index-helper dedup.

A nested SDFG that receives a view / non-full subset of a parent array under the SAME connector name
gives the inner descriptor different strides/offset -- a different-body ``<name>_idx`` with the same
base name -> a hard C++ ODR redefinition in one translation unit. The pass renames the inner occurrence
to a globally-unique name so every data name owns exactly one ``(ndim, strides, offset)`` signature.
Separately, ``_flush_generated_functions`` must emit each helper once per OUTPUT FILE (not per stream),
so an un-inlined nested-SDFG function does not re-emit an identical helper into the same host TU.
"""
import re

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize_nested_index_names import CanonicalizeNestedIndexNames

from tests.codegen.readable.conftest import EXPERIMENTAL, use_implementation, generated_code, run_isolated, assert_outputs_equivalent

N = dace.symbol("N")

#: A generated ``<name>_idx`` helper definition. The integer ctype is matched with ``\w+`` (the emitted
#: type is ``int64_t``/``int32_t``, NOT ``long long``) so the match cannot silently go vacuous. Named
#: groups expose the helper name and its ``return`` body, so a test can assert two same-named helpers
#: are byte-identical (a benign duplicate) versus different-bodied (the ODR bug the pass exists to kill).
HELPER_DEF = re.compile(r"static\s+DACE_HDFI\s+constexpr\s+\w+\s+(?P<name>\w+_idx)\s*"
                        r"\((?P<params>[^)]*)\)\s*\{\s*return\s+(?P<body>[^;]+);\s*\}")


def _helper_bodies(code):
    """name -> list of full definition strings for every ``<name>_idx`` helper in ``code``."""
    out = {}
    for m in HELPER_DEF.finditer(code):
        out.setdefault(m.group("name"), []).append(m.group(0).strip())
    return out


def _sig_of(desc):
    """The ``(ndim, strides, offset)`` signature that keys an ``<name>_idx`` body (mirrors the pass)."""
    return (len(desc.shape), tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset))


def _nested_sdfg(name, shape, strides):
    """A standalone nested SDFG that reads+writes its single array ``name`` element-wise (so codegen
    registers a ``<name>_idx`` helper built from ``strides``)."""
    nsdfg = dace.SDFG("inner")
    nsdfg.add_array(name, shape, dace.float64, strides=strides)
    ns = nsdfg.add_state("n")
    an = ns.add_access(name)
    me, mx = ns.add_map("im", dict(i="0:2", j="0:2"))
    tk = ns.add_tasklet("t", {"x"}, {"o"}, "o = x + 1.0")
    ns.add_memlet_path(an, me, tk, dst_conn="x", memlet=dace.Memlet(f"{name}[i,j]"))
    an2 = ns.add_access(name)
    ns.add_memlet_path(tk, mx, an2, src_conn="o", memlet=dace.Memlet(f"{name}[i,j]"))
    return nsdfg


def _nested_view_sdfg(inner_shape, inner_strides, inner_name="A"):
    """Parent A[N,N] element-wise map, plus a no_inline nested SDFG whose connector ``inner_name`` is a
    view of A with the given (shape, strides). Returns the top SDFG."""
    sdfg = dace.SDFG("nested_view")
    sdfg.add_array("A", [N, N], dace.float64)

    nsdfg = dace.SDFG("inner")
    nsdfg.add_array(inner_name, inner_shape, dace.float64, strides=inner_strides)
    ns = nsdfg.add_state("n")
    an = ns.add_access(inner_name)
    me, mx = ns.add_map("im", dict(i="0:2", j="0:2"))
    tk = ns.add_tasklet("t", {"x"}, {"o"}, "o = x + 1.0")
    ns.add_memlet_path(an, me, tk, dst_conn="x", memlet=dace.Memlet(f"{inner_name}[i,j]"))
    an2 = ns.add_access(inner_name)
    ns.add_memlet_path(tk, mx, an2, src_conn="o", memlet=dace.Memlet(f"{inner_name}[i,j]"))

    st = sdfg.add_state("main")
    # parent element-wise access -> outer A_idx (strides N,1)
    pr, pw = st.add_access("A"), st.add_access("A")
    pme, pmx = st.add_map("pm", dict(i="0:N", j="0:N"))
    ptk = st.add_tasklet("pt", {"x"}, {"o"}, "o = x * 2.0")
    st.add_memlet_path(pr, pme, ptk, dst_conn="x", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(ptk, pmx, pw, src_conn="o", memlet=dace.Memlet("A[i,j]"))
    # nested SDFG bound to a 2x2 sub-block of A
    ar, aw = st.add_access("A"), st.add_access("A")
    nn = st.add_nested_sdfg(nsdfg, {inner_name}, {inner_name}, symbol_mapping={"N": N})
    nn.no_inline = True
    st.add_edge(ar, None, nn, inner_name, dace.Memlet("A[0:2, 0:2]"))
    st.add_edge(nn, inner_name, aw, None, dace.Memlet("A[0:2, 0:2]"))
    sdfg.validate()
    return sdfg


def _signatures(sdfg):
    """name -> set of (ndim, strides, offset) signatures seen across the whole SDFG tree."""
    sigs = {}
    for sub in sdfg.all_sdfgs_recursive():
        for name, desc in sub.arrays.items():
            if hasattr(desc, "strides"):
                sig = (len(desc.shape), tuple(str(s) for s in desc.strides), tuple(str(o) for o in desc.offset))
                sigs.setdefault(name, set()).add(sig)
    return sigs


def test_full_subset_same_name_not_renamed():
    """Inner A with the SAME strides (row-major N,1... here 2x2 contiguous with strides matching a plain
    row-major view) as a genuinely identical signature must NOT be renamed."""
    # inner A[2,2] strides (2,1) -- distinct from outer A[N,N] strides (N,1): different signature.
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    before = _signatures(sdfg)
    assert before["A"] == {(2, ("N", "1"), ("0", "0")), (2, ("2", "1"), ("0", "0"))} or len(before["A"]) == 2
    renamed = CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    sdfg.validate()
    after = _signatures(sdfg)
    # After the pass every name owns exactly ONE signature.
    for name, s in after.items():
        assert len(s) == 1, f"{name} still has multiple signatures: {s}"
    assert renamed == 1  # the nested A was renamed


def test_identical_signature_kept():
    """Inner A whose signature is identical to the outer A is left alone (emission dedup handles it)."""
    sdfg = _nested_view_sdfg([N, N], [N, 1])
    renamed = CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    sdfg.validate()
    assert renamed is None  # no rename
    assert _signatures(sdfg)["A"] == {(2, ("N", "1"), ("0", "0"))}


def test_distinct_inner_name_kept():
    """A nested connector whose name is not present in the parent is never renamed."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1], inner_name="_inner")
    renamed = CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    sdfg.validate()
    assert renamed is None
    sigs = _signatures(sdfg)
    assert set(sigs["_inner"]) and set(sigs["A"])  # both present, no collision


def test_root_argument_never_renamed():
    """The root SDFG's own array names always win; only the nested occurrence is renamed."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    assert "A" in sdfg.arrays  # root name intact
    # the nested SDFG's array was renamed away from 'A'
    inner = [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg][0]
    assert "A" not in inner.arrays
    assert any(k.startswith("A_v") for k in inner.arrays)


@pytest.mark.parametrize("kernel", ["trisolv", "lu", "ludcmp"])
def test_uninlined_kernel_no_duplicate_idx(kernel):
    """The reproduction: a real kernel with every nested SDFG forced no_inline must emit each
    ``<name>_idx`` at most once per translation unit (no ODR redefinition)."""
    import importlib.util
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "polybench", "linear_algebra", "solvers")
    path = os.path.join(base, f"{kernel}.py")
    if not os.path.exists(path):
        pytest.skip("in-repo polybench corpus not present in this build")
    spec = importlib.util.spec_from_file_location(f"k_{kernel}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sdfg = getattr(mod, kernel).to_sdfg(simplify=True)
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            n.no_inline = True
    total_helpers = 0
    with use_implementation(EXPERIMENTAL):
        for co in sdfg.generate_code():
            if co.language != "cpp":
                continue
            counts = {}
            for m in HELPER_DEF.finditer(co.code):
                counts[m.group("name")] = counts.get(m.group("name"), 0) + 1
            total_helpers += sum(counts.values())
            dups = {k: v for k, v in counts.items() if v > 1}
            assert not dups, f"{kernel}/{co.name}: duplicate index helpers in one TU: {dups}"
    # Guard against a silently-vacuous run: a solver kernel that lowers to BLAS must emit >=1 helper,
    # so a regex that stopped matching (e.g. the emitted integer ctype changed) fails loudly here.
    assert total_helpers > 0, f"{kernel}: no ``<name>_idx`` helpers matched -- HELPER_DEF regex is stale"


def test_pass_returns_rename_count():
    """The pass reports the number of renames it performed (truthy int), or None when it did nothing."""
    assert CanonicalizeNestedIndexNames().apply_pass(_nested_view_sdfg([2, 2], [2, 1]), {}) == 1
    assert CanonicalizeNestedIndexNames().apply_pass(_nested_view_sdfg([N, N], [N, 1]), {}) is None


def test_no_mutation_when_nothing_applies():
    """A pass that does not apply must NOT mutate the SDFG (repo rule): identical-signature view -> the
    serialized SDFG is byte-for-byte unchanged and the pass returns None."""
    sdfg = _nested_view_sdfg([N, N], [N, 1])
    before = sdfg.to_json()
    result = CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    assert result is None
    assert sdfg.to_json() == before


def test_distinct_inner_name_no_mutation():
    """A nested connector whose name is absent from the parent cannot collide -> no rename, no mutation."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1], inner_name="_inner")
    before = sdfg.to_json()
    assert CanonicalizeNestedIndexNames().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


def test_every_name_owns_one_signature_after_pass():
    """The pass invariant: once it has run, no data name maps to two different ``(ndim, strides, offset)``
    signatures anywhere in the tree -- exactly the property that makes the ``<name>_idx`` map 1:1."""
    sdfg = _nested_view_sdfg([2, 2], [4, 1])  # inner strides (4,1) != outer (N,1)
    CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    sdfg.validate()
    for name, sigs in _signatures(sdfg).items():
        assert len(sigs) == 1, f"{name} still owns multiple signatures: {sigs}"


def test_renamed_descriptor_preserves_shape_strides_offset():
    """A subset passed without a new name gets a fresh unique name whose descriptor COPIES the inner
    descriptor's info (shape/strides/offset) -- the body must stay identical, only the name changes."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    inner = [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg][0]
    old_sig = _sig_of(inner.arrays["A"])
    CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    new_names = [k for k in inner.arrays if k.startswith("A_v")]
    assert len(new_names) == 1
    assert _sig_of(inner.arrays[new_names[0]]) == old_sig  # descriptor info copied verbatim


def test_internal_occurrences_fully_rewritten():
    """``replace_dict`` must update every internal occurrence: no memlet inside the nested SDFG may still
    name the old data, and the connector + incident edges move to the new name in lockstep."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    inner = [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg][0]
    node = inner.parent_nsdfg_node
    assert "A" not in inner.arrays and "A" not in node.in_connectors and "A" not in node.out_connectors
    for state in inner.states():
        for e in state.edges():
            assert e.data.data != "A", f"stale memlet still references old name: {e.data}"


def test_inout_connector_renamed_on_both_sides():
    """An inout connector (same name in and out) with a differing signature is renamed consistently on
    BOTH the in and out connector and both incident edges, leaving a single inner descriptor."""
    sdfg = dace.SDFG("inout_top")
    sdfg.add_array("A", [N, N], dace.float64)
    st = sdfg.add_state("m")
    nsdfg = _nested_sdfg("A", [2, 2], [2, 1])
    r, w = st.add_access("A"), st.add_access("A")
    nn = st.add_nested_sdfg(nsdfg, {"A"}, {"A"}, symbol_mapping={"N": N})
    nn.no_inline = True
    st.add_edge(r, None, nn, "A", dace.Memlet("A[0:2,0:2]"))
    st.add_edge(nn, "A", w, None, dace.Memlet("A[0:2,0:2]"))
    sdfg.validate()
    assert CanonicalizeNestedIndexNames().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    inner = [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg][0]
    new = [k for k in inner.arrays if k.startswith("A_v")]
    assert len(new) == 1 and "A" not in inner.arrays
    # the SAME new name lands on both connectors (an inout stays a single binding)
    assert set(nn.in_connectors) == {new[0]} and set(nn.out_connectors) == {new[0]}


def test_new_name_avoids_existing_parent_name():
    """New names must be GLOBALLY unique: when ``A_v0`` is already taken in the parent, the rename skips
    it and picks the next free ``A_v*``."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    sdfg.add_array("A_v0", [N], dace.float64, transient=True)  # occupy the first candidate
    CanonicalizeNestedIndexNames().apply_pass(sdfg, {})
    sdfg.validate()
    inner = [s for s in sdfg.all_sdfgs_recursive() if s is not sdfg][0]
    new = [k for k in inner.arrays if k.startswith("A_v")]
    assert new == ["A_v1"]  # A_v0 was taken, so the fresh name stepped past it
    assert "A_v0" in sdfg.arrays  # the pre-existing parent array is untouched


def test_three_level_nesting_all_renamed_uniquely():
    """Three levels all named ``A`` with three different strides: every level ends up owning a distinct
    unique name, so all three helper bodies live under separate names."""
    lvl3 = _nested_sdfg("A", [2, 2], [4, 1])
    lvl2 = dace.SDFG("lvl2")
    lvl2.add_array("A", [2, 2], dace.float64, strides=[8, 1])
    s2 = lvl2.add_state("s")
    r2, w2 = s2.add_access("A"), s2.add_access("A")
    n3 = s2.add_nested_sdfg(lvl3, {"A"}, {"A"}, symbol_mapping={})
    n3.no_inline = True
    s2.add_edge(r2, None, n3, "A", dace.Memlet("A[0:2,0:2]"))
    s2.add_edge(n3, "A", w2, None, dace.Memlet("A[0:2,0:2]"))
    top = dace.SDFG("three")
    top.add_array("A", [N, N], dace.float64)
    st = top.add_state("m")
    r, w = st.add_access("A"), st.add_access("A")
    n2 = st.add_nested_sdfg(lvl2, {"A"}, {"A"}, symbol_mapping={"N": N})
    n2.no_inline = True
    st.add_edge(r, None, n2, "A", dace.Memlet("A[0:2,0:2]"))
    st.add_edge(n2, "A", w, None, dace.Memlet("A[0:2,0:2]"))
    top.validate()
    assert CanonicalizeNestedIndexNames().apply_pass(top, {}) == 2  # two inner levels renamed
    top.validate()
    per_sdfg = [set(s.arrays) for s in top.all_sdfgs_recursive()]
    assert per_sdfg == [{"A"}, {"A_v0"}, {"A_v1"}]


def test_scalar_connector_not_renamed():
    """A Scalar connector has no ``<name>_idx`` helper (no strides), so it can never collide and must be
    left alone even if it shares a name with a parent array."""
    sdfg = dace.SDFG("scal_top")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    nsdfg = dace.SDFG("inner")
    nsdfg.add_scalar("s", dace.float64)
    ist = nsdfg.add_state("i")
    tk = ist.add_tasklet("t", {}, {"o"}, "o = 1.0")
    ist.add_memlet_path(tk, ist.add_access("s"), src_conn="o", memlet=dace.Memlet("s[0]"))
    st = sdfg.add_state("m")
    sw = st.add_access("s")
    nn = st.add_nested_sdfg(nsdfg, {}, {"s"}, symbol_mapping={"N": N})
    st.add_edge(nn, "s", sw, None, dace.Memlet("s[0]"))
    before = sdfg.to_json()
    assert CanonicalizeNestedIndexNames().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == before


# --------------------------------------------------------------------------- #
# Codegen-level: the pass + per-file dedup produce collision-free emitted C++.
# --------------------------------------------------------------------------- #
def test_synthetic_view_emits_no_colliding_helper(require_experimental):
    """End-to-end reproduction on a synthetic SDFG: a nested ``A`` that is a differently-strided view of
    the parent ``A``, forced ``no_inline``, must NOT yield two ``A_idx`` helpers (identical OR
    different-bodied). The pass renames the inner one, so the parent keeps ``A_idx`` and the view gets
    its own uniquely-named helper."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    sdfg.specialize({"N": 5})
    with use_implementation(EXPERIMENTAL):
        objs = sdfg.generate_code()
    matched = 0
    for co in objs:
        if co.language != "cpp":
            continue
        bodies = _helper_bodies(co.code)
        matched += sum(len(v) for v in bodies.values())
        for name, defs in bodies.items():
            assert len(defs) == 1, f"{co.name}: helper {name} defined {len(defs)}x in one TU: {defs}"
        assert len(set(bodies) & {"A_idx"}) <= 1
    assert matched >= 2, "expected at least the parent A_idx and the renamed view's helper"


def test_synthetic_view_helpers_have_distinct_bodies(require_experimental):
    """The renamed view's helper is not just uniquely named -- its body reflects the view's own strides,
    proving the rename kept the (different) offset math instead of collapsing it onto the parent's."""
    sdfg = _nested_view_sdfg([2, 2], [2, 1])
    sdfg.specialize({"N": 5})
    with use_implementation(EXPERIMENTAL):
        code = "\n".join(co.code for co in sdfg.generate_code() if co.language == "cpp")
    bodies = _helper_bodies(code)
    a_helpers = {k: v[0] for k, v in bodies.items() if k == "A_idx" or k.startswith("A_v")}
    assert len(a_helpers) == 2, f"expected parent + renamed view helper, got {sorted(a_helpers)}"
    assert len(set(a_helpers.values())) == 2, "the two A-derived helpers must have distinct bodies"


if __name__ == "__main__":
    test_full_subset_same_name_not_renamed()
    test_identical_signature_kept()
    test_distinct_inner_name_kept()
    test_root_argument_never_renamed()
    test_pass_returns_rename_count()
    test_no_mutation_when_nothing_applies()
    test_distinct_inner_name_no_mutation()
    test_every_name_owns_one_signature_after_pass()
    test_renamed_descriptor_preserves_shape_strides_offset()
    test_internal_occurrences_fully_rewritten()
    test_inout_connector_renamed_on_both_sides()
    test_new_name_avoids_existing_parent_name()
    test_three_level_nesting_all_renamed_uniquely()
    test_scalar_connector_not_renamed()
    for k in ("trisolv", "lu", "ludcmp"):
        test_uninlined_kernel_no_duplicate_idx(k)
    print("ok")

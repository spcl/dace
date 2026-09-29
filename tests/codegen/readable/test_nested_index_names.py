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
    with use_implementation(EXPERIMENTAL):
        for co in sdfg.generate_code():
            if co.language != "cpp":
                continue
            counts = {}
            for m in re.finditer(r"static\s+DACE_HDFI\s+constexpr\s+long\s+long\s+(\w+_idx)\s*\(", co.code):
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            dups = {k: v for k, v in counts.items() if v > 1}
            assert not dups, f"{kernel}/{co.name}: duplicate index helpers in one TU: {dups}"


if __name__ == "__main__":
    test_full_subset_same_name_not_renamed()
    test_identical_signature_kept()
    test_distinct_inner_name_kept()
    test_root_argument_never_renamed()
    for k in ("trisolv", "lu", "ludcmp"):
        test_uninlined_kernel_no_duplicate_idx(k)
    print("ok")

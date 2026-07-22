# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit test for ``_resolve_body_nsdfg_symbol_aliases``.

``MapFusion`` records fused-nest iter-var equivalence as a ``symbol_mapping`` alias
on the body NSDFG (``_loop_it_2 -> _loop_it_0``), not by renaming the inner body.
Tile walker keys per-lane classification off the OUTER iter-var names, so an aliased
inner access mis-classifies as loop-invariant -> scalar broadcast. Resolver inlines
the bare-symbol rename into the inner SDFG -> body references outer iter-var directly.
"""
import dace
from dace import symbolic
from dace.transformation.passes.vectorization.vectorize_multi_dim import _resolve_body_nsdfg_symbol_aliases


def _build_aliased_body_sdfg():
    """A map over ``_loop_it_0`` whose body NSDFG reads/writes via the aliased
    ``_loop_it_2`` (mapped ``_loop_it_2 -> _loop_it_0``)."""
    sdfg = dace.SDFG("alias_outer")
    sdfg.add_array("A", [8], dace.float64)
    sdfg.add_array("B", [8], dace.float64)
    state = sdfg.add_state()

    inner = dace.SDFG("alias_inner")
    inner.add_array("A", [8], dace.float64)
    inner.add_array("B", [8], dace.float64)
    inner.add_symbol("_loop_it_2", dace.int64)
    ist = inner.add_state()
    ra = ist.add_access("A")
    wb = ist.add_access("B")
    t = ist.add_tasklet("cp", {"a"}, {"b"}, "b = a")
    ist.add_edge(ra, None, t, "a", dace.Memlet("A[_loop_it_2]"))
    ist.add_edge(t, "b", wb, None, dace.Memlet("B[_loop_it_2]"))

    me, mx = state.add_map("m", {"_loop_it_0": "0:8"})
    nsdfg = state.add_nested_sdfg(inner, {"A"}, {"B"},
                                  symbol_mapping={"_loop_it_2": symbolic.pystr_to_symbolic("_loop_it_0")})
    a_out = state.add_access("A")
    b_out = state.add_access("B")
    state.add_memlet_path(a_out, me, nsdfg, dst_conn="A", memlet=dace.Memlet("A[0:8]"))
    state.add_memlet_path(nsdfg, mx, b_out, src_conn="B", memlet=dace.Memlet("B[0:8]"))
    return sdfg, nsdfg, inner


def test_resolve_alias_inlines_rename_into_inner_body():
    sdfg, nsdfg, inner = _build_aliased_body_sdfg()
    assert "_loop_it_2" in inner.symbols
    assert nsdfg.symbol_mapping["_loop_it_2"] == symbolic.pystr_to_symbolic("_loop_it_0")

    _resolve_body_nsdfg_symbol_aliases(sdfg)

    # The aliased inner symbol is gone; the identity target remains reachable.
    assert "_loop_it_2" not in inner.symbols
    assert "_loop_it_2" not in nsdfg.symbol_mapping
    assert "_loop_it_0" in nsdfg.symbol_mapping
    # Every inner memlet now references the outer iter-var directly.
    for st in inner.states():
        for e in st.edges():
            if e.data is not None and e.data.data is not None:
                syms = {str(s) for s in e.data.subset.free_symbols}
                assert "_loop_it_2" not in syms
                assert "_loop_it_0" in syms
    sdfg.validate()


def test_resolve_alias_leaves_offset_and_identity_mappings_untouched():
    """Only a bare non-identity symbol rename is inlined; an offset expression
    (a real value, not a rename) and an identity mapping are left as-is."""
    sdfg = dace.SDFG("noop_outer")
    sdfg.add_array("A", [8], dace.float64)
    state = sdfg.add_state()
    inner = dace.SDFG("noop_inner")
    inner.add_array("A", [8], dace.float64)
    inner.add_symbol("ii", dace.int64)
    ist = inner.add_state()
    ra = ist.add_access("A")
    t = ist.add_tasklet("r", {"a"}, set(), "pass")
    ist.add_edge(ra, None, t, "a", dace.Memlet("A[ii]"))
    nsdfg = state.add_nested_sdfg(inner, {"A"}, set(),
                                  symbol_mapping={
                                      "ii": symbolic.pystr_to_symbolic("_loop_it_0 + 1"),
                                      "_loop_it_0": symbolic.pystr_to_symbolic("_loop_it_0"),
                                  })
    me, mx = state.add_map("m", {"_loop_it_0": "0:8"})
    a_out = state.add_access("A")
    state.add_memlet_path(a_out, me, nsdfg, dst_conn="A", memlet=dace.Memlet("A[0:8]"))
    state.fill_scope_connectors()

    _resolve_body_nsdfg_symbol_aliases(sdfg)

    # An offset mapping is a value, not a rename -> untouched.
    assert nsdfg.symbol_mapping["ii"] == symbolic.pystr_to_symbolic("_loop_it_0 + 1")
    assert "ii" in inner.symbols


if __name__ == "__main__":
    test_resolve_alias_inlines_rename_into_inner_body()
    test_resolve_alias_leaves_offset_and_identity_mappings_untouched()
    print("ok")

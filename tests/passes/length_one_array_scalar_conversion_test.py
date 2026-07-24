# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the length-1-array <-> scalar conversion passes.

``ConvertLengthOneArraysToScalars`` rewrites every TRANSIENT length-1 ``Array`` (shape ``(1,)``) to a
true ``Scalar`` in place and strips the redundant ``[0]`` accessors; with
``stage_nontransients_arrays_into_scalars`` it additionally STAGES each non-transient length-1 array
into a fresh transient scalar (copy-in in a new start state, copy-out in a new sink state), leaving the
signature array untouched. ``ConvertScalarsToLengthOneArrays`` is the inverse. These are pure-SDFG (no
Fortran) tests of the Pass classes, covering the staging, ``filter`` gating, ``preserve_abi`` and the
``opaque``/View exemptions.

``preserve_abi`` (default) is the guarantee that a top-level non-transient descriptor is never
rewritten, so staging is the only route by which the body reaches the other form; clearing it opts
into an in-place rewrite that changes the SDFG's call signature and must therefore also cross every
NestedSDFG connector bound to that descriptor.
"""
import ctypes

import numpy as np

import dace
import dace.data as dd
import pytest

from dace.transformation.passes import (
    ConvertLengthOneArraysToScalars,
    ConvertScalarsToLengthOneArrays,
)

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass


def _io_sdfg() -> dace.SDFG:
    """SDFG with a non-transient length-1 input ``alpha`` and output ``beta``; ``beta = alpha * 2``."""
    sdfg = dace.SDFG("io")
    sdfg.add_array("alpha", [1], dace.float64, transient=False)
    sdfg.add_array("beta", [1], dace.float64, transient=False)
    st = sdfg.add_state("main")
    ra, wb = st.add_read("alpha"), st.add_write("beta")
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    st.add_edge(ra, None, t, "a", dace.Memlet("alpha[0]"))
    st.add_edge(t, "b", wb, None, dace.Memlet("beta[0]"))
    return sdfg


def _run(sdfg: dace.SDFG, alpha_val: float) -> float:
    a = np.array([alpha_val], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    sdfg(alpha=a, beta=b)
    return float(b[0])


# --- default: transient in place, non-transient untouched -------------------


def test_default_scalarizes_transient_length_one_array():
    sdfg = dace.SDFG("tr")
    sdfg.add_state("s")
    sdfg.add_array("a", [1], dace.float64, transient=True)
    sdfg.add_array("b", [10], dace.float64, transient=False)
    rewritten = ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    assert isinstance(sdfg.arrays["b"], dd.Array)  # multi-element array left alone


def test_default_skips_signature_arrays():
    """A non-transient (signature) length-1 array is NOT touched by default -- only staging does."""
    sdfg = dace.SDFG("sig")
    sdfg.add_state("s")
    sdfg.add_array("a", [1], dace.float64, transient=False)
    assert ConvertLengthOneArraysToScalars().apply_pass(sdfg, {}) is None
    assert isinstance(sdfg.arrays["a"], dd.Array)


def test_interstate_accessor_is_stripped():
    sdfg = dace.SDFG("istrip")
    s0, s1 = sdfg.add_state("s0"), sdfg.add_state("s1")
    sdfg.add_array("a", [1], dace.int64, transient=True)
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "a[0] + 1"}))
    ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    assert list(sdfg.all_interstate_edges())[0].data.assignments["k"] == "a + 1"


def test_scalarize_keeps_overlapping_name_subscript():
    """A scalarized name that is a suffix of another array must not eat that array's literal ``[0]``
    (scalarized ``ar`` vs multi-element ``bar``)."""
    sdfg = dace.SDFG("overlap")
    sdfg.add_array("ar", (1, ), dace.float64, transient=True)
    sdfg.add_array("bar", (4, ), dace.float64)
    s0, s1 = sdfg.add_state("s0"), sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "ar[0] + bar[0]"}))
    ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["ar"], dd.Scalar)
    assert isinstance(sdfg.arrays["bar"], dd.Array)
    assert list(sdfg.all_interstate_edges())[0].data.assignments["k"] == "ar + bar[0]"


def test_collapsed_memlet_preserves_dynamic():
    sdfg = dace.SDFG("dynmem")
    sdfg.add_array("a", (1, ), dace.float64, transient=True)
    sdfg.add_array("b", (1, ), dace.float64, transient=True)
    state = sdfg.add_state("s")
    an_a, an_b = state.add_access("a"), state.add_access("b")
    state.add_nedge(an_a, an_b, dace.Memlet(data="a", subset="0", dynamic=True))
    ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    assert state.edges()[0].data.dynamic is True


def test_recursive_descends_into_nested_sdfg():
    sdfg = dace.SDFG("outer")
    st = sdfg.add_state("s")
    nested = dace.SDFG("inner")
    nested.add_state("ns")
    nested.add_array("z", [1], dace.float64, transient=True)
    st.add_nested_sdfg(nested, {}, {})
    ConvertLengthOneArraysToScalars(recursive=True).apply_pass(sdfg, {})
    assert isinstance(nested.arrays["z"], dd.Scalar)


# --- staging non-transients -------------------------------------------------


def test_stage_keeps_signature_arrays_and_adds_scalars():
    sdfg = _io_sdfg()
    rewritten = ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    assert rewritten == {"alpha", "beta"}
    # Signature descriptors stay non-transient Arrays.
    assert isinstance(sdfg.arrays["alpha"], dd.Array) and not sdfg.arrays["alpha"].transient
    assert isinstance(sdfg.arrays["beta"], dd.Array)
    # A fresh transient scalar was staged for each.
    assert isinstance(sdfg.arrays["scal_alpha"], dd.Scalar) and sdfg.arrays["scal_alpha"].transient
    assert isinstance(sdfg.arrays["scal_beta"], dd.Scalar)
    labels = {s.label for s in sdfg.all_states()}
    assert "stage_copyin" in labels and "stage_copyout" in labels
    sdfg.validate()


def test_stage_is_numerically_correct():
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    assert _run(sdfg, 3.0) == pytest.approx(6.0)


def test_stage_read_only_input_gets_copyin_not_copyout():
    """A read-only non-transient input is staged with a copy-IN only (no spurious write-back)."""
    sdfg = dace.SDFG("ro")
    sdfg.add_array("inp", [1], dace.float64, transient=False)
    sdfg.add_array("out", [4], dace.float64, transient=False)
    st = sdfg.add_state("s")
    ri = st.add_read("inp")
    wo = st.add_write("out")
    me, mx = st.add_map("m", dict(i="0:4"))
    t = st.add_tasklet("t", {"a"}, {"o"}, "o = a")
    st.add_memlet_path(ri, me, t, dst_conn="a", memlet=dace.Memlet("inp[0]"))
    st.add_memlet_path(t, mx, wo, src_conn="o", memlet=dace.Memlet("out[i]"))
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    labels = {s.label for s in sdfg.all_states()}
    assert "stage_copyin" in labels  # inp is read
    assert "stage_copyout" not in labels  # inp is never written -> no copy-out
    sdfg.validate()


def test_forward_then_inverse_stays_correct():
    """X then X^-1 (both staging) leaves a valid, numerically-correct SDFG."""
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    ConvertScalarsToLengthOneArrays(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert _run(sdfg, 3.0) == pytest.approx(6.0)


def test_repeated_forward_finds_new_name_and_stays_correct():
    """Applying the forward pass twice must not collide on the scalar name it created before
    (``find_new_name``), and stays numerically correct."""
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    before = set(sdfg.arrays)
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    fresh = set(sdfg.arrays) - before
    assert fresh, "second application created no fresh staging scalar"
    assert all(f.startswith("scal_") for f in fresh)  # uniquified, not a collision
    sdfg.validate()
    assert _run(sdfg, 3.0) == pytest.approx(6.0)


# --- inverse round-trip + opaque exemptions ---------------------------------


def test_inverse_roundtrip_transient():
    sdfg = dace.SDFG("rt")
    sdfg.add_state("s")
    sdfg.add_array("a", [1], dace.float64, transient=True)
    ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    rewritten = ConvertScalarsToLengthOneArrays().apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Array)
    assert tuple(sdfg.arrays["a"].shape) == (1, )


def test_opaque_length_one_array_is_not_scalarized():
    sdfg = dace.SDFG("opaque_len1")
    sdfg.add_state("s")
    sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    rewritten = ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["req"], dd.Array)
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_opaque_scalar_is_not_arrayized():
    sdfg = dace.SDFG("opaque_scalar")
    sdfg.add_state("s")
    sdfg.add_scalar("comm", dace.dtypes.opaque("MPI_Comm"), transient=True)
    sdfg.add_scalar("k", dace.int64, transient=True)
    rewritten = ConvertScalarsToLengthOneArrays().apply_pass(sdfg, {})
    assert rewritten == {"k"}
    assert isinstance(sdfg.arrays["comm"], dd.Scalar)
    assert isinstance(sdfg.arrays["k"], dd.Array)


def test_passes_expose_property_options():
    assert set(ConvertLengthOneArraysToScalars.__properties__) == {
        "recursive", "stage_nontransients_arrays_into_scalars", "filter", "single_element", "preserve_abi"
    }
    assert set(ConvertScalarsToLengthOneArrays.__properties__) == {
        "recursive", "stage_nontransients_arrays_into_scalars", "filter", "preserve_abi"
    }
    for cls in (ConvertLengthOneArraysToScalars, ConvertScalarsToLengthOneArrays):
        inst = cls(recursive=False, stage_nontransients_arrays_into_scalars=True)
        assert inst.recursive is False
        assert inst.stage_nontransients_arrays_into_scalars is True
        assert inst.preserve_abi is True, "the ABI-safe route must be the default"


# --- preserve_abi -----------------------------------------------------------


def test_preserve_abi_stages_instead_of_touching_the_signature():
    """The guarantee: with ``preserve_abi`` the signature descriptors are byte-identical afterwards --
    the conversion reaches the body only through the staged transients."""
    sdfg = _io_sdfg()
    before = {nm: (type(d), tuple(d.shape)) for nm, d in sdfg.arrays.items()}
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    for nm, sig in before.items():
        assert (type(sdfg.arrays[nm]), tuple(sdfg.arrays[nm].shape)) == sig, f"{nm} left the signature"
    assert any(isinstance(d, dd.Scalar) and d.transient for d in sdfg.arrays.values())
    assert _run(sdfg, 3.0) == 6.0


def test_without_preserve_abi_the_signature_becomes_scalar():
    """Cleared, the non-transient is rewritten IN PLACE -- no staging transient, no copy-in/out, and
    the caller now binds a by-value scalar instead of a 1-element buffer."""
    sdfg = _io_sdfg()
    rewritten = ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True,
                                                preserve_abi=False).apply_pass(sdfg, {})
    assert rewritten == {"alpha", "beta"}
    assert isinstance(sdfg.arrays["alpha"], dd.Scalar) and not sdfg.arrays["alpha"].transient
    assert isinstance(sdfg.arrays["beta"], dd.Scalar) and not sdfg.arrays["beta"].transient
    assert not any(nm.startswith("scal_") for nm in sdfg.arrays)
    sdfg.validate()


def test_without_preserve_abi_inverse_restores_the_length_one_signature():
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True,
                                    preserve_abi=False).apply_pass(sdfg, {})
    ConvertScalarsToLengthOneArrays(stage_nontransients_arrays_into_scalars=True,
                                    preserve_abi=False).apply_pass(sdfg, {})
    for nm in ("alpha", "beta"):
        assert isinstance(sdfg.arrays[nm], dd.Array) and tuple(sdfg.arrays[nm].shape) == (1, )
        assert not sdfg.arrays[nm].transient
    sdfg.validate()
    assert _run(sdfg, 4.0) == 8.0


def test_in_place_signature_rewrite_reaches_the_nested_connector():
    """An in-place rewrite must cross the NestedSDFG connector: the inner descriptor is a SEPARATE
    object, so rewriting only the parent would leave the two ends of the connector disagreeing on the
    rank and validation would reject the SDFG."""
    inner = dace.SDFG("inner")
    inner.add_array("ia", [1], dace.float64, transient=False)
    inner.add_array("ib", [1], dace.float64, transient=False)
    ist = inner.add_state("is")
    it = ist.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0")
    ist.add_edge(ist.add_read("ia"), None, it, "a", dace.Memlet("ia[0]"))
    ist.add_edge(it, "b", ist.add_write("ib"), None, dace.Memlet("ib[0]"))

    sdfg = dace.SDFG("outer")
    sdfg.add_array("alpha", [1], dace.float64, transient=False)
    sdfg.add_array("beta", [1], dace.float64, transient=False)
    st = sdfg.add_state("main")
    nested = st.add_nested_sdfg(inner, {"ia"}, {"ib"})
    st.add_edge(st.add_read("alpha"), None, nested, "ia", dace.Memlet("alpha[0]"))
    st.add_edge(nested, "ib", st.add_write("beta"), None, dace.Memlet("beta[0]"))

    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True,
                                    preserve_abi=False).apply_pass(sdfg, {})
    assert isinstance(inner.arrays["ia"], dd.Scalar), "connector image not rewritten"
    assert isinstance(inner.arrays["ib"], dd.Scalar), "connector image not rewritten"
    sdfg.validate()


def test_preserve_abi_leaves_the_nested_connector_alone():
    """The mirror: staging repoints the body, so no nested non-transient may change."""
    inner = dace.SDFG("inner_keep")
    inner.add_array("ia", [1], dace.float64, transient=False)
    ist = inner.add_state("is")
    ist.add_tasklet("t", {"a"}, {}, "pass")

    sdfg = dace.SDFG("outer_keep")
    sdfg.add_array("alpha", [1], dace.float64, transient=False)
    st = sdfg.add_state("main")
    nested = st.add_nested_sdfg(inner, {"ia"}, {})
    st.add_edge(st.add_read("alpha"), None, nested, "ia", dace.Memlet("alpha[0]"))

    ConvertLengthOneArraysToScalars(stage_nontransients_arrays_into_scalars=True).apply_pass(sdfg, {})
    assert isinstance(inner.arrays["ia"], dd.Array) and tuple(inner.arrays["ia"].shape) == (1, )
    assert isinstance(sdfg.arrays["alpha"], dd.Array)


# --- filter knob ------------------------------------------------------------


def _three_transient_len1() -> dace.SDFG:
    """Three transient length-1 arrays referenced from one interstate edge."""
    sdfg = dace.SDFG("flt")
    for nm in ("keep_me", "skip_me", "local"):
        sdfg.add_array(nm, (1, ), dace.float64, transient=True)
    s0, s1 = sdfg.add_state("s0"), sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "keep_me[0] + skip_me[0] + local[0]"}))
    return sdfg


def test_filter_none_scalarizes_every_eligible():
    sdfg = _three_transient_len1()
    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})
    for name in ("keep_me", "skip_me", "local"):
        assert isinstance(sdfg.arrays[name], dd.Scalar), name


def test_filter_set_restricts_to_listed_names():
    sdfg = _three_transient_len1()
    ConvertLengthOneArraysToScalars(recursive=False, filter={"keep_me"}).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["keep_me"], dd.Scalar)
    assert isinstance(sdfg.arrays["skip_me"], dd.Array)
    assert isinstance(sdfg.arrays["local"], dd.Array)
    assigns = list(sdfg.all_interstate_edges())[0].data.assignments["k"]
    assert assigns == "keep_me + skip_me[0] + local[0]", assigns


def test_filter_with_unknown_name_does_nothing():
    sdfg = _three_transient_len1()
    assert ConvertLengthOneArraysToScalars(recursive=False, filter={"no_such_array"}).apply_pass(sdfg, {}) is None
    for name in ("keep_me", "skip_me", "local"):
        assert isinstance(sdfg.arrays[name], dd.Array), name


def test_filter_empty_set_converts_nothing_both_directions():
    """An EMPTY filter restricts eligibility to nothing (distinct from ``None`` = no restriction)."""
    fwd = _three_transient_len1()
    assert ConvertLengthOneArraysToScalars(recursive=False, filter=set()).apply_pass(fwd, {}) is None
    assert all(isinstance(fwd.arrays[n], dd.Array) for n in ("keep_me", "skip_me", "local"))
    inv = dace.SDFG("empty_inv")
    inv.add_state("s")
    inv.add_scalar("c", dace.float64, transient=True)
    inv.add_scalar("d", dace.float64, transient=True)
    assert ConvertScalarsToLengthOneArrays(recursive=False, filter=set()).apply_pass(inv, {}) is None
    assert isinstance(inv.arrays["c"], dd.Scalar) and isinstance(inv.arrays["d"], dd.Scalar)


def test_filter_only_gates_root_level_nested_recursion_unaffected():
    inner = dace.SDFG("inner")
    inner.add_array("inner_local", (1, ), dace.float64, transient=True)
    inner.add_state("s")
    outer = dace.SDFG("outer")
    outer.add_array("outer_arr", (1, ), dace.float64, transient=True)
    outer.add_array("outer_unrelated", (1, ), dace.float64, transient=True)
    ostate = outer.add_state()
    ostate.add_nested_sdfg(sdfg=inner, inputs=set(), outputs=set())
    ConvertLengthOneArraysToScalars(recursive=True, filter={"outer_arr"}).apply_pass(outer, {})
    assert isinstance(outer.arrays["outer_arr"], dd.Scalar)
    assert isinstance(outer.arrays["outer_unrelated"], dd.Array)  # not in filter
    assert isinstance(inner.arrays["inner_local"], dd.Scalar)  # recursion is not filter-gated


def test_filter_restricts_scalars_to_length_one_arrays():
    sdfg = dace.SDFG("inv_flt")
    sdfg.add_state("s")
    sdfg.add_scalar("keep_me", dace.float64, transient=True)
    sdfg.add_scalar("skip_me", dace.float64, transient=True)
    rewritten = ConvertScalarsToLengthOneArrays(recursive=False, filter={"keep_me"}).apply_pass(sdfg, {})
    assert rewritten == {"keep_me"}
    assert isinstance(sdfg.arrays["keep_me"], dd.Array)
    assert isinstance(sdfg.arrays["skip_me"], dd.Scalar)


# --- View / opaque / other_subset guards ------------------------------------


def test_length_one_view_is_not_scalarized():
    sdfg = dace.SDFG("len1_view")
    sdfg.add_state("s")
    sdfg.add_array("src", [4], dace.float64, transient=True)
    sdfg.add_view("vw", [1], dace.float64)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    assert sdfg.arrays["vw"].transient
    rewritten = ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert "vw" not in rewritten
    assert isinstance(sdfg.arrays["vw"], dd.View)
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_length_one_view_source_is_not_scalarized():
    sdfg = dace.SDFG("len1_view_src")
    st = sdfg.add_state("s")
    sdfg.add_array("src", [1], dace.float64, transient=True)
    sdfg.add_view("vw", [1], dace.float64)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    sn, vn = st.add_access("src"), st.add_access("vw")
    st.add_edge(sn, None, vn, "views", dace.Memlet(data="src", subset="0"))
    rewritten = ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert "src" not in rewritten
    assert isinstance(sdfg.arrays["src"], dd.Array) and not isinstance(sdfg.arrays["src"], dd.View)
    assert isinstance(sdfg.arrays["vw"], dd.View)
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_other_subset_of_scalarized_side_collapses():
    """A copy edge names one side in ``Memlet.data``; the opposite side is addressed by
    ``other_subset``. Scalarizing THAT side must collapse its ``other_subset`` too, else validation
    rejects the stale rank. Reproduces the npbench ``vadv`` ``(1, 1)`` MapFusion scratch case."""
    sdfg = dace.SDFG("len1_other_subset")
    st = sdfg.add_state("s")
    sdfg.add_array("big", [4, 4], dace.float64)
    sdfg.add_array("scratch", [1, 1], dace.float64, transient=True)
    rb, ws = st.add_access("big"), st.add_access("scratch")
    st.add_nedge(rb, ws, dace.Memlet(data="big", subset="1, 1", other_subset="0, 0"))
    sdfg.validate()
    rewritten = ConvertLengthOneArraysToScalars(single_element=True).apply_pass(sdfg, {})
    assert "scratch" in rewritten
    assert isinstance(sdfg.arrays["scratch"], dd.Scalar)
    assert next(iter(st.edges())).data.other_subset.dims() == 1
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the length-1-array <-> scalar conversion passes.

``ConvertLengthOneArraysToScalars`` rewrites every TRANSIENT length-1 ``Array`` (shape ``(1,)``) to a
true ``Scalar`` in place and strips the redundant ``[0]`` accessors; with ``preserve_abi`` it
additionally converts each non-transient length-1 array WITHOUT touching the signature, by STAGING it
into a fresh transient scalar (copy-in in a new start state, copy-out in a new sink state).
``ConvertScalarsToLengthOneArrays`` is the inverse. These are pure-SDFG (no Fortran) tests of the Pass
classes, covering the staging, ``preserve_abi``, ``filter`` gating and ``opaque``/View exemptions.
"""

import numpy as np

import dace
import dace.data as dd
import pytest

from dace.transformation.passes import (
    ConvertLengthOneArraysToScalars,
    ConvertScalarsToLengthOneArrays,
)


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


@pytest.mark.parametrize('preserve_abi', [False, True], ids=['transient', 'staged-argument'])
def test_interstate_accessor_with_a_computed_index_is_stripped(preserve_abi):
    """``a[j - 1]`` on a length-one array still names element 0 and must collapse like ``a[0]``.

    Only a literal ``0`` used to collapse, so a computed index was left subscripting the new
    scalar. npbench ``vexx_k`` reads a one-element k-point table this way at the fuzzed preset and
    the host code failed to compile: ``scal_index_xk[dace::int64(index_xkq_index)]``.
    """
    sdfg = dace.SDFG(f"istrip_computed_{preserve_abi}")
    s0, s1 = sdfg.add_state("s0"), sdfg.add_state("s1")
    sdfg.add_array("a", [1], dace.int64, transient=not preserve_abi)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "a[j - 1] + 1"}))
    ConvertLengthOneArraysToScalars(preserve_abi=preserve_abi).apply_pass(sdfg, {})

    for edge in sdfg.all_interstate_edges():
        if "k" in edge.data.assignments:
            value = edge.data.assignments["k"]
            assert "[" not in value, f"the scalar is still subscripted: {value}"
            break
    else:
        pytest.fail("the assignment of k vanished")


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
    rewritten = ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
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


def _signature(sdfg):
    """The caller-visible contract: each non-transient descriptor's kind, shape and dtype."""
    return {nm: (type(d), tuple(d.shape), d.dtype) for nm, d in sdfg.arrays.items() if not d.transient}


def test_preserve_abi_leaves_the_signature_byte_identical():
    """The guarantee the flag is named for, in both directions: staging routes the conversion through
    copy states, so no descriptor a caller binds to changes kind, shape or dtype."""
    sdfg = _io_sdfg()
    before = _signature(sdfg)
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    assert _signature(sdfg) == before, "forward pass moved the signature"
    ConvertScalarsToLengthOneArrays(preserve_abi=True).apply_pass(sdfg, {})
    assert _signature(sdfg) == before, "inverse pass moved the signature"


def test_preserve_abi_stages_a_signature_scalar_into_a_length_one_array():
    """The inverse direction end to end: a non-transient ``Scalar`` on the signature stays a Scalar and
    the body is repointed at a staged length-1 array, wired through the copy states."""
    sdfg = dace.SDFG("scal_sig")
    sdfg.add_scalar("alpha", dace.float64, transient=False)
    sdfg.add_array("beta", [1], dace.float64, transient=False)
    st = sdfg.add_state("main")
    t = st.add_tasklet("t", {"a"}, {"b"}, "b = a * 3.0")
    st.add_edge(st.add_read("alpha"), None, t, "a", dace.Memlet("alpha[0]"))
    st.add_edge(t, "b", st.add_write("beta"), None, dace.Memlet("beta[0]"))

    rewritten = ConvertScalarsToLengthOneArrays(preserve_abi=True).apply_pass(sdfg, {})
    assert rewritten == {"alpha"}
    assert isinstance(sdfg.arrays["alpha"], dd.Scalar) and not sdfg.arrays["alpha"].transient
    staged = sdfg.arrays["arr_alpha"]
    assert isinstance(staged, dd.Array) and staged.transient and tuple(staged.shape) == (1, )
    labels = {s.label for s in sdfg.all_states()}
    assert "stage_copyin" in labels, "a read signature scalar needs a copy-in"
    assert "stage_copyout" not in labels, "alpha is never written -- no copy-out"
    sdfg.validate()

    beta = np.array([0.0], dtype=np.float64)
    sdfg(alpha=2.0, beta=beta)
    assert beta[0] == pytest.approx(6.0)


def test_stage_is_numerically_correct():
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
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
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    labels = {s.label for s in sdfg.all_states()}
    assert "stage_copyin" in labels  # inp is read
    assert "stage_copyout" not in labels  # inp is never written -> no copy-out
    sdfg.validate()


def test_forward_then_inverse_stays_correct():
    """X then X^-1 (both staging) leaves a valid, numerically-correct SDFG."""
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    ConvertScalarsToLengthOneArrays(preserve_abi=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert _run(sdfg, 3.0) == pytest.approx(6.0)


def test_repeated_forward_is_idempotent_and_stays_correct():
    """Applying the forward pass twice must be a no-op the second time and stay numerically
    correct. Staging KEEPS the signature array, so without an already-staged check the array
    stays eligible forever and each re-run chains another copy hop onto the last."""
    sdfg = _io_sdfg()
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    before, before_states = set(sdfg.arrays), len(list(sdfg.all_states()))
    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    assert set(sdfg.arrays) == before, "second application re-staged an already-staged array"
    assert len(list(sdfg.all_states())) == before_states, "second application added copy states"
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
        "recursive", "preserve_abi", "filter", "single_element", "skip_gpu_outputs"
    }
    assert set(ConvertScalarsToLengthOneArrays.__properties__) == {"recursive", "preserve_abi", "filter"}
    for cls in (ConvertLengthOneArraysToScalars, ConvertScalarsToLengthOneArrays):
        inst = cls(recursive=False, preserve_abi=True)
        assert inst.recursive is False
        assert inst.preserve_abi is True


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


def test_control_flow_only_gate_gets_a_copyin():
    """A signature length-1 array read ONLY from control flow (an interstate assignment gating the
    kernel, QE's ``IF (.NOT.okvan) RETURN``). Staging repoints that reference to the scalar, so the
    scalar needs a copy-in: without one it is declared, read and never written, and an uninitialized
    stack value decides whether the kernel runs."""
    sdfg = dace.SDFG("gate_len1")
    sdfg.add_array("flag", (1, ), dace.bool_)
    sdfg.add_array("y", (4, ), dace.float64)
    sdfg.add_symbol("if_cond", dace.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    sg = sdfg.add_state("gate")
    s1 = sdfg.add_state("compute")
    s2 = sdfg.add_state("end")
    sdfg.add_edge(s0, sg, dace.InterstateEdge(assignments={"if_cond": "flag[0]"}))
    sdfg.add_edge(sg, s1, dace.InterstateEdge(condition="if_cond"))
    sdfg.add_edge(sg, s2, dace.InterstateEdge(condition="not if_cond"))
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    tasklet = s1.add_tasklet("set1", {}, {"out": None}, "out = 1.0")
    s1.add_edge(tasklet, "out", s1.add_write("y"), None, dace.Memlet("y[0]"))
    sdfg.validate()

    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})

    # preserve_abi keeps the signature array itself.
    assert "flag" in sdfg.arrays and not sdfg.arrays["flag"].transient
    staged = [n for n in sdfg.arrays if n.startswith("scal_flag")]
    assert len(staged) == 1, f"expected the gate to be staged once, got {staged}"
    scal = staged[0]
    assert any(scal in e.data.assignments.get("if_cond", "") for e in sdfg.all_interstate_edges()), \
        "the gate assignment was not repointed at the staged scalar"
    writers = sum(
        st.in_degree(nd) for st in sdfg.all_states() for nd in st.nodes()
        if isinstance(nd, dace.nodes.AccessNode) and nd.data == scal)
    assert writers > 0, f"{scal} is read from control flow but has no writer: the copy-in was dropped"
    sdfg.validate()


def test_control_flow_only_signature_scalar_gets_a_copyin():
    """Inverse direction of the gate case: a signature ``Scalar`` read only by an interstate assignment is
    staged into a length-1 array, and that array needs a copy-in or the gate reads uninitialized memory."""
    sdfg = dace.SDFG("gate_scalar")
    sdfg.add_scalar("flag", dace.int64)
    sdfg.add_array("y", (4, ), dace.float64)
    sdfg.add_symbol("if_cond", dace.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    sg = sdfg.add_state("gate")
    s1 = sdfg.add_state("compute")
    sdfg.add_edge(s0, sg, dace.InterstateEdge(assignments={"if_cond": "flag"}))
    sdfg.add_edge(sg, s1, dace.InterstateEdge(condition="if_cond > 0"))
    tasklet = s1.add_tasklet("set1", {}, {"out": None}, "out = 1.0")
    s1.add_edge(tasklet, "out", s1.add_write("y"), None, dace.Memlet("y[0]"))
    sdfg.validate()

    ConvertScalarsToLengthOneArrays(preserve_abi=True).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays["flag"], dd.Scalar) and not sdfg.arrays["flag"].transient
    staged = [n for n in sdfg.arrays if n.startswith("arr_flag")]
    assert len(staged) == 1, f"expected the gate to be staged once, got {staged}"
    arr = staged[0]
    assert any(f"{arr}[0]" in e.data.assignments.get("if_cond", "") for e in sdfg.all_interstate_edges()), \
        "the gate assignment was not repointed at the staged array"
    copyins = [(st.label, e.src.data) for st in sdfg.all_states() for e in st.edges()
               if isinstance(e.dst, dace.nodes.AccessNode) and e.dst.data == arr]
    assert copyins == [("stage_copyin", "flag")], f"{arr} is read from control flow; copy-ins found: {copyins}"
    sdfg.validate()


def _scatter_sdfg(tmp_is_scalar: bool) -> dace.SDFG:
    """``for i: A[(i+1) % 2] = B[i]`` staged through a single-value transient. The copy edge names the
    TRANSIENT, so the destination index lives in ``other_subset``."""
    sdfg = dace.SDFG("len1_other_subset_survives")
    st = sdfg.add_state("s")
    sdfg.add_array("A", [2], dace.int32)
    sdfg.add_array("B", [2], dace.int32)
    if tmp_is_scalar:
        sdfg.add_scalar("tmp", dace.int32, transient=True)
    else:
        sdfg.add_array("tmp", [1], dace.int32, transient=True)
    me, mx = st.add_map("m", {"i": "0:2"}, schedule=dace.dtypes.ScheduleType.Sequential)
    me.add_in_connector("IN_B")
    me.add_out_connector("OUT_B")
    mx.add_in_connector("IN_A")
    mx.add_out_connector("OUT_A")
    st.add_edge(st.add_read("B"), None, me, "IN_B", dace.Memlet("B[0:2]"))
    tmp = st.add_access("tmp")
    st.add_edge(me, "OUT_B", tmp, None, dace.Memlet("B[i]"))
    wa = st.add_access("A")
    st.add_edge(tmp, None, wa, None, dace.Memlet("tmp[0] -> [((i+1)%2)]"))
    st.add_edge(wa, None, mx, "IN_A", dace.Memlet("A[0:2]"))
    st.add_edge(mx, "OUT_A", st.add_access("A"), None, dace.Memlet("A[0:2]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("inverse", [False, True])
def test_other_subset_of_untouched_side_survives(inverse):
    """The rewritten side of a copy edge says NOTHING about the other side. Dropping the untouched
    side's ``other_subset`` silently redirects every write to element 0 -- a scatter turns into a
    single overwrite, with no validation error to catch it."""
    sdfg = _scatter_sdfg(tmp_is_scalar=inverse)
    if inverse:
        ConvertScalarsToLengthOneArrays().apply_pass(sdfg, {})
    else:
        ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    st = sdfg.states()[0]
    copy = next(e for e in st.edges() if e.data.data == "tmp" and e.data.other_subset is not None)
    assert str(copy.data.other_subset) != "0"
    sdfg.validate()

    a = np.zeros(2, np.int32)
    b = np.array([11, 22], np.int32)
    sdfg(A=a, B=b)
    assert a[0] == b[1] and a[1] == b[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_arrayize_rewrites_interstate_edge_assignment():
    """A scalar named on an interstate-edge assignment becomes ``name[0]`` once it is a length-1 array.

    Leaving the bare name there reads the array itself where a value is expected -- a silent
    miscompile rather than a validation error.
    """
    sdfg = dace.SDFG('arrayize_iedge')
    sdfg.add_scalar('arr', dace.float64, transient=True)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_symbol('a', dace.float64)

    s0 = sdfg.add_state('s0', is_start_block=True)
    t = s0.add_tasklet('w', {}, {'o'}, 'o = 3.0')
    s0.add_edge(t, 'o', s0.add_write('arr'), None, dace.Memlet('arr'))
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'a': 'arr'}))
    t2 = s1.add_tasklet('r', {}, {'o'}, 'o = a')
    s1.add_edge(t2, 'o', s1.add_write('out'), None, dace.Memlet('out[0]'))

    ConvertScalarsToLengthOneArrays().apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['arr'], dace.data.Array)
    assignments = [dict(e.data.assignments) for e in sdfg.all_interstate_edges()]
    assert assignments == [{'a': 'arr[0]'}], assignments
    sdfg.validate()


def test_arrayize_rewrites_conditional_guard_after_branch_removal():
    """A ConditionalBlock guard is rewritten even once its branch entries have become tuples.

    ``add_branch`` appends a list but ``remove_branch`` rebuilds the entries as tuples, so a pass
    that reassigns ``branch[0]`` crashes on any conditional a branch was ever removed from.
    """
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    sdfg = dace.SDFG('cond_after_removal')
    sdfg.add_scalar('arr', dace.float64, transient=True)
    sdfg.add_array('out', [1], dace.float64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    t = s0.add_tasklet('w', {}, {'o'}, 'o = 3.0')
    s0.add_edge(t, 'o', s0.add_write('arr'), None, dace.Memlet('arr'))

    cond = ConditionalBlock('cb')
    sdfg.add_node(cond)
    sdfg.add_edge(s0, cond, dace.InterstateEdge())
    keep = ControlFlowRegion('keep', sdfg=sdfg)
    drop = ControlFlowRegion('drop', sdfg=sdfg)
    cond.add_branch(CodeBlock('arr > 0'), keep)
    cond.add_branch(CodeBlock('arr < 0'), drop)
    bs = keep.add_state('bs', is_start_block=True)
    t2 = bs.add_tasklet('r', {}, {'o'}, 'o = 1.0')
    bs.add_edge(t2, 'o', bs.add_write('out'), None, dace.Memlet('out[0]'))
    cond.remove_branch(drop)

    ConvertScalarsToLengthOneArrays().apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['arr'], dace.data.Array)
    assert 'arr[0]' in cond.branches[0][0].as_string, cond.branches[0][0].as_string


def test_staged_scalar_of_a_host_written_device_array_stays_on_the_host():
    """A staged scalar host code writes must not inherit the array's device storage.

    A fully sequential program (npbench crc16) has no map to offload, so the GPU offload leaves every
    tasklet on the host while its signature arrays sit in ``GPU_Global``. Staging then produced a
    ``GPU_Global`` scalar that a host tasklet writes, which validation rejects.
    """
    sdfg = dace.SDFG('host_written_device_output')
    sdfg.add_array('crc', [1], dace.int64, transient=False, storage=dace.StorageType.GPU_Global)
    st = sdfg.add_state('main')
    t = st.add_tasklet('w', {}, {'o'}, 'o = 7')
    st.add_edge(t, 'o', st.add_write('crc'), None, dace.Memlet('crc[0]'))

    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['scal_crc'], dd.Scalar)
    assert sdfg.arrays['scal_crc'].storage is not dace.StorageType.GPU_Global
    assert sdfg.arrays['crc'].storage is dace.StorageType.GPU_Global
    sdfg.validate()


def test_staged_scalar_of_a_kernel_written_array_keeps_device_storage():
    """The other side of the rule: a GPU map writing the array keeps its staged scalar on the device."""
    sdfg = dace.SDFG('kernel_written_output')
    sdfg.add_array('res', [1], dace.float64, transient=False, storage=dace.StorageType.GPU_Global)
    st = sdfg.add_state('main')
    entry, exit_ = st.add_map('k', {'i': '0:1'}, schedule=dace.ScheduleType.GPU_Device)
    t = st.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    st.add_nedge(entry, t, dace.Memlet())
    st.add_memlet_path(t, exit_, st.add_write('res'), src_conn='o', memlet=dace.Memlet('res[0]'))

    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})

    assert sdfg.arrays['scal_res'].storage is dace.StorageType.GPU_Global


def test_a_rewritten_symbol_mapping_stays_symbolic():
    """A nest's symbol mapping must come out of the rewrite as a symbolic expression.

    ``rewrite_code_slots`` reprints every mapping value through an AST round trip, so even a value
    it renames nothing in comes back as a string. Readers check whether that value is symbolic, and
    a raw string is not: ``ConstantPropagation`` took ``'nat * nh'`` for a constant, substituted
    the caller's names into the nest and left it naming symbols nothing binds. npbench ``vexx_k``
    failed the GPU canonicalize column that way, with ``Missing symbols on nested SDFG``.
    """
    import sympy

    from dace.transformation.passes.length_one_array_scalar_conversion import (rewrite_code_slots,
                                                                               rewrite_refs_to_element)

    rows, cols = (dace.symbol(name, dtype=dace.int64) for name in ('rows', 'cols'))
    stride = dace.symbol('stride', dtype=dace.int64)

    inner = dace.SDFG('view_a_row')
    inner.add_array('src', [4], dace.float64, strides=[stride])
    inner.add_array('dst', [4], dace.float64)
    body = inner.add_state('copy', is_start_block=True)
    body.add_nedge(body.add_read('src'), body.add_write('dst'), dace.Memlet('src[0:4]'))

    sdfg = dace.SDFG('mapping_stays_symbolic')
    sdfg.add_array('a', [4], dace.float64)
    sdfg.add_array('out', [4], dace.float64)
    state = sdfg.add_state('call', is_start_block=True)
    nest = state.add_nested_sdfg(inner, {'src': None}, {'dst': None}, symbol_mapping={'stride': rows * cols})
    state.add_edge(state.add_read('a'), None, nest, 'src', dace.Memlet('a[0:4]'))
    state.add_edge(nest, 'dst', state.add_write('out'), None, dace.Memlet('out[0:4]'))

    # The pass's own rewrite, renaming a name this SDFG does not use: nothing is substituted,
    # and the value still comes back reprinted because the rewrite parses and unparses it.
    rewrite_code_slots(sdfg, lambda text: rewrite_refs_to_element(text, {'unrelated': 'scal_unrelated'}))

    value = nest.symbol_mapping['stride']
    assert isinstance(value, sympy.Basic), f'the mapping came back as {type(value).__name__}: {value!r}'
    assert {str(sym) for sym in value.free_symbols} == {'rows', 'cols'}, value
    assert int(value.subs({sympy.Symbol('rows'): 3, sympy.Symbol('cols'): 5})) == 15, value

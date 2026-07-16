# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the length-1-array <-> scalar conversion passes.

``ConvertLengthOneArraysToScalars`` rewrites every length-1 ``Array``
(shape ``(1,)``) to a true ``Scalar`` and strips the now-redundant
``[0]`` accessors.  ``ConvertScalarsToLengthOneArrays`` is the inverse.
These are pure-SDFG (no Fortran) tests of the Pass classes and the
underlying rewrite, covering the ``keep_program_outputs`` and ``filter``
gating knobs and the ``opaque``-dtype exemptions.
"""
import ctypes

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


def _sdfg_with_len1(transient: bool) -> dace.SDFG:
    sdfg = dace.SDFG("len1")
    sdfg.add_state("s")
    sdfg.add_array("a", [1], dace.float64, transient=transient)
    sdfg.add_array("b", [10], dace.float64, transient=False)
    return sdfg


def test_convert_length_one_arrays_to_scalars_basic():
    sdfg = _sdfg_with_len1(transient=False)
    rewritten = ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    # A genuine multi-element array is left alone.
    assert isinstance(sdfg.arrays["b"], dd.Array)


def test_transient_only_skips_signature_arrays():
    sdfg = _sdfg_with_len1(transient=False)
    rewritten = ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert rewritten is None
    assert isinstance(sdfg.arrays["a"], dd.Array)


def test_transient_only_rewrites_transient_arrays():
    sdfg = _sdfg_with_len1(transient=True)
    rewritten = ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_recursive_descends_into_nested_sdfg():
    sdfg = dace.SDFG("outer")
    st = sdfg.add_state("s")
    nested = dace.SDFG("inner")
    nested.add_state("ns")
    nested.add_array("z", [1], dace.float64, transient=True)
    st.add_nested_sdfg(nested, {}, {})
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=True).apply_pass(sdfg, {})
    assert isinstance(nested.arrays["z"], dd.Scalar)


def test_interstate_accessor_is_stripped():
    sdfg = dace.SDFG("istrip")
    s0 = sdfg.add_state("s0")
    s1 = sdfg.add_state("s1")
    sdfg.add_array("a", [1], dace.int64, transient=False)
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={"k": "a[0] + 1"}))
    ConvertLengthOneArraysToScalars().apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    edge = list(sdfg.all_interstate_edges())[0]
    assert edge.data.assignments["k"] == "a + 1"


def test_convert_scalars_to_length_one_arrays_roundtrip():
    sdfg = _sdfg_with_len1(transient=True)
    ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["a"], dd.Scalar)
    rewritten = ConvertScalarsToLengthOneArrays(transient_only=True).apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Array)
    assert tuple(sdfg.arrays["a"].shape) == (1, )


def test_keep_program_outputs_keeps_written_nontransient_array():
    """With ``keep_program_outputs`` a written non-transient length-1 array (a program
    output) stays an ``Array`` while a read-only non-transient one is still scalarized."""
    sdfg = dace.SDFG("prog_out")
    state = sdfg.add_state("s")
    sdfg.add_array("out", [1], dace.float64, transient=False)
    sdfg.add_array("inp", [1], dace.float64, transient=False)
    an_inp, an_out = state.add_access("inp"), state.add_access("out")
    state.add_nedge(an_inp, an_out, dace.Memlet(data="inp", subset="0"))
    ConvertLengthOneArraysToScalars(keep_program_outputs=True).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays["out"], dd.Array)
    assert isinstance(sdfg.arrays["inp"], dd.Scalar)


def test_opaque_length_one_array_is_not_scalarized():
    """A length-1 array of an ``opaque`` dtype (e.g. ``MPI_Request``)
    must stay an ``Array`` so a pointer-connector consumer can take its
    address; only the plain-dtype length-1 array next to it is folded."""
    sdfg = dace.SDFG("opaque_len1")
    sdfg.add_state("s")
    sdfg.add_array("req", [1], dace.dtypes.opaque("MPI_Request"), transient=True)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    rewritten = ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["req"], dd.Array)
    assert tuple(sdfg.arrays["req"].shape) == (1, )
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_opaque_scalar_is_not_arrayized():
    """The symmetric inverse: an ``opaque`` ``Scalar`` keeps its scalar
    form under ``ConvertScalarsToLengthOneArrays`` while a plain-dtype
    scalar is rewritten to a length-1 ``Array``."""
    sdfg = dace.SDFG("opaque_scalar")
    sdfg.add_state("s")
    sdfg.add_scalar("comm", dace.dtypes.opaque("MPI_Comm"), transient=True)
    sdfg.add_scalar("k", dace.int64, transient=True)
    rewritten = ConvertScalarsToLengthOneArrays(transient_only=True).apply_pass(sdfg, {})
    assert rewritten == {"k"}
    assert isinstance(sdfg.arrays["comm"], dd.Scalar)
    assert isinstance(sdfg.arrays["k"], dd.Array)


def test_passes_expose_property_options():
    # The forward pass additionally exposes ``keep_program_outputs`` / ``filter`` and
    # ``single_element`` (scalarize higher-rank all-ones arrays, e.g. a (1, 1) map-fusion
    # scratch buffer); the inverse exposes none of them.
    assert set(ConvertLengthOneArraysToScalars.__properties__) == {
        "recursive", "transient_only", "keep_program_outputs", "filter", "single_element"
    }
    assert set(ConvertScalarsToLengthOneArrays.__properties__) == {"recursive", "transient_only"}
    for cls in (ConvertLengthOneArraysToScalars, ConvertScalarsToLengthOneArrays):
        inst = cls(recursive=False, transient_only=True)
        assert inst.recursive is False
        assert inst.transient_only is True


def test_scalarize_rewrites_length_one_array():
    """A shape-``(1,)`` array becomes a true ``Scalar`` and its ``[0]`` accessor is dropped."""
    sdfg = dace.SDFG('scalarize')
    sdfg.add_array('a', (1, ), dace.float64)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'a[0] + 1'}))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['a'], dace.data.Scalar)
    assert list(sdfg.all_interstate_edges())[0].data.assignments['k'] == 'a + 1'


def test_scalarize_keeps_overlapping_name_subscript():
    """A scalarized name that is a suffix of another array must not eat that
    array's literal ``[0]`` index (scalarized ``ar`` vs multi-element ``bar``)."""
    sdfg = dace.SDFG('overlap')
    sdfg.add_array('ar', (1, ), dace.float64)
    sdfg.add_array('bar', (4, ), dace.float64)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'ar[0] + bar[0]'}))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['ar'], dace.data.Scalar)
    assert isinstance(sdfg.arrays['bar'], dace.data.Array)
    assert list(sdfg.all_interstate_edges())[0].data.assignments['k'] == 'ar + bar[0]'


def test_collapsed_memlet_preserves_dynamic():
    """Collapsing a scalarized array's memlet to element 0 keeps the dynamic flag."""
    sdfg = dace.SDFG('dynmem')
    sdfg.add_array('a', (1, ), dace.float64, transient=True)
    sdfg.add_array('b', (1, ), dace.float64)
    state = sdfg.add_state('s')
    an_a, an_b = state.add_access('a'), state.add_access('b')
    state.add_nedge(an_a, an_b, dace.Memlet(data='a', subset='0', dynamic=True))

    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})

    assert isinstance(sdfg.arrays['a'], dace.data.Scalar)
    assert state.edges()[0].data.dynamic is True


# ---------------------------------------------------------------------------
# ``filter`` knob: when None, every eligible array is scalarized; when a set is
# provided, only listed names are scalarized AND being in the filter overrides
# the ``transient_only`` check.
# ---------------------------------------------------------------------------
def _build_two_length_one_arrays_sdfg():
    """SDFG with two length-1 arrays: one transient and one non-transient. Both eligible
    under ``transient_only=False`` defaults."""
    sdfg = dace.SDFG('flt')
    sdfg.add_array('keep_me', (1, ), dace.float64, transient=False)
    sdfg.add_array('skip_me', (1, ), dace.float64, transient=False)
    sdfg.add_array('local', (1, ), dace.float64, transient=True)
    s0, s1 = sdfg.add_state('s0'), sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'keep_me[0] + skip_me[0] + local[0]'}))
    return sdfg


def test_filter_none_is_default_no_filtering():
    """``filter=None`` (the default) keeps existing behaviour: every length-1 array
    eligible under ``transient_only`` is scalarized."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False).apply_pass(sdfg, {})
    for name in ('keep_me', 'skip_me', 'local'):
        assert isinstance(sdfg.arrays[name], dace.data.Scalar), name


def test_filter_set_restricts_to_listed_names():
    """A non-empty filter set scalarizes only the listed arrays; the others stay arrays."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False, filter={'keep_me'}).apply_pass(sdfg, {})
    assert isinstance(sdfg.arrays['keep_me'], dace.data.Scalar)
    assert isinstance(sdfg.arrays['skip_me'], dace.data.Array)
    assert isinstance(sdfg.arrays['local'], dace.data.Array)
    # ``[0]`` only stripped for the scalarized one; the other two keep their subscripts.
    assigns = list(sdfg.all_interstate_edges())[0].data.assignments['k']
    assert assigns == 'keep_me + skip_me[0] + local[0]', assigns


def test_filter_overrides_transient_only_for_listed_name():
    """A name in the filter is scalarized even when ``transient_only=True`` would otherwise
    skip it; arrays not in the filter are NOT scalarized regardless of ``transient_only``
    (the filter is the exclusive allow-list at the root level)."""
    sdfg = _build_two_length_one_arrays_sdfg()
    ConvertLengthOneArraysToScalars(recursive=False, transient_only=True, filter={'keep_me'}).apply_pass(sdfg, {})
    # keep_me is non-transient, but the filter overrides the transient_only restriction.
    assert isinstance(sdfg.arrays['keep_me'], dace.data.Scalar)
    # skip_me is non-transient and not in the filter -> left alone.
    assert isinstance(sdfg.arrays['skip_me'], dace.data.Array)
    # ``local`` is transient and would otherwise pass transient_only, but the filter is
    # the exclusive allow-list at root level: only listed names are touched.
    assert isinstance(sdfg.arrays['local'], dace.data.Array)


def test_filter_with_unknown_name_does_nothing():
    """A filter that names no real array yields no rewrites; the pass returns ``None``."""
    sdfg = _build_two_length_one_arrays_sdfg()
    result = ConvertLengthOneArraysToScalars(recursive=False, filter={'no_such_array'}).apply_pass(sdfg, {})
    assert result is None
    for name in ('keep_me', 'skip_me', 'local'):
        assert isinstance(sdfg.arrays[name], dace.data.Array), name


def test_filter_only_gates_root_level_nested_recursion_unaffected():
    """The filter applies to root-SDFG names only. Inner SDFG recursion (transient-only)
    proceeds independently and rewrites the nested transient length-1 array even when the
    filter is restrictive at the root."""
    inner = dace.SDFG('inner')
    inner.add_array('inner_local', (1, ), dace.float64, transient=True)
    inner.add_state('s')

    outer = dace.SDFG('outer')
    outer.add_array('outer_arr', (1, ), dace.float64, transient=False)
    outer.add_array('outer_unrelated', (1, ), dace.float64, transient=False)
    ostate = outer.add_state()
    ostate.add_nested_sdfg(sdfg=inner, inputs=set(), outputs=set())

    ConvertLengthOneArraysToScalars(recursive=True, filter={'outer_arr'}).apply_pass(outer, {})
    assert isinstance(outer.arrays['outer_arr'], dace.data.Scalar)
    assert isinstance(outer.arrays['outer_unrelated'], dace.data.Array)
    # Inner recursion follows transient_only=True (the default for nested) and is NOT gated
    # by the outer filter; the inner transient length-1 array gets scalarized.
    assert isinstance(inner.arrays['inner_local'], dace.data.Scalar)


def test_length_one_view_is_not_scalarized():
    """A length-1 ``View`` aliases another array's storage through a
    ``views`` edge that a ``Scalar`` cannot carry, so it must stay a View
    even though it subclasses ``Array`` and has shape ``(1,)``.  The plain
    length-1 Array beside it is still folded.  (A Fortran scalar POINTER
    rebind lowered as a length-1-array view relies on this.)"""
    sdfg = dace.SDFG("len1_view")
    sdfg.add_state("s")
    sdfg.add_array("src", [4], dace.float64, transient=True)
    sdfg.add_view("vw", [1], dace.float64)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    # The View is transient, so without the View guard it would be folded.
    assert sdfg.arrays["vw"].transient
    rewritten = ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert "vw" not in rewritten
    assert isinstance(sdfg.arrays["vw"], dd.View)
    assert tuple(sdfg.arrays["vw"].shape) == (1, )
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_length_one_view_source_is_not_scalarized():
    """A length-1 Array that BACKS a length-1 View must stay an Array: a
    View needs an Array source to alias (a ``Scalar`` source is emitted
    ``const`` and a write through the view fails to compile).  Both the
    view and its source are kept; an unrelated length-1 array still folds."""
    sdfg = dace.SDFG("len1_view_src")
    st = sdfg.add_state("s")
    sdfg.add_array("src", [1], dace.float64, transient=True)
    sdfg.add_view("vw", [1], dace.float64)
    sdfg.add_array("a", [1], dace.float64, transient=True)
    sn = st.add_access("src")
    vn = st.add_access("vw")
    st.add_edge(sn, None, vn, "views", dace.Memlet(data="src", subset="0"))
    rewritten = ConvertLengthOneArraysToScalars(transient_only=True).apply_pass(sdfg, {})
    assert "src" not in rewritten
    assert isinstance(sdfg.arrays["src"], dd.Array) and not isinstance(sdfg.arrays["src"], dd.View)
    assert isinstance(sdfg.arrays["vw"], dd.View)
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


def test_other_subset_of_scalarized_side_collapses():
    """A copy edge names only ONE side in ``Memlet.data``; the opposite side is addressed by
    ``other_subset``. Scalarizing THAT side must collapse its ``other_subset`` too, or the edge
    keeps the pre-scalarization rank and validation rejects it with "Memlet other_subset does not
    match node dimension". Reproduces the npbench ``vadv`` failure, where ``single_element``
    scalarizes the ``(1, 1)`` MapFusion scratch and leaves a 2-D ``other_subset`` behind."""
    sdfg = dace.SDFG("len1_other_subset")
    st = sdfg.add_state("s")
    sdfg.add_array("big", [4, 4], dace.float64)
    sdfg.add_array("scratch", [1, 1], dace.float64, transient=True)
    rb = st.add_access("big")
    ws = st.add_access("scratch")
    # data == 'big' (stays 2-D); other_subset addresses 'scratch' (about to become a Scalar).
    st.add_nedge(rb, ws, dace.Memlet(data="big", subset="1, 1", other_subset="0, 0"))
    sdfg.validate()

    rewritten = ConvertLengthOneArraysToScalars(single_element=True, transient_only=True).apply_pass(sdfg, {})
    assert "scratch" in rewritten
    assert isinstance(sdfg.arrays["scratch"], dd.Scalar)
    edge = next(iter(st.edges()))
    assert edge.data.other_subset.dims() == 1, f"stale other_subset rank: {edge.data.other_subset}"
    sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

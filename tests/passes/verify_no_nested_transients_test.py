# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``VerifyNoNestedTransients`` / ``verify_no_nested_transients``."""
import pytest

import dace
from dace import memlet as mm
from dace.sdfg import SDFG
from dace.transformation.passes.verify_no_nested_transients import (
    VerifyNoNestedTransients, verify_no_nested_transients)


def _top_with_nested(inner: SDFG, in_conns=None, out_conns=None):
    top = SDFG("top_wrap")
    for name in (in_conns or ()):
        top.add_array(name, inner.arrays[name].shape, inner.arrays[name].dtype,
                      transient=False)
    for name in (out_conns or ()):
        top.add_array(name, inner.arrays[name].shape, inner.arrays[name].dtype,
                      transient=False)
    st = top.add_state()
    n = st.add_nested_sdfg(inner, set(in_conns or ()), set(out_conns or ()))
    for name in (in_conns or ()):
        top_r = st.add_read(name)
        st.add_edge(top_r, None, n, name,
                    mm.Memlet.from_array(name, top.arrays[name]))
    for name in (out_conns or ()):
        top_w = st.add_write(name)
        st.add_edge(n, name, top_w, None,
                    mm.Memlet.from_array(name, top.arrays[name]))
    return top


def test_no_nested_transients_passes_when_clean():
    """Fresh SDFG with a plain non-transient nested array -- check passes."""
    inner = SDFG("inner")
    inner.add_array("B", [8], dace.float64, transient=False)
    st = inner.add_state()
    bw = st.add_write("B")
    task = st.add_tasklet("set", {}, {"o"}, "o = 1.0")
    me, mx = st.add_map("fill", {"k": "0:8"})
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")
    st.add_edge(me, None, task, None, mm.Memlet())
    st.add_edge(task, "o", mx, "IN_B", mm.Memlet(data="B", subset="k"))
    st.add_edge(mx, "OUT_B", bw, None, mm.Memlet.from_array("B", inner.arrays["B"]))

    top = _top_with_nested(inner, out_conns=["B"])
    # No raise.
    verify_no_nested_transients(top)
    assert VerifyNoNestedTransients().apply_pass(top, {}) is None


def test_no_nested_transients_ignores_top_level_transient():
    """A transient at the root SDFG is fine -- only nested transients matter."""
    top = SDFG("top_only_root_transient")
    top.add_array("B", [8], dace.float64, transient=False)
    top.add_array("t", [16], dace.float64, transient=True)
    top.add_state()
    verify_no_nested_transients(top)


def test_no_nested_transients_allows_nested_scalar():
    """shape == (1,) transient inside a nested SDFG is thread-local
    scratch and MUST NOT trip the check."""
    inner = SDFG("inner_scalar_only")
    inner.add_array("s", [1], dace.float64, transient=True)
    inner.add_array("B", [1], dace.float64, transient=False)
    st = inner.add_state()
    task = st.add_tasklet("w", {}, {"o"}, "o = 3.14")
    s_acc = st.add_access("s")
    b_acc = st.add_write("B")
    st.add_edge(task, "o", s_acc, None, mm.Memlet(data="s", subset="0"))
    st.add_edge(s_acc, None, b_acc, None, mm.Memlet(data="B", subset="0"))

    top = _top_with_nested(inner, out_conns=["B"])
    verify_no_nested_transients(top)


def test_no_nested_transients_raises_on_multi_element_nested():
    """A multi-element transient inside a nested SDFG is exactly what
    this check rejects."""
    inner = SDFG("inner_bad")
    inner.add_array("B", [8], dace.float64, transient=False)
    inner.add_array("t", [8], dace.float64, transient=True)
    st = inner.add_state()
    # Trivial state; the array just needs to exist.
    st.add_access("t")

    top = _top_with_nested(inner, out_conns=["B"])
    with pytest.raises(ValueError, match="declares transient 't'"):
        verify_no_nested_transients(top)


def test_no_nested_transients_reports_all_offenders():
    """The error message lists every offender, not just the first one."""
    inner = SDFG("inner_two_bad")
    inner.add_array("B", [8], dace.float64, transient=False)
    inner.add_array("t1", [8], dace.float64, transient=True)
    inner.add_array("t2", [4, 4], dace.float64, transient=True)
    st = inner.add_state()
    st.add_access("t1")
    st.add_access("t2")

    top = _top_with_nested(inner, out_conns=["B"])
    with pytest.raises(ValueError) as exc:
        verify_no_nested_transients(top)
    msg = str(exc.value)
    assert "2 offender(s)" in msg
    assert "'t1'" in msg
    assert "'t2'" in msg


def test_no_nested_transients_checks_deeper_levels():
    """A transient two levels deep (nested SDFG inside a nested SDFG)
    must still be flagged."""
    deep = SDFG("deep_with_transient")
    deep.add_array("B", [8], dace.float64, transient=False)
    deep.add_array("t", [8], dace.float64, transient=True)
    deep.add_state().add_access("t")

    mid = _top_with_nested(deep, out_conns=["B"])
    mid.name = "mid_wrap"
    top = _top_with_nested(mid, out_conns=["B"])
    with pytest.raises(ValueError, match="declares transient 't'"):
        verify_no_nested_transients(top)


if __name__ == "__main__":
    test_no_nested_transients_passes_when_clean()
    test_no_nested_transients_ignores_top_level_transient()
    test_no_nested_transients_allows_nested_scalar()
    test_no_nested_transients_raises_on_multi_element_nested()
    test_no_nested_transients_reports_all_offenders()
    test_no_nested_transients_checks_deeper_levels()
    print("all VerifyNoNestedTransients tests passed")

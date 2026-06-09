# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for :class:`ValidateWCRBoundary` (design section 3.5)."""
import pytest

import dace
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.validate_wcr_boundary import (ValidateWCRBoundary)


def test_clean_sdfg_passes():
    """An SDFG with no WCR memlets triggers no failure."""
    sdfg = dace.SDFG("clean")
    sdfg.add_array("A", (16, ), dace.float64, transient=False)
    sdfg.add_state("s")
    assert ValidateWCRBoundary().apply_pass(sdfg, {}) is None


def test_canonical_wcr_an_to_mapexit_scalar_passes():
    """WCR from a Scalar AN to a MapExit -- the locked shape -- passes."""
    sdfg = dace.SDFG("canonical_wcr")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    sdfg.add_scalar("acc", dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:16"})
    acc = state.add_access("acc")
    out = state.add_access("Out")
    tasklet = state.add_tasklet("body", set(), {"_o"}, "_o = 1.0")
    state.add_memlet_path(me, tasklet, memlet=Memlet())
    state.add_edge(tasklet, "_o", acc, None, Memlet("acc[0]"))
    state.add_edge(acc, None, mx, None, Memlet("acc[0]", wcr="lambda a, b: a + b"))
    state.add_memlet_path(mx, out, memlet=Memlet("Out[0:16]"))
    assert ValidateWCRBoundary().apply_pass(sdfg, {}) is None


def test_refuses_wcr_to_non_mapexit():
    """A WCR memlet whose destination is NOT a MapExit triggers refusal."""
    sdfg = dace.SDFG("wcr_to_an")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    sdfg.add_scalar("acc", dace.float64, transient=True)
    state = sdfg.add_state("s")
    acc = state.add_access("acc")
    out = state.add_access("Out")
    state.add_edge(acc, None, out, None, Memlet("Out[0]", wcr="lambda a, b: a + b"))
    with pytest.raises(NotImplementedError, match=r"non-MapExit destination"):
        ValidateWCRBoundary().apply_pass(sdfg, {})


def test_refuses_wcr_from_non_accessnode():
    """A WCR memlet whose source is NOT an AccessNode triggers refusal."""
    sdfg = dace.SDFG("wcr_from_tasklet")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:16"})
    out = state.add_access("Out")
    tasklet = state.add_tasklet("body", set(), {"_o"}, "_o = 1.0")
    state.add_memlet_path(me, tasklet, memlet=Memlet())
    # WCR memlet from Tasklet -> MapExit (illegal -- must originate from an AN).
    state.add_edge(tasklet, "_o", mx, None, Memlet("Out[ii]", wcr="lambda a, b: a + b"))
    state.add_memlet_path(mx, out, memlet=Memlet("Out[0:16]"))
    with pytest.raises(NotImplementedError, match=r"non-AccessNode source"):
        ValidateWCRBoundary().apply_pass(sdfg, {})


def test_refuses_wcr_from_multi_element_array():
    """A WCR memlet from a multi-element Array AN triggers refusal."""
    sdfg = dace.SDFG("wcr_multi_element")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    sdfg.add_array("acc_arr", (8, ), dace.float64, transient=True)  # multi-element
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:8"})
    acc = state.add_access("acc_arr")
    out = state.add_access("Out")
    tasklet = state.add_tasklet("body", set(), {"_o"}, "_o = 1.0")
    state.add_memlet_path(me, tasklet, memlet=Memlet())
    state.add_edge(tasklet, "_o", acc, None, Memlet("acc_arr[ii]"))
    state.add_edge(acc, None, mx, None, Memlet("acc_arr[0:8]", wcr="lambda a, b: a + b"))
    state.add_memlet_path(mx, out, memlet=Memlet("Out[0:16]"))
    with pytest.raises(NotImplementedError, match=r"single-element AccessNode"):
        ValidateWCRBoundary().apply_pass(sdfg, {})


def test_walks_into_nested_sdfgs():
    """A WCR violation inside a NestedSDFG is detected (recursive walk)."""
    sdfg = dace.SDFG("outer")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    inner = dace.SDFG("inner")
    inner.add_array("Out", (16, ), dace.float64, transient=False)
    inner.add_scalar("acc", dace.float64, transient=True)
    istate = inner.add_state("body")
    acc = istate.add_access("acc")
    out = istate.add_access("Out")
    # Illegal WCR inside the inner state.
    istate.add_edge(acc, None, out, None, Memlet("Out[0]", wcr="lambda a, b: a + b"))
    state.add_nested_sdfg(inner, {"Out"}, {"Out"})
    with pytest.raises(NotImplementedError, match=r"non-MapExit destination"):
        ValidateWCRBoundary().apply_pass(sdfg, {})


def test_accepts_length1_array_source():
    """A length-1 Array AN is treated as single-element."""
    sdfg = dace.SDFG("len1_array")
    sdfg.add_array("Out", (16, ), dace.float64, transient=False)
    sdfg.add_array("acc1", (1, ), dace.float64, transient=True)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:16"})
    acc = state.add_access("acc1")
    out = state.add_access("Out")
    tasklet = state.add_tasklet("body", set(), {"_o"}, "_o = 1.0")
    state.add_memlet_path(me, tasklet, memlet=Memlet())
    state.add_edge(tasklet, "_o", acc, None, Memlet("acc1[0]"))
    state.add_edge(acc, None, mx, None, Memlet("acc1[0]", wcr="lambda a, b: a + b"))
    state.add_memlet_path(mx, out, memlet=Memlet("Out[0:16]"))
    assert ValidateWCRBoundary().apply_pass(sdfg, {}) is None

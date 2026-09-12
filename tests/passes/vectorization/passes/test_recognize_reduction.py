# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``utils.reductions.recognize_reduction`` (R-2a).

``recognize_reduction`` is the robust replacement for the fragile
single-op ``_extract_single_op`` reduction detection: it accepts an
arbitrary right-hand expression (compound product, indirect gather —
the spmv / cloudsc shapes) as long as the accumulator connector is
read-modify-written under one of the associative ``IDENTITY`` ops.
These tests build the tasklet shapes imperatively (no JSON fallback)
and pin exactly what is and is not recognised.
"""
import dace
import pytest

from dace.transformation.passes.vectorization.utils.reductions import (
    ReductionInfo,
    recognize_reduction,
)


def _rmw_sdfg(code: str, *, extra_in: bool = False, lang=dace.dtypes.Language.Python):
    """One state: ``acc`` read + (optional ``b``) -> tasklet -> ``acc``."""
    sdfg = dace.SDFG("rmw")
    sdfg.add_scalar("acc", dace.float64, transient=True)
    sdfg.add_scalar("b", dace.float64, transient=True)
    st = sdfg.add_state()
    a_in = st.add_access("acc")
    a_out = st.add_access("acc")
    inconns = {"a"}
    if extra_in:
        inconns.add("b")
    t = st.add_tasklet("upd", inconns, {"o"}, code, language=lang)
    st.add_edge(a_in, None, t, "a", dace.Memlet("acc[0]"))
    if extra_in:
        b_in = st.add_access("b")
        st.add_edge(b_in, None, t, "b", dace.Memlet("b[0]"))
    st.add_edge(t, "o", a_out, None, dace.Memlet("acc[0]"))
    return sdfg, st, t


def _no_rmw_sdfg(code: str):
    """``acc`` is written but never read by the tasklet (not a reduction)."""
    sdfg = dace.SDFG("nrmw")
    sdfg.add_scalar("acc", dace.float64, transient=True)
    sdfg.add_scalar("a", dace.float64, transient=True)
    sdfg.add_scalar("b", dace.float64, transient=True)
    st = sdfg.add_state()
    t = st.add_tasklet("w", {"a", "b"}, {"o"}, code)
    st.add_edge(st.add_access("a"), None, t, "a", dace.Memlet("a[0]"))
    st.add_edge(st.add_access("b"), None, t, "b", dace.Memlet("b[0]"))
    st.add_edge(t, "o", st.add_access("acc"), None, dace.Memlet("acc[0]"))
    return sdfg, st, t


@pytest.mark.parametrize("code,op,identity", [
    ("o = a + b", "+", "0"),
    ("o = a * b", "*", "1"),
    ("o = a | b", "|", "0"),
    ("o = a & b", "&", "~0"),
    ("o = a ^ b", "^", "0"),
])
def test_recognizes_infix_rmw(code, op, identity):
    sdfg, st, t = _rmw_sdfg(code, extra_in=True)
    info = recognize_reduction(st, t)
    assert info == ReductionInfo(op=op, accumulator="acc", identity=identity)


@pytest.mark.parametrize("code,op", [("o = max(a, b)", "max"), ("o = min(b, a)", "min")])
def test_recognizes_funcall_rmw_either_arg_position(code, op):
    # Accumulator may be the first (max(a,b)) or second (min(b,a)) arg.
    sdfg, st, t = _rmw_sdfg(code, extra_in=True)
    info = recognize_reduction(st, t)
    assert info is not None and info.op == op and info.accumulator == "acc"


def test_recognizes_compound_rhs_when_accumulator_is_direct_operand():
    # spmv-shaped post-SplitTasklets form: ``acc = acc + prod`` where the
    # *other* operand is itself a product/gather temp. Still a '+' RMW.
    sdfg = dace.SDFG("cmp")
    sdfg.add_scalar("acc", dace.float64, transient=True)
    sdfg.add_scalar("prod", dace.float64, transient=True)
    st = sdfg.add_state()
    t = st.add_tasklet("upd", {"a", "p"}, {"o"}, "o = a + p")
    st.add_edge(st.add_access("acc"), None, t, "a", dace.Memlet("acc[0]"))
    st.add_edge(st.add_access("prod"), None, t, "p", dace.Memlet("prod[0]"))
    st.add_edge(t, "o", st.add_access("acc"), None, dace.Memlet("acc[0]"))
    info = recognize_reduction(st, t)
    assert info == ReductionInfo(op="+", accumulator="acc", identity="0")


def test_rejects_non_rmw():
    sdfg, st, t = _no_rmw_sdfg("o = a + b")
    assert recognize_reduction(st, t) is None


def test_rejects_non_associative_subtraction_and_division():
    for code in ("o = a - b", "o = a / b"):
        sdfg, st, t = _rmw_sdfg(code, extra_in=True)
        assert recognize_reduction(st, t) is None


def test_rejects_accumulator_not_a_direct_operand():
    # ``acc = sqrt(acc) + b``: top-level op is '+', operands are a Call
    # and ``b``; the accumulator connector ``a`` is nested inside the
    # call, not a direct operand -> conservatively rejected.
    sdfg, st, t = _rmw_sdfg("o = sqrt(a) + b", extra_in=True)
    assert recognize_reduction(st, t) is None


def test_rejects_non_python_language():
    sdfg, st, t = _rmw_sdfg("o = a + b;", extra_in=True, lang=dace.dtypes.Language.CPP)
    assert recognize_reduction(st, t) is None


def test_rejects_multi_statement_body():
    sdfg, st, t = _rmw_sdfg("tmp = b\no = a + tmp", extra_in=True)
    assert recognize_reduction(st, t) is None


def test_rejects_non_tasklet():
    sdfg, st, t = _rmw_sdfg("o = a + b", extra_in=True)
    an = next(n for n in st.nodes() if isinstance(n, dace.nodes.AccessNode))
    assert recognize_reduction(st, an) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the length-1-array <-> scalar conversion passes.

``ConvertLengthOneArraysToScalars`` rewrites every length-1 ``Array``
(shape ``(1,)``) to a true ``Scalar`` and strips the now-redundant
``[0]`` accessors.  ``ConvertScalarsToLengthOneArrays`` is the inverse.
These are pure-SDFG (no Fortran) tests of the Pass classes and the
underlying ``replace_length_one_arrays_with_scalars`` function.
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
    for cls in (ConvertLengthOneArraysToScalars, ConvertScalarsToLengthOneArrays):
        assert set(cls.__properties__) == {"recursive", "transient_only"}
        inst = cls(recursive=False, transient_only=True)
        assert inst.recursive is False
        assert inst.transient_only is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

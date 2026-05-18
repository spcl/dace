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
    replace_length_one_arrays_with_scalars,
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


def test_passes_expose_property_options():
    for cls in (ConvertLengthOneArraysToScalars, ConvertScalarsToLengthOneArrays):
        assert set(cls.__properties__) == {"recursive", "transient_only"}
        inst = cls(recursive=False, transient_only=True)
        assert inst.recursive is False
        assert inst.transient_only is True


def test_function_entrypoint_matches_pass():
    sdfg = _sdfg_with_len1(transient=False)
    rewritten = replace_length_one_arrays_with_scalars(sdfg)
    assert rewritten == {"a"}
    assert isinstance(sdfg.arrays["a"], dd.Scalar)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

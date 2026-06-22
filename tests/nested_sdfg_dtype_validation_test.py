# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that validate() rejects inner/outer array base-type mismatches across
    nested SDFG connectors, while allowing differences that share a base type
    (e.g. a vector view of a scalar). """
import pytest
import dace
from dace.sdfg.validation import InvalidSDFGError


def _build(inner_a_dtype: dace.dtypes.typeclass,
           outer_a_dtype: dace.dtypes.typeclass) -> dace.SDFG:
    N = 8
    inner = dace.SDFG("nsdfg_dtype_inner")
    inner.add_array("A", [N], inner_a_dtype)
    inner.add_array("B", [N], dace.float64)
    ist = inner.add_state()
    ist.add_mapped_tasklet("m",
                           dict(i=f"0:{N}"),
                           dict(x=dace.Memlet("A[i]")),
                           "y = x * 2.0",
                           dict(y=dace.Memlet("B[i]")),
                           external_edges=True)

    outer = dace.SDFG("nsdfg_dtype_outer")
    outer.add_array("outside_A", [N], outer_a_dtype)
    outer.add_array("outside_B", [N], dace.float64)
    ost = outer.add_state()
    nsdfg = ost.add_nested_sdfg(inner, {"A"}, {"B"})
    ost.add_edge(ost.add_read("outside_A"), None, nsdfg, "A", dace.Memlet(f"outside_A[0:{N}]"))
    ost.add_edge(nsdfg, "B", ost.add_write("outside_B"), None, dace.Memlet(f"outside_B[0:{N}]"))
    return outer


def test_nested_sdfg_base_type_mismatch_rejected():
    # inner A is f32, outer outside_A is f64 -> reinterpret of the buffer -> reject.
    sdfg = _build(dace.float32, dace.float64)
    with pytest.raises(InvalidSDFGError):
        sdfg.validate()


def test_nested_sdfg_matching_base_type_ok():
    sdfg = _build(dace.float64, dace.float64)
    sdfg.validate()  # must not raise


if __name__ == "__main__":
    test_nested_sdfg_base_type_mismatch_rejected()
    test_nested_sdfg_matching_base_type_ok()

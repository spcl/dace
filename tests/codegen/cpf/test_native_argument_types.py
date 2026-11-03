# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The C rewrite of a native body types each helper argument from the names in scope at its site.

A reduction over a column of a slice reaches ``reduce_seq_sum`` with an offset of an offset,
``(__rsrc0 + _loop_it_0) + _i0 * M`` (polybench correlation, vexx_k), and a loop-carried accumulator
with an element of an offset pointer, ``(__rdo0 + _loop_it_1)[0]`` (gem, kmeans). Both point into a
typed array, so both are typed by it; C refused them while C++ rendered.
"""

import pytest

from dace.cpf_lowering import NativeSite, c_native_array_dtype, c_native_value_dtype

SITE = NativeSite({"__rsrc0": ("float64", True), "__rdo0": ("float32", True), "M": ("int64", False)}, "site")


@pytest.mark.parametrize(
    "argument",
    ["__rsrc0", "__rsrc0 + _i0", "(__rsrc0 + _loop_it_0) + _i0 * M", "((__rsrc0 + _loop_it_0) + _i0 * M)"],
)
def test_an_offset_of_an_offset_is_typed_by_the_array_it_points_into(argument):
    assert c_native_array_dtype(argument, SITE, "reduce_seq_sum") == "float64"


@pytest.mark.parametrize("argument", ["__rdo0[0]", "(__rdo0 + _loop_it_1)[0]", "((__rdo0 + _loop_it_1) + 2)[_i0]"])
def test_an_element_of_an_offset_pointer_is_typed_by_the_array_it_points_into(argument):
    assert c_native_value_dtype(argument, SITE, "reduce_seq_sum") == "float32"


@pytest.mark.parametrize("argument", ["(unknown + _loop_it_0) + _i0 * M", "(M + 1) + 2"])
def test_an_offset_into_no_typed_array_still_refuses(argument):
    with pytest.raises(NotImplementedError):
        c_native_array_dtype(argument, SITE, "reduce_seq_sum")


def test_an_element_of_an_offset_into_no_typed_array_still_refuses():
    with pytest.raises(NotImplementedError):
        c_native_value_dtype("(unknown + _loop_it_1)[0]", SITE, "reduce_seq_sum")


if __name__ == "__main__":
    for argument in [
        "__rsrc0",
        "__rsrc0 + _i0",
        "(__rsrc0 + _loop_it_0) + _i0 * M",
        "((__rsrc0 + _loop_it_0) + _i0 * M)",
    ]:
        test_an_offset_of_an_offset_is_typed_by_the_array_it_points_into(argument)
    for argument in ["__rdo0[0]", "(__rdo0 + _loop_it_1)[0]", "((__rdo0 + _loop_it_1) + 2)[_i0]"]:
        test_an_element_of_an_offset_pointer_is_typed_by_the_array_it_points_into(argument)
    for argument in ["(unknown + _loop_it_0) + _i0 * M", "(M + 1) + 2"]:
        test_an_offset_into_no_typed_array_still_refuses(argument)
    test_an_element_of_an_offset_into_no_typed_array_still_refuses()

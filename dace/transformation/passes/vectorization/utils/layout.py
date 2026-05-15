# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout / stride helpers for the vectorization pipeline."""
from typing import Union

import dace


def assert_strides_are_packed_C_or_packed_Fortran(sdfg: dace.SDFG) -> Union[str, None]:
    """
    Verify that all arrays in an SDFG are packed in consistent C or Fortran order.

    Each array must have a unit stride in its first or last dimension.
    One-dimensional arrays are allowed in both and default to ``"F"``.

    :param sdfg: The SDFG whose arrays are checked.
    :returns: ``"C"`` or ``"F"`` indicating the stride ordering, or
        ``None`` if no arrays are found.
    :raises AssertionError: If an array lacks a unit stride in both first
        and last dimension.
    :raises ValueError: If arrays have mixed C and Fortran stride ordering.
    """
    stride_type = None
    has_one_d_arrays = False
    current_type = None
    for arr_name, desc in sdfg.arrays.items():
        if not isinstance(desc, dace.data.Array):
            continue

        has_unit_stride = desc.strides[0] == 1 or desc.strides[-1] == 1
        assert has_unit_stride, f"Array {arr_name} needs unit stride in first or last dimension: {desc.strides}"

        if len(desc.shape) == 1:
            has_one_d_arrays = True
        else:
            current_type = "F" if desc.strides[0] == 1 else "C"

        if stride_type is None:
            stride_type = current_type
        elif stride_type != current_type:
            raise ValueError("All arrays must have consistent stride ordering (all F or all C)")

    if has_one_d_arrays and stride_type is None:
        stride_type = "F"

    return stride_type

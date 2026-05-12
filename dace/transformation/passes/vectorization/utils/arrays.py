# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Array-descriptor mutation helpers.

The three helpers in this module add or rebuild ``dace.data.Array``
descriptors on an SDFG. They are kept as separate functions
(rather than merged into one) deliberately:
``replace_arrays_with_new_shape`` is destructive (remove-then-readd),
``copy_arrays_with_a_new_shape`` is additive (new name, original kept),
``add_transient_arrays_from_list`` is bulk-add for arbitrary specs.
Callers rely on the distinct semantics.
"""
from typing import Any, Iterable, Set, Tuple

import dace
from dace import typeclass


def replace_arrays_with_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                  new_type: typeclass) -> None:
    """
    Replaces existing arrays in an SDFG with new shapes (and optionally a new dtype).

    Args:
        sdfg: The SDFG containing the arrays.
        array_namelist: Set of array names to replace.
        new_shape: The new shape for the arrays.
        new_type: Optional new data type for arrays.
    """
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.remove_data(arr_name, validate=False)
        sdfg.add_array(name=arr_name,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype if new_type is None else new_type,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)


def copy_arrays_with_a_new_shape(sdfg: dace.SDFG, array_namelist: Set[str], new_shape: Tuple[Any],
                                 name_suffix: str) -> None:
    """
    Creates copies of existing arrays with a new shape and a name suffix.

    Args:
        sdfg: The SDFG containing the arrays.
        array_namelist: Set of array names to copy.
        new_shape: Shape of the new arrays.
        name_suffix: Suffix to append to new array names.
    """
    for arr_name in array_namelist:
        arr = sdfg.arrays[arr_name]
        sdfg.add_array(name=arr_name + name_suffix,
                       shape=new_shape,
                       storage=arr.storage,
                       dtype=arr.dtype,
                       location=arr.location,
                       transient=arr.transient,
                       lifetime=arr.lifetime,
                       debuginfo=arr.debuginfo,
                       allow_conflicts=arr.allow_conflicts,
                       find_new_name=False,
                       alignment=arr.alignment,
                       may_alias=arr.may_alias)


def add_transient_arrays_from_list(sdfg: dace.SDFG, arr_name_shape_storage_dtype: Iterable[Tuple[str, Any, Any,
                                                                                                 Any]]) -> None:
    """
    Adds transient arrays to an SDFG given a list of (name, shape, storage, dtype) tuples.

    Args:
        sdfg: The SDFG to modify.
        arr_name_shape_storage_dtype: Iterable of array specifications.
    """

    for arr_name, shape, storage, dtype in arr_name_shape_storage_dtype:
        sdfg.add_array(
            name=arr_name,
            shape=shape,
            storage=storage,
            dtype=dtype,
            transient=True,
            find_new_name=False,
        )

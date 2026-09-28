# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from ordered_set import OrderedSet

from dace import data, dtypes
from dace.sdfg import SDFG
from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars,
                                                                           ConvertScalarsToLengthOneArrays)
import dace.transformation.passes.offloading.offloading_helpers as helpers


def change_single_element_containers(sdfg: SDFG, exceptions: OrderedSet[str]) -> OrderedSet[str]:
    """Device-written scalars become length-1 arrays (a kernel takes a scalar by value), the other length-1
    arrays scalars; ``exceptions`` are not asked again. Return the names asked for."""
    gpu_written = helpers.data_written_by_device_code(sdfg)
    to_len1_arrays = OrderedSet(
        name for name in sdfg.arrays
        if isinstance(sdfg.arrays[name], data.Scalar) and name in gpu_written and name not in exceptions)
    # ``__return`` stays by reference: the caller reads the result back through it.
    to_scalars = OrderedSet(name for name in sdfg.arrays if helpers.is_length1_array(name, sdfg)
                            and name not in gpu_written and name not in exceptions and not name.startswith("__return"))

    if to_len1_arrays:
        ConvertScalarsToLengthOneArrays(recursive=True, preserve_abi=True, filter=to_len1_arrays).apply_pass(sdfg, {})
        for name in to_len1_arrays:  # allocated once, not in every iteration of a busy loop
            sdfg.arrays[name].lifetime = dtypes.AllocationLifetime.SDFG
    if to_scalars:
        before = OrderedSet(sdfg.arrays)
        ConvertLengthOneArraysToScalars(recursive=True, preserve_abi=True, filter=to_scalars).apply_pass(sdfg, {})
        keep_on_host(sdfg, OrderedSet(sdfg.arrays) - before)
    return to_scalars | to_len1_arrays


def keep_on_host(sdfg: SDFG, staged: OrderedSet[str]) -> None:
    """A staged scalar inherits its array's storage, but no kernel writes it, so its readers are the host and
    kernels taking it by value: it lives on the host."""
    for name in staged:
        if isinstance(sdfg.arrays[name], data.Scalar) and helpers.is_array_stored_on_GPU(sdfg, name):
            sdfg.arrays[name].storage = dtypes.StorageType.Default

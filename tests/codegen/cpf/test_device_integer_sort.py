# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A host-level ``IntegerSort`` over device memory sorts on the device, through CUB's radix sort.

Without a device entry the node took its host ``isocpp`` lowering, a ``std::sort`` over the device pointer."""

import ctypes
import pathlib
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.libraries.sort.nodes.integer_sort import IntegerSort
from tests.codegen.cpf.conftest import entry_argtypes, require_gpu_backend
from tests.codegen.cpf.test_emission_cuda import nvcc_library


def device_sort_sdfg(name: str, dtype: dace.typeclass) -> dace.SDFG:
    """``out = sorted(keys)`` by an ``IntegerSort`` at host level, both arrays in device memory."""
    sdfg = dace.SDFG(name)
    for array in ("keys", "out"):
        sdfg.add_array(array, [dace.symbol("N")], dtype, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = IntegerSort("sort")
    state.add_node(node)
    state.add_edge(state.add_read("keys"), None, node, "_keys_in", dace.Memlet("keys[0:N]"))
    state.add_edge(node, "_keys_out", state.add_write("out"), None, dace.Memlet("out[0:N]"))
    return sdfg


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_device_integer_sort_is_a_cub_radix_sort(language: str):
    rendering = render(device_sort_sdfg(f"cpf_device_sort_{language}", dace.int64), language=language)
    assert "std::sort" not in rendering.code, "the sort must not run on the host over the device keys"
    assert "gpucub::DeviceRadixSort::SortKeys(" in rendering.device_code, rendering.device_code
    assert "get_scratch<SortTag>(" in rendering.device_code, rendering.device_code


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.int64, np.int32], ids=["int64", "int32"])
def test_a_device_integer_sort_sorts_signed_keys_ascending(dtype: type):
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = f"cpf_device_sort_run_{dtype.__name__}"
    rendering = render(device_sort_sdfg(name, dace.dtype_to_typeclass(dtype)), language="cuda")
    length = 100003  # not a multiple of any tile size, negative and repeated keys included
    keys = np.random.default_rng(1).integers(-(10**6), 10**6, length).astype(dtype)
    device = {"keys": cupy.asarray(keys), "out": cupy.zeros(length, dtype=dtype)}
    with tempfile.TemporaryDirectory() as directory:
        entry = ctypes.CDLL(str(nvcc_library(rendering, pathlib.Path(directory), name)))[name]
        entry.argtypes = entry_argtypes(rendering.sdfg)
        entry(*[ctypes.c_void_p(device[a].data.ptr) if a in device else length for a in rendering.sdfg.arglist()])
    np.testing.assert_array_equal(cupy.asnumpy(device["out"]), np.sort(keys))
    np.testing.assert_array_equal(cupy.asnumpy(device["keys"]), keys)


if __name__ == "__main__":
    test_a_device_integer_sort_is_a_cub_radix_sort("cuda")
    test_a_device_integer_sort_is_a_cub_radix_sort("hip")
    test_a_device_integer_sort_sorts_signed_keys_ascending(np.int64)
    test_a_device_integer_sort_sorts_signed_keys_ascending(np.int32)

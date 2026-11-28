# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A per-iteration private buffer of symbolic size: a stack VLA on the CPU, one global slab on the GPU.

A GPU thread has no variable-length array, and a per-thread ``new``/``delete`` in a kernel is a device heap call per
thread. The buffer of each GPU thread is one slice of a global buffer sized threads x largest size, allocated before
the launch."""

import ctypes
import pathlib
import re
import tempfile

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from tests.codegen.cpf.conftest import canonical_gpu_sdfg, entry_argtypes, render_gpu, require_gpu_backend
from tests.codegen.cpf.test_emission_cuda import nvcc_library
from tests.codegen.cpf.test_loop_comment_fixes_2 import bench_c

N = dace.symbol("N")


@dace.program
def triangular_rows(a: dace.float64[N, N], idx: dace.int64[N], c: dace.float64[N, N]):
    for i in dace.map[0:N]:
        tmp = np.empty(i + 1)
        for j in range(i + 1):
            tmp[j] = a[i, j] * 2.0
        for j in range(i + 1):
            c[i, j] = tmp[idx[j] % (i + 1)]


def assert_one_global_slab(host: str, device: str) -> None:
    """``tmp`` is no allocation of the kernel: the host allocates ``N * N`` once, each thread indexes its slice."""
    assert not re.search(r"= new\b|operator delete|double tmp\[", device), device
    assert re.search(r"cudaMalloc\(\(void\*\*\)&__cpf_persistent_tmp, ipow\(N, 2\) \* sizeof\(double\)\)", host), host
    assert "tmp[tmp_idx(i, j, N)] = " in device, device


def test_a_cpu_buffer_sized_by_the_iteration_is_a_stack_array():
    code = bench_c(triangular_rows, "cpf_private_rows_c")
    assert "double tmp[cpf_max_int64(1, (i + 1))];" in code, code
    assert not re.search(r"\b(malloc|free|aligned_alloc)\(", code), code


def test_a_gpu_buffer_sized_by_the_thread_is_one_global_slab():
    """The kernel did ``new double[tmp_size(i)]`` and ``operator delete[]`` in every thread."""
    rendering = render_gpu(triangular_rows, "cpf_private_rows_gpu", "cuda")
    assert_one_global_slab(rendering.code, rendering.device_code)


def test_a_forced_register_buffer_on_the_gpu_is_one_global_slab_not_a_vla():
    """``Register(force=True)``, a VLA on the CPU, was a per-thread VLA in the kernel, which nvcc rejects."""
    sdfg = canonical_gpu_sdfg(triangular_rows, "cpf_private_rows_forced")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    forced = [desc for sd in sdfg.all_sdfgs_recursive() for name, desc in sd.arrays.items() if name == "tmp"]
    for desc in forced:
        desc.storage = dtypes.StorageType.Register(force=True)
    rendering = render(sdfg, language="cuda")
    assert_one_global_slab(rendering.code, rendering.device_code)


@pytest.mark.gpu
def test_a_gpu_buffer_sized_by_the_thread_computes_what_numpy_computes():
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_private_rows_run"
    rendering = render_gpu(triangular_rows, name, "cuda")
    n = 300  # threads of two blocks, the last one partly masked
    generator = np.random.default_rng(3)
    host = {"a": generator.random((n, n)), "idx": generator.integers(0, 1000, n), "c": np.zeros((n, n))}
    expected = host["c"].copy()
    for i in range(n):
        expected[i, : i + 1] = 2.0 * host["a"][i, host["idx"][: i + 1] % (i + 1)]
    device = {arg: cupy.asarray(values) for arg, values in host.items()}
    with tempfile.TemporaryDirectory() as directory:
        entry = ctypes.CDLL(str(nvcc_library(rendering, pathlib.Path(directory), name)))[name]
        entry.argtypes = entry_argtypes(rendering.sdfg)
        entry(*[ctypes.c_void_p(device[arg].data.ptr) if arg in device else n for arg in rendering.sdfg.arglist()])
    np.testing.assert_array_equal(cupy.asnumpy(device["c"]), expected)


if __name__ == "__main__":
    test_a_cpu_buffer_sized_by_the_iteration_is_a_stack_array()
    test_a_gpu_buffer_sized_by_the_thread_is_one_global_slab()
    test_a_forced_register_buffer_on_the_gpu_is_one_global_slab_not_a_vla()
    test_a_gpu_buffer_sized_by_the_thread_computes_what_numpy_computes()

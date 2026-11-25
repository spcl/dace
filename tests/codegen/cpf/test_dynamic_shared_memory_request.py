# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A kernel with more shared memory than devices grant by default opts in through CPF's own helper, not DaCe's macro
(``seidel_2d`` on CUDA rendered ``DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY`` and failed the self-contained gate)."""

import ctypes
import pathlib
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from tests.codegen.cpf.conftest import assert_standalone_units, entry_argtypes, require_gpu_backend
from tests.codegen.cpf.test_emission_cuda import nvcc_library

N = dace.symbol("N")
S = dace.StorageType
KIB40 = 5120  # 40 KiB of doubles: two of them exceed the 48 KiB of static shared memory


@dace.program
def two_large_shared(A: dace.float64[N] @ S.GPU_Global, B: dace.float64[N] @ S.GPU_Global):
    for i in dace.map[0:N:32] @ dace.ScheduleType.GPU_Device:
        s1 = dace.define_local([KIB40], dace.float64, storage=S.GPU_Shared)
        s2 = dace.define_local([KIB40], dace.float64, storage=S.GPU_Shared)
        for j in dace.map[0:32] @ dace.ScheduleType.GPU_ThreadBlock:
            s1[j] = A[i + j]
            s2[j] = s1[j] * 2
            B[i + j] = s2[j] + 1


def render_two_large_shared(name: str):
    sdfg = two_large_shared.to_sdfg(simplify=False)
    sdfg.name = name
    return render(sdfg, language="cuda")


def test_a_dynamic_shared_memory_request_renders_self_contained():
    rendering = render_two_large_shared("cpf_dynsmem_request")

    assert_standalone_units(rendering, "cpf_dynsmem_request")
    assert "cpf_request_dynamic_shared_memory(cpf_dynsmem_request_" in rendering.device_code
    assert "cudaFuncSetAttribute(" in rendering.device_code


@pytest.mark.gpu
def test_a_dynamic_shared_memory_request_runs():
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_dynsmem_request_run"
    rendering = render_two_large_shared(name)
    with tempfile.TemporaryDirectory() as directory:
        entry = ctypes.CDLL(str(nvcc_library(rendering, pathlib.Path(directory), name)))[name]
        entry.argtypes = entry_argtypes(rendering.sdfg)
        length = 1024
        A = cupy.arange(length, dtype=cupy.float64)
        B = cupy.zeros(length, dtype=cupy.float64)
        arguments = {"A": ctypes.c_void_p(A.data.ptr), "B": ctypes.c_void_p(B.data.ptr), "N": length}
        entry(*(arguments[arg] for arg in rendering.sdfg.arglist()))
        np.testing.assert_array_equal(cupy.asnumpy(B), np.arange(length, dtype=np.float64) * 2 + 1)


if __name__ == "__main__":
    test_a_dynamic_shared_memory_request_renders_self_contained()
    test_a_dynamic_shared_memory_request_runs()

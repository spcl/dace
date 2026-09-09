# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests related to try_parse_state_struct
"""
import ctypes
import os

import pytest
import numpy as np

import dace
import dace.library
from dace import dtypes
from dace.codegen import codeobject, targets, compiler, compiled_sdfg, common


@pytest.fixture
def cuda_helper():
    return _cuda_helper()


def _cuda_helper():

    helper_code = f"""
    #include <dace/dace.h>

    extern "C" {{
        DACE_EXPORTED int host_to_gpu(void* gpu, void* host, size_t size) {{
            auto result = {common.get_gpu_backend()}Memcpy(gpu, host, size, {common.get_gpu_backend()}MemcpyHostToDevice);
            return result;
        }}
    }}
    """
    program = codeobject.CodeObject("cuda_helper", helper_code, "cpp", targets.cpu.CPUCodeGen, "CudaHelper")

    # Same naming the CUDA target itself uses (targets/cuda.py: language/target_type): a HIP build
    # emits .cpp under src/cuda/hip/, and it is that layout which makes CMake enable the HIP language
    # and define WITH_HIP. Hardcoding "cu" builds the helper as CUDA, so dace.h takes its CUDA branch
    # while the code above calls hipMemcpy -- "hipMemcpyHostToDevice was not declared in this scope".
    backend = common.get_gpu_backend()
    dummy_cuda_target = codeobject.CodeObject("dummy",
                                              "",
                                              "cu" if backend == 'cuda' else "cpp",
                                              targets.cuda.CUDACodeGen,
                                              "CudaDummy",
                                              target_type="" if backend == 'cuda' else backend)

    build_folder = dace.Config.get('default_build_folder')
    BUILD_PATH = os.path.join(build_folder, "cuda_helper")
    compiler.generate_program_folder(None, [program, dummy_cuda_target], BUILD_PATH)
    compiler.configure_and_compile(BUILD_PATH)

    checker_dll = compiled_sdfg.ReloadableDLL(compiler.get_binary_name(BUILD_PATH, "cuda_helper"))

    class CudaHelper:

        def __init__(self):
            self.dll = checker_dll
            checker_dll.load()

            self._host_to_gpu = checker_dll.get_symbol("host_to_gpu")
            self._host_to_gpu.restype = ctypes.c_int

        def __del__(self):
            self.dll.unload()

        def host_to_gpu(self, gpu_ptr: int, numpy_array: np.ndarray):
            size = ctypes.sizeof(dtypes._FFI_CTYPES[numpy_array.dtype.type]) * numpy_array.size
            result = ctypes.c_int(
                self._host_to_gpu(ctypes.c_void_p(gpu_ptr), ctypes.c_void_p(numpy_array.__array_interface__["data"][0]),
                                  ctypes.c_size_t(size)))
            if result.value != 0:
                raise ValueError("host_to_gpu returned nonzero result!")

    return CudaHelper()


@pytest.mark.gpu
def test_preallocate_transients_in_state_struct(cuda_helper):

    @dace.program
    def persistent_transient(A: dace.float32[3, 3]):
        persistent_transient = dace.define_local([3, 5],
                                                 dace.float32,
                                                 lifetime=dace.AllocationLifetime.Persistent,
                                                 storage=dace.StorageType.GPU_Global)
        return A @ persistent_transient

    sdfg: dace.SDFG = persistent_transient.to_sdfg()
    sdfg.apply_gpu_transformations()

    A = np.random.randn(3, 3).astype(np.float32)
    B = np.random.randn(3, 5).astype(np.float32)
    compiledsdfg = sdfg.compile()
    compiledsdfg._initialize(tuple())

    state_struct = compiledsdfg.get_state_struct()

    # copy the B array into the transient ptr
    ptr = getattr(state_struct, f'__{sdfg.cfg_id}_persistent_transient')
    cuda_helper.host_to_gpu(ptr, B.copy())
    result = np.zeros_like(B)
    compiledsdfg(A=A, __return=result)

    assert np.allclose(result, A @ B)


if __name__ == '__main__':
    test_preallocate_transients_in_state_struct(_cuda_helper())

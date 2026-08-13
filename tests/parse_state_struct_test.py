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

    dummy_cuda_target = codeobject.CodeObject("dummy", "", "cu", targets.cuda.CUDACodeGen, "CudaDummy")

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
    # The public initialize() works on both interfaces; the ctypes-internal
    # _initialize(tuple()) does not exist on the nanobind interface.
    compiledsdfg.initialize(A=A)

    state_struct = compiledsdfg.get_state_struct()

    # copy the B array into the transient ptr
    ptr = getattr(state_struct, f'__{sdfg.cfg_id}_persistent_transient')
    cuda_helper.host_to_gpu(ptr, B.copy())
    # Take the returned array instead of passing a __return buffer: explicit
    # return buffers are refused on the nanobind interface unless the module
    # was compiled with compiler.nanobind_allow_return_override.
    result = compiledsdfg(A=A)

    assert np.allclose(result, A @ B)


def test_get_state_struct_refused_in_production_folder_mode(monkeypatch):
    """The ctypes ``get_state_struct`` recovers the layout by parsing the generated
    ``src/cpu/<name>.cpp``, which ``production`` folder mode trims away (it also places
    the library directly in the build folder instead of in ``build/``, so the source path
    would not even resolve). It must refuse explicitly instead of failing on a missing file.
    """
    from dace.config import set_temporary

    # The env vars override set_temporary, so both pins must be dropped first.
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    monkeypatch.delenv('DACE_compiler_build_folder_mode', raising=False)

    @dace.program
    def prod_state_struct_probe(A: dace.float64[8]):
        A += 1.0

    # A DISTINCT program for the positive control: the build folder is derived from the
    # program name, and an existing folder's FOLDER_MODE marker overrides the config - so
    # reusing the name would run the development half inside the production folder.
    @dace.program
    def dev_state_struct_probe(A: dace.float64[8]):
        A += 1.0

    with set_temporary('compiler', 'interface', value='ctypes'), \
            set_temporary('compiler', 'build_folder_mode', value='production'):
        csdfg = prod_state_struct_probe.to_sdfg().compile()
        # Resolved at construction from the folder's FOLDER_MODE marker, not from the config.
        assert csdfg.build_folder_mode == 'production'
        csdfg.initialize(A=np.zeros(8))
        with pytest.raises(NotImplementedError, match='production'):
            csdfg.get_state_struct()

    # Development mode keeps the sources, so the same query works there.
    with set_temporary('compiler', 'interface', value='ctypes'), \
            set_temporary('compiler', 'build_folder_mode', value='development'):
        csdfg = dev_state_struct_probe.to_sdfg().compile()
        assert csdfg.build_folder_mode == 'development'
        csdfg.initialize(A=np.zeros(8))
        assert isinstance(csdfg.get_state_struct(), ctypes.Structure)


if __name__ == '__main__':
    test_preallocate_transients_in_state_struct(_cuda_helper())
    test_get_state_struct_refused_in_production_folder_mode(pytest.MonkeyPatch())

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the NVIDIA cuTENSOR backend."""
import dace.library
from dace.codegen import common


@dace.library.environment
class cuTensor:
    """Build/link configuration and per-node setup code for cuTENSOR-backed library nodes."""

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cutensor"]
    cmake_compile_flags = []
    cmake_link_flags = ["-L -lcutensor"]
    cmake_files = []

    headers = {'frame': ["dace/dace_cutensor.h"], 'cuda': ["dace/dace_cutensor.h"]}
    state_fields = ["dace::linalg::CuTensorHandle cutensor_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    # cuTENSOR v2 type mapping: dtype -> (tensor data type, compute descriptor,
    # C scalar type for alpha/beta). The scalar type follows the compute
    # descriptor (real even when the tensors are complex). Only float/complex
    # are listed: integer descriptors are accepted by the API but crash at
    # execution, and reinterpreting ints as floats corrupts negatives (their
    # bit patterns are NaN, which the GPU canonicalizes on multiply). Callers
    # fall back to the pure expansion for unsupported dtypes.
    TYPE_MAP = {
        dace.float16: ('CUTENSOR_R_16F', 'CUTENSOR_COMPUTE_DESC_16F', '__half'),
        dace.float32: ('CUTENSOR_R_32F', 'CUTENSOR_COMPUTE_DESC_32F', 'float'),
        dace.float64: ('CUTENSOR_R_64F', 'CUTENSOR_COMPUTE_DESC_64F', 'double'),
        dace.complex64: ('CUTENSOR_C_32F', 'CUTENSOR_COMPUTE_DESC_32F', 'float'),
        dace.complex128: ('CUTENSOR_C_64F', 'CUTENSOR_COMPUTE_DESC_64F', 'double'),
    }

    @staticmethod
    def is_available() -> bool:
        """Whether a build on this machine can reach cuTENSOR at all.

        It is NVIDIA's library, but the question is the platform rather than the backend: HIP's
        NVIDIA platform is the CUDA toolkit underneath and links it like any CUDA build. An AMD
        build has hipTensor instead, which DaCe does not wrap.
        """
        return common.get_gpu_backend() != 'hip' or common.get_hip_platform() == 'nvidia'

    @staticmethod
    def handle_setup_code(node):
        return dace.library.reject_gpu_location(node) + """\
cutensorHandle_t &__dace_cutensor_handle = __state->cutensor_handle.Get();
// cutensorSetStream(__dace_cutensor_handle, __dace_current_stream);\n"""

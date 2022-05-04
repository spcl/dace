# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class cuTensor:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cutensor"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_cutensor.h"], 'cuda': ["../include/dace_cutensor.h"]}
    state_fields = ["dace::linalg::CuTensorHandle cutensor_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "gpu" not in node.location:
            location = 0
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};
cutensorHandle_t &__dace_tensor_handle = __state->cutensor_handle.Get(__dace_cuda_device);
// cutensorSetStream(__dace_tensor_handle, __dace_current_stream);\n"""

        return code.format(location=location)

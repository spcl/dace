# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class cuSPARSE:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cusparse"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_cusparse.h"], 'cuda': ["../include/dace_cusparse.h"]}
    state_fields = ["dace::sparse::CusparseHandle cusparse_handle;"]
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
cusparseHandle_t &__dace_cusparse_handle = __state->cusparse_handle.Get(__dace_cuda_device);
cusparseSetStream(__dace_cusparse_handle, __dace_current_stream);\n"""

        return code.format(location=location)

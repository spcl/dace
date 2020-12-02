# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class cuBLAS:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cublas"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["../include/dace_cublas.h"]
    init_code = ""
    finalize_code = ""

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "cuda_device" not in node.location:
            location = 0
        else:
            try:
                location = int(location["cuda_device"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};
auto &__dace_cublas_handle = dace::blas::CublasHandle::Get(__dace_cuda_device);
cublasSetStream(__dace_cublas_handle, __dace_current_stream);\n"""

        return code.format(location=location)

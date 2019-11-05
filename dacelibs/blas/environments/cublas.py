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
        if not location:
            location = 0
        else:
            try:
                location = int(location)
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
auto &handle = dacelib::blas::CublasHelper<>::Get().Handle({location});
cublasSetStream(handle, __dace_current_stream);\n"""

        return code.format(location=location)

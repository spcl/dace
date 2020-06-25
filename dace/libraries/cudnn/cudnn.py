import dace.library


@dace.library.environment
class cuDNN:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cudnn"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = []
    init_code = ""
    finalize_code = ""
import dace.library
import os

def _find_cuda_include():
    if 'CUDADIR' in os.environ:
        return [os.path.join(os.environ['CUDADIR'], 'include')]
    else:
        return []

@dace.library.environment
class cuDNN:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = _find_cuda_include()
    cmake_libraries = ["cudnn"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["cudnn.h"]
    init_code ='''
    '''
    finalize_code = ""
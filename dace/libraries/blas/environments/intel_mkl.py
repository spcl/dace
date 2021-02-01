# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings


def _find_mkl_include():
    if 'MKLROOT' in os.environ:
        return [os.path.join(os.environ['MKLROOT'], 'include')]
    # Anaconda
    elif 'CONDA_PREFIX' in os.environ:
        base_path = os.environ['CONDA_PREFIX']
        # Anaconda on Windows
        candpath = os.path.join(base_path, 'Library', 'include')
        if os.path.isfile(os.path.join(candpath, 'mkl.h')):
            return [candpath]
        # Anaconda on other platforms
        candpath = os.path.join(base_path, 'include')
        if os.path.isfile(os.path.join(candpath, 'mkl.h')):
            return [candpath]
        warnings.warn(
            'Anaconda Python is installed but the MKL include directory cannot '
            'be found. Please install MKL includes with '
            '"conda install mkl-include" or set the MKLROOT environment '
            'variable')
        return []
    else:
        return []


def _find_mkl_lib():
    if 'MKLROOT' in os.environ:
        prefix = Config.get('compiler', 'linker', 'library_prefix')
        suffix = Config.get('compiler', 'linker', 'library_suffix')
        libfile = os.path.join(os.environ['MKLROOT'], 'lib',
                               prefix + 'mkl_rt' + suffix)
        if os.path.isfile(libfile):
            return [libfile]

    path = ctypes.util.find_library('mkl_rt')
    if path:
        # Attempt to link on Windows
        if path.endswith('.dll'):
            libfile = os.path.join(os.path.dirname(os.path.abspath(path)), '..',
                                   'lib', 'mkl_rt.lib')
            if os.path.isfile(libfile):
                return [libfile]
            elif 'CONDA_PREFIX' in os.environ:
                warnings.warn(
                    'Anaconda Python is installed but the MKL library file '
                    'cannot be found for linkage. Please install libraries with '
                    '"conda install mkl-devel" or set the MKLROOT environment '
                    'variable')
                return []
            else:
                return []

        return [path]

    # If all else fails, let CMake find the library
    return ['mkl_rt']


@dace.library.environment
class IntelMKL:

    cmake_minimum_version = None
    cmake_packages = []  #["BLAS"]
    cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
    cmake_includes = _find_mkl_include()
    cmake_libraries = _find_mkl_lib()
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["mkl.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

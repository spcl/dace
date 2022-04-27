# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings


@dace.library.environment
class IntelMKL:
    """ 
    An environment for the Intel Math Kernel Library (MKL), which implements
    the BLAS library and other functions.
    """

    cmake_minimum_version = None
    cmake_packages = []  #["BLAS"]
    cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["mkl.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_includes():
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
            warnings.warn('Anaconda Python is installed but the MKL include directory cannot '
                          'be found. Please install MKL includes with '
                          '"conda install mkl-include" or set the MKLROOT environment '
                          'variable')
            return []
        else:
            return []

    @staticmethod
    def cmake_libraries():
        if 'MKLROOT' in os.environ:
            prefix = Config.get('compiler', 'library_prefix')
            suffix = Config.get('compiler', 'library_extension')
            libfile = os.path.join(os.environ['MKLROOT'], 'lib', prefix + 'mkl_rt.' + suffix)
            if os.path.isfile(libfile):
                return [libfile]
            # Try with ${MKLROOT}/lib/intel64 (oneAPI on Linux)
            libfile = os.path.join(os.environ['MKLROOT'], 'lib', 'intel64', prefix + 'mkl_rt.' + suffix)
            if os.path.isfile(libfile):
                return [libfile]

        path = ctypes.util.find_library('mkl_rt')
        if not path:
            path = ctypes.util.find_library('mkl_rt.1')
        if path:
            # Attempt to link on Windows
            if path.endswith('.dll'):
                libfile = os.path.join(os.path.dirname(os.path.abspath(path)), '..', 'lib', 'mkl_rt.lib')
                if os.path.isfile(libfile):
                    return [libfile]
                elif 'CONDA_PREFIX' in os.environ:
                    warnings.warn('Anaconda Python is installed but the MKL library file '
                                  'cannot be found for linkage. Please install libraries with '
                                  '"conda install mkl-devel" or set the MKLROOT environment '
                                  'variable')
                    return []
                else:
                    return []

            return [path]

        # If all else fails, let CMake find the library
        return ['mkl_rt']

    @staticmethod
    def is_installed():
        return IntelMKL.cmake_libraries() and IntelMKL.cmake_includes()

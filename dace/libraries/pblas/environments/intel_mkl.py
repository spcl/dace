# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings


@dace.library.environment
class IntelMKLScaLAPACK:
    """ 
    An environment for the Intel Math Kernel Library (MKL), which implements
    the PBLAS library.
    """

    cmake_minimum_version = None
    cmake_packages = []  #["BLAS"]
    cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
    cmake_compile_flags = []
    cmake_link_flags = ["-L /lib/x86_64-linux-gnu -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmpich -lpthread -lm -ldl"]
    cmake_files = []

    headers = ["mkl_scalapack.h", "mkl_blacs.h", "mkl_pblas.h", "../include/dace_blas.h"]
    state_fields = [
        "MKL_INT __mkl_scalapack_context;",
        "MKL_INT __mkl_scalapack_rank, __mkl_scalapack_size;",
        "MKL_INT __mkl_scalapack_prows = 0, __mkl_scalapack_pcols = 0;",
        "MKL_INT __mkl_scalapack_myprow = 0, __mkl_scalapack_mypcol = 0;",
        "MKL_INT __mkl_int_zero = 0, __mkl_int_one = 1, __mkl_int_negone = -1;",
        "bool __mkl_scalapack_grid_init = false;"
    ]
    init_code = """
    blacs_pinfo(&__state->__mkl_scalapack_rank, &__state->__mkl_scalapack_size);
    blacs_get(&__state->__mkl_int_negone, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context);
    """
    finalize_code = """
    if (__state->__mkl_scalapack_grid_init) {{
        blacs_gridexit(&__state->__mkl_scalapack_context);
    }}
    // blacs_exit(&__state->__mkl_int_zero);
    """ # actually if we finalize in the dace program we break pytest :)
    dependencies = []

    libraries = ["mkl_scalapack_lp64", "mkl_blacs_intelmpi_lp64",
                 "mkl_intel_lp64", "mkl_sequential", "mkl_core", "mkl_avx2"]

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
            warnings.warn(
                'Anaconda Python is installed but the MKL include directory cannot '
                'be found. Please install MKL includes with '
                '"conda install mkl-include" or set the MKLROOT environment '
                'variable')
            return []
        else:
            return []

    @staticmethod
    def cmake_libraries():
        libfiles = ["/lib/x86_64-linux-gnu/lib" + name + ".so"
                    for name in IntelMKLScaLAPACK.libraries + ["mpich"]]
        return libfiles
        if 'MKLROOT' in os.environ:
            prefix = Config.get('compiler', 'library_prefix')
            suffix = Config.get('compiler', 'library_extension')
            libfiles = [os.path.join(os.environ['MKLROOT'], 'lib',
                        prefix + name + "." + suffix)
                        for name in IntelMKLScaLAPACK.libraries]
            if all([os.path.isfile(f) for f in libfiles]):
                return libfiles + ["/lib/x86_64-linux-gnu/libmpich.so"]

        path = ctypes.util.find_library('mkl_scalapack_lp64')
        if path:
            # Attempt to link on Windows
            if path.endswith('.dll'):
                libfiles = [os.path.join(os.path.dirname(os.path.abspath(path)),
                            '..', 'lib', name + '.lib')
                            for name in IntelMKLScaLAPACK.libraries]
                if all([os.path.isfile(f) for f in libfiles]):
                    return libfiles
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
        return IntelMKLScaLAPACK.libraries + ["mpich"]

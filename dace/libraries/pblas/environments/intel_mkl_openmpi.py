# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings


@dace.library.environment
class IntelMKLScaLAPACKOpenMPI:
    """ An environment for the Intel Math Kernel Library (MKL), which implements the ScaLAPACK library using OpenMPI.
    """

    # NOTE: MKL ScaLAPACK linking needs special options depending on the
    # compiler, MPI vendor and machine (e.g., CRAY). The following work for a
    # typical Ubuntu installation on an AVX2 machine with OpenMPI.

    cmake_minimum_version = None
    cmake_packages = ["MPI"]
    cmake_packages = []
    cmake_variables = {}
    cmake_compile_flags = []
    cmake_libraries = []
    cmake_files = []

    headers = ["mkl.h", "mkl_scalapack.h", "mkl_blacs.h", "mkl_pblas.h"]
    state_fields = [
        "MKL_INT __mkl_scalapack_context;", "MKL_INT __mkl_scalapack_rank, __mkl_scalapack_size;",
        "MKL_INT __mkl_scalapack_prows = 0, __mkl_scalapack_pcols = 0;",
        "MKL_INT __mkl_scalapack_myprow = 0, __mkl_scalapack_mypcol = 0;",
        "MKL_INT __mkl_int_zero = 0, __mkl_int_one = 1, __mkl_int_negone = -1;",
        "bool __mkl_scalapack_grid_init = false;"
    ]
    init_code = """
    blacs_pinfo(&__state->__mkl_scalapack_rank, &__state->__mkl_scalapack_size);
    blacs_get(&__state->__mkl_int_negone, &__state->__mkl_int_zero, &__state->__mkl_scalapack_context);
    if (!__state->__mkl_scalapack_grid_init) {{\n
        __state->__mkl_scalapack_prows = Py;\n
        __state->__mkl_scalapack_pcols = Px;\n
        blacs_gridinit(&__state->__mkl_scalapack_context, \"C\", &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols);\n
        blacs_gridinfo(&__state->__mkl_scalapack_context, &__state->__mkl_scalapack_prows, &__state->__mkl_scalapack_pcols, &__state->__mkl_scalapack_myprow, &__state->__mkl_scalapack_mypcol);\n
        __state->__mkl_scalapack_grid_init = true;\n
    }}\n
    """
    finalize_code = """
    if (__state->__mkl_scalapack_grid_init) {{
        blacs_gridexit(&__state->__mkl_scalapack_context);
    }}
    // blacs_exit(&__state->__mkl_int_zero);
    """
    dependencies = []

    @staticmethod
    def _find_mkl_lib_path() -> str:

        if 'MKLROOT' in os.environ:
            prefix = Config.get('compiler', 'library_prefix')
            suffix = Config.get('compiler', 'library_extension')
            libpath = os.path.join(os.environ['MKLROOT'], 'lib')
            libfile = os.path.join(os.environ['MKLROOT'], 'lib', f"{prefix}mkl_scalapack_lp64.{suffix}")
            if not os.path.isfile(libfile):
                # Try with ${MKLROOT}/lib/intel64 (oneAPI on Linux)
                libpath = os.path.join(os.environ['MKLROOT'], 'lib', 'intel64')
                libfile = os.path.join(os.environ['MKLROOT'], 'lib', 'intel64', f"{prefix}mkl_scalapack_lp64.{suffix}")
            if os.path.isfile(libfile):
                return libpath

        libfile = ctypes.util.find_library('mkl_scalapack_lp64')
        if libfile:
            return os.path.abspath(libfile)

        if 'CONDA_PREFIX' in os.environ:
            warnings.warn('Anaconda Python is installed but the MKL library file cannot be found for linkage. Please '
                          'install libraries with "conda install mkl-devel" or set the MKLROOT environment variable.')
        else:
            warnings.warn('MKL was not found. Please install MKL or set hte MKLROOT environment variable.')

        return ""

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
    def cmake_link_flags():

        libpath = IntelMKLScaLAPACKOpenMPI._find_mkl_lib_path()

        return [
            f"-Wl,--whole-archive {libpath}/libmkl_scalapack_lp64.a -Wl,--no-whole-archive -Wl,--start-group {libpath}/libmkl_intel_lp64.a {libpath}/libmkl_gnu_thread.a {libpath}/libmkl_core.a -Wl,--whole-archive {libpath}/libmkl_blacs_openmpi_lp64.a -Wl,--no-whole-archive -Wl,--end-group -lmpi -lgomp -lpthread -lm -ldl"
        ]

# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings
from typing import Union


@dace.library.environment
class IntelMKLScaLAPACKMPICH:
    """ An environment for the Intel Math Kernel Library (MKL), which implements the ScaLAPACK library using MPICH.
    """

    # NOTE: MKL ScaLAPACK linking needs special options depending on the
    # compiler, MPI vendor and machine (e.g., CRAY). The following work for a
    # typical Ubuntu installation on an AVX2 machine with MPICH.

    cmake_minimum_version = None
    cmake_packages = ["MPI"]
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
    # NOTE: The last library (mkl_avx2) must be set to whatever matches the
    # target hardware, e.g., mkl_avx512
    libraries = ["mkl_scalapack_lp64", "mkl_blacs_intelmpi_lp64", "mkl_intel_lp64", "mkl_gnu_thread", "mkl_core"]
    simd = "mkl_avx2"

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
            return os.path.dirname(os.path.abspath(libfile))

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
    def cmake_libraries():

        libpath = IntelMKLScaLAPACKMPICH._find_mkl_lib_path()

        prefix = Config.get('compiler', 'library_prefix')
        suffix = Config.get('compiler', 'library_extension')

        libfiles = [os.path.join(libpath, f"{prefix}{name}.{suffix}") for name in IntelMKLScaLAPACKMPICH.libraries]

        simd_libfile = os.path.join(libpath, f"{prefix}{IntelMKLScaLAPACKMPICH.simd}.{suffix}")
        if not os.path.isfile(simd_libfile):
            for num in range(1, 6):
                simd_libfile = os.path.join(libpath, f"{prefix}{IntelMKLScaLAPACKMPICH.simd}.{suffix}.{num}")
                if os.path.isfile(simd_libfile):
                    break
        if os.path.isfile(simd_libfile):
            libfiles.append(simd_libfile)

        return libfiles

    @staticmethod
    def cmake_link_flags():

        mpi_libs = ''
        mpi_link = ''

        import tempfile

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:

            os.chdir(tmp_dir)
            with open("CMakeLists.txt", "w") as f: 
                f.write("find_package(MPI)")
            os.mkdir('build')
            os.chdir(os.path.join(tmp_dir, 'build'))
            os.system("cmake ..")
            output = os.popen("cmake -LA ..")

            # Get MPI libraries names and link flags
            libs = []
            vars = {}
            line = output.readline()
            while not line.startswith('-- Cache values'):
                line = output.readline()
            for line in output:
                try:
                    tokens = line.strip('\n').split(':', maxsplit=1)
                    name = tokens[0]
                    tokens = tokens[1].split("=", maxsplit=1)
                    value = tokens[1]
                    vars[name] = value
                    if name == "MPI_CXX_LIB_NAMES":
                        libs = value.split(';')
                    if name == "MPI_CXX_LINK_FLAGS":
                        mpi_link = value
                except IndexError:
                    print(f"Line {line} does not define a CMake variable.")

            # Get MPI libraries paths
            if libs:
                for l in libs:
                    try:
                        name = f"MPI_{l}_LIBRARY"
                        libfile = vars[name]
                        mpi_libs += f"-L {os.path.dirname(os.path.abspath(libfile))} -l{l} "
                    except KeyError:
                        print(f"CMake variable {name} was not found.")
            
            os.chdir(cwd)

        libpath = IntelMKLScaLAPACKMPICH._find_mkl_lib_path()

        return [
            f"-L {libpath} -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 {mpi_link} {mpi_libs} -lgomp -lpthread -lm -ldl"
        ]

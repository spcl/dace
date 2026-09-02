# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.libraries.pblas.environments.thread_level import MPI_THREAD_LEVEL_GUARD


@dace.library.environment
class ScaLAPACKOpenMPI:
    """ An environment for the reference ScaLAPACK library using OpenMPI. """

    cmake_minimum_version = None
    # mpi.h is not on the default include path (it lives under the OpenMPI package
    # directory), so the MPI package must be resolved to get its header directory.
    cmake_packages = ["MPI"]
    cmake_variables = {}
    cmake_compile_flags = ["-I${MPI_CXX_HEADER_DIR}"]
    cmake_link_flags = ["-lscalapack-openmpi"]
    cmake_includes = []
    cmake_libraries = ['libscalapack-openmpi.so']
    cmake_files = []

    headers = ["mpi.h", "cstdio", "../include/scalapack.h"]
    state_fields = [
        "int __scalapack_context;", "int __scalapack_rank, __scalapack_size;",
        "int __scalapack_prows = 0, __scalapack_pcols = 0;", "int __scalapack_myprow = 0, __scalapack_mypcol = 0;",
        "int __int_zero = 0, __int_one = 1, __int_negone = -1;", "bool __scalapack_grid_init = false;"
    ]
    init_code = """
    """ + MPI_THREAD_LEVEL_GUARD + """
    Cblacs_pinfo(&__state->__scalapack_rank, &__state->__scalapack_size);
    Cblacs_get(__state->__int_negone, __state->__int_zero, &__state->__scalapack_context);
    if (!__state->__scalapack_grid_init) {{\n
        __state->__scalapack_prows = Py;\n
        __state->__scalapack_pcols = Px;\n
        char order = \'C\';
        Cblacs_gridinit(&__state->__scalapack_context, &order, __state->__scalapack_prows, __state->__scalapack_pcols);\n
        Cblacs_gridinfo(__state->__scalapack_context, &__state->__scalapack_prows, &__state->__scalapack_pcols, &__state->__scalapack_myprow, &__state->__scalapack_mypcol);\n
        __state->__scalapack_grid_init = true;\n
    }}\n
    """
    finalize_code = """
    // Cblacs_gridexit frees the grid communicator; illegal once MPI is finalized.
    int __scalapack_mpi_finalized = 0;
    MPI_Finalized(&__scalapack_mpi_finalized);
    if (__state->__scalapack_grid_init && !__scalapack_mpi_finalized) {{
        Cblacs_gridexit(__state->__scalapack_context);
    }}
    // Cblacs_exit(__state->__int_zero);
    """
    dependencies = []

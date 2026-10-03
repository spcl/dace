# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.libraries.pblas.environments.thread_level import MPI_THREAD_LEVEL_GUARD


@dace.library.environment
class ScaLAPACKMPICH:
    """ An environment for the reference ScaLAPACK library using MPICH. """

    cmake_minimum_version = None
    # mpi.h is not on the default include path (it lives under the MPICH package
    # directory), so the MPI package must be resolved to get its header directory.
    cmake_packages = ["MPI"]
    cmake_variables = {}
    cmake_compile_flags = ["-I${MPI_CXX_HEADER_DIR}"]
    cmake_link_flags = ["-lscalapack-mpich"]
    cmake_includes = []
    cmake_libraries = ['libscalapack-mpich.so']
    cmake_files = []

    headers = ["mpi.h", "cstdio", "../include/scalapack.h", "../include/blacs_grid.h"]
    state_fields = [
        "int __scalapack_rank, __scalapack_size;", "int __int_zero = 0, __int_one = 1;",
        "std::vector<DaceBlacsGrid<int>> __scalapack_grids;"
    ]
    init_code = """
    """ + MPI_THREAD_LEVEL_GUARD + """
    Cblacs_pinfo(&__state->__scalapack_rank, &__state->__scalapack_size);
    """
    finalize_code = """
    // Cblacs_gridexit frees the grid communicator; illegal once MPI is finalized.
    int __scalapack_mpi_finalized = 0;
    MPI_Finalized(&__scalapack_mpi_finalized);
    if (!__scalapack_mpi_finalized) {
        for (const auto& __grid : __state->__scalapack_grids) {
            Cblacs_gridexit(__grid.context);
        }
    }
    __state->__scalapack_grids.clear();
    // Cblacs_exit(__state->__int_zero);
    """
    dependencies = []

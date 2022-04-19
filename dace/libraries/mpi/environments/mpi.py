# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class MPI:

    cmake_minimum_version = "3.6"
    cmake_packages = ["MPI"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["${MPI_CXX_LIBRARIES}"]
    cmake_compile_flags = ["-I${MPI_CXX_HEADER_DIR}"]
    cmake_link_flags = ["${MPI_LINKER_FLAGS}"]

    headers = ["mpi.h"]
    state_fields = []
    init_code = "int t; MPI_Initialized(&t);  if (!t) MPI_Init(NULL, NULL);"
    finalize_code = "// MPI_Finalize();"  # actually if we finalize in the dace program we break pytest :)
    dependencies = []

# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace.config import Config
import dace.library
import ctypes.util
import warnings


@dace.library.environment
class ScaLAPACK_MPICH:
    """ An environment for the reference ScaLAPACK library using MPICH. """

    cmake_minimum_version = None
    cmake_packages = ["MPI"]
    cmake_variables = {}
    cmake_compile_flags = []
    cmake_link_flags = ["-lscalapack-mpich"]
    cmake_includes = []
    cmake_libraries = []
    cmake_files = []

    headers = ["../include/scalapack.h"]
    state_fields = [
        "int __scalapack_context;", "int __scalapack_rank, __scalapack_size;",
        "int __scalapack_prows = 0, __scalapack_pcols = 0;",
        "int __scalapack_myprow = 0, __scalapack_mypcol = 0;",
        "int __int_zero = 0, __int_one = 1, __int_negone = -1;",
        "bool __scalapack_grid_init = false;"
    ]
    init_code = """
    Cblacs_pinfo(&__state->__scalapack_rank, &__state->__scalapack_size);
    Cblacs_get(&__state->__int_negone, &__state->__int_zero, &__state->__scalapack_context);
    if (!__state->__scalapack_grid_init) {{\n
        __state->__scalapack_prows = Py;\n
        __state->__scalapack_pcols = Px;\n
        Cblacs_gridinit(&__state->__scalapack_context, \"C\", &__state->__scalapack_prows, &__state->__scalapack_pcols);\n
        Cblacs_gridinfo(&__state->__scalapack_context, &__state->__scalapack_prows, &__state->__scalapack_pcols, &__state->__scalapack_myprow, &__state->__scalapack_mypcol);\n
        __state->__scalapack_grid_init = true;\n
    }}\n
    """
    finalize_code = """
    if (__state->__scalapack_grid_init) {{
        Cblacs_gridexit(&__state->__scalapack_context);
    }}
    // Cblacs_exit(&__state->__int_zero);
    """
    dependencies = []

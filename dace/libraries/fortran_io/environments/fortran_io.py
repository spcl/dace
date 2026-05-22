# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Build environment for the Fortran-I/O library nodes.

Compiles the shipped ``dace_fortran_io.f90`` wrappers into the program (via
:data:`cmake_files`) and links ``libgfortran`` so the real Fortran runtime
performs each transfer.  The C++ tasklets call the wrappers through the
prototypes in ``dace_fortran_io.h``, found on the include path added here.
"""
import os

import dace.library

#: This library's directory, where ``dace_fortran_io.{f90,h}`` and
#: ``fortran_io.cmake`` ship together.
_LIB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dace.library.environment
class FortranIO:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = [_LIB_DIR]
    cmake_libraries = ["gfortran"]
    cmake_compile_flags = [f"-I{_LIB_DIR}"]
    cmake_link_flags = []
    cmake_files = [os.path.join(_LIB_DIR, "fortran_io.cmake")]

    headers = ["dace_fortran_io.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

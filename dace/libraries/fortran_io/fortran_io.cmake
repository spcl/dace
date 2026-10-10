# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
#
# Compile the shipped iso_c_binding Fortran I/O wrappers (``dace_fortran_io.f90``)
# into one relocatable object and add it to the program's link.  Mirrors the
# MLIR target's cmake: a custom command produces the object, which is appended
# to ``DACE_OBJECTS`` so the generated ``add_library`` depends on (and builds)
# it.  ``CMAKE_CURRENT_LIST_DIR`` is this library directory, where the source
# ships alongside this file.

set(DACE_FORTRAN_IO_SRC ${CMAKE_CURRENT_LIST_DIR}/dace_fortran_io.f90)
set(DACE_FORTRAN_IO_OBJ ${CMAKE_CURRENT_BINARY_DIR}/dace_fortran_io.o)

add_custom_command(
  OUTPUT ${DACE_FORTRAN_IO_OBJ}
  COMMAND gfortran -c -fPIC -O2 -J${CMAKE_CURRENT_BINARY_DIR} ${DACE_FORTRAN_IO_SRC} -o ${DACE_FORTRAN_IO_OBJ}
  DEPENDS ${DACE_FORTRAN_IO_SRC}
  VERBATIM
)

set(DACE_OBJECTS ${DACE_OBJECTS} ${DACE_FORTRAN_IO_OBJ})

# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Fortran external-file I/O as DaCe library nodes.

The HLFIR bridge lowers Fortran ``READ`` / ``WRITE`` and namelist I/O to the
nodes in this library (:class:`~dace.libraries.fortran_io.nodes.read.Read`,
:class:`~dace.libraries.fortran_io.nodes.write.Write`).  Each node fuses the
``open`` / transfer / ``close`` of one I/O statement so the file handle never
has to be threaded between nodes.

The transfer runs through the *real* Fortran runtime: the shipped
``dace_fortran_io.f90`` exposes ``iso_c_binding`` (``bind(c)``) wrappers around
``open`` / ``read`` / ``write`` / ``close``, the :class:`FortranIO` environment
compiles it into the program and links ``libgfortran``, and each node expands
to a C++ tasklet that calls those wrappers through the standardized C-interop
ABI.  This gives exact Fortran list-directed semantics with no hand-coded
runtime structs.
"""
from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "fortran_io")

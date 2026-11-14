# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python backend: an SDFG as one Python function over NumPy arrays.

Selected with ``compiler.backend: python``. :mod:`~dace.codegen.python.prepare` lowers the SDFG to the form the
emitter takes (structured control flow, Python tasklets, no shadowed names), :mod:`~dace.codegen.python.emit` writes
the function and :mod:`~dace.codegen.python.compiled` loads and calls it.
"""

from typing import TYPE_CHECKING

from dace.dtypes import StorageType

if TYPE_CHECKING:
    from dace import SDFG, data

#: The storages whose arrays are CuPy arrays; every other array is a NumPy one.
CUPY_STORAGES = frozenset({StorageType.GPU_Global, StorageType.GPU_Shared})


def arguments(sdfg: "SDFG") -> dict[str, "data.Data"]:
    """The arguments of the function: Python evaluates the ends and steps of the subsets it reads, which
    ``SDFG.arglist`` leaves out of a view, so it takes every free symbol."""
    return sdfg.arglist(free_symbols=sdfg.free_symbols)


class PythonTarget:
    """The target of the Python code object; ``generate_program_folder`` writes it to ``src/python``."""

    target_name = "python"

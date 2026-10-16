# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Python backend: an SDFG as one Python function over NumPy arrays.

Selected with ``compiler.backend: python``. :mod:`~dace.codegen.python.prepare` lowers the SDFG to the form the
emitter takes (structured control flow, Python tasklets, no shadowed names), :mod:`~dace.codegen.python.emit` writes
the function and :mod:`~dace.codegen.python.compiled` loads and calls it.
"""

from dace.dtypes import StorageType

#: The storages whose arrays are CuPy arrays; every other array is a NumPy one.
CUPY_STORAGES = frozenset({StorageType.GPU_Global, StorageType.GPU_Shared})


class PythonTarget:
    """The target of the Python code object; ``generate_program_folder`` writes it to ``src/python``."""

    target_name = "python"

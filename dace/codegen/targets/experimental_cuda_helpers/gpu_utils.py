# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Small shared helpers for the experimental CUDA codegen (block-size math, schedule checks)."""

from dace import Config, data as dt
from dace.codegen import common
from dace.codegen.dispatcher import DefinedType

# CUDA / HIP launch grids and blocks have exactly three dimensions
# (x, y, z); accessor helpers index into that fixed-width tuple.
CUDA_GRID_DIMS = 3


def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx >= CUDA_GRID_DIMS:
        raise ValueError(f'idx must be in 0..{CUDA_GRID_DIMS - 1}, got {idx}')
    return ('x', 'y', 'z')[idx]


def generate_sync_debug_call() -> str:
    """Return backend sync + error-check calls when ``compiler.cuda.syncdebug`` is set,
    or an empty string otherwise. Backend prefix is resolved via ``common.get_gpu_backend()``.
    """
    backend: str = common.get_gpu_backend()
    sync_call: str = ""
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        sync_call = (f"DACE_GPU_CHECK({backend}GetLastError());\n"
                     f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")

    return sync_call


def get_defined_type(data: dt.Data) -> DefinedType:
    """Return the ``DefinedType`` for a data descriptor.

    Only scalars and arrays are supported; extend if others are needed.

    :param data: the data descriptor to classify.
    :returns: ``DefinedType.Scalar`` for a ``Scalar``, ``DefinedType.Pointer`` for an ``Array``.
    :raises NotImplementedError: if ``data`` is neither a ``Scalar`` nor an ``Array``.
    """
    if isinstance(data, dt.Scalar):
        return DefinedType.Scalar
    elif isinstance(data, dt.Array):
        return DefinedType.Pointer
    else:
        raise NotImplementedError(f"Data type '{type(data).__name__}' is not supported for defined type inference."
                                  "Only Scalars and Arrays are expected for Kernels.")

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Small shared helpers for the experimental CUDA codegen (block-size math, schedule checks)."""

from dace import Config
from dace.codegen import common

# CUDA/HIP launch grids and blocks always have three dimensions (x, y, z).
CUDA_GRID_DIMS = 3


def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx >= CUDA_GRID_DIMS:
        raise ValueError(f'idx must be in 0..{CUDA_GRID_DIMS - 1}, got {idx}')
    return ('x', 'y', 'z')[idx]


def generate_sync_debug_call() -> str:
    """Return backend sync + error-check calls when ``compiler.cuda.syncdebug`` is set, else empty string."""
    if not Config.get_bool('compiler', 'cuda', 'syncdebug'):
        return ""
    backend: str = common.get_gpu_backend()
    return (f"DACE_GPU_CHECK({backend}GetLastError());\n"
            f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")

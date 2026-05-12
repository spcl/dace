# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List, Set

from dace import Config, data as dt, dtypes
from dace.sdfg import nodes, SDFGState
from dace.codegen import common
from dace.codegen.dispatcher import DefinedType
from dace.transformation.dataflow.add_threadblock_map import (product, to_3d_dims, validate_block_size_limits)
from dace.transformation.helpers import get_parent_map

# CUDA / HIP launch grids and blocks have exactly three dimensions
# (x, y, z); accessor helpers index into that fixed-width tuple.
CUDA_GRID_DIMS = 3


def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx >= CUDA_GRID_DIMS:
        raise ValueError(f'idx must be in 0..{CUDA_GRID_DIMS - 1}, got {idx}')
    return ('x', 'y', 'z')[idx]


def generate_sync_debug_call() -> str:
    """
    Generate backend sync and error-check calls as a string if
    synchronous debugging is enabled.

    Parameters
    ----------
    backend : str
        Backend API prefix (e.g., 'cuda').

    Returns
    -------
    str
        The generated debug call code, or an empty string if debugging is disabled.
    """
    backend: str = common.get_gpu_backend()
    sync_call: str = ""
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        sync_call = (f"DACE_GPU_CHECK({backend}GetLastError());\n"
                     f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n")

    return sync_call


def get_defined_type(data: dt.Data) -> DefinedType:
    """
    Return the DefinedType for a data descriptor.
    Currently supports only scalars and arrays; extend if others are needed.
    """
    if isinstance(data, dt.Scalar):
        return DefinedType.Scalar
    elif isinstance(data, dt.Array):
        return DefinedType.Pointer
    else:
        raise NotImplementedError(f"Data type '{type(data).__name__}' is not supported for defined type inference."
                                  "Only Scalars and Arrays are expected for Kernels.")


def is_within_schedule_types(state: SDFGState, node: nodes.Node, schedules: Set[dtypes.ScheduleType]) -> bool:
    """
    Checks if the given node is enclosed within a Map whose schedule type
    matches any in the `schedules` set.

    Parameters
    ----------
    state : SDFGState
        The State where the node resides
    node : nodes.Node
        The node to check.
    schedules : set[dtypes.ScheduleType]
        A set of schedule types to match (e.g., {dtypes.ScheduleType.GPU_Device}).

    Returns
    ----------
    bool
        True if the node is enclosed by a Map with a schedule type in `schedules`, False otherwise.
    """
    current = node

    while current is not None:
        if isinstance(current, nodes.MapEntry):
            if current.map.schedule in schedules:
                return True

        parent = get_parent_map(state, current)
        if parent is None:
            return False
        current, state = parent

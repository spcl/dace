# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helper functions for transformations and passes that work on GPU code."""

import math
from typing import List, Tuple

from dace import config, dtypes
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.scope import is_devicelevel_gpu


def gpu_kernels(sdfg: SDFG) -> List[Tuple[SDFG, SDFGState, nodes.MapEntry]]:
    """
    Returns the GPU kernels of an SDFG and its nested SDFGs, i.e., the device and persistent maps that are not
    themselves in device code.

    :param sdfg: The SDFG to search.
    :return: A list of (SDFG, state, map entry) for each kernel map.
    """
    result = []
    for node, state in sdfg.all_nodes_recursive():
        if (
            isinstance(node, nodes.MapEntry)
            and node.map.schedule in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent)
            and not is_devicelevel_gpu(state.sdfg, state, node)
        ):
            result.append((state.sdfg, state, node))
    return result


def dynamic_map_block_dims() -> Tuple[int, ...]:
    """
    Returns the thread-block dimensions of dynamic thread-block maps (``compiler.cuda.dynamic_map_block_size``).

    :return: The thread-block size in each dimension.
    :raises NotImplementedError: If the configuration entry is ``max``.
    """
    value = config.Config.get("compiler", "cuda", "dynamic_map_block_size")
    if value == "max":
        raise NotImplementedError("max dynamic block size unimplemented")
    return tuple(int(x) for x in value.split(","))


def dynamic_map_block_size() -> int:
    """
    Returns the total thread-block size of dynamic thread-block maps (``compiler.cuda.dynamic_map_block_size``).

    :return: The number of threads in a thread-block.
    """
    return math.prod(dynamic_map_block_dims())

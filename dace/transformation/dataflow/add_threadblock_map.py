# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Provides a transformation to add missing GPU_ThreadBlock maps to
GPU_Device maps, along with helper functions.
"""
import warnings

import sympy

import dace
from dace import Config, dtypes, symbolic
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.transformation import helpers, transformation
from dace.transformation.dataflow.tiling import MapTiling

# TODO: Move these helper functions to a separate utility module or class
import functools
from typing import List


def to_3d_dims(dim_sizes: List) -> List:
    """
    Converts a list of dimension sizes to a 3D format.

    If the list has more than three dimensions, all dimensions beyond the second are
    collapsed into the third (via multiplication). If the list has fewer than three
    entries, it is padded with 1s to ensure a fixed length of three.

    Examples:
        [x]             → [x, 1, 1]
        [x, y]          → [x, y, 1]
        [x, y, z]       → [x, y, z]
        [x, y, z, u, v] → [x, y, z * u * v]
    """

    if len(dim_sizes) > 3:
        # multiply everything from the 3rd onward into d[2]
        dim_sizes[2] = product(dim_sizes[2:])
        dim_sizes = dim_sizes[:3]

    # pad with 1s if necessary
    dim_sizes += [1] * (3 - len(dim_sizes))

    return dim_sizes


def product(iterable):
    """
    Computes the symbolic product of elements in the iterable using sympy.Mul.

    This is equivalent to: ```functools.reduce(sympy.Mul, iterable, 1)```.

    Purpose: This function is used to improve readability of the codeGen.
    """
    return functools.reduce(sympy.Mul, iterable, 1)


def validate_block_size_limits(kernel_map_entry: nodes.MapEntry, block_size: List):
    """
    Validates that the given block size for a kernel does not exceed typical CUDA hardware limits.

    These limits are not enforced by the CUDA compiler itself, but are configurable checks
    performed by DaCe during GPU code generation. They are based on common hardware
    restrictions and can be adjusted via the configuration system.

    Specifically, this function checks:
    - That the total number of threads in the block does not exceed `compiler.cuda.block_size_limit`.
    - That the number of threads in the last (z) dimension does not exceed
      `compiler.cuda.block_size_lastdim_limit`.

    Raises:
        ValueError: If either limit is exceeded.
    """

    kernel_map_label = kernel_map_entry.map.label

    total_block_size = product(block_size)
    limit = Config.get('compiler', 'cuda', 'block_size_limit')
    lastdim_limit = Config.get('compiler', 'cuda', 'block_size_lastdim_limit')

    if (total_block_size > limit) == True:
        raise ValueError(f'Block size for kernel "{kernel_map_label}" ({block_size}) '
                         f'is larger than the possible number of threads per block ({limit}). '
                         'The kernel will potentially not run, please reduce the thread-block size. '
                         'To increase this limit, modify the `compiler.cuda.block_size_limit` '
                         'configuration entry.')

    if (block_size[-1] > lastdim_limit) == True:
        raise ValueError(f'Last block size dimension for kernel "{kernel_map_label}" ({block_size}) '
                         'is larger than the possible number of threads in the last block dimension '
                         f'({lastdim_limit}). The kernel will potentially not run, please reduce the '
                         'thread-block size. To increase this limit, modify the '
                         '`compiler.cuda.block_size_lastdim_limit` configuration entry.')


@make_properties
class AddThreadBlockMap(transformation.SingleStateTransformation):
    """
    Adds an explicit `GPU_ThreadBlock` map into `GPU_Device`-scheduled maps ("kernel maps")
    that do not already contain one.

    This pass only applies to simple kernels — i.e., `GPU_Device` maps that are not nested within,
    and do not contain, other GPU-scheduled maps. Such special cases (e.g., dynamic parallelism
    or persistent kernels) are skipped and left to be handled by the `CUDACodeGen` backend.
    """
    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG, permissive=False):
        """
        Determines whether the transformation can be applied to a map entry.

        The transformation applies only to GPU_Device-scheduled maps ("kernel maps") that:
        1. Are not already nested within any GPU-scheduled maps,
        2. Do not themselves contain any inner GPU-scheduled maps.
        """
        map_entry = self.map_entry

        # Only applicable to GPU_Device maps
        if map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False

        # Check if any inner maps are GPU-scheduled (e.g., GPU_ThreadBlock)
        for _, inner_entry in helpers.get_internal_scopes(state, map_entry):
            if inner_entry.map.schedule in dtypes.GPU_SCHEDULES:
                return False  # Already has GPU-scheduled inner scope — does not apply

        # Check if the map is nested inside another GPU-scheduled map
        parent_map_tuple = helpers.get_parent_map(state, map_entry)
        while parent_map_tuple is not None:
            parent_map, parent_state = parent_map_tuple

            if parent_map.map.schedule in dtypes.GPU_SCHEDULES:
                return False  # Nested inside a GPU scope — does not apply

            parent_map_tuple = helpers.get_parent_map(parent_state, parent_map)

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        """
        The `self.map_entry`, a `GPU_Device`-scheduled map, together with its enclosed
        subgraph is a kernel function. This kernel lacks an explicit nested
        `GPU_ThreadBlock` map. This function produces a semantically equivalent kernel
        with an explicit `GPU_ThreadBlock` map.

        This is achieved by applying the `MapTiling` transformation to `self.map_entry`,
        using a computed block size. Essentially `self.map_entry` becomes then the thread block map and
        the new inserted parent map (added by `MapTiling`) is the new kernel (`GPU_Device`-scheduled) map.
        The schedules are set accordingly. A final consistency check verifies that the resulting thread
        block map's range fits into the computed block size.

        Raises:
            ValueError: If the overapproximated extent of the thread block map does not match
                        the derived block size.
        """
        gpu_block_size = self.preprocess_default_dims()
        kernel_map_entry = self.map_entry

        # Reverse for map tiling to prioritize later dimensions for better memory/performance
        reversed_block_size = gpu_block_size[::-1]

        # Get tile size
        num_dims = len(kernel_map_entry.map.params)

        # Reverse for map tiling to prioritize later dimensions for better memory/performance
        reversed_block_size = gpu_block_size[::-1]

        len_diff = num_dims - len(reversed_block_size)
        if len_diff > 0:
            # More dimensions than block size elements - pad with 1s
            tile_sizes = [1] * len_diff + reversed_block_size
        else:
            # Fewer or equal dimensions - truncate from the beginning
            tile_sizes = reversed_block_size[-num_dims:]

        # Apply map tiling transformation
        MapTiling.apply_to(sdfg=sdfg,
                           options={
                               "prefix": "b",
                               "tile_sizes": tile_sizes,
                               "tile_trivial": True,
                               "skew": False
                           },
                           map_entry=kernel_map_entry)

        # After tiling: kernel_map_entry is now the thread block map, configure its schedule
        thread_block_map_entry = kernel_map_entry
        thread_block_map_entry.map.schedule = dtypes.ScheduleType.GPU_ThreadBlock

        # Set the new kernel_entry's gpu_block_size attribute
        new_kernel_entry, *_ = helpers.get_parent_map(state, kernel_map_entry)
        new_kernel_entry.map.gpu_block_size = gpu_block_size

    def preprocess_default_dims(self):
        """
        Computes a 3D GPU thread block size for a kernel `MapEntry` without an explicit `GPU_ThreadBlock` map.

        Assumes that `self.map_entry` is a GPU kernel map (i.e., schedule is `ScheduleType.GPU_Device`) without
        an explicit thread block map.

        If set, the `gpu_block_size` property on the map `self.map_entry` is used. Otherwise, a default is taken from
        `Config('compiler', 'cuda', 'default_block_size')`, with basic validation and dimension normalization.

        Returns:
            List[int]: A normalized [blockDim.x, blockDim.y, blockDim.z] list representing the GPU block size.

        Raises:
            NotImplementedError: If the configuration sets the block size to `"max"`.
            ValueError: If the computed block size exceeds hardware limits.

        Warnings:
            - If falling back to the default block size from configuration.
            - If the default block size has more dimensions than the kernel iteration space and gets linearized.
        """
        kernel_map_entry = self.map_entry
        preset_block_size = kernel_map_entry.map.gpu_block_size

        if preset_block_size is not None:
            block_size = to_3d_dims(preset_block_size)

        else:
            kernel_map = kernel_map_entry.map
            kernel_map_label = kernel_map.label
            default_block_size_config = Config.get('compiler', 'cuda', 'default_block_size')

            # 1) Warn that we are falling back to config
            warnings.warn(
                f'No `gpu_block_size` property specified on map "{kernel_map_label}". '
                f'Falling back to the configuration entry `compiler.cuda.default_block_size`: {default_block_size_config}. '
                'You can either specify the block size to use with the gpu_block_size property, '
                'or by adding nested `GPU_ThreadBlock` maps, which map work to individual threads. '
                'For more information, see https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

            # 2) Reject unsupported 'max' setting
            if default_block_size_config == 'max':
                raise NotImplementedError('max dynamic block size unimplemented')

            # 3) Parse & normalize the default block size to 3D
            default_block_size = [int(x) for x in default_block_size_config.split(',')]
            default_block_size = to_3d_dims(default_block_size)

            # 4) Normalize the total iteration space size (len(X),len(Y),len(Z)…) to 3D
            # This is needed for X
            raw_domain = list(kernel_map.range.size(True))[::-1]
            kernel_domain_size = to_3d_dims(raw_domain)

            # 5) If block has more "active" dims than the grid, collapse extras
            active_block_dims = max(1, sum(1 for b in default_block_size if b != 1))
            active_grid_dims = max(1, sum(1 for g in kernel_domain_size if g != 1))

            if active_block_dims > active_grid_dims:
                tail_product = product(default_block_size[active_grid_dims:])
                block_size = default_block_size[:active_grid_dims] + [1] * (3 - active_grid_dims)
                block_size[active_grid_dims - 1] *= tail_product
                warnings.warn(f'Default block size has more dimensions ({active_block_dims}) than kernel dimensions '
                              f'({active_grid_dims}) in map "{kernel_map_label}". Linearizing block '
                              f'size to {block_size}. Consider setting the ``gpu_block_size`` property.')
            else:
                block_size = default_block_size

        # Validate that the block size does not exeed any limits
        validate_block_size_limits(kernel_map_entry, block_size)

        # Note order is [blockDim.x, blockDim.y, blockDim.z]
        return block_size

    def update_names():
        pass

    @staticmethod
    def annotates_memlets():
        return False

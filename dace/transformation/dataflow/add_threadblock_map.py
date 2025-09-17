# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""
import warnings

import sympy

import dace
from dace import Config, dtypes, symbolic
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.codegen.targets.experimental_cuda_helpers import gpu_utils
from dace.transformation import helpers, transformation
from dace.transformation.dataflow.tiling import MapTiling


@make_properties
class AddThreadBlockMap(transformation.SingleStateTransformation):
    """
    Ensures that all `GPU_Device`-scheduled maps (kernel maps) in the SDFG
    without an explicit `GPU_ThreadBlock` or `GPU_ThreadBlock_Dynamic` map
    are nested within one.

    This is achieved by applying the `MapTiling` transformation to each such map,
    inserting a corresponding thread block scope.
    """
    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

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
            block_size = gpu_utils.to_3d_dims(preset_block_size)

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
            default_block_size = gpu_utils.to_3d_dims(default_block_size)

            # 4) Normalize the total iteration space size (len(X),len(Y),len(Z)…) to 3D
            # This is needed for X
            raw_domain = list(kernel_map.range.size(True))[::-1]
            kernel_domain_size = gpu_utils.to_3d_dims(raw_domain)

            # 5) If block has more "active" dims than the grid, collapse extras
            active_block_dims = max(1, sum(1 for b in default_block_size if b != 1))
            active_grid_dims = max(1, sum(1 for g in kernel_domain_size if g != 1))

            if active_block_dims > active_grid_dims:
                tail_product = gpu_utils.product(default_block_size[active_grid_dims:])
                block_size = default_block_size[:active_grid_dims] + [1] * (3 - active_grid_dims)
                block_size[active_grid_dims - 1] *= tail_product
                warnings.warn(f'Default block size has more dimensions ({active_block_dims}) than kernel dimensions '
                              f'({active_grid_dims}) in map "{kernel_map_label}". Linearizing block '
                              f'size to {block_size}. Consider setting the ``gpu_block_size`` property.')
            else:
                block_size = default_block_size

        # Validate that the block size does not exeed any limits
        gpu_utils.validate_block_size_limits(kernel_map_entry, block_size)

        # Note order is [blockDim.x, blockDim.y, blockDim.z]
        return block_size

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        """
        Determines whether the transformation can be applied to the given map entry.

        The transformation only applies to maps with a GPU_Device schedule (i.e., kernel map entries).
        It is not applicable if a nested GPU_ThreadBlock or GPU_ThreadBlock_Dynamic map exists
        within the kernel scope, as that indicates the thread-block schedule is already defined.
        The same restriction applies in the case of dynamic parallelism (nested kernel launches).
        """
        # Only applicable to GPU_Device maps
        if self.map_entry.map.schedule != dtypes.ScheduleType.GPU_Device:
            return False

        # Traverse inner scopes (ordered outer -> inner)
        for _, inner_entry in helpers.get_internal_scopes(graph, self.map_entry):
            schedule = inner_entry.map.schedule

            if schedule in {
                    dtypes.ScheduleType.GPU_ThreadBlock,
                    dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
            }:
                # Already scheduled with thread block — cannot apply
                return False

            if schedule == dtypes.ScheduleType.GPU_Device:
                # Found another kernel launch — safe to apply
                return True

        # No thread block schedule found - do apply
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        """
        Ensures that `self.map_entry`, a `GPU_Device`-scheduled map, is explicitly nested
        within a `GPU_ThreadBlock` map.

        This is achieved by applying the `MapTiling` transformation to `self.map_entry`,
        using a computed block size. Essentially `self.map_entry` becomes the thread block map and
        the new inserted parent map is the new kernel map. The schedules are set accordingly.
        A final consistency check verifies that the resulting thread block map's range fits into the
        computed block size.

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

        # Catch any unexpected mismatches of inserted threadblock map's block size and the used block size
        tb_size = gpu_utils.to_3d_dims(
            [symbolic.overapproximate(sz) for sz in thread_block_map_entry.map.range.size()[::-1]])
        max_block_size = [sympy.Max(sz, bbsz) for sz, bbsz in zip(tb_size, gpu_block_size)]

        if max_block_size != gpu_block_size:
            raise ValueError(f"Block size mismatch: the overapproximated extent of the thread block map "
                             f"({tb_size}) is not enclosed by the derived block size ({gpu_block_size}). "
                             "They are expected to be equal or the derived block size to be larger.")

    @staticmethod
    def annotates_memlets():
        return False

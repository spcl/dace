import warnings
from typing import Dict, List, Set, Tuple

import sympy

from dace import SDFG, SDFGState, dtypes, symbolic
from dace.codegen.targets.experimental_cuda_helpers import gpu_utils
from dace.sdfg import nodes
from dace.transformation import helpers, pass_pipeline as ppl


class InferGPUGridAndBlockSize(ppl.Pass):
    """
    Infers the 3D CUDA launch configuration (grid and block sizes) for all GPU_Device map entries in the SDFG.

    This pass assumes the `AddThreadBlockMap` transformation has already been applied, ensuring that each kernel
    either has an explicit thread block map. However it is applicable as long as each GPU_Device scheduled map
    has an inner explicit GPU_ThreadBlock scheduled map.

    Block sizes are determined based on:
    - Whether an explicit GPU_ThreadBlock map was inserted by `AddThreadBlockMap`. In this case,
      the `gpu_block_size` attribute holds this information.
    - Existing nested thread block maps and also the `gpu_block_size`, if present.

    Grid sizes are computed from the kernel map's range, normalized to a 3D shape.

    NOTE:
        This pass does not handle dynamic parallelism (i.e., nested GPU_Device maps),
        nor does it support GPU_ThreadBlock_Dynamic maps inside kernels. Behavior is unclear in
        such cases.
    """

    def apply_pass(self, sdfg: SDFG,
                   kernels_with_added_tb_maps: Set[nodes.MapEntry]) -> Dict[nodes.MapEntry, Tuple[List, List]]:
        """
        Analyzes the given SDFG to determine the 3D grid and block sizes for all GPU_Device map entries.

        Returns:
            A dictionary mapping each GPU_Device MapEntry node to a tuple (grid_dimensions, block_dimensions).
        """
        # Collect all GPU_Device map entries across the SDFG
        kernel_maps: Set[Tuple[
            nodes.MapEntry,
            SDFGState,
        ]] = set()
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.schedule == dtypes.ScheduleType.GPU_Device:
                kernel_maps.add((node, state))

        kernel_dimensions_map: Dict[nodes.MapEntry, Tuple[List, List]] = dict()
        for map_entry, state in kernel_maps:
            # Compute grid size
            raw_grid = map_entry.map.range.size(True)[::-1]
            grid_size = gpu_utils.to_3d_dims(raw_grid)

            # Compute Block size
            if map_entry in kernels_with_added_tb_maps:
                block_size = self._get_inserted_gpu_block_size(map_entry)
            else:
                block_size = self._infer_gpu_block_size(state, map_entry)

            block_size = gpu_utils.to_3d_dims(block_size)
            gpu_utils.validate_block_size_limits(map_entry, block_size)

            kernel_dimensions_map[map_entry] = (grid_size, block_size)

        return kernel_dimensions_map

    def _get_inserted_gpu_block_size(self, kernel_map_entry: nodes.MapEntry) -> List:
        """
        Returns the block size from a kernel map entry with an inserted thread-block map.

        Assumes the `gpu_block_size` attribute is set by the AddThreadBlockMap transformation.
        """
        gpu_block_size = kernel_map_entry.map.gpu_block_size

        if gpu_block_size is None:
            raise ValueError("Expected 'gpu_block_size' to be set. This kernel map entry should have been processed "
                             "by the AddThreadBlockMap transformation.")

        return gpu_block_size

    def _infer_gpu_block_size(self, state: SDFGState, kernel_map_entry: nodes.MapEntry) -> List:
        """
        Infers the GPU block size for a kernel map entry based on nested GPU_ThreadBlock maps.

        If the `gpu_block_size` attribute is set, it is assumed to be user-defined (not set by
        a transformation like `AddThreadBlockMap`), and all nested thread-block maps must fit within it.
        Otherwise, the block size is inferred by overapproximating the range sizes of all inner
        GPU_ThreadBlock maps of kernel_map_entry.


        Example:
            for i in dace.map[0:N:32] @ GPU_Device:
                for j in dace.map[0:32] @ GPU_ThreadBlock:
                    ...
                for l in dace.map[0:23] @ GPU_ThreadBlock:
                    for k in dace.map[0:16] @ GPU_ThreadBlock:
                        ...

        Inferred GPU block size is [32, 1, 1]
        """
        # Identify nested threadblock maps
        threadblock_maps = self._get_internal_threadblock_maps(state, kernel_map_entry)

        # guard check
        if not threadblock_maps:
            state.sdfg.save("failure.sdfg")
            raise ValueError(f"{self.__class__.__name__} expects at least one explicit nested GPU_ThreadBlock map, "
                             "as it assumes AddThreadBlockMap was applied beforehand.\n"
                             f"Check for issues in that transformation or ensure AddThreadBlockMap was applied.")

        # Overapproximated block size enclosing all inner ThreadBlock maps
        block_size = kernel_map_entry.map.gpu_block_size
        detected_block_sizes = [block_size] if block_size is not None else []
        for tb_map in threadblock_maps:

            # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
            # and collapse to GPU-compatible 3D dimensions
            tb_size = [symbolic.overapproximate(s) for s in tb_map.range.size()[::-1]]
            tb_size = gpu_utils.to_3d_dims(tb_size)

            if block_size is None:
                block_size = tb_size
            else:
                block_size = [sympy.Max(sz1, sz2) for sz1, sz2 in zip(block_size, tb_size)]

            if block_size != tb_size or len(detected_block_sizes) == 0:
                detected_block_sizes.append(tb_size)

        # Check for conflicting or multiple thread-block sizes
        # - If gpu_block_size is explicitly defined (by the user) and conflicts with detected map sizes, raise an error
        # - Otherwise, emit a warning when multiple differing sizes are detected, and over-approximate
        if len(detected_block_sizes) > 1:
            kernel_map_label = kernel_map_entry.map.label

            if kernel_map_entry.map.gpu_block_size is not None:
                raise ValueError('Both the `gpu_block_size` property and internal thread-block '
                                 'maps were defined with conflicting sizes for kernel '
                                 f'"{kernel_map_label}" (sizes detected: {detected_block_sizes}). '
                                 'Use `gpu_block_size` only if you do not need access to individual '
                                 'thread-block threads, or explicit block-level synchronization (e.g., '
                                 '`__syncthreads`). Otherwise, use internal maps with the `GPU_Threadblock` or '
                                 '`GPU_ThreadBlock_Dynamic` schedules. For more information, see '
                                 'https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

            else:
                warnings.warn('Multiple thread-block maps with different sizes detected for '
                              f'kernel "{kernel_map_label}": {detected_block_sizes}. '
                              f'Over-approximating to block size {block_size}.\n'
                              'If this was not the intent, try tiling one of the thread-block maps to match.')

        return block_size

    def _get_internal_threadblock_maps(self, state: SDFGState,
                                       kernel_map_entry: nodes.MapEntry) -> List[nodes.MapEntry]:
        """
        Returns GPU_ThreadBlock MapEntries nested within a given the GPU_Device scheduled kernel map
        (kernel_map_entry).

        Returns:
            A List of GPU_ThreadBlock scheduled maps.
        """
        threadblock_maps = []

        for _, scope in helpers.get_internal_scopes(state, kernel_map_entry):
            if isinstance(scope, nodes.MapEntry) and scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                threadblock_maps.append(scope)

        return threadblock_maps

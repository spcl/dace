# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis pass that infers CUDA grid and block dimensions for GPU device maps."""
import warnings
from typing import Dict, List, Optional, Set, Tuple

import sympy

from dace import SDFG, SDFGState, dtypes, symbolic
from dace.codegen.targets.cuda import default_block_size, gpu_scope_maps_recursive, thread_block_extent
from dace.sdfg import nodes
from dace.transformation import gpu_helpers, pass_pipeline as ppl
from dace.transformation.dataflow.add_threadblock_map import to_3d_dims, validate_block_size_limits


class InferGPUGridAndBlockSize(ppl.Pass):
    """Infer the 3D grid and block sizes of every ``GPU_Device`` map; nested ``GPU_Device`` maps are not handled.

    Without a thread-block map the kernel spans threads: its block is ``gpu_block_size`` or the default.
    """

    def apply_pass(self, sdfg: SDFG,
                   kernels_with_added_tb_maps: Set[nodes.MapEntry]) -> Dict[nodes.MapEntry, Tuple[List, List]]:
        """Map each ``GPU_Device`` entry to ``(grid, block)``; ``kernels_with_added_tb_maps`` read ``gpu_block_size``.

        :raises ValueError: Explicit and inferred block sizes conflict.
        """
        kernel_dimensions_map: Dict[nodes.MapEntry, Tuple[List, List]] = dict()
        for _, state, map_entry in gpu_helpers.gpu_kernels(sdfg):
            raw_grid = map_entry.map.range.size(True)[::-1]
            grid_size = to_3d_dims(raw_grid)

            if map_entry in kernels_with_added_tb_maps:
                block_size = self.get_inserted_gpu_block_size(map_entry)
            else:
                block_size = self.infer_gpu_block_size(state, map_entry)
            if block_size is None:
                block_size = map_entry.map.gpu_block_size or default_block_size(map_entry, grid_size, False)
                block_size = to_3d_dims(list(block_size))
                grid_size = [symbolic.int_ceil(g, b) for g, b in zip(grid_size, block_size)]

            block_size = to_3d_dims(block_size)
            validate_block_size_limits(map_entry, block_size)

            kernel_dimensions_map[map_entry] = (grid_size, block_size)

        return kernel_dimensions_map

    def get_inserted_gpu_block_size(self, kernel_map_entry: nodes.MapEntry) -> List:
        """The ``gpu_block_size`` of a kernel whose thread-block map ``AddThreadBlockMap`` inserted."""
        gpu_block_size = kernel_map_entry.map.gpu_block_size

        if gpu_block_size is None:
            raise ValueError("Expected 'gpu_block_size' to be set. This kernel map entry should have been processed "
                             "by the AddThreadBlockMap transformation.")

        return gpu_block_size

    def infer_gpu_block_size(self, state: SDFGState, kernel_map_entry: nodes.MapEntry) -> Optional[List]:
        """The block size over the nested ``GPU_ThreadBlock`` maps (a set ``gpu_block_size`` must match), or ``None``."""
        threadblock_maps = [(tb_map, sym_map)
                            for tb_map, sym_map in gpu_scope_maps_recursive(state.scope_subgraph(kernel_map_entry))
                            if tb_map.schedule == dtypes.ScheduleType.GPU_ThreadBlock]

        if not threadblock_maps:
            return None

        # Thread-block sizes are 3D, so a user-set ``gpu_block_size`` is too, or it never matches them.
        block_size = kernel_map_entry.map.gpu_block_size
        if block_size is not None:
            block_size = to_3d_dims(list(block_size))
        detected_block_sizes = [block_size] if block_size is not None else []
        for tb_map, sym_map in threadblock_maps:
            tb_size = thread_block_extent(tb_map, sym_map)

            if block_size is None:
                block_size = tb_size
            else:
                block_size = [sympy.Max(sz1, sz2) for sz1, sz2 in zip(block_size, tb_size)]

            # Distinct sizes, not the running max: a size that only grows the max would be accepted silently.
            if tb_size not in detected_block_sizes:
                detected_block_sizes.append(tb_size)

        # Conflicting with a user-set ``gpu_block_size`` is an error; differing map sizes alone only warn.
        if len(detected_block_sizes) > 1:
            kernel_map_label = kernel_map_entry.map.label

            if kernel_map_entry.map.gpu_block_size is not None:
                raise ValueError('Both the ``gpu_block_size`` property and internal thread-block '
                                 'maps were defined with conflicting sizes for kernel '
                                 f'"{kernel_map_label}" (sizes detected: {detected_block_sizes}). '
                                 'Use ``gpu_block_size`` only if you do not need access to individual '
                                 'thread-block threads, or explicit block-level synchronization (e.g., '
                                 '``__syncthreads``). Otherwise, use internal maps with the ``GPU_Threadblock`` or '
                                 '``GPU_ThreadBlock_Dynamic`` schedules. For more information, see '
                                 'https://spcldace.readthedocs.io/en/latest/optimization/gpu.html')

            else:
                warnings.warn('Multiple thread-block maps with different sizes detected for '
                              f'kernel "{kernel_map_label}": {detected_block_sizes}. '
                              f'Over-approximating to block size {block_size}.\n'
                              'If this was not the intent, try tiling one of the thread-block maps to match.')

        return block_size

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis pass that infers CUDA grid and block dimensions for GPU device maps."""
import warnings
from typing import Dict, List, Set, Tuple

import sympy

from dace import SDFG, SDFGState, dtypes, symbolic
from dace.sdfg import nodes
from dace.transformation import helpers, pass_pipeline as ppl
from dace.transformation.dataflow.add_threadblock_map import to_3d_dims, validate_block_size_limits


class InferGPUGridAndBlockSize(ppl.Pass):
    """
    Infer the 3D CUDA launch configuration (grid and block sizes) for every ``GPU_Device`` map.

    Requires each kernel to have an inner explicit ``GPU_ThreadBlock`` map (normally inserted by
    ``AddThreadBlockMap``). Block size comes from ``gpu_block_size`` or the nested thread-block maps;
    grid size is the kernel range normalized to 3D. Nested ``GPU_Device`` maps and
    ``GPU_ThreadBlock_Dynamic`` maps are not handled.
    """

    def apply_pass(self, sdfg: SDFG,
                   kernels_with_added_tb_maps: Set[nodes.MapEntry]) -> Dict[nodes.MapEntry, Tuple[List, List]]:
        """
        Determine the 3D grid and block sizes for all ``GPU_Device`` map entries.

        :param kernels_with_added_tb_maps: kernel map entries whose thread-block map was inserted
                                           by ``AddThreadBlockMap`` (their block size is read from
                                           ``gpu_block_size`` rather than inferred).
        :returns: a dict mapping each ``GPU_Device`` ``MapEntry`` to ``(grid_dimensions,
                  block_dimensions)``.
        :raises ValueError: if a kernel has neither a set ``gpu_block_size`` nor a nested
                            ``GPU_ThreadBlock`` map, or if explicit and inferred block sizes conflict.
        """
        kernel_maps: Set[Tuple[
            nodes.MapEntry,
            SDFGState,
        ]] = set()
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.schedule == dtypes.ScheduleType.GPU_Device:
                kernel_maps.add((node, state))

        kernel_dimensions_map: Dict[nodes.MapEntry, Tuple[List, List]] = dict()
        for map_entry, state in kernel_maps:
            raw_grid = map_entry.map.range.size(True)[::-1]
            grid_size = to_3d_dims(raw_grid)

            if map_entry in kernels_with_added_tb_maps:
                block_size = self._get_inserted_gpu_block_size(map_entry)
            else:
                block_size = self._infer_gpu_block_size(state, map_entry)

            block_size = to_3d_dims(block_size)
            validate_block_size_limits(map_entry, block_size)

            kernel_dimensions_map[map_entry] = (grid_size, block_size)

        return kernel_dimensions_map

    def _get_inserted_gpu_block_size(self, kernel_map_entry: nodes.MapEntry) -> List:
        """Return the block size of a kernel whose thread-block map was inserted by ``AddThreadBlockMap``
        (its ``gpu_block_size`` attribute is assumed set)."""
        gpu_block_size = kernel_map_entry.map.gpu_block_size

        if gpu_block_size is None:
            raise ValueError("Expected 'gpu_block_size' to be set. This kernel map entry should have been processed "
                             "by the AddThreadBlockMap transformation.")

        return gpu_block_size

    def _infer_gpu_block_size(self, state: SDFGState, kernel_map_entry: nodes.MapEntry) -> List:
        """Infer the GPU block size from nested ``GPU_ThreadBlock`` maps.

        A set ``gpu_block_size`` is treated as user-defined and all nested thread-block maps must fit
        within it; otherwise the block size over-approximates the range sizes of all inner
        ``GPU_ThreadBlock`` maps.
        """
        threadblock_maps = self._get_internal_threadblock_maps(state, kernel_map_entry)

        if not threadblock_maps:
            raise ValueError(f"{self.__class__.__name__} expects at least one explicit nested GPU_ThreadBlock map, "
                             "as it assumes AddThreadBlockMap was applied beforehand.\n"
                             f"Check for issues in that transformation or ensure AddThreadBlockMap was applied.")

        # Overapproximated block size enclosing all inner ThreadBlock maps. Normalize a user-set
        # ``gpu_block_size`` to 3D so it compares like the (always-3D) thread-block sizes below; a
        # non-3D value (e.g. [128]) would otherwise never match and be flagged as a false conflict,
        # and would mis-zip in the elementwise ``Max``.
        block_size = kernel_map_entry.map.gpu_block_size
        if block_size is not None:
            block_size = to_3d_dims(list(block_size))
        detected_block_sizes = [block_size] if block_size is not None else []
        for tb_map in threadblock_maps:

            # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
            tb_size = [symbolic.overapproximate(s) for s in tb_map.range.size()[::-1]]
            tb_size = to_3d_dims(tb_size)

            if block_size is None:
                block_size = tb_size
            else:
                block_size = [sympy.Max(sz1, sz2) for sz1, sz2 in zip(block_size, tb_size)]

            # Collect the DISTINCT thread-block sizes seen (a user-set ``gpu_block_size`` seeds
            # ``detected_block_sizes``). Comparing against the running elementwise max ``block_size``
            # instead would miss a size that only grows the max -- e.g. [64,1,1] after a declared
            # [32,1,1] -- silently accepting the conflict and overriding the user's block size.
            if tb_size not in detected_block_sizes:
                detected_block_sizes.append(tb_size)

        # A user-set ``gpu_block_size`` conflicting with detected map sizes is an error; multiple
        # differing map sizes without one only warn (and over-approximate).
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

    def _get_internal_threadblock_maps(self, state: SDFGState,
                                       kernel_map_entry: nodes.MapEntry) -> List[nodes.MapEntry]:
        """Return the ``GPU_ThreadBlock`` ``MapEntry`` nodes nested within ``kernel_map_entry``."""
        threadblock_maps = []

        for _, scope in helpers.get_internal_scopes(state, kernel_map_entry):
            if isinstance(scope, nodes.MapEntry) and scope.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                threadblock_maps.append(scope)

        return threadblock_maps

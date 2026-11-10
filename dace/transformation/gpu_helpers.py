# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Helper functions for transformations and passes that work on GPU code."""

import functools
import itertools
import math
from dataclasses import dataclass

import sympy

from dace import config, dtypes, symbolic
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import StateSubgraphView


def gpu_kernels(sdfg: SDFG) -> list[tuple[SDFG, SDFGState, nodes.MapEntry]]:
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
            and node.map.schedule in dtypes.GPU_KERNEL_SCHEDULES
            and not is_devicelevel_gpu(state.sdfg, state, node)
        ):
            result.append((state.sdfg, state, node))
    return result


def dynamic_map_block_dims() -> tuple[int, ...]:
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


def gpu_scope_maps(subgraph: SDFGState | StateSubgraphView) -> list[tuple[nodes.Map, dict[sympy.Symbol, sympy.Basic]]]:
    """
    Returns the ``GPU_Device``, ``GPU_ThreadBlock`` and ``GPU_ThreadBlock_Dynamic`` maps of a (kernel scope)
    subgraph, including those in nested SDFGs.

    :param subgraph: The subgraph to search, e.g., the scope subgraph of a kernel map.
    :return: A list of (map, symbol mapping), where the symbol mapping restates the free symbols of the map's range
             in the symbols of ``subgraph``'s SDFG.
    """
    res = []
    for node in subgraph.nodes():
        if isinstance(node, nodes.NestedSDFG):
            for state in node.sdfg.states():
                for submap, sym_map in gpu_scope_maps(state):
                    for k in sym_map.values():
                        for kk, vv in node.symbol_mapping.items():
                            sym_map[k] = sym_map[k].subs(symbolic.symbol(kk), vv)
                    res.append((submap, sym_map))
        elif isinstance(node, nodes.MapEntry) and node.schedule in (
            dtypes.ScheduleType.GPU_Device,
            dtypes.ScheduleType.GPU_ThreadBlock,
            dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
        ):
            res.append((node.map, {symbolic.symbol(k): symbolic.symbol(k) for k in node.map.range.free_symbols}))
    return res


@dataclass
class KernelDimensions:
    """The grid and thread-block dimensions of a GPU kernel (see :func:`kernel_dimensions`)."""

    grid_size: list[symbolic.SymbolicType]  #: Three grid dimensions (before persistent-grid and chiplet changes).
    block_size: list[symbolic.SymbolicType]  #: Three thread-block dimensions.
    has_tbmap: bool  #: Whether the kernel contains ``GPU_ThreadBlock`` maps.
    has_dtbmap: bool  #: Whether the kernel contains ``GPU_ThreadBlock_Dynamic`` maps.
    #: Grid dimensions contributed by nested ``GPU_Device`` maps (appended after the kernel map's own dimensions).
    extra_grid_dims: list[symbolic.SymbolicType]
    #: Per nested ``GPU_Device`` map, the offset of its range within the extra grid dimensions.
    extra_dim_offsets: dict[nodes.Map, list[symbolic.SymbolicType]]
    #: The block sizes of ``gpu_block_size`` and of the thread-block maps, if they differ.
    detected_block_sizes: list[list[symbolic.SymbolicType]]
    #: Whether the block size fell back to ``compiler.cuda.default_block_size``.
    default_block_size: bool
    #: ``(block dimensions, grid dimensions)`` if the default block size had more dimensions than the grid and
    #: was linearized, otherwise ``None``.
    linearized_default_ndims: tuple[int, int] | None


def kernel_dimensions(kernel_entry: nodes.MapEntry, subgraph: SDFGState | StateSubgraphView) -> KernelDimensions:
    """
    Determines a GPU kernel's grid and thread-block dimensions from its map scopes, as code generation does.

    Ruleset for kernel dimensions:

        1. If only one map (device-level) exists, of an integer set ``S``, the block size is
           ``compiler.cuda.default_block_size`` (``32x1x1`` by default) and the grid size is ``ceil(|S|/32)`` in
           the first dimension.
        2. If nested thread-block maps exist ``(T_1,...,T_n)``, grid size is ``|S|`` and block size is
           ``max(|T_1|,...,|T_n|)`` with block specialization.
        3. If block size can be overapproximated, it is (for dynamically-sized blocks that are bounded by a
           predefined size).
        4. If nested device maps exist, they generate extra grid dimensions (block size 1) as the sum of all
           their sizes ``(|T_1| + ... + |T_n|)``.

    To make use of the grid/block 3D registers, kernels use up to 3 dimensions and flatten the rest into the
    third dimension. Does not modify the SDFG and emits no warnings (code generation reports the fallbacks).

    :param kernel_entry: The entry node of the kernel map (``GPU_Device`` or ``GPU_Persistent``).
    :param subgraph: The scope subgraph of the kernel map.
    :return: The kernel's dimensions.
    :raises NotImplementedError: If nested device maps differ in dimensionality, or the default block size is
                                 ``max``.
    """
    kernel_map = kernel_entry.map
    grid_size = kernel_map.range.size(True)[::-1]
    sub_maps = gpu_scope_maps(subgraph)

    # Introduce extra grid dimensions based on device sub-maps
    extra_dim_offsets: dict[nodes.Map, list[symbolic.SymbolicType]] = {}
    extra_grid_dims: list[symbolic.SymbolicType] | None = None
    for submap, sym_map in sub_maps:
        if submap.schedule != dtypes.ScheduleType.GPU_Device or submap is kernel_map:
            continue
        if extra_grid_dims is not None and len(submap.params) != len(extra_grid_dims):
            raise NotImplementedError(
                "Multiple GPU_Device sub-ranges with different dimensionality not yet implemented (found: "
                f"{len(submap.params)}, existing: {len(extra_grid_dims)}, map: {kernel_entry})"
            )

        # Add and overapproximate sizes
        gsize = [s.subs(list(sym_map.items())) for s in submap.range.size()[::-1]]
        gsize = [symbolic.overapproximate(s) for s in gsize]
        if extra_grid_dims is None:
            extra_grid_dims = gsize
            extra_dim_offsets[submap] = [0] * len(submap.params)
        else:
            extra_dim_offsets[submap] = extra_grid_dims
            extra_grid_dims = [(sz + gsz) for sz, gsz in zip(extra_grid_dims, gsize)]
    if extra_grid_dims is None:
        extra_grid_dims = []
    grid_size.extend(extra_grid_dims)

    # Linearize (flatten) rest of dimensions to third
    if len(grid_size) > 3:
        grid_size[2] = functools.reduce(sympy.Mul, grid_size[2:], 1)
        del grid_size[3:]

    # Extend to 3 dimensions if necessary
    grid_size = grid_size + [1] * (3 - len(grid_size))

    has_dtbmap = any(tbmap.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic for tbmap, _ in sub_maps)
    tb_maps_sym_map = [
        (tbmap, sym_map) for tbmap, sym_map in sub_maps if tbmap.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]

    # Map thread-block size override (copied: the map's property is not modified)
    block_size = None
    if kernel_map.gpu_block_size is not None:
        block_size = list(kernel_map.gpu_block_size)
        # Complement to three dimensions
        block_size += [1] * (3 - len(block_size))
        # Linearize (flatten) rest of dimensions to third
        if len(block_size) > 3:
            block_size[2] = functools.reduce(sympy.Mul, block_size[2:], 1)
            del block_size[3:]

    default_block_size = False
    linearized_default_ndims = None
    detected_block_sizes = []
    if len(tb_maps_sym_map) == 0:
        # No thread-block maps
        if block_size is None:
            if has_dtbmap:
                block_size = list(dynamic_map_block_dims())
            else:
                default_block_size = True
                value = config.Config.get("compiler", "cuda", "default_block_size")
                if value == "max":
                    raise NotImplementedError("max dynamic block size unimplemented")
                block_size = [int(b) for b in value.split(",")]

                block_ndim = max(1, sum(1 if b != 1 else 0 for b in block_size))
                grid_ndim = max(1, sum(1 if g != 1 else 0 for g in grid_size))
                if block_ndim > grid_ndim:
                    linearized_remainder = math.prod(block_size[grid_ndim:])
                    block_size = block_size[:grid_ndim] + [1] * (3 - grid_ndim)
                    block_size[grid_ndim - 1] *= linearized_remainder
                    linearized_default_ndims = (block_ndim, grid_ndim)

        assert len(block_size) >= 1 and len(block_size) <= 3

        # Grid size = ceil(|S|/32) for first dimension, rest = |S|
        grid_size = list(itertools.starmap(symbolic.int_ceil, zip(grid_size, block_size)))

    else:
        # Find all thread-block maps to determine overall block size
        detected_block_sizes = [block_size] if block_size is not None else []
        for tbmap, sym_map in tb_maps_sym_map:
            tbsize = [s.subs(list(sym_map.items())) for s in tbmap.range.size()[::-1]]

            # Over-approximate block size (e.g. min(N,(i+1)*32)-i*32 --> 32)
            # The partial trailing thread-block is emitted as an if-condition
            # that returns on some of the participating threads
            tbsize = [symbolic.overapproximate(s) for s in tbsize]

            # Linearize (flatten) rest of dimensions to third
            if len(tbsize) > 3:
                tbsize[2] = functools.reduce(sympy.Mul, tbsize[2:], 1)
                del tbsize[3:]

            # Extend to 3 dimensions if necessary
            tbsize = tbsize + [1] * (3 - len(tbsize))

            if len(detected_block_sizes) == 0:
                block_size = tbsize
            else:
                block_size = list(itertools.starmap(sympy.Max, zip(block_size, tbsize)))

            if block_size != tbsize or len(detected_block_sizes) == 0:
                detected_block_sizes.append(tbsize)

    return KernelDimensions(
        grid_size=grid_size,
        block_size=block_size,
        has_tbmap=len(tb_maps_sym_map) > 0,
        has_dtbmap=has_dtbmap,
        extra_grid_dims=extra_grid_dims,
        extra_dim_offsets=extra_dim_offsets,
        detected_block_sizes=detected_block_sizes,
        default_block_size=default_block_size,
        linearized_default_ndims=linearized_default_ndims,
    )

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowers GPU maps that run in strides of the thread block or of the grid (``Map.gpu_strided``)."""

import copy
import itertools
from typing import Any

import numpy
import sympy

from dace import dtypes, nodes, subsets, symbolic
from dace.config import Config
from dace.frontend import operations
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, dealias, graph
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def _wrap_in_map(
    state: SDFGState, entry: nodes.MapEntry, params: list[str], ndrange: list, schedule: dtypes.ScheduleType
) -> nodes.MapEntry:
    """Encloses the scope of ``entry`` in a new map, through which every edge of the scope passes unchanged (the
    memlets of the enclosed scope cover those of each iteration of the new map)."""
    exit_node = state.exit_node(entry)
    outer_entry, outer_exit = state.add_map(
        f"{entry.map.label}_outer", dict(zip(params, ndrange)), schedule, debuginfo=entry.map.debuginfo
    )
    for e in list(state.in_edges(entry)):
        if e.dst_conn is None:
            state.add_nedge(e.src, outer_entry, copy.deepcopy(e.data))
            state.add_nedge(outer_entry, entry, copy.deepcopy(e.data))
        else:
            # Data into the scope (``IN_x``) or a dynamic map input (any other name)
            name = outer_entry.next_connector(e.dst_conn.removeprefix("IN_"))
            outer_entry.add_in_connector(f"IN_{name}")
            outer_entry.add_out_connector(f"OUT_{name}")
            state.add_edge(e.src, e.src_conn, outer_entry, f"IN_{name}", copy.deepcopy(e.data))
            state.add_edge(outer_entry, f"OUT_{name}", entry, e.dst_conn, copy.deepcopy(e.data))
        state.remove_edge(e)
    if state.in_degree(entry) == 0:  # A map without inputs: the new map encloses it with an empty edge
        state.add_nedge(outer_entry, entry, Memlet())
    for e in list(state.out_edges(exit_node)):
        if e.src_conn is None:
            state.add_nedge(exit_node, outer_exit, copy.deepcopy(e.data))
            state.add_nedge(outer_exit, e.dst, copy.deepcopy(e.data))
        else:
            name = outer_exit.next_connector(e.src_conn[4:])
            outer_exit.add_in_connector(f"IN_{name}")
            outer_exit.add_out_connector(f"OUT_{name}")
            state.add_edge(exit_node, e.src_conn, outer_exit, f"IN_{name}", copy.deepcopy(e.data))
            state.add_edge(outer_exit, f"OUT_{name}", e.dst, e.dst_conn, copy.deepcopy(e.data))
        state.remove_edge(e)
    if state.out_degree(exit_node) == 0:
        state.add_nedge(exit_node, outer_exit, Memlet())
    return outer_entry


def _fresh_name(sdfg: SDFG, state: SDFGState, base: str) -> str:
    taken = set(sdfg.symbols) | set(sdfg.arrays)
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            taken |= set(node.map.params)
    name, index = base, 0
    while name in taken:
        index += 1
        name = f"{base}_{index}"
    return name


def _enclosing_kernel(state: SDFGState, entry: nodes.MapEntry) -> tuple[nodes.MapEntry, SDFGState] | None:
    """The map of the GPU kernel that runs ``entry``, across nested SDFGs."""
    node = entry
    while True:
        scope = state.entry_node(node)
        while scope is None and state.sdfg.parent_nsdfg_node is not None:
            node, state = state.sdfg.parent_nsdfg_node, state.sdfg.parent
            scope = state.entry_node(node)
        if scope is None:
            return None
        if scope.map.schedule in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent):
            return scope, state
        node = scope


def _block_size(kernel: nodes.MapEntry, kernel_state: SDFGState) -> list:
    """The thread block of a kernel, (x, y, z): the kernel map's ``gpu_block_size``, or else the extent of its thread-
    block maps that are not strided (the last parameter of a map is x), or else the configured default."""
    if kernel.map.gpu_block_size is not None:
        return list(kernel.map.gpu_block_size) + [1] * (3 - len(kernel.map.gpu_block_size))
    block = None
    for node, _ in kernel_state.scope_subgraph(kernel).all_nodes_recursive():
        if (
            isinstance(node, nodes.MapEntry)
            and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
            and not node.map.gpu_strided
        ):
            sizes = list(node.map.range.size()[::-1]) + [1] * 3
            block = sizes[:3] if block is None else [sympy.Max(a, b) for a, b in zip(block, sizes)]
    if block is None:
        block = [int(b) for b in Config.get("compiler", "cuda", "default_block_size").split(",")]
    for size in block:
        if symbolic.issymbolic(size):
            raise ValueError(
                f'The thread block of kernel "{kernel.map.label}" depends on symbols ({block}): set its '
                "gpu_block_size to stride thread-block maps by it"
            )
    return [int(b) for b in block]


#: The most elements of a container that one thread accumulates in registers, per update of a strided map
MAX_ACCUMULATED_ELEMENTS = 8


def _accumulable(state: SDFGState, edge: graph.MultiConnectorEdge, params: set[str]) -> Any:
    """The identity of the reduction an edge out of a strided map's loop applies, if each thread can accumulate it
    in registers: a built-in reduction of a few elements, the same in every iteration of the loop. None otherwise."""
    if edge.data.is_empty() or edge.data.wcr is None:
        return None
    desc = state.sdfg.arrays[edge.data.data]
    if desc.storage not in (dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global):
        return None
    identity = dtypes.reduction_identity(desc.dtype, operations.detect_reduction_type(edge.data.wcr))
    if identity is None:
        return None
    sizes = edge.data.subset.size()
    if any(symbolic.issymbolic(size) for size in sizes) or edge.data.subset.num_elements() > MAX_ACCUMULATED_ELEMENTS:
        return None
    if any(step != 1 for _, _, step in edge.data.subset):
        return None
    for leaf in state.memlet_tree(edge).leaves():
        if leaf.data.subset is None or leaf.data.subset.free_symbols & params:
            return None  # An update of a location that changes with the iteration
    if edge.data.subset.num_elements() > 1:
        # The update splits into one per element, which must end at the container
        outer_exit = edge.dst
        out_edges = list(state.out_edges_by_connector(outer_exit, "OUT_" + edge.dst_conn[3:]))
        if len(out_edges) != 1 or not isinstance(out_edges[0].dst, nodes.AccessNode):
            return None
    return identity


def _literal(value: Any) -> str:
    return repr(numpy.asarray(value).item())  # A Python literal of a NumPy scalar


def _accumulate_per_thread(state: SDFGState, inner_entry: nodes.MapEntry, outer_entry: nodes.MapEntry) -> None:
    """Makes each thread of a block-strided map accumulate the updates of its loop (WCR memlets whose location does
    not change across the loop) in registers, and then apply them once: a register per element, set to the
    reduction's identity before the loop, and a tasklet after it that applies all the updates of the thread together
    (which the threads of a warp combine before updating memory)."""
    sdfg = state.sdfg
    inner_exit, outer_exit = state.exit_node(inner_entry), state.exit_node(outer_entry)
    params = set(inner_entry.map.params)
    inputs, outputs, code = {}, {}, []
    for edge in list(state.edges_between(inner_exit, outer_exit)):
        identity = _accumulable(state, edge, params)
        if identity is None:
            continue
        desc = sdfg.arrays[edge.data.data]
        sizes = [int(size) for size in edge.data.subset.size()]
        name, _ = sdfg.add_transient(
            f"{edge.data.data}_acc",
            sizes,
            desc.dtype,
            storage=dtypes.StorageType.Register,
            find_new_name=True,
        )
        elements = list(itertools.product(*(range(size) for size in sizes)))

        # Before the loop: the identity
        init = state.add_tasklet(
            f"init_{name}",
            {},
            {f"_out{i}": desc.dtype for i in range(len(elements))},  # Typed: codegen no longer infers them
            "\n".join(f"_out{i} = {_literal(identity)}" for i in range(len(elements))),
        )
        state.add_nedge(outer_entry, init, Memlet())
        initialized = state.add_access(name)
        for i, element in enumerate(elements):
            state.add_edge(init, f"_out{i}", initialized, None, Memlet.simple(name, ",".join(map(str, element))))
        state.add_nedge(initialized, inner_entry, Memlet())

        # In the loop: the registers
        accumulator = state.add_access(name)
        offset = subsets.Range.from_indices([begin for begin, _, _ in edge.data.subset])
        state.remove_edge(edge)
        local = subsets.Range([(0, size - 1, 1) for size in sizes])
        new_edge = state.add_edge(
            inner_exit, edge.src_conn, accumulator, None, Memlet(data=name, subset=local, wcr=edge.data.wcr)
        )
        moved = list(state.memlet_tree(new_edge))
        for moved_edge in moved:
            if moved_edge is not new_edge:
                moved_edge.data.subset.offset(offset, True)
                moved_edge.data.data = name
        dealias.rebase_reconnected_edges(moved, offset)

        # After the loop: one update per element
        if len(elements) == 1:
            targets = [(edge.dst_conn, copy.deepcopy(edge.data))]
        else:
            (out_edge,) = state.out_edges_by_connector(outer_exit, "OUT_" + edge.dst_conn[3:])
            state.remove_edge(out_edge)
            outer_exit.remove_in_connector(edge.dst_conn)
            outer_exit.remove_out_connector(out_edge.src_conn)
            targets = []
            for element in elements:
                location = subsets.Range.from_indices([b + e for b, e in zip(offset.min_element(), element)])
                memlet = Memlet(data=edge.data.data, subset=location, wcr=edge.data.wcr)
                connector = outer_exit.next_connector(edge.data.data)
                outer_exit.add_in_connector(f"IN_{connector}")
                outer_exit.add_out_connector(f"OUT_{connector}")
                state.add_edge(outer_exit, f"OUT_{connector}", out_edge.dst, out_edge.dst_conn, copy.deepcopy(memlet))
                targets.append((f"IN_{connector}", memlet))
        for element, (connector, memlet) in zip(elements, targets):
            k = len(code)
            inputs[f"_in{k}"] = (accumulator, Memlet.simple(name, ",".join(map(str, element))), desc.dtype)
            outputs[f"_out{k}"] = (connector, memlet, desc.dtype)
            code.append(f"_out{k} = _in{k}")
    if not code:
        return
    flush = state.add_tasklet(
        f"{inner_entry.map.label}_flush",
        {connector: dtype for connector, (_, _, dtype) in inputs.items()},
        {connector: dtype for connector, (_, _, dtype) in outputs.items()},
        "\n".join(code),
    )
    for connector, (accumulator, memlet, _) in inputs.items():
        state.add_edge(accumulator, None, flush, connector, memlet)
    for connector, (exit_connector, memlet, _) in outputs.items():
        state.add_edge(flush, connector, outer_exit, exit_connector, memlet)


@transformation.explicit_cf_compatible
class LowerStridedGPUMaps(ppl.Pass):
    """Lowers the GPU maps whose ``gpu_strided`` property is set.

    * A ``GPU_ThreadBlock`` map ``i in [b:e:s]`` becomes a thread-block map ``t in [0:B]`` (``B`` the thread block of
      the kernel, per dimension) around a sequential map ``i in [b + t * s : e : B * s]``: each thread loops over its
      iterations. The range of the map may exceed the thread block, or depend on data (a dynamic range). Reductions
      into the same location in every iteration (e.g., ``total[0] += a[i]``) accumulate in registers per thread, and
      update memory once per thread, combined across each warp.
    * A ``GPU_Device`` map is enclosed in a ``GPU_Persistent`` map of one iteration, whose grid is as many thread
      blocks as fit on the device at once, and over which the device map's iterations loop (a grid-stride loop). The
      kernel is launched cooperatively only if it synchronizes its grid.
    """

    CATEGORY: str = "Code Generation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        strided = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.gpu_strided
        ]
        # Device maps first: their thread blocks do not change, and thread-block maps then find their kernel
        for entry, state in strided:
            if entry.map.schedule == dtypes.ScheduleType.GPU_Device:
                self._grid_strided(state, entry)
        blocks = {}
        for entry, state in strided:
            if entry.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                kernel = _enclosing_kernel(state, entry)
                if kernel is None:
                    raise ValueError(f'Thread-block map "{entry.map.label}" is not in a GPU kernel')
                if kernel[0] not in blocks:  # Before any of the kernel's maps is lowered
                    blocks[kernel[0]] = _block_size(*kernel)
        for entry, state in strided:
            if entry.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                self._block_strided(state, entry, blocks[_enclosing_kernel(state, entry)[0]])
            elif entry.map.schedule not in (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_Persistent):
                raise ValueError(f'Map "{entry.map.label}" is strided, but neither a thread-block nor a device map')
        return len(strided) or None

    @staticmethod
    def _block_strided(state: SDFGState, entry: nodes.MapEntry, block: list[int]) -> None:
        smap = entry.map
        if len(smap.params) > 3:
            raise ValueError(f'Strided thread-block map "{smap.label}" has more than three dimensions')
        threads, outer_range, inner_range = [], [], []
        for i, (param, (begin, end, step)) in enumerate(zip(smap.params, smap.range)):
            size = block[len(smap.params) - 1 - i]  # The last parameter strides by x
            thread = _fresh_name(state.sdfg, state, f"{param}_thread")
            threads.append(thread)
            outer_range.append(f"0:{size}")
            inner_range.append((begin + symbolic.pystr_to_symbolic(thread) * step, end, step * size))
        smap.range = subsets.Range(inner_range)
        smap.schedule = dtypes.ScheduleType.Sequential
        smap.gpu_strided = False
        outer = _wrap_in_map(state, entry, threads, outer_range, dtypes.ScheduleType.GPU_ThreadBlock)
        _accumulate_per_thread(state, entry, outer)

    @staticmethod
    def _grid_strided(state: SDFGState, entry: nodes.MapEntry) -> None:
        smap = entry.map
        if state.entry_node(entry) is not None or state.sdfg.parent_nsdfg_node is not None:
            raise ValueError(f'Strided device map "{smap.label}" is not a kernel (it is nested in another scope)')
        if len(smap.params) > 1:
            raise ValueError(f'Strided device map "{smap.label}" must have one dimension (a persistent kernel)')
        smap.gpu_strided = False
        outer = _wrap_in_map(
            state,
            entry,
            [_fresh_name(state.sdfg, state, "grid")],
            ["0:1"],
            dtypes.ScheduleType.GPU_Persistent,
        )
        # The kernel: launched by the enclosing map, which takes over its launch properties
        outer.map.gpu_strided = True  # Sized by occupancy
        for prop in ("gpu_block_size", "gpu_launch_bounds", "gpu_min_warps_per_eu", "gpu_maxnreg"):
            setattr(outer.map, prop, copy.deepcopy(getattr(smap, prop)))

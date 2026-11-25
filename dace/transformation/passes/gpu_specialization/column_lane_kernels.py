# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a column kernel one thread block per kernel iteration and one thread per column.

A column physics kernel (CloudSC) runs one thread per block of columns after the device offload, with a level
loop inside it and every map over the columns a serial loop in that thread::

    GPU_Device map jb:                    # one thread per block of NPROMA columns
        for jk in levels:
            map jl (Sequential): a[jl] = ...      # the whole block of columns, serially
            map jl (Sequential): b[jl] = a[jl] ...

This pass gives the kernel one thread per column instead. A ``GPU_ThreadBlock`` lane map of ``__tid`` wraps the kernel
body, and lane ``t`` runs column ``t`` (``t + B``, ... when the block of ``B`` lanes is narrower than the columns) of
every map over the columns, the column maps. What runs outside them, the level loop included, runs on every lane::

    GPU_Device map jb:
        GPU_ThreadBlock map __tid:
            for jk in levels:
                map jl = __tid + 1 : klon + 1 : B: a[jl] = ...
                map jl = __tid + 1 : klon + 1 : B: b[jl] = a[jl] ...

The lanes need no barrier when no lane reads what another wrote: every container the kernel writes is reached
along its column dimension at the lane's own column (``a[jl - 1]`` in a map starting at ``jl = 1``), or over all
columns by a node that works element by element (a fill, a copy, the memlet of a scope), which then fills each
lane's columns only. Code outside the column maps writes only per-thread containers, so every lane computes the same
values in it, and the conditions of the control flow around the barriers are the same on every lane. A kernel that
breaks any of this is left to :class:`~dace.transformation.passes.gpu_specialization.block_tile_kernels.
BlockTileKernels`; so is a kernel without a sequential loop around its column maps, which is the shape that pass
serves.

The column buffers the kernel allocates per thread (``zsolqa(klon, 5, 5)``) are one per block once the lanes share
them: they move out of the kernel, one slice per block, the columns innermost, so adjacent lanes touch adjacent
elements.
"""

import ast
from typing import Any

from ordered_set import OrderedSet

from dace import SDFG, SDFGState, data, dtypes, properties, symbolic
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import ConditionalBlock, LoopRegion, SymbolResolver
from dace.subsets import Range
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.warp_tiling import bind_lane_symbol
from dace.transformation.passes.gpu_specialization.block_tile_kernels import barrier, is_kernel, lane_strided_hint

#: The most lanes a block gets: a kernel at 255 registers per thread still launches with this many.
MAX_LANES = 256
#: What the lane map of a column kernel is, for a CPF reader.
COLUMN_LANES_HINT = (
    "parallel -- column lanes: the threads of the one thread block that runs each iteration of the kernel loop, one "
    "per column; every loop over the columns below runs its lane's columns only, so no lane reads what another wrote"
)
#: How a column map shares its columns out, for a CPF reader.
COLUMN_STRIDED = (
    "column lanes, lane t of the block takes column t (and t + the lane count, ... when the columns outnumber the "
    "lanes): every loop over the columns gives a lane the same columns"
)
LANE = symbolic.symbol("__tid")

#: A container as ``(owning SDFG, name)``.
Container = tuple[SDFG, str]


class Refused(Exception):
    """The kernel is no column kernel; the message says why."""


def same(a, b) -> bool:
    return symbolic.simplify(as_expr(symbolic.pystr_to_symbolic(a) - symbolic.pystr_to_symbolic(b))) == 0


def column_dims(desc: data.Data, extent) -> list[int]:
    if not isinstance(desc, data.Array):
        return []
    return [i for i, size in enumerate(desc.shape) if same(size, extent)]


def lane_axis(entry: nodes.MapEntry, extent) -> int | None:
    """The parameter of ``entry`` that walks ``extent`` columns with unit step, the innermost such one."""
    found = None
    for i, (begin, end, step) in enumerate(entry.map.range):
        if step == 1 and same(end - begin + 1, extent):
            found = i
    return found


def column_extent(state: SDFGState, kernel: nodes.MapEntry) -> Any | None:
    """The extent most maps inside ``kernel`` walk with unit step that is neither ``SMALL`` nor a constant below a
    warp: the columns."""
    resolver = SymbolResolver()
    counts: dict[Any, int] = {}
    for inner_state, inner in xfh.get_internal_scopes(state, kernel):
        if not isinstance(inner, nodes.MapEntry):
            continue
        small = xfh.small_map_axes(inner_state, inner, resolver)
        for (begin, end, step), is_small in zip(inner.map.range, small):
            extent = symbolic.simplify(as_expr(end - begin + 1))
            if step != 1 or is_small or (extent.is_Number and extent < 32):
                continue
            key = next((k for k in counts if same(k, extent)), extent)
            counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else None


class Region:
    """What a column kernel's lanes run: the column maps, and around them the code every lane runs (the non-column
    nodes, per state, and the SDFGs holding them)."""

    def __init__(self, extent) -> None:
        self.extent = extent
        self.column_maps: list[tuple[SDFGState, nodes.MapEntry, int]] = []
        self.shared_nodes: list[tuple[SDFGState, nodes.Node]] = []
        self.sdfgs: list[SDFG] = []

    def walk(self, state: SDFGState, children: dict, scope) -> None:
        for node in children[scope]:
            if isinstance(node, nodes.MapEntry):
                axis = lane_axis(node, self.extent)
                if axis is not None:
                    self.column_maps.append((state, node, axis))
                    continue
                self.shared_nodes.append((state, node))
                self.walk(state, children, node)
            elif isinstance(node, nodes.NestedSDFG):
                self.shared_nodes.append((state, node))
                self.walk_sdfg(node.sdfg)
            else:
                self.shared_nodes.append((state, node))

    def walk_sdfg(self, sdfg: SDFG) -> None:
        self.sdfgs.append(sdfg)
        for state in sdfg.states():
            self.walk(state, state.scope_children(), None)


def in_level_loop(state: SDFGState, top: SDFG) -> bool:
    """Whether a sequential loop between ``top`` and ``state`` encloses it."""
    graph = state.parent_graph
    while graph is not None and graph is not top:
        if isinstance(graph, LoopRegion):
            return True
        if isinstance(graph, SDFG):
            parent = graph.parent
            graph = parent.parent_graph if parent is not None else None
        else:
            graph = graph.parent_graph
    return False


def written_containers(region: Region, state: SDFGState, kernel: nodes.MapEntry) -> OrderedSet:
    """The containers the kernel writes, named in each SDFG that holds one (through nested SDFG connectors both ways)."""
    written: OrderedSet = OrderedSet()
    links: list[tuple[Container, Container]] = []
    scope = state.scope_subgraph(kernel)
    graphs = [(state, scope.edges())] + [(st, st.edges()) for sdfg in region.sdfgs for st in sdfg.states()]
    for st, edges in graphs:
        for edge in edges:
            if edge.data.is_empty():
                continue
            if isinstance(edge.dst, nodes.AccessNode):
                written.add((st.sdfg, edge.dst.data))
            for node, conn in ((edge.dst, edge.dst_conn), (edge.src, edge.src_conn)):
                if isinstance(node, nodes.NestedSDFG) and conn is not None:
                    links.append(((st.sdfg, edge.data.data), (node.sdfg, conn)))
    changed = True
    while changed:
        changed = False
        for outer, inner in links:
            if (outer in written) != (inner in written):
                written.update((outer, inner))
                changed = True
    return written


def full_columns(subset: Range, dim: int, extent) -> bool:
    begin, end, step = subset[dim]
    return begin == 0 and step == 1 and same(end, extent - 1)


def lane_column(subset: Range, dim: int, param: str, begin) -> bool:
    """``subset`` reaches column ``param - begin`` along ``dim`` and mentions ``param`` nowhere else."""
    p = symbolic.symbol(param)
    for i, (lo, hi, step) in enumerate(subset):
        mentions = p in symbolic.pystr_to_symbolic(lo).free_symbols | symbolic.pystr_to_symbolic(hi).free_symbols
        if i == dim:
            if not (same(lo, hi) and same(lo, p - begin)):
                return False
        elif mentions:
            return False
    return True


class LanePlan:
    """The checked rewrites of one column kernel: the fills and copies every lane runs over all columns, and the
    column buffers to move out of the kernel."""

    def __init__(self) -> None:
        self.fills: list[tuple[SDFGState, Any]] = []
        self.copies: list[tuple[SDFGState, Any]] = []
        self.fences: OrderedSet = OrderedSet()
        self.temporaries: OrderedSet = OrderedSet()
        self.buffers: OrderedSet = OrderedSet()


def per_thread(owner: SDFG, name: str, region: Region) -> bool:
    """A transient each lane holds its own copy of: declared in an SDFG the lanes run, in registers or default storage,
    or passed into a nested SDFG from one."""
    if owner not in region.sdfgs:
        return False
    desc = owner.arrays[name]
    if not desc.transient:
        node = owner.parent_nsdfg_node
        outer = [
            edge.data.data
            for edge in owner.parent.all_edges(node)
            if name in (edge.src_conn, edge.dst_conn) and not edge.data.is_empty()
        ]
        return bool(outer) and all(per_thread(owner.parent_sdfg, other, region) for other in outer)
    return (
        desc.storage
        in (
            dtypes.StorageType.Register,
            dtypes.StorageType.Default,
        )
        and desc.lifetime == dtypes.AllocationLifetime.Scope
    )


def check_shared_code(region: Region, written: OrderedSet, plan: LanePlan) -> None:
    """Code every lane runs writes per-thread containers, or columns element by element; it reads no column another
    lane writes."""
    seen = set()
    for state, node in region.shared_nodes:
        for edge in state.all_edges(node):
            if edge in seen or edge.data.is_empty():
                continue
            seen.add(edge)
            # A scope or a nested SDFG relays a write-conflict resolution from inside, where it is checked
            if edge.data.wcr is not None and not isinstance(edge.src, (nodes.ExitNode, nodes.NestedSDFG)):
                raise Refused(f"every lane would accumulate into '{edge.data.data}' in {state.label}")
            check_shared_edge(state, edge, region, written, plan)


def written_column(sdfg: SDFG, name: str, region: Region, written: OrderedSet) -> int | None:
    """The column dimension of a container the kernel writes, ``None`` for any other container."""
    dims = column_dims(sdfg.arrays[name], region.extent)
    if (sdfg, name) not in written or not dims:
        return None
    if len(dims) > 1:
        raise Refused(f"'{name}' has more than one column dimension")
    return dims[0]


def check_shared_edge(state: SDFGState, edge, region: Region, written: OrderedSet, plan: LanePlan) -> None:
    sdfg = state.sdfg
    if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
        dims = [written_column(sdfg, node.data, region, written) for node in (edge.src, edge.dst)]
        subsets = (edge.data.src_subset, edge.data.dst_subset)
        if dims[1] is None:
            check_shared_write(sdfg, edge.dst.data, state, region)
            if dims[0] is not None:
                plan.fences.add(state)
            return
        if None in dims or not all(full_columns(s, d, region.extent) for s, d in zip(subsets, dims)):
            raise Refused(f"a copy in {state.label} moves '{edge.data.data}' across columns")
        plan.copies.append((state, edge))
        return
    name = edge.data.data
    dim = written_column(sdfg, name, region, written)
    if dim is None:
        if isinstance(edge.dst, nodes.AccessNode):
            check_shared_write(sdfg, edge.dst.data, state, region)
        return
    if not full_columns(edge.data.subset, dim, region.extent):
        # Every lane reads the same columns once the lanes that write them are done
        if isinstance(edge.src, nodes.AccessNode):
            plan.fences.add(state)
            return
        raise Refused(f"'{name}' is reached across the lanes' columns in {state.label}: {edge.data.subset}")
    if isinstance(edge.src, FillLibraryNode):
        plan.fills.append((state, edge))
    elif not any(isinstance(n, (nodes.EntryNode, nodes.ExitNode, nodes.NestedSDFG)) for n in (edge.src, edge.dst)):
        raise Refused(f"'{name}' is reached over all columns by one node in {state.label}")


def check_shared_write(sdfg: SDFG, name: str, state: SDFGState, region: Region) -> None:
    """Every lane writes the same value: only into its own copy."""
    if not per_thread(sdfg, name, region):
        raise Refused(f"every lane would write '{name}' in {state.label}")


def check_column_map(
    state: SDFGState, entry: nodes.MapEntry, axis: int, region: Region, written: OrderedSet, plan: LanePlan
) -> None:
    """A column map reaches each column container the kernel writes at its lane's column, and keeps what it writes
    elsewhere to itself."""
    sdfg = state.sdfg
    param, begin = entry.map.params[axis], entry.map.range[axis][0]
    scope = state.scope_subgraph(entry)
    inside = set(scope.nodes())
    exit_node = state.exit_node(entry)
    for edge in list(scope.edges()) + list(state.out_edges(exit_node)):
        if edge.data.is_empty() or edge.data.data not in sdfg.arrays:
            continue
        name = edge.data.data
        dims = column_dims(sdfg.arrays[name], region.extent)
        if dims and (sdfg, name) in written:
            if edge.src is exit_node:
                continue
            if len(dims) > 1:
                raise Refused(f"'{name}' has more than one column dimension")
            if lane_column(edge.data.subset, dims[0], param, begin):
                continue
            nested = [
                (n, c)
                for n, c in ((edge.src, edge.src_conn), (edge.dst, edge.dst_conn))
                if isinstance(n, nodes.NestedSDFG)
            ]
            if not nested or not full_columns(edge.data.subset, dims[0], region.extent):
                raise Refused(
                    f"column map {entry.map.label} reaches '{name}' off its lane's column: {edge.data.subset}"
                )
            check_nested_column(nested[0][0], nested[0][1], symbolic.symbol(param), begin, region.extent)
            continue
        if dims or not (isinstance(edge.dst, nodes.AccessNode) or edge.data.wcr is not None):
            continue
        # A write to no column: the lane's own temporary, which nothing outside the map reads
        if edge.data.wcr is not None or edge.src is exit_node:
            raise Refused(f"column map {entry.map.label} writes '{name}' for all lanes")
        if not lane_temporary(sdfg, name, region):
            raise Refused(f"column map {entry.map.label} writes '{name}', which code outside it reads")
        plan.temporaries.add((sdfg, name))


def lane_temporary(sdfg: SDFG, name: str, region: Region) -> bool:
    """A transient that every column map holding it writes before it reads it, and nothing else touches."""
    if not sdfg.arrays[name].transient:
        return False
    scopes = {}
    for state, entry, axis in region.column_maps:
        if state.sdfg is sdfg:
            scopes.setdefault(state, set()).update(state.scope_subgraph(entry).nodes())
    return all(
        node in scopes.get(state, ()) and any(not e.data.is_empty() for e in state.in_edges(node))
        for state in sdfg.states()
        for node in state.data_nodes()
        if node.data == name
    )


def check_nested_column(node: nodes.NestedSDFG, name: str, param, begin, extent) -> None:
    """A nested SDFG that receives all columns of ``name`` reaches it at the column of the symbol it maps ``param`` to
    only."""
    inner = node.sdfg
    dims = column_dims(inner.arrays[name], extent)
    lanes = [sym for sym, value in node.symbol_mapping.items() if same(value, param)]
    if len(dims) != 1 or not lanes:
        raise Refused(f"nested SDFG {node.label} reaches '{name}' by no lane symbol")
    for state in inner.states():
        for edge in state.edges():
            if edge.data.is_empty() or edge.data.data != name:
                continue
            if any(lane_column(edge.data.subset, dims[0], lane, begin) for lane in lanes):
                continue
            if not full_columns(edge.data.subset, dims[0], extent):
                raise Refused(f"nested SDFG {node.label} reaches '{name}' off its lane's column: {edge.data.subset}")
            # A scope relays all columns; the memlets inside it are checked on their own
            if any(isinstance(n, (nodes.EntryNode, nodes.ExitNode)) for n in (edge.src, edge.dst)):
                continue
            if indexes_lane_column(edge, dims[0], lanes, begin):
                continue
            nested = edge.dst if isinstance(edge.dst, nodes.NestedSDFG) else edge.src
            conn = edge.dst_conn if nested is edge.dst else edge.src_conn
            if not isinstance(nested, nodes.NestedSDFG):
                raise Refused(f"nested SDFG {node.label} reaches '{name}' off its lane's column: {edge.data.subset}")
            for lane in lanes:
                check_nested_column(nested, conn, symbolic.symbol(lane), begin, extent)
    for edge in inner.all_interstate_edges():
        if name not in edge.data.read_symbols():
            continue
        codes = [edge.data.condition.as_string, *edge.data.assignments.values()]
        indices = [index for code in codes for index in subscripts(code, name)]
        if not all(
            index is not None and any(same(index[dims[0]], symbolic.symbol(lane) - begin) for lane in lanes)
            for index in indices
        ):
            raise Refused(f"nested SDFG {node.label} reads '{name}' off its lane's column in its control flow")


def indexes_lane_column(edge, dim: int, lanes: list[str], begin) -> bool:
    """A tasklet given all columns through ``edge`` subscripts them in its code at a lane's column only (an indirect
    read, ``__src[lane - 1, __pick - 1]``)."""
    tasklet, conn = (edge.dst, edge.dst_conn) if isinstance(edge.dst, nodes.Tasklet) else (edge.src, edge.src_conn)
    if not isinstance(tasklet, nodes.Tasklet) or tasklet.language != dtypes.Language.Python:
        return False
    indices = subscripts(tasklet.code.as_string, conn)
    return bool(indices) and all(
        index is not None
        and len(index) > dim
        and any(same(index[dim], symbolic.symbol(lane) - begin) for lane in lanes)
        for index in indices
    )


def subscripts(code: str, name: str) -> list[list[str] | None]:
    """The index of each read of ``name`` in ``code``, ``None`` for a read without one."""
    found = []
    tree = ast.parse(code)
    subscripted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == name:
            subscripted.add(node.value)
            parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            found.append([ast.unparse(part) for part in parts])
    found += [
        None for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name and node not in subscripted
    ]
    return found


def check_conditions(region: Region, written: OrderedSet, plan: LanePlan) -> None:
    """The control flow every lane runs reads no column a lane writes, nor a lane's own temporary."""
    for sdfg in region.sdfgs:
        read = [name for edge in sdfg.all_interstate_edges() for name in edge.data.read_symbols()]
        for cfg in sdfg.all_control_flow_regions():
            codes = []
            if isinstance(cfg, LoopRegion):
                codes = [cfg.loop_condition, cfg.init_statement, cfg.update_statement]
            elif isinstance(cfg, ConditionalBlock):
                codes = [condition for condition, branch in cfg.branches]
            read += [name for code in codes if code is not None for name in code.get_free_symbols()]
        for name in read:
            if name in sdfg.arrays and (
                written_column(sdfg, name, region, written) is not None or (sdfg, name) in plan.temporaries
            ):
                raise Refused(f"the control flow reads column container '{name}'")


def plan_kernel(state: SDFGState, kernel: nodes.MapEntry) -> tuple[Region, LanePlan]:
    extent = column_extent(state, kernel)
    if extent is None:
        raise Refused("no map over the columns")
    region = Region(extent)
    region.walk(state, state.scope_children(), kernel)
    if not region.column_maps:
        raise Refused("no column map")
    top = state.sdfg
    if not any(in_level_loop(st, top) for st, entry, axis in region.column_maps):
        raise Refused("no sequential loop around the column maps")
    written = written_containers(region, state, kernel)
    plan = LanePlan()
    check_shared_code(region, written, plan)
    for st, entry, axis in region.column_maps:
        check_column_map(st, entry, axis, region, written, plan)
    check_conditions(region, written, plan)
    fenced_maps = [entry.map.label for st, entry, axis in region.column_maps if st in plan.fences]
    if fenced_maps:
        raise Refused(f"columns written by {fenced_maps} are read across the lanes in the same state")
    for sdfg in region.sdfgs:
        for name, desc in sdfg.arrays.items():
            if desc.transient and per_thread(sdfg, name, region) and column_dims(desc, extent):
                plan.buffers.add((sdfg, name))
    return region, plan


def lift_buffers(state: SDFGState, kernel: nodes.MapEntry, buffers: OrderedSet) -> None:
    """One copy of each column buffer per block, outside the kernel, the block's slice contiguous."""
    from dace.sdfg import dealias
    from dace.transformation.passes.move_array_out_of_kernel import move_array, plan_lift

    for owner, name in buffers:
        plan = plan_lift(name, owner, (kernel, state), False)
        owner.arrays[name].storage = dtypes.StorageType.GPU_Global
        move_array(plan)
        dealias.convert_legacy_nested_sdfgs(state.sdfg)


def can_lift(state: SDFGState, kernel: nodes.MapEntry, buffers: OrderedSet) -> None:
    from dace.transformation.passes.move_array_out_of_kernel import plan_lift

    for owner, name in buffers:
        try:
            if plan_lift(name, owner, (kernel, state), False) is None:
                raise Refused(f"column buffer '{name}' is sized inside the kernel")
        except NotImplementedError as ex:
            raise Refused(f"column buffer '{name}' cannot move out of the kernel: {ex}") from ex


def wrap_in_lanes(state: SDFGState, kernel: nodes.MapEntry, lanes) -> nodes.MapEntry:
    """A ``GPU_ThreadBlock`` map of ``lanes`` lanes around the kernel body, nested in an SDFG of its own."""
    exit_node = state.exit_node(kernel)
    lane_entry, lane_exit = state.add_map(
        "column_lanes", {str(LANE): f"0:{lanes}"}, dtypes.ScheduleType.GPU_ThreadBlock
    )
    for edge in state.out_edges(kernel):
        xfh.reconnect_edge_through_map(state, edge, lane_entry, True)
    for edge in state.in_edges(exit_node):
        xfh.reconnect_edge_through_map(state, edge, lane_exit, False)
    lane_entry.specialization_hint = COLUMN_LANES_HINT
    return lane_entry


def stride_column(subset: Range, dim: int, lanes) -> None:
    """``subset``'s columns along ``dim`` become the lane's: ``__tid``, ``__tid + lanes``, ..."""
    begin, end, step = subset[dim]
    subset[dim] = (begin + LANE, end, lanes)


def fence(state: SDFGState) -> None:
    """Barriers around ``state``, which reads columns other lanes write: the writes finish before it, and its reads
    before the next writes. The control flow around it is the same on every lane, so every lane reaches both."""
    parent = state.parent_graph
    start = parent.start_block is state
    barrier(parent.add_state_before(state, f"{state.label}_lanes_enter", is_start_block=start))
    barrier(parent.add_state_after(state, f"{state.label}_lanes_leave"))


def apply_lanes(state: SDFGState, kernel: nodes.MapEntry, region: Region, plan: LanePlan) -> None:
    # As few blocks of columns as fit MAX_LANES lanes, split evenly: every lane runs, and the launch knows the width
    lanes = symbolic.int_ceil(region.extent, symbolic.int_ceil(region.extent, MAX_LANES))
    lift_buffers(state, kernel, plan.buffers)
    wrap_in_lanes(state, kernel, lanes)
    top = state.sdfg
    users: OrderedSet = OrderedSet()
    for st, entry, axis in region.column_maps:
        begin, end, step = entry.map.range[axis]
        entry.map.range[axis] = (begin + LANE, end, lanes)
        entry.specialization_hint = lane_strided_hint(st, entry, COLUMN_STRIDED)
        users.add(st.sdfg)
    # The lift put a block dimension in front of a moved buffer's columns, so they are looked up again
    for st, edge in plan.fills:
        stride_column(edge.data.subset, column_dims(st.sdfg.arrays[edge.data.data], region.extent)[0], lanes)
        users.add(st.sdfg)
    for st, edge in plan.copies:
        for node, subset in ((edge.src, edge.data.src_subset), (edge.dst, edge.data.dst_subset)):
            stride_column(subset, column_dims(st.sdfg.arrays[node.data], region.extent)[0], lanes)
        users.add(st.sdfg)
    for st in plan.fences:
        fence(st)
    for sdfg in users:
        bind_lane_symbol(sdfg, top)
    kernel.map.gpu_block_size = None
    kernel.map.gpu_launch_bounds = str(MAX_LANES)


@properties.make_properties
class ColumnLaneKernels(ppl.Pass):
    """Run each column kernel one block per kernel iteration and one lane per column."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Give every column kernel its lanes.

        :param sdfg: the offloaded SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many kernels now run one lane per column, or ``None`` if none do.
        """
        kernels = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and is_kernel(state, node)
        ]
        done = 0
        for kernel, state in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            try:
                region, plan = plan_kernel(state, kernel)
                can_lift(state, kernel, plan.buffers)
            except Refused:
                continue
            apply_lanes(state, kernel, region, plan)
            done += 1
        return done or None

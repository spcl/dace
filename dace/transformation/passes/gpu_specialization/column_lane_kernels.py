# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a column kernel one thread block per kernel iteration and one thread per column.

A column physics kernel (CloudSC) runs one thread per block of columns after the device offload, with a level
loop inside it and every map over the columns a serial loop in that thread::

    GPU_Device map jb:                    # one thread per block of NPROMA columns
        for jk in levels:
            map jl (Sequential): a[jl] = ...      # the whole block of columns, serially
            map jl (Sequential): b[jl] = a[jl] ...

This pass gives the kernel a lane per column instead, block-stride over columns in chunks of the block's lane
count ``B``: a ``GPU_ThreadBlock`` lane map of ``__tid`` over ``B`` lanes wraps the kernel body, and lane ``t`` runs
columns ``t, t + B, ...`` of every map over the columns, the column maps. ``B`` is a compile-time integer: the column
extent when it is a constant up to ``MAX_LANES``, else ``BLOCK_LANES``. What runs outside the column maps, the level
loop included, runs on every lane::

    GPU_Device map jb:
        GPU_ThreadBlock map __tid = 0 : B:
            for __col in range(__tid, klon, B):       # chunked; one chunk (B = klon) drops the loop, __col = __tid
                for jk in levels:
                    map jl = __col + 1: a = ...
                    map jl = __col + 1: b = a ...

A kernel whose lanes read across columns (a fence, below) keeps every lane on the same path to its barriers: the
level loop runs once per lane, and each column map strides over the lane's columns itself
(``map jl = __tid + 1 : klon + 1 : B``). The form is fixed at compile time; nothing is versioned at run time.

The lanes need no barrier when no lane reads what another wrote: every container the kernel writes is reached
along its column dimension at the lane's own column (``a[jl - 1]`` in a map starting at ``jl = 1``), or over all
columns by a node that works element by element (a fill, a copy, the memlet of a scope), which then fills the lane's
column only. Code outside the column maps writes only per-thread containers, so every lane computes the same values
in it, and the conditions of the control flow around the barriers are the same on every lane. A read of another
lane's column in that code is fenced by barriers. A kernel that breaks any of this is left to
:class:`~dace.transformation.passes.gpu_specialization.block_tile_kernels.BlockTileKernels`; so is a kernel without a
sequential loop around its column maps, which is the shape that pass serves.

A column buffer the kernel allocates per thread (``zsolqa(klon, 5, 5)``) loses its column dimension when a lane runs
a whole chunk at a time: each lane's copy is its current column (``zsolqa(5, 5)``), reused per chunk, in registers
when small, and code generation gives a larger one a per-thread slice of device memory, the threads innermost. A
buffer whose columns lanes read across, and every buffer of a strided kernel, is shared instead: one copy per block,
outside the kernel, the columns innermost.
"""

import ast
from typing import Any

from ordered_set import OrderedSet

from dace import SDFG, SDFGState, data, dtypes, properties, symbolic
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import ConditionalBlock, LoopRegion, SymbolResolver
from dace.subsets import Range
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.warp_tiling import bind_lane_symbol
from dace.transformation.passes.gpu_specialization.block_tile_kernels import barrier, is_kernel, lane_strided_hint

#: What the lane map of a column kernel is, for a CPF reader; ``{lanes}`` is the block's lane count.
COLUMN_LANES_HINT = (
    "parallel -- column lanes: the {lanes} threads of the one thread block that runs each iteration of the kernel "
    "loop, a lane per column, block-stride over columns in chunks of {lanes}; every loop over the columns below runs "
    "its lane's columns only, so no lane reads what another wrote"
)
#: How a column map shares its columns out, for a CPF reader; ``{lanes}`` is the block's lane count.
COLUMN_STRIDED = (
    "column lanes, a lane per column, block-stride over columns in chunks of {lanes}: lane t takes columns t, "
    "t + {lanes}, ..., the same columns in every loop over the columns"
)
#: The most lanes a block gets: a constant column extent up to it is one chunk, a lane per column. At the 255
#: registers a CloudSC thread takes, 256 threads fill a multiprocessor's register file.
MAX_LANES = 256
#: The lanes of a block whose columns may outnumber ``MAX_LANES``: a symbolic or larger column extent.
BLOCK_LANES = 128
LANE = symbolic.symbol("__tid")
#: The column a lane runs in the current chunk, the chunk loop's variable.
COLUMN = symbolic.symbol("__col")

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
        self.cross_lane: OrderedSet = OrderedSet()
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
                plan.cross_lane.add((sdfg, edge.src.data))
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
            plan.cross_lane.add((sdfg, edge.src.data))
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
    lane_entry.specialization_hint = COLUMN_LANES_HINT.format(lanes=lanes)
    return lane_entry


def fence(state: SDFGState) -> None:
    """Barriers around ``state``, which reads columns other lanes write: the writes finish before it, and its reads
    before the next writes. The control flow around it is the same on every lane, so every lane reaches both."""
    parent = state.parent_graph
    start = parent.start_block is state
    barrier(parent.add_state_before(state, f"{state.label}_lanes_enter", is_start_block=start))
    barrier(parent.add_state_after(state, f"{state.label}_lanes_leave"))


def memlet_sides(edge) -> list[tuple[str, Range]]:
    """The containers ``edge`` reaches and the subset of each: its memlet's, and the other subset's access node. A
    fill inside a map writes through the map exit, so the memlet names the container, not an endpoint."""
    memlet = edge.data
    sides = [(memlet.data, memlet.subset)]
    others = [
        node
        for node in (edge.src, edge.dst)
        if isinstance(node, nodes.AccessNode) and node.data != memlet.data and memlet.other_subset is not None
    ]
    sides += [(node.data, memlet.other_subset) for node in others]
    return sides


def lane_count(extent) -> int:
    """The block's lanes: the column extent when it is a constant up to ``MAX_LANES``, else ``BLOCK_LANES``."""
    extent = symbolic.pystr_to_symbolic(extent)
    return int(extent) if extent.is_Number and int(extent) <= MAX_LANES else BLOCK_LANES


def kernel_body(state: SDFGState, kernel: nodes.MapEntry) -> nodes.NestedSDFG | None:
    """The nested SDFG that is all of ``kernel``'s body, if one is."""
    inner = [node for node in state.scope_children()[kernel] if node is not state.exit_node(kernel)]
    return inner[0] if len(inner) == 1 and isinstance(inner[0], nodes.NestedSDFG) else None


def chunk_loop(body: SDFG, extent, lanes: int) -> LoopRegion:
    """Run all of ``body`` once per column of the lane: ``for __col in range(__tid, extent, lanes)``."""
    loop = LoopRegion(
        "column_chunks",
        f"{COLUMN} < {symbolic.symstr(extent)}",
        str(COLUMN),
        f"{COLUMN} = {LANE}",
        f"{COLUMN} = {COLUMN} + {lanes}",
    )
    start = body.start_block
    blocks, edges = list(body.nodes()), list(body.edges())
    for block in blocks:
        body.remove_node(block)
    for block in blocks:
        loop.add_node(block, is_start_block=block is start)
    for edge in edges:
        loop.add_edge(edge.src, edge.dst, edge.data)
    body.add_node(loop, is_start_block=True)
    sdutil.set_nested_sdfg_parent_references(body)
    body.reset_cfg_list()
    return loop


def apply_lanes(state: SDFGState, kernel: nodes.MapEntry, region: Region, plan: LanePlan) -> None:
    """Give ``kernel`` a block of ``lane_count`` lanes, a lane per column, block-stride over columns in chunks of the
    lane count. The form is fixed at compile time, by the extent and by whether lanes read across:

    - one chunk (a constant extent up to ``MAX_LANES``): lane ``t`` runs column ``t``;
    - chunked (no fence): the lane runs the whole body once per column of its, ``__col = t, t + lanes, ...``; no
      barrier sits in the body, so the lanes may run different chunk counts;
    - strided (a fence): every column map runs ``t, t + lanes, ...`` itself, and every lane runs the code around them
      once, so all lanes reach each barrier; a lane then owns more than one column of a buffer, which stays per block.

    One chunk and chunked contract the per-thread buffers to the lane's column: the lane reuses its copy per chunk.
    """
    lanes = lane_count(region.extent)
    one_chunk = same(region.extent, lanes)
    body = kernel_body(state, kernel)
    strided = not one_chunk and (bool(plan.fences) or body is None)
    lifted = OrderedSet(plan.buffers) if strided else OrderedSet()
    for owner, name in OrderedSet() if strided else plan.buffers:
        dim = column_dims(owner.arrays[name], region.extent)[0]
        try:
            Contraction(owner, name, dim, apply=False, cross_lane=plan.cross_lane).run()
        except Refused:
            lifted.add((owner, name))
            continue
        Contraction(owner, name, dim, apply=True, cross_lane=plan.cross_lane).run()
    lift_buffers(state, kernel, lifted)
    wrap_in_lanes(state, kernel, lanes)
    column = LANE if one_chunk or strided else COLUMN
    users: OrderedSet = OrderedSet()
    for st, entry, axis in region.column_maps:
        begin, end, step = entry.map.range[axis]
        entry.map.range[axis] = (begin + LANE, end, lanes) if strided else (begin + column, begin + column, 1)
        entry.specialization_hint = lane_strided_hint(st, entry, COLUMN_STRIDED.format(lanes=lanes))
        users.add(st.sdfg)
    # A contracted buffer has no columns left; the lift put a block dimension in front of a moved one's
    for st, edge in plan.fills + plan.copies:
        for name, subset in memlet_sides(edge):
            dims = column_dims(st.sdfg.arrays[name], region.extent)
            if dims:
                begin, end, step = subset[dims[0]]
                subset[dims[0]] = (begin + LANE, end, lanes) if strided else (column, column, 1)
        users.add(st.sdfg)
    for st in plan.fences:
        fence(st)
    top = state.sdfg
    if column is COLUMN:
        chunk_loop(body.sdfg, region.extent, lanes)
        for sdfg in users:
            bind_lane_symbol(sdfg, body.sdfg, str(COLUMN))
        bind_lane_symbol(body.sdfg, top)
    else:
        for sdfg in users:
            bind_lane_symbol(sdfg, top)
    kernel.map.gpu_block_size = None


def drop_index(index: list, dim: int) -> list:
    return [part for i, part in enumerate(index) if i != dim]


class Contraction:
    """Drop the column dimension of a per-thread column buffer: every lane reaches only its own column, so each lane's
    copy is that column alone. With ``apply`` unset it only checks that every access can follow (raising
    :class:`Refused`); set, it rewrites the descriptor, every memlet, the subscripts in tasklet code and in the
    control flow, and the nested SDFGs the buffer is passed into."""

    def __init__(
        self, owner: SDFG, name: str, dim: int, apply: bool, cross_lane: OrderedSet, done: OrderedSet | None = None
    ) -> None:
        self.owner, self.name, self.dim, self.apply, self.cross_lane = owner, name, dim, apply, cross_lane
        # A container reached through an input and an output connector of one nested SDFG is contracted once
        self.done = OrderedSet() if done is None else done

    def run(self) -> None:
        if (self.owner, self.name) in self.done:
            return
        self.done.add((self.owner, self.name))
        if (self.owner, self.name) in self.cross_lane:
            raise Refused(f"lanes read each other's columns of '{self.name}'")
        desc = self.owner.arrays[self.name]
        rank = len(desc.shape)
        shape, strides, total = contracted_layout(desc, self.dim)
        for state in self.owner.states():
            for edge in state.edges():
                self.memlet(state, edge, rank)
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    self.tasklet(state, node, rank)
        self.control_flow(rank)
        if self.apply:
            desc.set_shape(shape, strides=strides, total_size=total)

    def memlet(self, state: SDFGState, edge, rank: int) -> None:
        memlet = edge.data
        if memlet.is_empty():
            return
        sides = []
        if memlet.data == self.name:
            sides.append("subset")
        ends = [n for n in (edge.src, edge.dst) if isinstance(n, nodes.AccessNode) and n.data == self.name]
        # The other subset names this container too unless the edge copies from or to another one
        others = [
            n
            for n in (edge.src, edge.dst)
            if isinstance(n, nodes.AccessNode) and n.data not in (self.name, memlet.data)
        ]
        if memlet.data == self.name and memlet.other_subset is not None and not others:
            sides.append("other_subset")
        elif memlet.data != self.name and ends:
            if memlet.other_subset is None:
                raise Refused(f"a copy reaches '{self.name}' at an implicit subset")
            sides.append("other_subset")
        for side in sides:
            subset = getattr(memlet, side)
            if subset is None or len(subset) != rank:
                raise Refused(f"'{self.name}' is reached with a subset of another rank: {subset}")
            if self.apply:
                kept = drop_index(subset.ndrange(), self.dim) or [(0, 0, 1)]
                setattr(memlet, side, Range(kept))
        if memlet.data == self.name:
            for node, conn in ((edge.src, edge.src_conn), (edge.dst, edge.dst_conn)):
                if isinstance(node, nodes.NestedSDFG) and conn is not None:
                    self.nested(node, conn, rank)

    def nested(self, node: nodes.NestedSDFG, conn: str, rank: int) -> None:
        inner = node.sdfg.arrays[conn]
        if (node.sdfg, conn) in self.done or len(inner.shape) == rank:
            Contraction(node.sdfg, conn, self.dim, self.apply, self.cross_lane, self.done).run()
            return
        raise Refused(f"nested SDFG {node.label} holds '{self.name}' in rank {len(inner.shape)}, not {rank}")

    def tasklet(self, state: SDFGState, node: nodes.Tasklet, rank: int) -> None:
        conns = [
            conn
            for edge, conn in [(e, e.dst_conn) for e in state.in_edges(node)]
            + [(e, e.src_conn) for e in state.out_edges(node)]
            if conn is not None and not edge.data.is_empty() and edge.data.data == self.name
        ]
        for conn in conns:
            if f"{conn}[" not in node.code.as_string:
                continue
            if node.language != dtypes.Language.Python:
                raise Refused(f"tasklet {node.label} subscripts '{conn}' in {node.language.name}")
            rewritten = self.rewrite(node.code.as_string, conn, rank)
            if self.apply:
                node.code = properties.CodeBlock(rewritten, node.language)

    def control_flow(self, rank: int) -> None:
        for edge in self.owner.all_interstate_edges():
            if self.name not in edge.data.read_symbols():
                continue
            condition = self.rewrite(edge.data.condition.as_string, self.name, rank)
            assignments = {k: self.rewrite(v, self.name, rank) for k, v in edge.data.assignments.items()}
            if self.apply:
                edge.data.condition = properties.CodeBlock(condition)
                edge.data.assignments = assignments
        for cfg in self.owner.all_control_flow_regions():
            if isinstance(cfg, LoopRegion):
                for attr in ("loop_condition", "init_statement", "update_statement"):
                    code = getattr(cfg, attr)
                    if code is not None and self.name in code.get_free_symbols():
                        raise Refused(f"loop {cfg.label} reads '{self.name}' in its header")
            elif isinstance(cfg, ConditionalBlock):
                for i, (condition, branch) in enumerate(cfg.branches):
                    if condition is not None and self.name in condition.get_free_symbols():
                        rewritten = self.rewrite(condition.as_string, self.name, rank)
                        if self.apply:
                            cfg.branches[i] = (properties.CodeBlock(rewritten), branch)

    def rewrite(self, code: str, name: str, rank: int) -> str:
        """``code`` with the column index dropped from every subscript of ``name``."""
        dim = self.dim

        class Drop(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == name:
                    raise Refused(f"'{name}' is read whole in: {code}")
                return node

            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                if not (isinstance(node.value, ast.Name) and node.value.id == name):
                    self.generic_visit(node)
                    return node
                node.slice = self.visit(node.slice)
                parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if len(parts) != rank:
                    raise Refused(f"'{name}' is subscripted in rank {len(parts)}, not {rank}: {code}")
                kept = drop_index(parts, dim) or [ast.Constant(0)]
                node.slice = kept[0] if len(kept) == 1 else ast.Tuple(kept, ast.Load())
                return node

        return ast.unparse(Drop().visit(ast.parse(code)))


def contracted_layout(desc: data.Array, dim: int) -> tuple[list, list, Any]:
    """Shape, packed strides and size of ``desc`` without dimension ``dim``, the other dimensions in their order."""
    if desc.is_packed_fortran_strides():
        order = list(range(len(desc.shape)))
    elif desc.is_packed_c_strides():
        order = list(reversed(range(len(desc.shape))))
    else:
        raise Refused(f"cannot contract {desc}: only packed C or Fortran strides are")
    keep = [i for i in range(len(desc.shape)) if i != dim]
    if not keep:
        return [1], [1], 1
    shape = [desc.shape[i] for i in keep]
    clone = desc.clone()
    clone.set_shape(shape)
    strides, total = clone.strides_from_layout(*[keep.index(i) for i in order if i != dim])
    return shape, list(strides), total


@properties.make_properties
class ColumnLaneKernels(ppl.Pass):
    """Run each column kernel one block per kernel iteration, a lane per column, block-stride over columns."""

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

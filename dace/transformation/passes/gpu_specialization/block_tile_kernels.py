# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give a kernel's inner parallel maps the threads of one block.

After the device offload a kernel runs one THREAD per outer iteration, and every map nested inside it
is a serial loop in that thread (``SequentializeNestedDeviceScopes``)::

    GPU_Device map i:                 # one thread per row
        y_i = 0
        map j (Sequential):           # the whole row, serially
            y_i += A[i, j] * x[j]

This pass makes the outer map run one BLOCK per iteration instead: :class:`~dace.transformation.
dataflow.warp_tiling.WarpTiling` wraps the kernel body in a ``GPU_ThreadBlock`` lane map of
:data:`~dace.libraries.standard.block_reduce.BLOCK_COLLECTIVE_THREADS` lanes, strides the inner
parallel maps across the lanes (lane ``t`` takes ``j = t, t + B, ...``, so adjacent lanes read
adjacent elements), and folds each lane's partial of a scalar reduction with ``gpucub::BlockReduce``.

Everything outside the strided maps runs once per LANE, not once per iteration. That is sound when
running it several times changes nothing. A tasklet that updates a shared container it also reads
(``b[i] -= s``, lu's ``A[i, j] -= s``) would apply once per lane, so it runs on lane 0 alone, between
two ``__syncthreads()`` every lane reaches: the first keeps lane 0 from writing what other lanes still
read, the second publishes the write. A kernel is left alone when its body outside the strided maps
accumulates (a write-conflict-resolved memlet), holds a library node (its own lowering may need the
block), when a strided map leaves a lane-private container other code reads afterwards (each lane
would hold only its own share of it), or when code outside the strided maps fills a per-thread buffer of
runtime size (each lane would allocate and fill all of it).
"""

import copy
from typing import Any

import sympy
from ordered_set import OrderedSet

from dace import SDFG, Memlet, SDFGState, data, dtypes, properties, subsets, symbolic
from dace import graphlib as nx
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS, gpu_block_implementation, spread_over_lanes
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SymbolResolver
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.warp_tiling import WarpTiling

#: What a block-tiled kernel's lane map is, for a CPF reader.
BLOCK_TILE_HINT = (
    "parallel -- block tile: this kernel runs one block per outer iteration, and these are the lanes of the "
    "block\nwhy: the inner parallel maps are strided across the lanes, so one outer iteration uses a "
    "whole block instead of one thread; code outside them runs on every lane, or on lane 0 alone "
    "between two barriers when it updates data other lanes read"
)
#: What a map strided across the lanes of a block-tiled kernel is, for a CPF reader.
LANE_STRIDED_HINT = (
    "parallel -- lane-strided: lane t of the block takes iterations t, t + B, ... (B lanes)\n"
    "why: the map is parallel, and adjacent lanes touching adjacent elements coalesce the accesses"
)

#: Storage whose containers each lane holds its own copy of.
LANE_PRIVATE_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def is_kernel(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A ``GPU_Device`` map no other ``GPU_Device`` map encloses."""
    if entry.map.schedule != dtypes.ScheduleType.GPU_Device:
        return False
    return not any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.GPU_Device
        for scope in xfh.get_parent_map_and_loop_scopes(state.sdfg, entry, state)
    )


def strided_maps(state: SDFGState, kernel: nodes.MapEntry) -> list[tuple[SDFGState, nodes.MapEntry]]:
    """The inner maps to stride across the lanes: the immediate ones that are not provably narrower than
    the block and do not span ``SMALL`` axes only (``nblks``: the offload put them in the kernel to run
    sequentially, and a handful of iterations would leave all but a handful of lanes idle). A map is
    parallel by definition; the ``Sequential`` schedule inference gives a map nested in a kernel only
    says one thread would walk it."""
    resolver = SymbolResolver()
    return [
        (inner_state, inner)
        for inner_state, inner in xfh.get_internal_scopes(state, kernel, immediate=True)
        if isinstance(inner, nodes.MapEntry)
        and (inner.map.range.size()[-1] < BLOCK_COLLECTIVE_THREADS) != True
        and not all(xfh.small_map_axes(inner_state, inner, resolver))
    ]


#: A container each lane of a block-tiled kernel holds its own copy of, as ``(owning SDFG, name)``.
LanePrivate = OrderedSet[tuple[SDFG, str]]


def per_thread_allocation(sdfg: SDFG, name: str) -> bool:
    """A kernel holding every access to such a transient gives each thread a copy, ``GPU_Global`` included."""
    desc = sdfg.arrays[name]
    return (
        desc.transient
        and desc.lifetime == dtypes.AllocationLifetime.Scope
        and desc.storage != dtypes.StorageType.GPU_Shared
    )


def lane_private_containers(state: SDFGState, kernel: nodes.MapEntry) -> LanePrivate:
    """Register and default-storage transients, and the kernel's per-thread allocations."""
    inside = set(state.scope_subgraph(kernel).nodes())
    accessed_outside = OrderedSet(
        node.data for st in state.sdfg.states() for node in st.data_nodes() if node not in inside
    )
    owners = [(state.sdfg, accessed_outside)] + [
        (nested, OrderedSet())
        for node in inside
        if isinstance(node, nodes.NestedSDFG)
        for nested in node.sdfg.all_sdfgs_recursive()
    ]
    return OrderedSet(
        (sdfg, name)
        for sdfg, outside in owners
        for name, desc in sdfg.arrays.items()
        if desc.transient
        and (desc.storage in LANE_PRIVATE_STORAGE or (name not in outside and per_thread_allocation(sdfg, name)))
    )


def fills_per_lane_buffer(state: SDFGState, kernel: nodes.MapEntry, strided) -> bool:
    """Whether code outside the strided maps writes a per-thread buffer of runtime size.

    That code runs on every lane, so each lane would allocate and fill the whole buffer: icon_scatter's staging
    copies, which the device codegen then hoisted out of the kernel sized for every thread (~54 GB at preset M). A
    buffer only the strided maps write holds one lane's share and is decided by :func:`strided_map_is_safe`.
    """
    striding: set[nodes.Node] = set()
    for inner_state, entry in strided:
        for node in inner_state.scope_subgraph(entry).nodes():
            striding.add(node)
            if isinstance(node, nodes.NestedSDFG):
                striding.update(pair[0] for pair in node.sdfg.all_nodes_recursive())
    for owner, name in lane_private_containers(state, kernel):
        desc = owner.arrays[name]
        if not isinstance(desc, data.Array) or not desc.free_symbols:
            continue
        for owner_state in owner.states():
            for node in owner_state.data_nodes():
                if node.data == name and node not in striding and owner_state.in_degree(node) > 0:
                    return True
    return False


def reads_outside(sdfg: SDFG, name: str, inner: set[nodes.Node]) -> bool:
    """Whether any node of ``sdfg`` outside ``inner`` reads ``name``."""
    return any(
        isinstance(node, nodes.AccessNode) and node.data == name and node not in inner and state.out_degree(node) > 0
        for state in sdfg.states()
        for node in state.nodes()
    )


def give_blocks_their_containers(state: SDFGState, kernel: nodes.MapEntry, strided) -> list[SDFGState]:
    """Make a lane-private array a strided map fills and later code reads one per block: lifted out of the kernel,
    one slice per kernel iteration (:class:`MoveArrayOutOfKernel`), so the lanes fill the block's copy together and
    the code after them reads it whole. Only a fill in a state of a nested SDFG that nothing else in the state
    touches is lifted; the returned states are the fills, which :func:`fence_with_barriers` fences once the kernel
    is tiled. An array the lift refuses stays lane-private, and the kernel untiled."""
    from dace.sdfg import dealias
    from dace.transformation.passes.move_array_out_of_kernel import (
        MoveArrayOutOfKernel,
        is_move_candidate,
        move_array,
        plan_lift,
    )

    mover = MoveArrayOutOfKernel()
    private = lane_private_containers(state, kernel)
    fills: list[SDFGState] = []
    for inner_state, entry in strided:
        if inner_state.sdfg is state.sdfg:
            continue
        inner = set(inner_state.scope_subgraph(entry).nodes())
        owner = inner_state.sdfg
        for edge in inner_state.out_edges(inner_state.exit_node(entry)):
            name = edge.data.data
            if edge.data.is_empty() or edge.data.wcr is not None or (owner, name) not in private:
                continue
            desc = owner.arrays[name]
            if not is_move_candidate(desc, mover.register_limit):
                continue
            if not reads_outside(owner, name, inner):
                continue
            touched = [node for node in inner_state.data_nodes() if node.data == name and node not in inner]
            if any(inner_state.out_degree(node) > 0 or inner_state.in_degree(node) > 1 for node in touched):
                continue
            try:
                plan = plan_lift(name, owner, (kernel, state), mover.layout == "coalesced")
            except NotImplementedError:
                continue
            if plan is None:
                continue
            desc.storage = dtypes.StorageType.GPU_Global
            move_array(plan)
            dealias.convert_legacy_nested_sdfgs(state.sdfg)
            private.discard((owner, name))
            if inner_state not in fills:
                fills.append(inner_state)
    return fills


def fence_with_barriers(fill: SDFGState) -> None:
    """Barrier states before and after ``fill``: every lane's share is written before any lane reads the block's
    copy, and the reads of a previous trip finish before it is overwritten."""
    parent = fill.parent_graph
    start = parent.start_block is fill
    barrier(parent.add_state_before(fill, f"{fill.label}_lanes_enter", is_start_block=start))
    barrier(parent.add_state_after(fill, f"{fill.label}_lanes_leave"))


def strided_map_is_safe(state: SDFGState, entry: nodes.MapEntry, private: LanePrivate) -> bool:
    """A strided map's outputs survive lane splitting: a lane-private output is one element reduced
    with a known identity (folded across the lanes), or nothing outside the map reads it."""
    inner = set(state.scope_subgraph(entry).nodes())
    for edge in state.out_edges(state.exit_node(entry)):
        if edge.data.is_empty() or (state.sdfg, edge.data.data) not in private:
            continue
        if edge.data.wcr is not None:
            if required(edge.data.subset).num_elements() != 1:
                return False
            continue
        if reads_outside(state.sdfg, edge.data.data, inner):
            return False
    return True


def updates_shared(node: nodes.Tasklet, state: SDFGState, private: LanePrivate) -> bool:
    """Whether ``node`` writes a shared (not lane-private) container that it, or a node leading to it, reads:
    once per lane is wrong. The read may sit in an earlier tasklet (``s = acc[k] + t; acc[k] = s``)."""
    upstream = nx.ancestors(state._nx, node) | {node}
    # Only an access node or a scope entry hands data on; any other edge carries a write, a sibling kernel's included
    read = {
        edge.data.data
        for edge in state.edges()
        if edge.dst in upstream
        and not edge.data.is_empty()
        and isinstance(edge.src, (nodes.AccessNode, nodes.EntryNode))
    }
    return any(
        edge.data.data in read and (state.sdfg, edge.data.data) not in private
        for edge in state.out_edges(node)
        if not edge.data.is_empty()
    )


def accumulates_outside(state: SDFGState, edges, skipped: set[nodes.Node]) -> bool:
    """Whether a write-conflict-resolved edge leaves a node outside the strided maps: every lane would add.
    In canonical form a WCR is ``tasklet -wcr-> MapExit* -wcr-> access node``, so a map exit only relays
    the tasklet's write (checked where the tasklet is); any other source accumulates here."""
    return any(
        edge.data.wcr is not None and edge.src not in skipped and not isinstance(edge.src, nodes.MapExit)
        for edge in edges
    )


def block_states(block) -> list[SDFGState]:
    return [block] if isinstance(block, SDFGState) else list(block.states())


def lane_zero_blocks(sdfg: SDFG, skipped: set[nodes.Node], private: LanePrivate) -> list | None:
    """The largest control-flow blocks of ``sdfg`` that hold no strided-map node and write shared data: each runs
    whole on lane 0, so a sequential loop over a block's container is one guarded region, not a guard per tasklet.
    ``None`` if code every lane runs reads a symbol or a lane-private container such a block assigns."""
    found = []

    def updates(block) -> bool:
        # A read in one state and the write in a later one is a read-modify-write too: any shared write counts.
        return any(
            isinstance(edge.dst, nodes.AccessNode) and (st.sdfg, edge.dst.data) not in private
            for st in block_states(block)
            for edge in st.edges()
            if not edge.data.is_empty()
        )

    def walk(region) -> None:
        for block in region.nodes():
            if not any(node in skipped for st in block_states(block) for node in st.nodes()):
                if updates(block):
                    found.append(block)
            elif isinstance(block, ConditionalBlock):
                for _, branch in block.branches:
                    walk(branch)
            elif not isinstance(block, SDFGState):
                walk(block)

    walk(sdfg)
    if not found:
        return found
    inside = {st for block in found for st in block_states(block)}
    regions = [block for block in found if not isinstance(block, SDFGState)]
    assigned = {lhs for block in regions for edge in block.all_interstate_edges() for lhs in edge.data.assignments}
    assigned |= {
        region.loop_variable
        for block in regions
        for region in block.all_control_flow_regions()
        if isinstance(region, LoopRegion) and region.loop_variable
    }
    written = {
        edge.data.data
        for st in inside
        for edge in st.edges()
        if not edge.data.is_empty() and isinstance(edge.dst, nodes.AccessNode)
    }
    written = {name for name in written if (sdfg, name) in private}
    for st in sdfg.states():
        if st in inside:
            continue
        if st.free_symbols & assigned:
            return None
        if any(isinstance(node, nodes.AccessNode) and node.data in written for node in st.nodes()):
            return None
    found_edges = {edge for block in regions for edge in block.all_interstate_edges()}
    if any(edge.data.free_symbols & assigned for edge in sdfg.all_interstate_edges() if edge not in found_edges):
        return None
    return found


def run_block_on_lane_zero(block) -> None:
    """Nest ``block`` under ``if (__tid == 0)`` in its region, fenced by barriers every lane reaches."""
    parent = block.parent_graph
    sdfg = block.sdfg
    start = parent.start_block is block
    ins, outs = list(parent.in_edges(block)), list(parent.out_edges(block))
    parent.remove_node(block)
    guard = ConditionalBlock(f"{block.label}_lane_zero", sdfg=sdfg, parent=parent)
    branch = ControlFlowRegion(f"{block.label}_lane_zero_body", sdfg=sdfg)
    branch.add_node(block, is_start_block=True)
    guard.add_branch("__tid == 0", branch)
    parent.add_node(guard, is_start_block=start)
    for edge in ins:
        parent.add_edge(edge.src, guard, edge.data)
    for edge in outs:
        parent.add_edge(guard, edge.dst, edge.data)
    barrier(parent.add_state_before(guard, f"{block.label}_lane_zero_enter", is_start_block=start))
    barrier(parent.add_state_after(guard, f"{block.label}_lane_zero_leave"))
    nested = sdfg
    while nested.parent_nsdfg_node is not None and "__tid" not in nested.symbols:
        nested.add_symbol("__tid", dtypes.int32)
        nested.parent_nsdfg_node.symbol_mapping["__tid"] = "__tid"
        nested = nested.parent_sdfg
    sdfg.reset_cfg_list()


def single_lane_nodes(
    state: SDFGState, kernel: nodes.MapEntry, strided: list[tuple[SDFGState, nodes.MapEntry]]
) -> tuple[list[nodes.Tasklet], list[nodes.Tasklet], list] | None:
    """The tasklets that must run on lane 0 alone for the kernel body to run once per lane, with ``strided``
    split across the lanes, the other tasklets that read what the strided maps wrote, which must wait at a
    barrier for every lane's share, and the control-flow blocks that run whole on lane 0
    (:func:`lane_zero_blocks`); ``None`` if no such split is sound."""
    private = lane_private_containers(state, kernel)
    if not strided or not all(strided_map_is_safe(s, entry, private) for s, entry in strided):
        return None
    # Nesting the lane body follows every edge into it back to its source; an ordering edge off the
    # kernel entry has no connector to follow. One that only binds an input-less node to the scope does not order.
    if any(edge.data.is_empty() and state.in_degree(edge.dst) > 1 for edge in state.out_edges(kernel)):
        return None
    skipped: set[nodes.Node] = set()
    for s, entry in strided:
        skipped |= set(s.scope_subgraph(entry).nodes())
    scope = state.scope_subgraph(kernel)
    if accumulates_outside(state, scope.edges(), skipped):
        return None
    shared_by_lanes = {
        (st.sdfg, edge.data.data)
        for st, entry in strided
        for edge in st.out_edges(st.exit_node(entry))
        if not edge.data.is_empty() and (st.sdfg, edge.data.data) not in private
    }
    single: list[nodes.Tasklet] = []
    waiting: list[nodes.Tasklet] = []
    blocks: list = []
    pending = [(state, scope.nodes())]
    while pending:
        current, nodes_in_scope = pending.pop()
        for node in nodes_in_scope:
            if node in skipped:
                continue
            if isinstance(node, nodes.NestedSDFG):
                found = lane_zero_blocks(node.sdfg, skipped, private)
                if found is None:
                    return None
                blocks += found
                on_lane_zero = {st for block in found for st in block_states(block)}
                for inner in node.sdfg.states():
                    if inner in on_lane_zero:
                        continue
                    if accumulates_outside(inner, inner.edges(), skipped):
                        return None
                    pending.append((inner, inner.nodes()))
            elif isinstance(node, nodes.LibraryNode):
                return None
            elif isinstance(node, nodes.Tasklet) and updates_shared(node, current, private):
                if any(edge.data.is_empty() for edge in current.all_edges(node)):
                    return None
                single.append(node)
            elif isinstance(node, nodes.Tasklet) and any(
                (current.sdfg, edge.data.data) in shared_by_lanes
                for edge in current.in_edges(node)
                if not edge.data.is_empty()
            ):
                waiting.append(node)
    return single, waiting, blocks


def barrier(state: SDFGState) -> nodes.Tasklet:
    """A ``__syncthreads()`` tasklet (CUDA and HIP spell it alike)."""
    return state.add_tasklet("lane_barrier", {}, {}, "__syncthreads();", dtypes.Language.CPP)


def run_on_lane_zero(tasklet: nodes.Tasklet, state: SDFGState) -> None:
    """Nest ``tasklet`` under ``if (__tid == 0)``, fenced by barriers every lane reaches."""
    wrapper = xfh.nest_state_subgraph(state.sdfg, state, SubgraphView(state, [tasklet]), name="lane_zero")
    inner = wrapper.sdfg
    body = inner.start_state
    inner.remove_node(body)
    branch = ControlFlowRegion("lane_zero_body", sdfg=inner)
    branch.add_node(body, is_start_block=True)
    guard = ConditionalBlock("lane_zero_guard", sdfg=inner)
    guard.add_branch("__tid == 0", branch)
    inner.add_node(guard, is_start_block=True)
    inner.add_symbol("__tid", dtypes.int32)
    wrapper.symbol_mapping["__tid"] = "__tid"
    inner.reset_cfg_list()
    before, after = barrier(state), barrier(state)
    for pred in state.predecessors(wrapper):
        state.add_edge(pred, None, before, None, Memlet())
    for succ in state.successors(wrapper):
        state.add_edge(after, None, succ, None, Memlet())
    state.add_edge(before, None, wrapper, None, Memlet())
    state.add_edge(wrapper, None, after, None, Memlet())


def fence_after(exit_node: nodes.MapExit, state: SDFGState, shared: set[str]) -> None:
    """Hold every lane at a barrier once the lane-strided map of ``exit_node`` stored its share of ``shared``: what
    runs next, the same map in the next trip of a loop around it included, reads the other lanes' elements."""
    written = [edge.dst for edge in state.out_edges(exit_node) if not edge.data.is_empty() and edge.data.data in shared]
    if not written:
        return
    fence = barrier(state)
    for node in OrderedSet(written):
        for succ in list(state.successors(node)):
            state.add_edge(fence, None, succ, None, Memlet())
        state.add_edge(node, None, fence, None, Memlet())


def run_after_barrier(node: nodes.Tasklet, state: SDFGState) -> None:
    """Hold every lane at a barrier before ``node`` reads what the other lanes wrote."""
    fence = barrier(state)
    for pred in state.predecessors(node):
        state.add_edge(pred, None, fence, None, Memlet())
    state.add_edge(fence, None, node, None, Memlet())


def cta_level_sdfgs(state: SDFGState, kernel: nodes.MapEntry) -> list[SDFG] | None:
    """The SDFGs a whole CTA runs: the kernel body, one nested SDFG, and those nested in it outside every map.
    ``None`` unless the body is one nested SDFG."""
    body = [n for n in state.scope_children()[kernel] if not isinstance(n, (nodes.AccessNode, nodes.MapExit))]
    if len(body) != 1 or not isinstance(body[0], nodes.NestedSDFG):
        return None
    found = [body[0].sdfg]
    for sdfg in found:
        for inner_state in sdfg.states():
            scope = inner_state.scope_dict()
            found += [n.sdfg for n in inner_state.nodes() if isinstance(n, nodes.NestedSDFG) and scope[n] is None]
    return found


def cta_arrays(sdfgs: list[SDFG]) -> list[tuple[SDFG, str]]:
    """The arrays of ``sdfgs`` code outside every map touches: one copy per CTA, which all its lanes use."""
    found = []
    for sdfg in sdfgs:
        outside = {
            node.data
            for inner_state in sdfg.states()
            for node, scope in inner_state.scope_dict().items()
            if scope is None and isinstance(node, nodes.AccessNode)
        }
        found += [
            (sdfg, name)
            for name, desc in sdfg.arrays.items()
            if desc.transient and type(desc) is data.Array and desc.total_size != 1 and name in outside
        ]
    return found


def stride_over_ctas(state: SDFGState, kernel: nodes.MapEntry) -> None:
    """One CTA per kernel iteration, on a grid of at most the CTAs the device holds resident: the kernel map walks the
    CTAs and a sequential map inside it walks each CTA's iterations ``c, c + G, ...``, so per-CTA buffers scale with
    the grid, not with the iterations."""
    from dace.sdfg.propagation import propagate_memlets_state
    from dace.transformation.passes.gpu_specialization.resident_blocks import resident_blocks

    (param,), ((begin, end, step),) = kernel.map.params, kernel.map.range
    kernel_exit = state.exit_node(kernel)
    ctas = sympy.Min(resident_blocks(BLOCK_COLLECTIVE_THREADS), symbolic.int_ceil(end + 1 - begin, step))
    cta = f"__cta_{param}"
    first = symbolic.symstr(begin + symbolic.symbol(cta) * step)
    tasks, tasks_exit = state.add_map(
        f"{kernel.map.label}_tasks",
        {param: f"{first}:{symbolic.symstr(end + 1)}:{symbolic.symstr(ctas * step)}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    kernel.map.params = [cta]
    kernel.map.range = subsets.Range([(0, ctas - 1, 1)])
    outer: dict[str, Memlet] = {}
    for edge in list(state.out_edges(kernel)):
        state.remove_edge(edge)
        conn = pass_through(tasks, edge.src_conn)
        if conn is not None and conn not in outer:
            outer[conn] = copy.deepcopy(edge.data)
            state.add_edge(kernel, edge.src_conn, tasks, f"IN_{conn[4:]}", outer[conn])
        state.add_edge(tasks, conn, edge.dst, edge.dst_conn, edge.data)
    for edge in list(state.in_edges(kernel_exit)):
        state.remove_edge(edge)
        conn = pass_through(tasks_exit, edge.dst_conn, True)
        if conn is not None and conn not in outer:
            outer[conn] = copy.deepcopy(edge.data)
            state.add_edge(tasks_exit, f"OUT_{conn[3:]}", kernel_exit, edge.dst_conn, outer[conn])
        state.add_edge(edge.src, edge.src_conn, tasks_exit, conn, edge.data)
    if state.in_degree(tasks) == 0:
        state.add_nedge(kernel, tasks, Memlet())
    if state.out_degree(tasks_exit) == 0:
        state.add_nedge(tasks_exit, kernel_exit, Memlet())
    propagate_memlets_state(state.sdfg, state)


def pass_through(scope: nodes.Node, conn: str | None, into: bool = False) -> str | None:
    """The connector of ``scope`` an edge on ``conn`` of its enclosing scope node now uses, added if new."""
    if conn is None:
        return None
    name = conn[3:] if into else conn[4:]
    scope.add_in_connector(f"IN_{name}")
    scope.add_out_connector(f"OUT_{name}")
    return f"IN_{name}" if into else f"OUT_{name}"


def lift_per_cta(state: SDFGState, kernel: nodes.MapEntry, nest: nodes.NestedSDFG, name: str) -> None:
    """Give the body's ``name`` one row per CTA of a global array ``[CTAs, capacity]`` allocated outside the kernel:
    in the body, ``name`` becomes a view of its CTA's row, keeping its own shape and strides."""
    from dace.sdfg.propagation import propagate_memlets_state

    body = nest.sdfg
    inner = body.arrays[name]
    cta = kernel.map.params[0]
    outer, desc = state.sdfg.add_array(
        f"{name}_cta",
        [kernel.map.range.num_elements(), inner.total_size],
        inner.dtype,
        storage=dtypes.StorageType.GPU_Global,
        transient=True,
        find_new_name=True,
    )
    whole = body.add_datadesc(outer, copy.deepcopy(desc), find_new_name=True)
    body.arrays[whole].transient = False
    view = data.View.view(inner)
    view.storage = dtypes.StorageType.GPU_Global
    body.remove_data(name, validate=False)
    body.add_datadesc(name, view)
    if cta not in body.symbols:
        body.add_symbol(cta, dtypes.int64)
        nest.symbol_mapping[cta] = cta
    row = f"{whole}[{cta}, 0:{symbolic.symstr(inner.total_size)}]"
    # A view is bound to its container by one edge, so an access both written and read becomes two views: the
    # written one stores into the CTA's row, and the read one loads from it after.
    for inner_state in body.states():
        for node in [n for n in inner_state.data_nodes() if n.data == name]:
            container = inner_state.add_access(whole)
            reads = [e for e in inner_state.out_edges(node) if not e.data.is_empty()]
            if not any(not e.data.is_empty() for e in inner_state.in_edges(node)):
                node.add_in_connector("views", force=True)
                inner_state.add_edge(container, None, node, "views", Memlet(row))
                continue
            node.add_out_connector("views", force=True)
            inner_state.add_edge(node, "views", container, None, Memlet(row))
            if reads:
                reader = inner_state.add_access(name)
                reader.add_in_connector("views", force=True)
                inner_state.add_edge(container, None, reader, "views", Memlet(row))
                for edge in reads:
                    inner_state.remove_edge(edge)
                    inner_state.add_edge(reader, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
    tasks = state.entry_node(nest)
    exits = [state.exit_node(tasks), state.exit_node(kernel)]
    nest.add_in_connector(whole)
    nest.add_out_connector(whole, force=True)
    full = Memlet.from_array(outer, desc)
    state.add_memlet_path(state.add_read(outer), kernel, tasks, nest, dst_conn=whole, memlet=copy.deepcopy(full))
    state.add_memlet_path(nest, *exits, state.add_write(outer), src_conn=whole, memlet=full)
    propagate_memlets_state(state.sdfg, state)


def define_lane(sdfgs: list[SDFG]) -> None:
    """Bind ``__tid``, the thread's lane in its CTA, on entry to the body ``sdfgs[0]`` and map it into the others."""
    from dace.sdfg import InterstateEdge

    body = sdfgs[0]
    lane, _ = body.add_scalar(
        "__lane_id", dtypes.int32, storage=dtypes.StorageType.Register, transient=True, find_new_name=True
    )
    first = body.start_block
    enter = body.add_state_before(first, "cta_lane", is_start_block=True)
    body.edges_between(enter, first)[0].data = InterstateEdge(assignments={"__tid": lane})
    tasklet = enter.add_tasklet("lane", {}, {"__out"}, "__out = threadIdx.x;", dtypes.Language.CPP)
    enter.add_edge(tasklet, "__out", enter.add_write(lane), None, Memlet(f"{lane}[0]"))
    for sdfg in sdfgs:
        sdfg.add_symbol("__tid", dtypes.int32)
        if sdfg is not body:
            sdfg.parent_nsdfg_node.symbol_mapping["__tid"] = "__tid"


def tile_per_cta(state: SDFGState, kernel: nodes.MapEntry) -> bool:
    """Run ``kernel`` one CTA per iteration when its body fills per-iteration arrays outside its inner maps.

    The arrays code outside every inner map touches get one copy per CTA in global memory (a row of an array with a
    leading CTA dimension); every inner map is strided across the lanes, every library node takes its thread-block
    lowering, and a statement writing a per-CTA array runs on lane 0 between two barriers. Sequential loops and
    scalar statements run on every lane alike. ``False``, with nothing changed, if a per-CTA array has no capacity
    known at the launch.
    """
    if len(kernel.map.params) != 1:
        return False
    sdfgs = cta_level_sdfgs(state, kernel)
    if sdfgs is None:
        return False
    nest = sdfgs[0].parent_nsdfg_node
    arrays = cta_arrays(sdfgs)
    launch = {str(s) for s in state.sdfg.free_symbols} | set(state.sdfg.symbols)
    if any(
        owner is not nest.sdfg
        or not {str(s) for s in sympy.sympify(owner.arrays[name].total_size).free_symbols} <= launch
        or any(scope is not None for st in owner.states() for n, scope in st.scope_dict().items() if getattr(n, "data", None) == name)
        for owner, name in arrays
    ):
        return False
    stride_over_ctas(state, kernel)
    for _, name in arrays:
        lift_per_cta(state, kernel, nest, name)
    define_lane(sdfgs)
    # Library nodes first: a node's block lowering is refused next to thread-block maps of another shape, and the
    # lane maps below have the very shape its own takes.
    for sdfg in sdfgs:
        for inner_state in sdfg.states():
            scope = inner_state.scope_dict()
            for node in [n for n in inner_state.nodes() if scope[n] is None]:
                if isinstance(node, nodes.LibraryNode):
                    block = gpu_block_implementation(node, inner_state, sdfg)
                    if block is not None:
                        node.implementation = block
                        node.auto_select_implementation = False
                        fence_after(node, inner_state, {e.data.data for e in inner_state.out_edges(node)})
                    else:
                        run_on_lane_zero(node, inner_state)
                elif isinstance(node, nodes.Tasklet) and any(
                    not e.data.is_empty() and not lane_private(sdfg, e.data.data) for e in inner_state.out_edges(node)
                ):
                    run_on_lane_zero(node, inner_state)
    for sdfg in sdfgs:
        spread_over_lanes(sdfg, f"{kernel.map.label}_cta")
    kernel.map.gpu_block_size = None
    return True


def lane_private(sdfg: SDFG, name: str) -> bool:
    """Whether each lane holds its own ``name``: a transient in registers or default storage."""
    desc = sdfg.arrays[name]
    return desc.transient and desc.storage in LANE_PRIVATE_STORAGE


@properties.make_properties
class BlockTileKernels(ppl.Pass):
    """Run each eligible kernel one block per outer iteration, its inner parallel maps across the lanes."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.States | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """Tile every kernel :func:`single_lane_nodes` accepts.

        :param sdfg: the offloaded SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many kernels were tiled, or ``None`` if none were.
        """
        kernels = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and is_kernel(state, node)
        ]
        tiled = 0
        for kernel, state in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            strided = strided_maps(state, kernel)
            fills = give_blocks_their_containers(state, kernel, strided)
            if fills_per_lane_buffer(state, kernel, strided):
                tiled += tile_per_cta(state, kernel)
                continue
            lanes = single_lane_nodes(state, kernel, strided)
            if lanes is None:
                continue
            single, waiting, blocks = lanes
            private = lane_private_containers(state, kernel)
            shared = {
                entry: {
                    edge.data.data
                    for edge in owner.out_edges(owner.exit_node(entry))
                    if not edge.data.is_empty() and (owner.sdfg, edge.data.data) not in private
                }
                for owner, entry in strided
            }
            # WarpTiling strides the non-serial maps; the serial pinning re-applies to each lane's loop.
            for inner in [entry for owner, entry in strided]:
                inner.map.schedule = dtypes.ScheduleType.Default
                inner.specialization_hint = LANE_STRIDED_HINT
            WarpTiling.apply_to(
                state.sdfg,
                options={"warp_size": BLOCK_COLLECTIVE_THREADS, "replicate_maps": False},
                verify=False,
                mapentry=kernel,
            )
            # WarpTiling nests the body, so each tasklet is found again by identity.
            owners = {node: owner for node, owner in sdfg.all_nodes_recursive() if node in single or node in waiting}
            for node in single:
                run_on_lane_zero(node, owners[node])
            for node in waiting:
                run_after_barrier(node, owners[node])
            for block in blocks:
                run_block_on_lane_zero(block)
            for fill in fills:
                fence_with_barriers(fill)
            exits = [
                (node, owner)
                for node, owner in sdfg.all_nodes_recursive()
                if isinstance(node, nodes.MapExit) and owner.entry_node(node) in shared
            ]
            for node, owner in exits:
                fence_after(node, owner, shared[owner.entry_node(node)])
            for lane_map in state.scope_children()[kernel]:
                if (
                    isinstance(lane_map, nodes.MapEntry)
                    and lane_map.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
                ):
                    lane_map.specialization_hint = BLOCK_TILE_HINT
            # The lane map sizes the block now; a declared block size beside it is a conflict.
            kernel.map.gpu_block_size = None
            tiled += 1
        return tiled or None

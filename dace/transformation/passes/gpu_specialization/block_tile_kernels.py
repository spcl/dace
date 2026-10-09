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
block), or when a strided map leaves a lane-private container other code reads afterwards (each lane
would hold only its own share of it).
"""

from typing import Any

from dace import SDFG, Memlet, SDFGState, dtypes, properties
from dace import graphlib as nx
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
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
    the block. A map is parallel by definition; the ``Sequential`` schedule inference gives a map
    nested in a kernel only says one thread would walk it."""
    return [
        (inner_state, inner)
        for inner_state, inner in xfh.get_internal_scopes(state, kernel, immediate=True)
        if isinstance(inner, nodes.MapEntry) and (inner.map.range.size()[-1] < BLOCK_COLLECTIVE_THREADS) != True
    ]


def lane_private(sdfg: SDFG, name: str) -> bool:
    desc = sdfg.arrays[name]
    return desc.transient and desc.storage in LANE_PRIVATE_STORAGE


def reads_outside(sdfg: SDFG, name: str, inner: set[nodes.Node]) -> bool:
    """Whether any node of ``sdfg`` outside ``inner`` reads ``name``."""
    return any(
        isinstance(node, nodes.AccessNode) and node.data == name and node not in inner and state.out_degree(node) > 0
        for state in sdfg.states()
        for node in state.nodes()
    )


def strided_map_is_safe(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A strided map's outputs survive lane splitting: a lane-private output is one element reduced
    with a known identity (folded across the lanes), or nothing outside the map reads it."""
    inner = set(state.scope_subgraph(entry).nodes())
    for edge in state.out_edges(state.exit_node(entry)):
        if edge.data.is_empty() or not lane_private(state.sdfg, edge.data.data):
            continue
        if edge.data.wcr is not None:
            if required(edge.data.subset).num_elements() != 1:
                return False
            continue
        if reads_outside(state.sdfg, edge.data.data, inner):
            return False
    return True


def updates_shared(node: nodes.Tasklet, state: SDFGState) -> bool:
    """Whether ``node`` writes a shared (not lane-private) container that it, or a node leading to it, reads:
    once per lane is wrong. The read may sit in an earlier tasklet (``s = acc[k] + t; acc[k] = s``)."""
    upstream = nx.ancestors(state._nx, node) | {node}
    read = {edge.data.data for edge in state.edges() if edge.dst in upstream and not edge.data.is_empty()}
    return any(
        edge.data.data in read and not lane_private(state.sdfg, edge.data.data)
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


def single_lane_nodes(
    state: SDFGState, kernel: nodes.MapEntry, strided: list[tuple[SDFGState, nodes.MapEntry]]
) -> tuple[list[nodes.Tasklet], list[nodes.Tasklet]] | None:
    """The tasklets that must run on lane 0 alone for the kernel body to run once per lane, with ``strided``
    split across the lanes, and the other tasklets that read what the strided maps wrote, which must wait at a
    barrier for every lane's share; ``None`` if no such split is sound."""
    if not strided or not all(strided_map_is_safe(s, entry) for s, entry in strided):
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
        if not edge.data.is_empty() and not lane_private(st.sdfg, edge.data.data)
    }
    single: list[nodes.Tasklet] = []
    waiting: list[nodes.Tasklet] = []
    pending = [(state, scope.nodes())]
    while pending:
        current, nodes_in_scope = pending.pop()
        for node in nodes_in_scope:
            if node in skipped:
                continue
            if isinstance(node, nodes.NestedSDFG):
                for inner in node.sdfg.states():
                    if accumulates_outside(inner, inner.edges(), skipped):
                        return None
                    pending.append((inner, inner.nodes()))
            elif isinstance(node, nodes.LibraryNode):
                return None
            elif isinstance(node, nodes.Tasklet) and updates_shared(node, current):
                if any(edge.data.is_empty() for edge in current.all_edges(node)):
                    return None
                single.append(node)
            elif isinstance(node, nodes.Tasklet) and any(
                (current.sdfg, edge.data.data) in shared_by_lanes
                for edge in current.in_edges(node)
                if not edge.data.is_empty()
            ):
                waiting.append(node)
    return single, waiting


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


def run_after_barrier(node: nodes.Tasklet, state: SDFGState) -> None:
    """Hold every lane at a barrier before ``node`` reads what the other lanes wrote."""
    fence = barrier(state)
    for pred in state.predecessors(node):
        state.add_edge(pred, None, fence, None, Memlet())
    state.add_edge(fence, None, node, None, Memlet())


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
            lanes = single_lane_nodes(state, kernel, strided)
            if lanes is None:
                continue
            single, waiting = lanes
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

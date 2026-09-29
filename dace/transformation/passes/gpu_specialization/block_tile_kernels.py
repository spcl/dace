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

Everything outside the strided maps runs once per LANE, not once per iteration. That is only sound
when running it several times changes nothing, so a kernel is left alone when its body outside the
strided maps accumulates (a write-conflict-resolved memlet), updates a container it also reads, holds
a library node (its own lowering may need the block), or when a strided map leaves a lane-private
container other code reads afterwards (each lane would hold only its own share of it).
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS
from dace.sdfg import nodes
from dace.transformation import helpers as xfh, pass_pipeline as ppl
from dace.transformation.dataflow.warp_tiling import WarpTiling

#: Storage whose containers each lane holds its own copy of.
LANE_PRIVATE_STORAGE = (dtypes.StorageType.Register, dtypes.StorageType.Default)


def is_kernel(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A ``GPU_Device`` map no other ``GPU_Device`` map encloses."""
    if entry.map.schedule != dtypes.ScheduleType.GPU_Device:
        return False
    return not any(
        isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.GPU_Device
        for scope in xfh.get_parent_map_and_loop_scopes(state.sdfg, entry, state))


def strided_maps(state: SDFGState, kernel: nodes.MapEntry) -> List[Tuple[SDFGState, nodes.MapEntry]]:
    """The inner maps to stride across the lanes: the immediate ones that are not provably narrower than
    the block. A map is parallel by definition; the ``Sequential`` schedule inference gives a map
    nested in a kernel only says one thread would walk it."""
    return [(inner_state, inner) for inner_state, inner in xfh.get_internal_scopes(state, kernel, immediate=True)
            if isinstance(inner, nodes.MapEntry) and (inner.map.range.size()[-1] < BLOCK_COLLECTIVE_THREADS) != True]


def lane_private(sdfg: SDFG, name: str) -> bool:
    desc = sdfg.arrays[name]
    return desc.transient and desc.storage in LANE_PRIVATE_STORAGE


def reads_outside(sdfg: SDFG, name: str, inner: Set[nodes.Node]) -> bool:
    """Whether any node of ``sdfg`` outside ``inner`` reads ``name``."""
    return any(
        isinstance(node, nodes.AccessNode) and node.data == name and node not in inner and state.out_degree(node) > 0
        for state in sdfg.all_states() for node in state.nodes())


def strided_map_is_safe(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A strided map's outputs survive lane splitting: a lane-private output is one element reduced
    with a known identity (folded across the lanes), or nothing outside the map reads it."""
    inner = set(state.scope_subgraph(entry).nodes())
    for edge in state.out_edges(state.exit_node(entry)):
        if edge.data.is_empty() or not lane_private(state.sdfg, edge.data.data):
            continue
        if edge.data.wcr is not None:
            if edge.data.subset.num_elements() != 1:
                return False
            continue
        if reads_outside(state.sdfg, edge.data.data, inner):
            return False
    return True


def redundant_node_is_safe(node: nodes.Node, state: SDFGState) -> bool:
    """A node every lane executes: no library node, no update of a shared container it reads."""
    if isinstance(node, nodes.LibraryNode):
        return False
    if not isinstance(node, nodes.Tasklet):
        return True
    read = {edge.data.data for edge in state.in_edges(node) if not edge.data.is_empty()}
    return not any(edge.data.data in read and not lane_private(state.sdfg, edge.data.data)
                   for edge in state.out_edges(node) if not edge.data.is_empty())


def accumulates_outside(state: SDFGState, edges, skipped: Set[nodes.Node]) -> bool:
    """Whether a write-conflict-resolved edge leaves a node outside the strided maps: every lane would add."""
    return any(edge.data.wcr is not None and edge.src not in skipped for edge in edges)


def lanes_are_safe(state: SDFGState, kernel: nodes.MapEntry, strided: List[Tuple[SDFGState, nodes.MapEntry]]) -> bool:
    """Whether running the kernel body once per lane, with ``strided`` split across the lanes, is sound."""
    if not strided or not all(strided_map_is_safe(s, entry) for s, entry in strided):
        return False
    skipped: Set[nodes.Node] = set()
    for s, entry in strided:
        skipped |= set(s.scope_subgraph(entry).nodes())
    scope = state.scope_subgraph(kernel)
    if accumulates_outside(state, scope.edges(), skipped):
        return False
    pending = [(state, scope.nodes())]
    while pending:
        current, nodes_in_scope = pending.pop()
        for node in nodes_in_scope:
            if node in skipped:
                continue
            if isinstance(node, nodes.NestedSDFG):
                for inner in node.sdfg.all_states():
                    if accumulates_outside(inner, inner.edges(), skipped):
                        return False
                    pending.append((inner, inner.nodes()))
            elif not redundant_node_is_safe(node, current):
                return False
    return True


@properties.make_properties
class BlockTileKernels(ppl.Pass):
    """Run each eligible kernel one block per outer iteration, its inner parallel maps across the lanes."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.States | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Tile every kernel :func:`lanes_are_safe` accepts.

        :param sdfg: the offloaded SDFG, in place.
        :param pipeline_results: unused.
        :returns: how many kernels were tiled, or ``None`` if none were.
        """
        kernels = [(node, state) for node, state in sdfg.all_nodes_recursive()
                   if isinstance(node, nodes.MapEntry) and is_kernel(state, node)]
        tiled = 0
        for kernel, state in kernels:
            if xfh.gpu_map_has_explicit_threadblocks(state, kernel):
                continue
            strided = strided_maps(state, kernel)
            if not lanes_are_safe(state, kernel, strided):
                continue
            # WarpTiling strides the non-serial maps; the serial pinning re-applies to each lane's loop.
            for inner in [entry for owner, entry in strided]:
                inner.map.schedule = dtypes.ScheduleType.Default
            WarpTiling.apply_to(state.sdfg,
                                options={
                                    'warp_size': BLOCK_COLLECTIVE_THREADS,
                                    'replicate_maps': False
                                },
                                verify=False,
                                mapentry=kernel)
            # The lane map sizes the block now; a declared block size beside it is a conflict.
            kernel.map.gpu_block_size = None
            tiled += 1
        return tiled or None

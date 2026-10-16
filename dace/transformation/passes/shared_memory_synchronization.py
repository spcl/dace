# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that inserts ``__syncthreads()`` barriers around GPU shared-memory accesses."""

import warnings

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.tileops.expansions import TILE_THREADS_MAP
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg.narrowing import as_map_entry
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG, Node
from dace.sdfg.scope import is_in_scope
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, transformation
from dace.transformation import pass_pipeline as ppl


def is_shared_memory_write(node: Node, state: SDFGState) -> bool:
    """Whether ``node`` is a ``GPU_Shared`` access node with a non-empty incoming edge."""
    return (
        isinstance(node, AccessNode)
        and node.desc(state).storage == dtypes.StorageType.GPU_Shared
        and any(not edge.data.is_empty() for edge in state.in_edges(node))
    )


@properties.make_properties
@transformation.explicit_cf_compatible
class DefaultSharedMemorySync(ppl.Pass):
    """Insert ``__syncthreads()`` tasklets after ``GPU_ThreadBlock`` map exits and collaborative writes of shared memory.

    Barriers stay outside thread-block maps, where thread divergence would deadlock them. So a shared-memory write in
    a sequential map or loop nested in a thread-block map only warns, a write-then-read inside one thread-block map
    is not synchronized, and nested thread-block maps sync at the outermost exit.
    """

    def apply_pass(self, sdfg: SDFG, _):
        tb_map_exits: dict[MapExit, SDFGState] = {}
        collaborative_smem_copies: dict[AccessNode, SDFGState] = {}
        for node, parent_state in sdfg.all_nodes_recursive():
            if isinstance(node, MapExit) and node.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                # The barriers of a tile node's threads are placed by InsertTileSync
                if not node.map.label.startswith(TILE_THREADS_MAP):
                    tb_map_exits[node] = parent_state
            elif isinstance(node, AccessNode) and self.is_collaborative_smem_write(node, parent_state):
                collaborative_smem_copies[node] = parent_state

        self.insert_synchronization_after_nodes(self.identify_synchronization_tb_exits(tb_map_exits))
        self.insert_synchronization_after_nodes(collaborative_smem_copies)

    def is_collaborative_smem_write(self, node: AccessNode, state: SDFGState) -> bool:
        """Whether ``node`` is shared memory written in a kernel but outside any thread-block map."""
        if node.desc(state).storage != dtypes.StorageType.GPU_Shared:
            return False
        # The barriers of a tile node's threads are placed by InsertTileSync
        if all(
            isinstance(pred, NestedSDFG) and pred.sdfg.name.startswith(TILE_THREADS_MAP)
            for pred in state.predecessors(node)
        ):
            return False
        if all(
            isinstance(pred, MapExit) and pred.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
            for pred in state.predecessors(node)
        ):
            return False
        if all(edge.data.is_empty() for edge in state.in_edges(node)):
            return False
        return is_in_scope(state.sdfg, state, node, [dtypes.ScheduleType.GPU_Device]) and not is_in_scope(
            state.sdfg, state, node, [dtypes.ScheduleType.GPU_ThreadBlock]
        )

    def identify_synchronization_tb_exits(self, tb_map_exits: dict[MapExit, SDFGState]) -> dict[MapExit, SDFGState]:
        """The thread-block exits that write shared memory and need a barrier after them."""
        sync_requiring_exits: dict[MapExit, SDFGState] = {}
        for map_exit, state in tb_map_exits.items():
            map_entry = state.entry_node(map_exit)
            writes_to_smem, race_cond_danger, has_tb_parent = self.tb_exits_analysis(map_entry, map_exit, state)
            if has_tb_parent or not writes_to_smem:
                continue
            if race_cond_danger:
                warnings.warn(
                    f"Race condition danger: LoopRegion or Sequential Map inside ThreadBlock map {map_entry} "
                    "writes to GPU shared memory. No synchronization occurs for intermediate steps, "
                    "because '__syncthreads()' is only called outside the ThreadBlock map to avoid potential deadlocks."
                    "Please consider moving the LoopRegion or Sequential Map outside the ThreadBlock map."
                )
            sync_requiring_exits[map_exit] = state
        return sync_requiring_exits

    def tb_exits_analysis(self, map_entry: MapEntry, map_exit: MapExit, state: SDFGState) -> tuple[bool, bool, bool]:
        """``(writes shared memory, race danger, nested in another thread-block map)`` of a thread-block map.

        The race danger is a shared write inside a sequential map or loop, even a single-iteration one.
        """
        nested_sdfgs = [n.sdfg for n in state.all_nodes_between(map_entry, map_exit) if isinstance(n, NestedSDFG)]
        race_cond_danger = any(self.writes_to_smem_inside_loopregion(sd) for sd in nested_sdfgs) or any(
            as_map_entry(inner_scope).map.schedule == dtypes.ScheduleType.Sequential
            and self.map_writes_to_smem(inner_scope, inner_state)
            for inner_state, inner_scope in helpers.get_internal_scopes(state, map_entry)
        )
        return (
            self.map_writes_to_smem(map_entry, state),
            race_cond_danger,
            nested_in_threadblock_map(state, map_entry),
        )

    def writes_to_smem_inside_loopregion(self, sdfg: SDFG) -> bool:
        """Whether ``sdfg``, nested SDFGs included, writes shared memory inside a loop region."""
        for node in sdfg.nodes():
            if isinstance(node, LoopRegion):
                if any(is_shared_memory_write(subnode, parent) for subnode, parent in node.all_nodes_recursive()):
                    return True
            elif isinstance(node, NestedSDFG) and self.writes_to_smem_inside_loopregion(node.sdfg):
                return True
        return False

    def sdfg_writes_to_smem(self, sdfg: SDFG) -> bool:
        return any(is_shared_memory_write(node, state) for node, state in sdfg.all_nodes_recursive())

    def map_writes_to_smem(self, map_entry: MapEntry, state: SDFGState) -> bool:
        """Whether the map writes shared memory at its exit, in its scope or through a nested SDFG."""
        map_exit = state.exit_node(map_entry)
        if any(
            isinstance(edge.dst, AccessNode)
            and edge.dst.desc(state).storage == dtypes.StorageType.GPU_Shared
            and not edge.data.is_empty()
            for edge in state.out_edges(map_exit)
        ):
            return True
        return any(
            is_shared_memory_write(node, state)
            or (isinstance(node, NestedSDFG) and self.sdfg_writes_to_smem(node.sdfg))
            for node in state.all_nodes_between(map_entry, required(map_exit))
        )

    def insert_synchronization_after_nodes(self, nodes: dict[Node, SDFGState]):
        """Insert a ``__syncthreads()`` tasklet after each given node."""
        for node, state in nodes.items():
            sync_tasklet = state.add_tasklet(
                name="sync_threads",
                inputs=OrderedSet(),
                outputs=OrderedSet(),
                code="__syncthreads();\n",
                language=dtypes.Language.CPP,
            )
            for succ in state.successors(node):
                state.add_edge(sync_tasklet, None, succ, None, dace.Memlet())
            state.add_edge(node, None, sync_tasklet, None, dace.Memlet())


def nested_in_threadblock_map(state: SDFGState, map_entry: MapEntry) -> bool:
    """Whether another ``GPU_ThreadBlock`` map sits between the enclosing kernel and ``map_entry``."""
    parent = helpers.get_parent_map(state, map_entry)
    while parent:
        parent_map, parent_state = parent
        if as_map_entry(parent_map).map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
            return True
        if as_map_entry(parent_map).map.schedule == dtypes.ScheduleType.GPU_Device:
            return False
        parent = helpers.get_parent_map(parent_state, parent_map)
    return False

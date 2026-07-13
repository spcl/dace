# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that inserts ``__syncthreads()`` barriers around GPU shared-memory accesses."""
import warnings
from typing import Dict, Set, Tuple

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG, Node
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class DefaultSharedMemorySync(ppl.Pass):
    """Insert ``__syncthreads()`` tasklets after GPU_ThreadBlock (TB) MapExits
    that write shared memory, and after collaborative shared-memory writes.

    Barriers are kept outside TB maps because calling ``__syncthreads()`` under
    thread divergence deadlocks (worse than a race). Consequences: shared-memory
    writes inside a Sequential map / LoopRegion nested in a TB map only get a
    warning (race risk, no intermediate sync); write-then-read of shared memory
    within one TB map is silently unsynchronized (split into sequential TB maps
    instead); nested TB maps sync only at the outermost TB exit.
    """

    def __init__(self):
        """Initialize the synchronization pass."""
        # Cache each node's parent state during apply_pass()
        self._node_to_parent_state: Dict[Node, SDFGState] = dict()

    def apply_pass(self, sdfg: SDFG, _):
        """Insert ``__syncthreads()`` barriers so shared-memory writes are visible to subsequent reads.

        Collects TB MapExits and collaborative shared-memory write AccessNodes,
        determines which TB exits need a barrier, then inserts barriers after
        those exits and after the collaborative writes.

        :param sdfg: SDFG to insert barriers into (modified in place).
        """

        # 1. Find all GPU_ThreadBlock-scheduled Maps and all collaborative writes to
        #    GPU shared memory, and cache each node's parent state for convenience.
        tb_map_exits: Dict[MapExit, SDFGState] = dict()
        collaborative_smem_copies: Dict[AccessNode, SDFGState] = dict()
        for node, parent_state in sdfg.all_nodes_recursive():
            self._node_to_parent_state[node] = parent_state
            if isinstance(node, MapExit) and node.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                tb_map_exits[node] = parent_state
            elif isinstance(node, AccessNode) and self.is_collaborative_smem_write(node, parent_state):
                collaborative_smem_copies[node] = parent_state

        # 2. Identify TB MapExits requiring a synchronization barrier
        sync_requiring_exits = self.identify_synchronization_tb_exits(tb_map_exits)

        # 3. Insert synchronization barriers for previous TB MapExits
        self.insert_synchronization_after_nodes(sync_requiring_exits)

        # 4. Insert synchronization after collaborative shared memory writes
        self.insert_synchronization_after_nodes(collaborative_smem_copies)

    def is_collaborative_smem_write(self, node: AccessNode, state: SDFGState) -> bool:
        """Whether ``node`` is a collaborative shared-memory write: written
        cooperatively at device level but not within a thread-block map.

        :param node: Candidate access node.
        :param state: State containing ``node``.
        :returns: True if ``node`` is a collaborative shared-memory write.
        """
        # 1. node is not stored in shared memory - skip
        if node.desc(state).storage != dtypes.StorageType.GPU_Shared:
            return False

        # 2. It is not a collaborative write if the result comes from a ThreadBlock map.
        if all(
                isinstance(pred, MapExit) and pred.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
                for pred in state.predecessors(node)):
            return False

        # 3. If all in edges are empty, there is no write - and no sync necessary
        if all(edge.data.is_empty() for edge in state.in_edges(node)):
            return False

        # 4. It is a collaborative copy if it is within a kernel but not within a GPU_ThreadBlock map
        if (not helpers.is_within_schedule_types(state, node, [dtypes.ScheduleType.GPU_Device])
                or helpers.is_within_schedule_types(state, node, [dtypes.ScheduleType.GPU_ThreadBlock])):
            return False

        return True

    def identify_synchronization_tb_exits(self, tb_map_exits: Dict[MapExit, SDFGState]) -> Dict[MapExit, SDFGState]:
        """TB exits after which ``__syncthreads()`` must be called.

        :param tb_map_exits: GPU_ThreadBlock MapExits mapped to their state.
        :returns: Subset of ``tb_map_exits`` that write shared memory and need
            a barrier.
        """
        sync_requiring_exits: Dict[MapExit, SDFGState] = {}

        for map_exit, state in tb_map_exits.items():

            map_entry = state.entry_node(map_exit)
            writes_to_smem, race_cond_danger, has_tb_parent = self.tb_exits_analysis(map_entry, map_exit, state)

            # Skip: if this TB map is nested inside another TB map in the same kernel
            # (i.e., before reaching the GPU_Device map), synchronization responsibility belongs
            # to the outermost such TB map in the kernel.
            if has_tb_parent:
                continue

            # Warn user: potential race condition detected.
            elif race_cond_danger and writes_to_smem:
                warnings.warn(
                    f"Race condition danger: LoopRegion or Sequential Map inside ThreadBlock map {map_entry} "
                    "writes to GPU shared memory. No synchronization occurs for intermediate steps, "
                    "because '__syncthreads()' is only called outside the ThreadBlock map to avoid potential deadlocks."
                    "Please consider moving the LoopRegion or Sequential Map outside the ThreadBlock map.")
                sync_requiring_exits[map_exit] = state

            # TB map writes to shared memory: synchronization is needed
            elif writes_to_smem:
                sync_requiring_exits[map_exit] = state

        return sync_requiring_exits

    def tb_exits_analysis(self, map_entry: MapEntry, map_exit: MapExit, state: SDFGState) -> Tuple[bool, bool, bool]:
        """Analyze a GPU_ThreadBlock map.

        :param map_entry: TB map entry node.
        :param map_exit: TB map exit node.
        :param state: Parent state containing the map.
        :returns: ``(writes_to_shared_memory, race_cond_danger,
            has_parent_tb_map)``. ``writes_to_shared_memory`` covers writes at
            the MapExit or inside the scope. ``race_cond_danger`` flags shared
            writes inside a Sequential map or LoopRegion (single-iteration
            ones are still flagged though they cannot race).
            ``has_parent_tb_map`` is True if another TB map sits between the
            enclosing GPU_Device map and this one.
        """
        writes_to_shared_memory = False
        race_cond_danger = False
        has_parent_tb_map = False

        # 1. Check if the ThreadBlock (TB) map writes to shared memory
        for edge in state.out_edges(map_exit):
            is_smem: bool = (isinstance(edge.dst, AccessNode)
                             and edge.dst.desc(state).storage == dtypes.StorageType.GPU_Shared)
            if is_smem and not edge.data.is_empty():
                writes_to_shared_memory = True
                break

        # 2. Search between map entry and exit:
        #    - Detect writes to shared memory (unless already found)
        #    - Collect nested SDFGs for later analysis
        nested_sdfgs: Set[NestedSDFG] = set()

        for node in state.all_nodes_between(map_entry, map_exit):
            if not writes_to_shared_memory and isinstance(node, AccessNode):
                if (node.desc(state).storage == dtypes.StorageType.GPU_Shared
                        and any(not edge.data.is_empty() for edge in state.in_edges(node))):
                    writes_to_shared_memory = True

            elif isinstance(node, NestedSDFG):
                nested_sdfgs.add(node)

        # 3. Recursively analyze nested SDFGs:
        #    - Detect shared memory writes (only if not already found)
        #    - Check for potential race conditions in loop regions (only if not already flagged)
        for nsdfg in nested_sdfgs:
            subs_sdfg = nsdfg.sdfg
            if not writes_to_shared_memory:
                writes_to_shared_memory = self.sdfg_writes_to_smem(subs_sdfg)

            if not race_cond_danger:
                race_cond_danger = self.writes_to_smem_inside_loopregion(subs_sdfg)

        # 4. Check for race condition danger in sequential maps that use shared memory
        #    (only if not already flagged)
        if not race_cond_danger:
            race_cond_danger = any(
                inner_scope.map.schedule == dtypes.ScheduleType.Sequential and self.map_writes_to_smem(inner_scope)
                for _, inner_scope in helpers.get_internal_scopes(state, map_entry))

        # 5. Check if this TB map is nested within another TB map
        parent = helpers.get_parent_map(state, map_entry)

        while parent:
            parent_map, parent_state = parent
            if parent_map.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                has_parent_tb_map = True
                break
            if parent_map.map.schedule == dtypes.ScheduleType.GPU_Device:
                break
            parent = helpers.get_parent_map(parent_state, parent_map)

        return writes_to_shared_memory, race_cond_danger, has_parent_tb_map

    def writes_to_smem_inside_loopregion(self, sdfg: SDFG) -> bool:
        """True if the SDFG writes shared memory inside a LoopRegion
        (recursive, including nested SDFGs)."""
        for node in sdfg.nodes():
            if isinstance(node, LoopRegion):
                for subnode, parent in node.all_nodes_recursive():
                    if (isinstance(subnode, AccessNode)
                            and subnode.desc(parent).storage == dtypes.StorageType.GPU_Shared
                            and any(not edge.data.is_empty() for edge in parent.in_edges(node))):
                        return True

            elif isinstance(node, NestedSDFG):
                if self.writes_to_smem_inside_loopregion(node.sdfg):
                    return True

        return False

    def sdfg_writes_to_smem(self, sdfg: SDFG) -> bool:
        """True if the SDFG has a GPU_Shared AccessNode with a non-empty
        incoming edge (i.e. writes shared memory)."""
        for node, state in sdfg.all_nodes_recursive():
            if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Shared
                    and any(not edge.data.is_empty() for edge in state.in_edges(node))):
                return True
        return False

    def map_writes_to_smem(self, map_entry: MapEntry) -> bool:
        """True if the map writes shared memory -- at its MapExit, within its
        scope, or via a nested SDFG."""
        state = self._node_to_parent_state[map_entry]
        map_exit = state.exit_node(map_entry)

        # 1. Check if MapExit writes directly to shared memory
        for edge in state.out_edges(map_exit):
            if (isinstance(edge.dst, AccessNode) and edge.dst.desc(state).storage == dtypes.StorageType.GPU_Shared
                    and not edge.data.is_empty()):
                return True

        # 2. Inspect nodes inside the map scope
        for node in state.all_nodes_between(map_entry, map_exit):
            if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Shared
                    and any(not edge.data.is_empty() for edge in state.in_edges(node))):
                return True

            if isinstance(node, NestedSDFG) and self.sdfg_writes_to_smem(node.sdfg):
                return True

        return False

    def insert_synchronization_after_nodes(self, nodes: Dict[Node, SDFGState]):
        """Insert a ``__syncthreads()`` tasklet after each given node.

        :param nodes: Nodes mapped to their parent state.
        """
        for node, state in nodes.items():

            sync_tasklet = state.add_tasklet(name="sync_threads",
                                             inputs=set(),
                                             outputs=set(),
                                             code="__syncthreads();\n",
                                             language=dtypes.Language.CPP)

            for succ in state.successors(node):
                state.add_edge(sync_tasklet, None, succ, None, dace.Memlet())

            state.add_edge(node, None, sync_tasklet, None, dace.Memlet())

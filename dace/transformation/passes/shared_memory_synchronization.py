# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import warnings
from typing import Dict, Set, Tuple

import dace
from dace import SDFG, SDFGState, dtypes, properties
from dace.codegen.targets.experimental_cuda_helpers import gpu_utils
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG, Node
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class DefaultSharedMemorySync(ppl.Pass):
    """
    This pass inserts synchronization tasklets that call "__syncthreads()".
    This is for GPUs.

    Synchronization is added after GPU_ThreadBlock (TB) MapExits if the TB map
    writes to shared memory or after collaborative writes to shared memory (smem).

    Important notes:
    - Calling "__syncthreads()" inside a TB map can lead to deadlocks, 
      for example when only a subset of threads participates (thread divergence). 
      Therefore, users must **not** write to shared memory inside a Sequential 
      map or LoopRegion that is nested within a TB map.

    - If shared memory is still written sequentially within a TB map, the missing 
      intermediate synchronizations may lead to race conditions and incorrect results. 
      Because deadlocks are worse than race conditions, this pass avoids inserting 
      synchronization inside TB maps, but it will warn the user about potential risks.

    - When writing to and reading from shared memory within the same TB map, 
      users must ensure that no synchronization is required, since barriers 
      are not inserted automatically in this case (again, to avoid deadlocks). 
      If synchronization is needed, the computation should instead be split 
      across sequential TB maps. There is no warning for race conditions in this 
      case for misbehavior.

    - In nested TB maps (e.g., GPU_Device map -> TB map -> TB map ...), 
      synchronization is only inserted at the outermost TB map's exit, 
      again to avoid deadlocks.
    """

    def __init__(self):
        """Initialize the synchronization pass."""
        # Cache each node's parent state during apply_pass()
        self._node_to_parent_state: Dict[Node, SDFGState] = dict()

    def apply_pass(self, sdfg: SDFG, _) -> None:
        """
        Insert synchronization barriers (`__syncthreads()`) where needed to ensure
        shared memory writes are synchronied for potential subsequent reads.

        This pass performs the following steps:
        1. Collect all ThreadBlock-scheduled MapExits and candidate collaborative
           shared-memory writes (AccessNodes).
        2. Analyze ThreadBlock MapExits for synchronization requirements.
        3. Insert synchronization barriers after both MapExits and collaborative
           shared-memory writes as needed.
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
        """
        Determine whether the given AccessNode corresponds to a collaborative
        shared-memory (smem) write, i.e., whether it is written cooperatively
        by GPU threads at the device level but not within a thread block map.

        Parameters
        ----------
        node : AccessNode
            The candidate access node.
        state : SDFGState
            The state in which the node resides.

        Returns
        -------
        bool
            True if the node is a collaborative smem write, False otherwise.
        """
        # 1. node is not stored in shared memory - skip
        if node.desc(state).storage != dtypes.StorageType.GPU_Shared:
            return False
        
        # 2. No writes to the shared memory - skip
        if state.in_degree(node) == 0:
            return False 
        
        # 3. It is a collaborative copy if it is within a kernel but not within a GPU_ThreadBlock map
        if (not gpu_utils.is_within_schedule_types(state, node, [dtypes.ScheduleType.GPU_Device]) 
            or gpu_utils.is_within_schedule_types(state, node, [dtypes.ScheduleType.GPU_ThreadBlock])):
            return False

        return True
        
    def identify_synchronization_tb_exits(self, tb_map_exits: Dict[MapExit, SDFGState]) -> Dict[MapExit, SDFGState]:
        """
        Identify ThreadBlock exits after which "__syncthreads()" should be called.

        Parameters
        ----------
        tb_map_exits : Dict[MapExit, SDFGState]
            Mapping from GPU_ThreadBlock - scheduled MapExit nodes to their parent SDFGState.

        Returns
        -------
        Dict[MapExit, SDFGState]
            Subset of `tb_map_exits` where any AccessNode between the entry and exit
            uses GPU shared memory, indicating a synchronization barrier is needed.
        """
        #------------------------- helper function -------------------------
        sync_requiring_exits: Dict[MapExit, SDFGState] = {}

        for map_exit, state in tb_map_exits.items():

            # process
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
        """
        Analyze a GPU_ThreadBlock-scheduled map to determine:
        - whether it writes to shared memory,
        - whether such writes may cause race conditions, and
        - whether it is nested within another GPU_ThreadBlock map inside the kernel.

        Returns a tuple of three booleans:

        1. `writes_to_shared_memory`:
            True if the map writes to GPU shared memory. This includes writes
            directly at the MapExit or within the map scope.

        2. `race_cond_danger`:
           True if there is a potential race condition due to shared memory writes
           inside either:
             - a sequentially scheduled map, or
             - a loop region.
           (Note: single-iteration loops/sequential maps are not treated differently;
           they are still marked as dangerous, even though they cannot cause races.)

        3. `has_parent_tb_map`:
           True if this ThreadBlock map is nested inside another ThreadBlock map
           (i.e., there exists another TB map between the enclosing GPU_Device
            map and the current TB map).

        Parameters
        ----------
        map_entry : MapEntry
            The entry node of the ThreadBlock map.
        map_exit : MapExit
            The exit node of the ThreadBlock map.
        state : SDFGState
            The parent state containing the map.

        Returns
        -------
        Tuple[bool, bool, bool]
            A tuple:
            `(writes_to_shared_memory, race_cond_danger, has_parent_tb_map)`
        """
        # Initially, the flags are all set to False
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
                # Check if this AccessNode writes to shared memory
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

        # 6. Return the results
        return writes_to_shared_memory, race_cond_danger, has_parent_tb_map

    def writes_to_smem_inside_loopregion(self, sdfg: SDFG) -> bool:
        """
        Return True if the SDFG writes to GPU shared memory (smem) inside
        a LoopRegion. This check is recursive and includes nested SDFGs.
        """
        for node in sdfg.nodes():
            if isinstance(node, LoopRegion):
                # Traverse all nodes inside the loop region
                for subnode, parent in node.all_nodes_recursive():
                    if (isinstance(subnode, AccessNode)
                            and subnode.desc(parent).storage == dtypes.StorageType.GPU_Shared
                            and any(not edge.data.is_empty() for edge in parent.in_edges(node))):
                        return True

            elif isinstance(node, NestedSDFG):
                # Recurse into nested SDFGs
                if self.writes_to_smem_inside_loopregion(node.sdfg):
                    return True

        return False

    def sdfg_writes_to_smem(self, sdfg: SDFG) -> bool:
        """
        Return True if the SDFG writes to GPU shared memory (smem),
        i.e., contains an AccessNode with GPU_Shared storage that has
        at least one non-empty incoming edge.
        """
        for node, state in sdfg.all_nodes_recursive():
            if (isinstance(node, AccessNode) and node.desc(state).storage == dtypes.StorageType.GPU_Shared
                    and any(not edge.data.is_empty() for edge in state.in_edges(node))):
                return True
        return False

    def map_writes_to_smem(self, map_entry: MapEntry) -> bool:
        """
        Return True if the map writes to GPU shared memory (smem).

        A map is considered to write to smem if:
        - Any AccessNode with GPU_Shared storage is written to at the MapExit, or
        - Such writes occur within the map scope, or
        - A nested SDFG within the map writes to smem.
        """
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

        # No writes to shared memory found
        return False

    def insert_synchronization_after_nodes(self, nodes: Dict[Node, SDFGState]) -> None:
        """
        Insert synchronization tasklets (calling `__syncthreads()`) after the given
        GPU-related nodes.

        Parameters
        ----------
        nodes : Dict[Node, SDFGState]
            Mapping from SDFG nodes to their parent states after which a
            synchronization tasklet should be inserted.
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

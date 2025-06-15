from typing import Union, Dict, Set

import functools
import sympy

import dace
from dace import SDFG, properties, SDFGState
from dace import dtypes
from dace.codegen import common
from dace.config import Config
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg import nodes, InterstateEdge
from dace.sdfg.graph import Edge

from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowBlock
from dace.sdfg.nodes import AccessNode, Map, MapEntry, MapExit

from dace.transformation.passes import analysis as ap

@properties.make_properties
@transformation.explicit_cf_compatible
class DefaultSharedMemorySync(ppl.Pass):
    """
    A DaCe transformation pass that automatically inserts GPU synchronization barriers
    (__syncthreads()) for shared memory access patterns.
    
    This pass ensures proper synchronization in two scenarios:
    1. Pre-synchronization: Before consuming shared memory data (AccessNode -> CodeNode/MapEntry)
    2. Post-synchronization: After shared memory reuse in sequential loops/maps within GPU kernels
    
    The pass traverses the SDFG hierarchy and identifies shared memory access patterns
    that require synchronization to prevent race conditions in GPU code.
    
    NOTE: This implementation handles commonly observed patterns. Unsupported cases
    raise NotImplementedError with context for extending the implementation once comming across 
    another constellation which was not observed in the used common examples.
    """


    def __init__(self):
        """Initialize the synchronization pass."""

        # Track which scopes (sequential maps and Loops) have already been
        # synchronized to avoid duplicate barriers
        self._synchronized_scopes: Set[Union[MapExit, LoopRegion]] = set()
        
        # Map from MapExit nodes to their containing states for post-synchronization
        self._map_exit_to_state: Dict[MapExit, SDFGState] = dict()

        # Keep track of processed nested sdfgs
        self._processed_nsdfg = set()
        


    def apply_pass(self, sdfg: SDFG, _) -> None:
        """
        Apply the synchronization pass to the entire SDFG.
        
        Args:
            sdfg: The SDFG to process (expected to be top-level)
            _: Unused pass pipeline argument
        """
        # Start processing from the top-level with empty scope stack
        # The scope stack tracks nested execution contexts (maps, loops)
        enclosing_scopes = []
        self._process_sdfg(sdfg, enclosing_scopes)



    def _process_sdfg(self, sdfg: SDFG, enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Recursively traverse all nodes in an SDFG, handling different node types.
        
        Args:
            sdfg: The SDFG to traverse
            enclosing_scopes: Stack of execution scopes (e.g., maps, loops) enclosing the SDFG as a whole.
        """
        for sdfg_elem in sdfg.nodes():
            self._process_sdfg_element(sdfg, sdfg_elem, enclosing_scopes)

            
    def _process_sdfg_element(self, sdfg: SDFG, element: any,  enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Identifies the type of the SDFG element and processes it using the corresponding handler.

        Args:
            sdfg: The current SDFG we are in (innermost if nested)
            enclosing_scopes: Stack of enclosing execution scopes (maps, loops) wrapping the current SDFG
        """
        if isinstance(element, LoopRegion):
            self._process_loop_region(sdfg, element, enclosing_scopes)

        elif isinstance(element, SDFGState):
            self._process_state(sdfg, element, enclosing_scopes)

        elif isinstance(element, ConditionalBlock):
            self._process_conditionalBlock(sdfg, element, enclosing_scopes)

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}: Unsupported node type '{type(element).__name__}' "
                f"encountered during SDFG traversal. Please extend the implementation to handle this case."
            )

    def _process_loop_region(self, sdfg: SDFG, loop_region: LoopRegion, 
                           enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Process a loop region by adding it to the scope stack and traversing its contents.
        
        Args:
            sdfg: The containing SDFG
            loop_region: The loop region to process
            enclosing_scopes: Current scope stack which wraps around state
        """
        # Create a new scope stack with this loop region added
        nested_scopes = enclosing_scopes.copy()
        nested_scopes.insert(0, loop_region) # Not append! :) careful

        # Process all states within the loop region
        for node in loop_region.nodes():
            if isinstance(node, SDFGState):
                self._process_state(sdfg, node, nested_scopes)
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: Unexpected node type '{type(node).__name__}' "
                    f"found inside LoopRegion. SDFGState nodes were expected. Extend if you think"
                    "the node type is also valid"
                )


    def _process_state(self, sdfg: SDFG, state: SDFGState, 
                      enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Process a single SDFG state, analyzing edges for shared memory access patterns.
        
        Args:
            sdfg: The containing SDFG
            state: The state to process
            enclosing_scopes: Current scope stack which wrapp around state (NOT of each individual node)
        """
        # Track destination nodes that already have synchronization tasklets
        # This prevents creating duplicate barriers for the same consumer
        nodes_with_sync: Dict[nodes.Node, nodes.Tasklet] = {}

        # Analyze each edge in the state for shared memory access patterns
        for edge in state.edges():
            source_node, dest_node = edge.src, edge.dst

            # Skip edges that don't involve shared memory reads
            # (either source is not shared memory, or it's a memory-to-memory copy)
            if not self._is_shared_memory_access_node(sdfg, source_node) or isinstance(dest_node, nodes.AccessNode):
                continue


            # Handle different types of shared memory consumers
            if isinstance(dest_node, (nodes.CodeNode, nodes.MapEntry)):
                # Direct consumption by computation or map entry
                self._insert_pre_synchronization_barrier(source_node, dest_node, state, nodes_with_sync)
                
            elif isinstance(dest_node, nodes.NestedSDFG):
                # Consumption by nested SDFG - synchronize and recurse
                # NOTE: For nesting, we append all scopes which wrap around the nestedSDFG
                self._insert_pre_synchronization_barrier(source_node, dest_node, state, nodes_with_sync)
                nested_scopes = self._build_nested_scope_stack(state, dest_node, enclosing_scopes)
                self._process_sdfg(dest_node.sdfg, nested_scopes)
                self._processed_nsdfg.add(dest_node)
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: Unsupported destination node type '{type(dest_node).__name__}' "
                    f"for shared memory access. Currently supported: CodeNode, MapEntry, AccessNode, NestedSDFG."
                )

            # Check if post-synchronization is needed and apply shared
            self._handle_shared_memory_post_synchronization(state, source_node, enclosing_scopes)

        
        # It may be the case that nestedSDFG were not recursed previously. Process them in that case
        for node in state.nodes():

            # Guards
            if not isinstance(node, nodes.NestedSDFG):
                continue
            if node in self._processed_nsdfg:
                continue

            # not yet processed NestedSDFG
            nested_scopes = self._build_nested_scope_stack(state, node, enclosing_scopes)
            self._process_sdfg(node.sdfg, nested_scopes)
            self._processed_nsdfg.add(node)


    def _process_conditionalBlock(self, sdfg: SDFG, cond_block: ConditionalBlock, 
                                  enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Processes a ConditionalBlock by visiting each clause body and its elements.

        Args:
            sdfg: The current SDFG context.
            cond_block: The ConditionalBlock to process (e.g., if-elif-else structure).
            enclosing_scopes: Stack of execution scopes (e.g., maps, loops) enclosing the SDFG as a whole.
        """
        clause_bodies: list[ControlFlowBlock] = cond_block.nodes()
        
        for body in clause_bodies:
            for sdfg_elem in body.nodes():
                self._process_sdfg_element(sdfg, sdfg_elem, enclosing_scopes)

        

    def _is_shared_memory_access_node(self, sdfg: SDFG, node: nodes.Node) -> bool:
        """
        Check if a node represents a GPU shared memory access.
        
        Args:
            sdfg: The containing SDFG
            node: The node to check
            
        Returns:
            True if the node is an AccessNode with GPU_Shared storage
        """
        return (
            isinstance(node, nodes.AccessNode)
            and node.desc(sdfg).storage == dtypes.StorageType.GPU_Shared
        )



    def _insert_pre_synchronization_barrier(self, source_node: nodes.Node, dest_node: nodes.Node, 
                                          state: SDFGState, nodes_with_sync: Dict[nodes.Node, nodes.Tasklet]) -> None:
        """
        Insert a __syncthreads() barrier before shared memory consumption.
        Reuses existing barriers when multiple shared memory sources feed the same destination.
        
        Args:
            source_node: The shared memory AccessNode
            dest_node: The consuming node
            state: The containing state
            nodes_with_sync: Map tracking existing synchronization tasklets
        """
        if dest_node in nodes_with_sync:
            # Reuse existing synchronization barrier for this destination
            existing_barrier = nodes_with_sync[dest_node]
            state.add_edge(source_node, None, existing_barrier, None, dace.Memlet())
        else:
            # Create a new synchronization barrier
            sync_barrier = state.add_tasklet(
                name="pre_sync_barrier",
                inputs=set(),
                outputs=set(),
                code="__syncthreads();\n",
                language=dtypes.Language.CPP
            )

            # Connect: shared_memory -> sync_barrier -> consumer
            state.add_edge(source_node, None, sync_barrier, None, dace.Memlet())
            state.add_edge(sync_barrier, None, dest_node, None, dace.Memlet())
            nodes_with_sync[dest_node] = sync_barrier

    def _build_nested_scope_stack(self, state: SDFGState, nested_sdfg_node: nodes.NestedSDFG,
                                enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> list[Union[MapExit, LoopRegion]]:
        """
        Copy the 'enclosing_scopes' stack and extend it with all maps in 'state' that enclose 'nested_sdfg_node'.
        It is assumed that the 'enclosing_scopes' stack contains all maps and loops that wrap around 'state', but
        not individual nodes within 'state'.
        
        Args:
            state: The state containing the nested SDFG
            nested_sdfg_node: The NestedSDFG node
            enclosing_scopes: Current scope stack
            
        Returns:
            Updated scope stack including maps enclosing the nested SDFG
        """
        scope_dict = state.scope_dict()
        updated_scopes = enclosing_scopes.copy()

        # Walk up the scope hierarchy, adding all enclosing maps
        current_map = scope_dict[nested_sdfg_node]
        while current_map is not None:

            # Add MapExit node to scope, since it is only needed
            # for post synchronization anyways
            map_exit = state.exit_node(current_map)
            updated_scopes.append(map_exit)

            # add the current state in which the map_exit is contained,
            # needed for potential post synchronization barriers
            self._map_exit_to_state[map_exit] = state
            
            # move up in the nested map hierarchy
            current_map = scope_dict[current_map]

        return updated_scopes


    def _handle_shared_memory_post_synchronization(self, state: SDFGState, shared_mem_node: nodes.Node,
                                                  enclosing_scopes: list[Union[MapExit, LoopRegion]]) -> None:
        """
        Handle post-synchronization for shared memory reuse in sequential execution contexts.
        
        When shared memory is reused across iterations in a for loop or sequential map within
        a GPU kernel, we need post-synchronization barriers to prevent race conditions.
        
        Args:
            state: The state containing the shared memory access
            shared_mem_node: The shared memory AccessNode
            enclosing_scopes: Current scope stack
        """
        scope_dict = state.scope_dict()
        complete_scope_stack = enclosing_scopes.copy()

        # Build complete scope stack including maps inside the current state
        # enclosing the shared memory node. Analogous as in _build_nested_scope_stack()
        current_map = scope_dict[shared_mem_node]
        while current_map is not None:

            map_exit = state.exit_node(current_map)
            complete_scope_stack.append(map_exit)
            self._map_exit_to_state[map_exit] = state
            current_map = scope_dict[current_map]

        # Analyze scope stack to find synchronization requirements
        inside_gpu_kernel = False
        innermost_sequential_scope = None

        # Process scopes from outermost to innermost
        while complete_scope_stack:
            scope = complete_scope_stack.pop(0)

            if isinstance(scope, MapExit):
                schedule = scope.schedule
                if schedule == dtypes.ScheduleType.Sequential and innermost_sequential_scope is None:

                    # Special: Skip if there is only one iteration
                    size_per_dim = scope.map.range.size()
                    number_total_iterations = functools.reduce(sympy.Mul, size_per_dim, 1)
                    if number_total_iterations.is_number and number_total_iterations <= 1:
                        continue

                    innermost_sequential_scope = scope

                elif schedule == dtypes.ScheduleType.GPU_Device:
                    inside_gpu_kernel = True
                    break
            elif isinstance(scope, LoopRegion) and innermost_sequential_scope is None:

                # Special: Skip if there is only one iteration
                start = ap.get_init_assignment(scope)
                end = ap.get_loop_end(scope)
                stride = ap.get_loop_stride(scope)
                nr_iter = (end - start) / stride

                if nr_iter.is_number and nr_iter <= 1:
                    continue

                innermost_sequential_scope = scope

        # Validate that shared memory is used within GPU kernel context
        if not inside_gpu_kernel:
            raise ValueError(
                "Shared memory usage detected outside GPU kernel context. "
                "GPU shared memory is only valid within GPU_Device scheduled maps."
            )

        # No post synchronization needed if there's no sequential iteration context
        if innermost_sequential_scope is None:
            return


        # Apply appropriate post-synchronization based on scope type
        if isinstance(innermost_sequential_scope, MapExit):
            self._add_post_sync_for_sequential_map(innermost_sequential_scope)
        elif isinstance(innermost_sequential_scope, LoopRegion):
            # two options, see docstrings
            self._add_post_sync_tasklets_for_loop_region(innermost_sequential_scope)
            # self._add_post_sync_state_for_loop_region(innermost_sequential_scope)


    
    def _add_post_sync_for_sequential_map(self, seq_map_exit: MapExit) -> None:
        """
        Add post-synchronization barrier after a sequential map that may reuse shared memory.
        
        Args:
            seq_map_exit: The MapExit node of the sequential map
        """
        # Avoid duplicate synchronization
        if seq_map_exit in self._synchronized_scopes:
            return
        
        # Find the state containing this map
        containing_state = self._map_exit_to_state[seq_map_exit]
        
        # Create post-synchronization barrier
        post_sync_barrier = containing_state.add_tasklet(
            name="post_sync_barrier",
            inputs=set(),
            outputs=set(),
            code="__syncthreads();\n",
            language=dtypes.Language.CPP
        )

        # Insert barrier before the map exit and all other predecessors
        incoming_edges = containing_state.in_edges(seq_map_exit)
        for edge in incoming_edges:

            predecessor = edge.src
            containing_state.add_edge(predecessor, None, post_sync_barrier, None, dace.Memlet())
            containing_state.add_edge(post_sync_barrier, None, seq_map_exit, None, dace.Memlet())
            

        # Mark as synchronized
        self._synchronized_scopes.add(seq_map_exit)

    def _add_post_sync_state_for_loop_region(self, loop_region: LoopRegion) -> None:
        """
        Add post-synchronization barrier for a loop region that reuses shared memory arrays.
        It adds a new state, which contains only a synchronization tasklet that connects
        to all sink blocks of the loop region.
        
        Args:
            loop_region: The LoopRegion that needs post-synchronization
        """

        sink_blocks: list[ControlFlowBlock] = []
        for block in loop_region.nodes():

            if loop_region.out_degree(block) == 0:
                sink_blocks.append(block)

        # No sync needed
        if len(sink_blocks) < 0:
            return
        
        # Add new state which synchronizates all sink nodes of the loop
        syn_block = loop_region.add_state("sync_state")
        syn_block.add_tasklet(
            name="post_sync_barrier",
            inputs=set(),
            outputs=set(),
            code="__syncthreads();\n",
            language=dtypes.Language.CPP
        )


        for block in sink_blocks:
            loop_region.add_edge(block, syn_block, InterstateEdge())

        # Mark as synchronized
        self._synchronized_scopes.add(loop_region)


    def _add_post_sync_tasklets_for_loop_region(self, loop_region: LoopRegion) -> None:
        """
        Add post-synchronization barrier for a loop region that reuses shared memory arrays.
        Determines all sink blocks in the LoopRegion, and then, for each sink block, adds a new synchronization
        tasklet that connects to all sink nodes within that sink block.
        
        Args:
            loop_region: The LoopRegion that needs post-synchronization
        """

        sink_blocks: list[SDFGState] = []
        for block in loop_region.nodes():

            if not isinstance(block, SDFGState):
                raise NotImplementedError(f"Block {block} is expected to be an SDFG state. But it is of type {type(block)}. "
                                           "Extend use case if this should be valid."
                                           )
            
            if loop_region.out_degree(block) == 0:
                sink_blocks.append(block)

        # No sync needed
        if len(sink_blocks) < 0:
            return
        

        # For each sink block, synchronize at the end
        for block in sink_blocks:
            
            sink_nodes: list[nodes.Node] = block.sink_nodes()

            # All sink nodes in the same block (= state) get the same sync tasklet
            post_sync_barrier = block.add_tasklet(
                name="post_sync_barrier",
                inputs=set(),
                outputs=set(),
                code="__syncthreads();\n",
                language=dtypes.Language.CPP
            )

            for snode in sink_nodes:
                block.add_edge(snode, None, post_sync_barrier, None, dace.Memlet())

            
        # Mark as synchronized
        self._synchronized_scopes.add(loop_region)
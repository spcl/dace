from typing import Union, Dict, Set, List, Tuple

import dace
from dace import SDFG, properties, SDFGState
from dace import dtypes
from dace.codegen import common
from dace.config import Config
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg import nodes
from dace.sdfg.graph import Graph, NodeT


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(ppl.Pass):
    """
    Assigns GPU streams to relevant nodes and inserts synchronization tasklets where needed.

    Strategy Overview:
    ------------------
    - GPU stream assignment is based on weakly connected components (WCCs) within each state.
    - "Relevant nodes" in a WCC are assigned to the same stream.
        Relevant nodes include:
          * AccessNodes in GPU memory,
          * GPU-scheduled nodes (Maps or Library nodes),
          * Nodes directly connected to the above.
    - For top-level states (not within nested SDFGs), each new WCC starts on a new stream (starting from 0).
    - In nested SDFGs:
        * Stream assignment is inherited from the parent component,
        * All internal components share the parent's stream (consider revisiting this for performance tuning).
    - GPU stream IDs wrap around according to the `max_concurrent_streams` configuration.
    - Synchronization tasklets are inserted using a simple heuristic:
        * At the end of a state, if outputs certain patterns regarding GPU memory occur,
        * After a node, if its outputs cross GPU boundaries and are reused downstream.

    Example:
    --------
    A state with the following independent chains:
        K1 → K2
        K3 → K4 → K5
        K6

    would be scheduled as:
        K1, K2     → stream 0  
        K3, K4, K5 → stream 1  
        K6         → stream 2  

    (assuming no limit on the number of concurrent streams)

    Note:
    -----
    These refer to **backend GPU streams** (e.g., CUDA or HIP), not DaCe symbolic streams.
    """

    def __init__(self):
        # max configured number of concurrent streams
        self._max_concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

        # needed to call correct backend synchronization functions
        self._backend: str = common.get_gpu_backend() 

        # This is expected to be set by the calling backend code generator before applying the pass
        self._gpu_stream_access_template: str = ""

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, Union[int, str]]:
        """
        Assigns GPU streams to nodes and inserts synchronization tasklets where needed.
        """

        # 1. Traverse each top-level state and assign stream IDs to eligible nodes (starting from stream 0).
        assigned_nodes = dict()
        for state in sdfg.states():
            self._assign_gpu_streams_in_state(sdfg, False, state, assigned_nodes, 0)

        # 2. If only one stream is used set all assignments to "nullptr".
        num_assigned_streams = max(assigned_nodes.values(), default=0)  # self.max_concurrent_streams == -1 (default) also handled here
        if num_assigned_streams == 0:  
            for k in assigned_nodes.keys():
                assigned_nodes[k] = "nullptr"

        # 3. Insert synchronization tasklets based on stream usage.
        self._insert_gpu_stream_sync_tasklet(sdfg, assigned_nodes)

        return assigned_nodes

    def _assign_gpu_streams_in_state(self, sdfg: SDFG, in_nested_sdfg: bool, state: SDFGState, assigned_nodes: Dict, gpu_stream:int):
        """
        Processes connected components in a state, assigning each to a different GPU stream if not inside a nested SDFG.
        If inside a nested SDFG, components inherit the stream from the parent state/component.

        Stream assignment is performed only for components that contain GPU-related nodes;
        components without such nodes are skipped.
        """
        components = self._get_weakly_connected_nodes(state)

        for component in components:
            nodes_assigned_before = len(assigned_nodes)

            for node in component:
                
                if self._is_relevant_for_gpu_stream(node, sdfg, state):
                    assigned_nodes[node] = gpu_stream
                
                if isinstance(node, nodes.NestedSDFG):
                    for nested_state in node.sdfg.states():
                        self._assign_gpu_streams_in_state(node.sdfg, True, nested_state, assigned_nodes, gpu_stream)
                    
            # Move to next stream if we assigned streams to any node in this component (careful: if nested, states are in same component)
            if not in_nested_sdfg and len(assigned_nodes) > nodes_assigned_before:
                gpu_stream = self._next_stream(gpu_stream)

    def _get_weakly_connected_nodes(self, graph: Graph) -> List[Set[NodeT]]:
        """
        Returns all weakly connected components in the given directed graph.

        A weakly connected component is a maximal group of nodes such that each pair 
        of nodes is connected by a path when ignoring edge directions.

        :param graph: A directed graph (Graph) instance.
        :return: A list of sets, each containing the nodes of one weakly connected component.
        """
        visited: Set[NodeT] = set()
        components: List[Set[NodeT]] = []

        for node in graph.nodes():
            if node in visited:
                continue

            # Start a new weakly connected component
            component: Set[NodeT] = set()
            stack = [node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        return components

    def _is_relevant_for_gpu_stream(self, node: nodes.Node, sdfg: SDFG, state: SDFGState) -> bool:
        """
        Determines whether a node is relevant for GPU stream assignment.

        A node is considered relevant if:
        - It is an AccessNode accessing GPU global memory,
        - It is a GPU-scheduled map entry/exit node (i.e., a kernel),
        - It is a GPU-scheduled library node,
        - Or it is directly connected (via in/out edges) to such a node.

        Args:
            node: The node to check.
            sdfg: The SDFG for memory/storage context.
            state: The state in which the node resides.

        Returns:
            True if the node is relevant for GPU stream assignment, False otherwise.
        """

        node_and_neighbors = list(state.neighbors(node))
        node_and_neighbors.append(node)

        for n in node_and_neighbors:
            # GPU global memory access nodes
            if (isinstance(n, nodes.AccessNode) and 
                n.desc(sdfg).storage == dtypes.StorageType.GPU_Global):
                return True
            
            # GPU-scheduled map entry/exit nodes (kernels)
            if (isinstance(n, (nodes.EntryNode, nodes.ExitNode)) and 
                n.schedule in dtypes.GPU_SCHEDULES):
                return True
            
            # GPU-scheduled library nodes
            if (isinstance(n, nodes.LibraryNode) and 
                n.schedule in dtypes.GPU_SCHEDULES):
                return True

        return False
        
    def _next_stream(self, gpu_stream: int) -> int:
        """
        Returns the next CUDA stream index based on the configured concurrency policy.
        
        - If max_concurrent_streams == 0: unlimited streams → increment stream index
        - If max_concurrent_streams == -1: default → always return 0
        - Else: wrap around within the allowed number of streams
        """
        if self._max_concurrent_streams == 0:
            return gpu_stream + 1
        elif self._max_concurrent_streams == -1:
            return 0
        else:
            return (gpu_stream + 1) % self._max_concurrent_streams

    def _insert_gpu_stream_sync_tasklet(self, sdfg: SDFG, assigned_nodes: Dict) -> None:
        """
        Inserts GPU stream synchronization tasklets at required locations:
        - At the end of a state, for streams used in the state.
        - After specific nodes, if their outputs need to synchronize before reuse.
        """
        sync_state, sync_node = self._identify_sync_locations(sdfg, assigned_nodes)

        #----------------- Insert synchronization tasklets at the end of each state -----------------
        for state, streams in sync_state.items():

            # Important: get sink nodes before adding the tasklet
            sink_nodes = list(state.sink_nodes())

            # Generate sync code for all streams used in this state
            sync_code_lines = []
            for stream in streams:

                if stream == "nullptr":
                    gpu_stream_access_expr = "nullptr"
                else:
                    gpu_stream_access_expr = self._gpu_stream_access_template.format(gpu_stream=stream)

                sync_code_lines.append(f"DACE_GPU_CHECK({self._backend}StreamSynchronize({gpu_stream_access_expr}));")

            sync_code = "\n".join(sync_code_lines)

            tasklet = state.add_tasklet(
                name=f"gpu_stream_sync_{state}", inputs=set(), outputs=set(),
                code=sync_code,
                language=dtypes.Language.CPP
                )
            
            for sink_node in sink_nodes:
                state.add_edge(sink_node, None, tasklet, None, dace.Memlet())


        #----------------- Insert synchronization tasklets after specific nodes -----------------

        for node, state in sync_node.items():

            # get correct stream access expr
            stream = assigned_nodes.get(node, "nullptr")
            if stream == "nullptr":
                gpu_stream_access_expr = "nullptr"
            else:
                gpu_stream_access_expr = self._gpu_stream_access_template.format(gpu_stream=stream)

            tasklet = state.add_tasklet(
                name=f"gpu_stream_sync_{stream}", inputs=set(), outputs=set(),
                code=f"DACE_GPU_CHECK({self._backend}StreamSynchronize({gpu_stream_access_expr}));\n",
                language=dtypes.Language.CPP
                )
            
            # important: First get the successors, then add the tasklet
            successors = list(state.successors(node))
            state.add_edge(node, None, tasklet, None, dace.Memlet())

            for succ in successors :
                state.add_edge(tasklet, None, succ, None, dace.Memlet())
            
    def _identify_sync_locations(self, sdfg: SDFG, assigned_nodes: Dict) -> Tuple[Dict[SDFGState, Set[str]], Dict[nodes.Node, SDFGState]]:
        """
        Heuristically identifies GPU stream synchronization points in an SDFG.

        Synchronization is needed:
        - At the end of a state, if we copy to/from GPU AccessNodes.
        - Immediately after a node, if data leaves GPU memory and is further used.
        
        Returns:
            - sync_state: Maps each SDFGState to a set of stream IDs to sync at the end of the state.
            - sync_node: Maps individual nodes to the state where a sync is required after the node.
        """

        # ------------------ Helper predicates -----------------------------

        def is_gpu_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(state.parent).storage == dtypes.StorageType.GPU_Global

        def is_nongpu_accessnode(node, state):
            return isinstance(node, nodes.AccessNode) and node.desc(state.parent).storage not in dtypes.GPU_MEMORY_STORAGES_EXPERIMENTAL_CUDACODEGEN
        
        def is_kernel_exit(node):
            return isinstance(node, nodes.ExitNode) and node.schedule == dtypes.ScheduleType.GPU_Device
        
        def is_sink_node(node, state):
            return state.out_degree(node) == 0

        # ------------------ Sync detection logic -----------------------------

        sync_state: Dict[SDFGState, Set[str]] = {}
        sync_node: Dict[nodes.Node, SDFGState] = {}

        for edge, state in sdfg.all_edges_recursive():
            src, dst = edge.src, edge.dst

            # Ensure state is initialized in sync_state
            if state not in sync_state:
                sync_state[state] = set()

            # --- Heuristics for when to sync ---
            if is_gpu_accessnode(src, state) and is_nongpu_accessnode(dst, state) and is_sink_node(dst, state):
                sync_state[state].add(assigned_nodes[dst])

            elif is_gpu_accessnode(src, state) and is_nongpu_accessnode(dst, state) and not is_sink_node(dst, state):
                sync_state[state].add(assigned_nodes[dst])
                sync_node[dst] = state

            elif is_nongpu_accessnode(src, state) and is_gpu_accessnode(dst, state):
                sync_state[state].add(assigned_nodes[dst])

            elif is_kernel_exit(src) and is_gpu_accessnode(dst, state) and is_sink_node(dst, state):
                sync_state[state].add(assigned_nodes[dst])

            else:
                continue
        
            # Check that state is indeed a SDFGState when added to the dictionary, to be on the safe side
            if not isinstance(state, SDFGState):
                raise NotImplementedError(f"Unexpected parent type '{type(state).__name__}' for edge '{edge}'. "
                                           "Expected 'SDFGState'. Please handle this case explicitly.")

        # Remove states with no syncs
        sync_state = {state: streams for state, streams in sync_state.items() if len(streams) > 0}

        return sync_state, sync_node

    def set_gpu_stream_access_template(self, expr_template: str):
        """
        Sets the stream access expression template. The string should include
        a `{gpu_stream}` placeholder. This function is expected to be called from a
        gpu code generator.
        """
        if "{gpu_stream}" not in expr_template:
            raise ValueError("self._gpu_stream_access_template must include '{gpu_stream}' placeholder.")
        self._gpu_stream_access_template = expr_template
from typing import Union, Dict, Set

import dace
from dace import SDFG, properties, SDFGState
from dace import dtypes
from dace.codegen import common
from dace.config import Config
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg import nodes
from dace.sdfg.graph import Edge


@properties.make_properties
@transformation.explicit_cf_compatible
class NaiveGPUStreamScheduler(ppl.Pass):
    """
    Assigns GPU streams to relevant nodes based on connected components.
    Also, it adds synchronization tasklets where required.

    Strategy:
    - "Relevant nodes" in connected components within a state are assigned the same stream.
    - Each state (except for nested states) starts fresh with stream 0.
    - States in nested SDFGs inherit the parent component's stream.
    - Only nodes that are either ("relevant nodes"):
        * in GPU memory (AccessNodes in GPU memory),
        * GPU scheduled (e.g., maps/kernels or library nodes),
        * or directly connected to such nodes,
      are assigned a stream.
    - GPU stream IDs wrap around based on the max_concurrent_streams config.

    Example:
        A state with K1->K2, K3->K4->K5, K6 becomes:
        K1,K2 → stream0
        K3,K4,K5 → stream1
        K6 → stream2
        (assuming no limit on the number of CUDA streams)

    NOTE: These are backend streams (CUDA/HIP), not DaCe streams.
    """
    
    # max configured number of concurrent streams
    max_concurrent_streams = int(Config.get('compiler', 'cuda', 'max_concurrent_streams'))

    # needed to call correct backend synchronization functions and in correct language
    backend: str = common.get_gpu_backend() 
    language = 'cu' if backend == 'cuda' else 'cpp'

    # This is expected to be set by the calling target codegenerator. 
    gpu_stream_access_template: str = ""

    def apply_pass(self, sdfg: SDFG, _) -> Dict[nodes.Node, Union[int, str]]:
        """
        Assigns GPU streams and adds synchronization tasklets.
        """

        assigned_nodes = self._assign_streams_to_sdfg(sdfg)
        
        num_assigned_streams = max(assigned_nodes.values(), default=0) 

        # If all use 0 stream or max_concurrent_stream is -1 (only default stream)
        # then assign to all nodes the nullptr.
        if num_assigned_streams == 0: # note: self.max_concurrent_streams == -1 implies num_assigned_streams == 0
            for k in assigned_nodes.keys():
                assigned_nodes[k] = "nullptr"

        self._add_sync_tasklet(sdfg, assigned_nodes)

        return assigned_nodes

    def _assign_streams_to_sdfg(self, sdfg: SDFG, assigned_nodes=None, visited=None) -> Dict:
        """
        Traverse all SDFG states and assign streams to connected components.
        Each state (exluding nested states) restarts stream assignment from 0.
        """
        if assigned_nodes is None:
            assigned_nodes = dict()
        if visited is None:
            visited = set()
            
        for state in sdfg.states():
            self._assign_streams_to_state_recursively(sdfg, state, assigned_nodes, visited, 0)
    
        return assigned_nodes
    
    def _assign_streams_to_state_recursively(self, sdfg: SDFG, state: SDFGState, assigned_nodes: Dict, visited: Set, gpu_stream:int):
        """
        Processes connected components in a state, assigning each to a different GPU stream,
        but only if they contain GPU-related nodes (otherwise, stream assignment is skipped).

        Nested SDFGs inherit the GPU stream of their parent state/component.
        """
        for source_node in state.source_nodes():
            if source_node in visited:
                continue  # Skip already processed components
                
            nodes_assigned_before = len(assigned_nodes)
            
            # Process all nodes in this connected component
            for edge in state.dfs_edges(source_node):

                # get both ends of the edge
                src = edge.src
                dst = edge.dst

                # both are visited, potentially again
                visited.add(src)
                visited.add(dst)

                # Either they are gpu nodes are directly connected to them, 
                # so they get assigned to the current gpu_stream
                if self._is_gpu_node(src, sdfg) or self._is_gpu_node(dst, sdfg):
                    assigned_nodes[src] = gpu_stream
                    assigned_nodes[dst] = gpu_stream

                # Recursively process nested SDFG states with same stream
                if isinstance(src, nodes.NestedSDFG):
                    for nested_state in src.sdfg.states():
                        self._assign_streams_to_state_recursively(src.sdfg, nested_state, assigned_nodes, visited, gpu_stream)
                
                if isinstance(dst, nodes.NestedSDFG):
                    for nested_state in dst.sdfg.states():
                        self._assign_streams_to_state_recursively(dst.sdfg, nested_state, assigned_nodes, visited, gpu_stream)


            # Move to next stream if we assigned any nodes in this component
            if len(assigned_nodes) > nodes_assigned_before:
                gpu_stream = self._next_stream(gpu_stream)

    def _is_gpu_node(self, node: nodes.Node, sdfg: SDFG) -> bool:
        """
        Determine if a node is a gpu node.
        
        This includes GPU-scheduled library nodes, kernels (maps), and GPU global memory
        access nodes.
        
        Args:
            node: Node to check
            sdfg: SDFG for context
            
        Returns:
            True if node is a gpu node
        """
        # GPU global memory access nodes
        if (isinstance(node, nodes.AccessNode) and 
            node.desc(sdfg).storage == dtypes.StorageType.GPU_Global):
            return True
        
        # GPU-scheduled map entry/exit nodes (kernels)
        if (isinstance(node, (nodes.EntryNode, nodes.ExitNode)) and 
            node.schedule in dtypes.GPU_SCHEDULES):
            return True
        
        # GPU-scheduled library nodes
        if (isinstance(node, nodes.LibraryNode) and 
            node.schedule in dtypes.GPU_SCHEDULES):
            return True
        
        return False

    def _next_stream(self, gpu_stream: int) -> int:
        """
        Returns the next CUDA stream index based on the configured concurrency policy.
        
        - If max_concurrent_streams == 0: unlimited streams → increment stream index
        - If max_concurrent_streams == -1: default → always return 0
        - Else: wrap around within the allowed number of streams
        """
        if self.max_concurrent_streams == 0:
            return gpu_stream + 1
        elif self.max_concurrent_streams == -1:
            return 0
        else:
            return (gpu_stream + 1) % self.max_concurrent_streams

    def _add_sync_tasklet(self, sdfg: SDFG, assigned_nodes: dict):
        """
        Adds a synchronization tasklet for each sink node in a connected component,
        but only for top-level states (not inside nested SDFGs).

        Specifically:
        - If a sink node is an AccessNode and has been assigned a GPU stream,
          a tasklet is inserted after it to call stream synchronization.
        - This ensures proper synchronization.
        """
        for state in sdfg.states():
            for snode in state.sink_nodes():

                if isinstance(snode, nodes.AccessNode) and snode in assigned_nodes.keys():
                    
                    # get correct stream access expr
                    stream = assigned_nodes[snode]
                    if stream == "nullptr":
                        gpu_stream_access_expr = "nullptr"
                    else:
                        gpu_stream_access_expr = self.gpu_stream_access_template.format(gpu_stream=stream)
                    # Add tasklet and connect it to the sink node
                    tasklet = state.add_tasklet(
                        name=f"sync_{stream}", inputs=set(), outputs=set(),
                        code=f"DACE_GPU_CHECK({self.backend}StreamSynchronize({gpu_stream_access_expr}));\n",
                        language=dtypes.Language.CPP
                        )
                    
                    state.add_edge(snode, None, tasklet, None, dace.Memlet())
                else:
                    continue

    def set_gpu_stream_access_template(self, expr_template: str):
        """
        Sets the stream access expression template. The string should include
        a `{gpu_stream}` placeholder. This function is expected to be called from a
        gpu code generator.
        """
        if "{gpu_stream}" not in expr_template:
            raise ValueError("gpu_stream_access_template must include '{gpu_stream}' placeholder.")
        self.gpu_stream_access_template = expr_template
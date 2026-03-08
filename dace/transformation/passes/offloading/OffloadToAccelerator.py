import dace
from dace import dtypes, properties
from dace.sdfg import nodes, SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union
import numpy as np

@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """
    Docstring for OffloadToAccelerator
    """
    
    CATEGORY: str = 'Offload To Accelerator'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False
    
    #def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
    #    return set()
    
    #def report(self, pass_retval: Any) -> Optional[str]:
    #    """
    #    Returns a user-readable string report based on the results of this pass.
    #
    #    :param pass_retval: The return value from applying this pass.
    #    :return: A string with the user-readable report, or None if nothing to report.
    #    """
    #    return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """

        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)

        self.find_and_insert_copies(sdfg)
        print("program is ready")


    def set_toplevel_to_GPU(self, sdfg: SDFG, type:Type):
        assert type in (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)

        for state in sdfg.states():
            scope_dict = state.scope_dict() 

            for node in state.nodes():
                if not isinstance(node, type): # filter
                    continue
                    
                if scope_dict[node] is None: # toplevel node -> offload
                    self.set_schedule(node)

                else: # within nested scope -> must not offload (defensive check)
                    if self.has_GPU_schedule(node):
                        raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass." \
                        "All maps must have default or CPU schedule before pass." \
                        f"Node {node} has schedule type {self.get_schedule(node)}" )

    
    def find_and_insert_copies(self, sdfg):
        gpu_access_nodes = set()

        for state in sdfg.states():
            for node in state.nodes():
                if self.is_schedule_node(node) and self.has_GPU_schedule(node): # map entry, exit or library node

                    if isinstance(node, nodes.MapEntry):
                        #print(f"checking map entry node {node}")
                        for input in self.get_all_input_nodes(state, node):
                            self._insert_copies(sdfg, state, input, node, gpu_access_nodes)

                    elif isinstance(node, nodes.MapExit):
                        #print(f"checking map exit node {node}")
                        for output in self.get_all_output_nodes(state, node):
                            self._insert_copies(sdfg, state, output, node, gpu_access_nodes)
                        
                    elif isinstance(node, nodes.LibraryNode):
                        #print(f"checking library node {node}")
                        neighbours = self.get_all_input_nodes(state, node)
                        neighbours |= self.get_all_output_nodes(state, node)
                        for neighbour in neighbours:
                            self._insert_copies(sdfg, state, neighbour, node, gpu_access_nodes)

                    else:
                        raise RuntimeError(f"unknown schedule node {node} in find_and_insert_copies")


    def _insert_copies(self, sdfg:dace.SDFG, state:dace.SDFGState, node:nodes.Node, prev:nodes.Node, gpu_access_nodes:set):
        # if wrong node or already visited, return
        if not isinstance(node, nodes.AccessNode): return
        if node in gpu_access_nodes: return
        
        # otherwise, mark as data needed on GPU
        gpu_access_nodes.add(node)
        
        # get neighbours = input + output - prev
        inputs = self.get_all_input_nodes(state, node)
        neighbours = inputs | self.get_all_output_nodes(state, node) 
        if prev in neighbours: neighbours.remove(prev) # prevent cylical recursion

        # check all neighbours and insert copies where needed
        #print(f"checking node {node} with neighbours {neighbours}\n")
        for neighbour in neighbours:

            if self.is_schedule_node(neighbour):
                # if neighbour is schedule node it must have GPU schedule, no action needed
                if not self.has_GPU_schedule(neighbour):
                    raise RuntimeError(f"Node {node} is expected to be scheduled for GPU but isn't.")
            
            elif isinstance(neighbour, nodes.AccessNode):
                # if the neighbouring access node is already marked as GPU, no action needed
                # otherwise, assume current node's data lives on GPU and recurse on the neighbour
                if not neighbour in gpu_access_nodes:
                    self._insert_copies(sdfg, state, neighbour, node, gpu_access_nodes)

            else:
                # different node, i.e. control flow -> data needed on CPU: insert copy
                raise NotImplemented(f"Copy needed between {node} and {neighbour}")

        # if its a non-transient node, a copy from/to userspace is also needed
        if not node.desc(sdfg)._transient:
      
            if not inputs and node.has_reads(state):
                print(f"copy input node {node} to GPU at beginning of the program")
                # raise NotImplemented
            elif node.has_writes(state): # can also be a result if it isn't returned (pass by pointer), so checking for no outputs doesn't work
                print(f"copy result node {node} back from GPU at the end of the program")
                # raise NotImplemented  
        
            # Q: Heat3d: array B is partially overwritten but never read from 
            #    do we still need an initial copy of B to the GPU or can be do a partial copy back?
            # A?: all arrays need to be copied back because of sideeffects?


    def is_schedule_node(self, node):
        return isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit) or isinstance(node, nodes.LibraryNode)

    def get_schedule(self, node):
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            return node.map.schedule
        elif isinstance(node, nodes.LibraryNode):
            return node.schedule
        else:
            assert False
        
    def set_schedule(self, node):
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            node.map.schedule = dtypes.ScheduleType.GPU_Device
        elif isinstance(node, nodes.LibraryNode):
            node.schedule = dtypes.ScheduleType.GPU_Device
        else:
            assert False

    def has_GPU_schedule(self, node):
        GPU_SCHEDULES = (
            dtypes.ScheduleType.GPU_Device,
            dtypes.ScheduleType.GPU_Default
        )
        return self.get_schedule(node) in GPU_SCHEDULES

    def get_all_input_nodes(self, state, node, type=None):
        input_nodes = set()
        for e in state.in_edges(node):
            root_src = state.memlet_tree(e).root().edge.src
            if not type or isinstance(root_src, type):
                input_nodes.add(root_src)
        return input_nodes

    def get_all_output_nodes(self, state, node, type=None):
        output_nodes = set()
        for e in state.out_edges(node):
            root_dst = state.memlet_tree(e).root().edge.dst
            if not type or isinstance(root_dst, type):
                output_nodes.add(root_dst)
        return output_nodes
        

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

        for state in sdfg.bfs_nodes():
            print(state)
        print()
        for state in sdfg.states():
            print(state)
        return
        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)

        gpu_set, cpu_set = self.copy_analysis(sdfg)
        for access_node in gpu_set:
            access_node.desc(sdfg).storage = dace.dtypes.StorageType.GPU_Global
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
        return self.get_schedule(node) in dtypes.GPU_SCHEDULES

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
    
    ######

    def BFS(self, start, get_neighbour, process, result):

        # BFS
        visited = set()
        queue = []
        visited = [start]

        while queue:
            node = queue.popleft()
            
            process(node, result)

            for neighbor in get_neighbour(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result


    def data_used_by_map(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dict[str, dace.dtypes.DeviceType]:

        data_dict = {}

        # BFS
        visited = set()
        queue = []
        visited = [map_entry]

        while queue:
            node = queue.popleft()
            
            # process node
            if isinstance(node, nodes.AccessNode):
                
                pass
            #

            for neighbor in self.get_all_output_nodes(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def data_used_by_state(self): pass

    def data_used_by_sdfg(self): pass



    ## V2: interstate

    def copy_analysis(self, sdfg: SDFG):
        pre_gpu_set, pre_cpu_set = set(), set()
        last_state = None

        # ASSUME: states are arranged as a line graph
        for state in sdfg.states():
            print(f"\n\nstate: {state}")

            # get locations of arrays within this state
            gpu_set, cpu_set = set(), set()
            for node in state.data_nodes():
                self._find_data_location(state, node, gpu_set, cpu_set)
            
            # check whether copies are needed within state
            copy_within_state = gpu_set & cpu_set
            if copy_within_state:
                #raise NotImplemented(f"cannot handle copies within states: {state} needs copies of {copy_within_state}")
                print(f"copy within state {state}: {copy_within_state}")

            # check whether copies are needed between states
            if last_state:
                copy_to_gpu = pre_cpu_set & gpu_set
                copy_to_cpu = pre_gpu_set & cpu_set
                print(pre_gpu_set)
                print(f"copy to cpu {cpu_set}: {copy_to_cpu}")
                if copy_to_gpu:
                    print(f"copy to gpu {last_state} -> {state}: {copy_to_gpu}")
                if copy_to_cpu:
                    print(f"copy to cpu {last_state} -> {state}: {copy_to_cpu}")

            # check the interstate edge condition
            pre_gpu_set = gpu_set
            pre_cpu_set = cpu_set
            last_state = state

            print(f"gpu_data: {pre_gpu_set}")
            print(f"cpu_data: {pre_cpu_set}")

        return pre_gpu_set, pre_cpu_set
            
    
    def _find_data_location(self, state:dace.SDFGState, node:nodes.AccessNode, gpu_set:Set, cpu_set:Set):
        """
        pre condition: node is nodes.AccessNode
        post condition: node in gpu_set or neighbour in cpu_set
        """
        assert isinstance(node, nodes.AccessNode) # defensive
        if node in gpu_set or node in cpu_set:
            return
        
        # get all new neighbours
        neighbours : Set = self.get_all_input_nodes(state, node) | self.get_all_output_nodes(state, node)
        
        for neighbour in neighbours:
            
            if self.is_schedule_node(neighbour):
                if not self.has_GPU_schedule(neighbour):
                    raise RuntimeError(f"Node {node} is expected to be scheduled for GPU but isn't.")
                gpu_set.add(node)
                
            elif isinstance(neighbour, nodes.AccessNode):
                # recurse on neighbour
                # Assumption: access nodes are acylic in sdfgs.
                self._find_data_location(state, neighbour, gpu_set, cpu_set)
                # post condition: neighbour in gpu_set or neighbour in cpu_set
                
                if neighbour in gpu_set and neighbour in cpu_set:
                    if not (node in gpu_set or node in cpu_set): 
                        # if neighbour is both, and this node is still undefined
                        # assume GPU schedule for this node
                        gpu_set.add(node)
                    else:
                        # if the node is not undefined, leave as is
                        pass

                elif neighbour in gpu_set: 
                    gpu_set.add(node)

                else:
                    cpu_set.add(node)
    
            else:
                # different node, i.e. control flow -> data needed on CPU
                cpu_set.add(node)




    ## V1: intrastate
        
    def find_all_intrastate_copies(self, sdfg):
        gpu_access_nodes = set()

        for state in sdfg.states():
            for node in state.nodes():
                if self.is_schedule_node(node) and self.has_GPU_schedule(node): # map entry, exit or library node

                    if isinstance(node, nodes.MapEntry):
                        #print(f"checking map entry node {node}")
                        for input in self.get_all_input_nodes(state, node):
                            self._find_intrastate_copies(sdfg, state, input, node, gpu_access_nodes)

                    elif isinstance(node, nodes.MapExit):
                        #print(f"checking map exit node {node}")
                        for output in self.get_all_output_nodes(state, node):
                            self._find_intrastate_copies(sdfg, state, output, node, gpu_access_nodes)
                        
                    elif isinstance(node, nodes.LibraryNode):
                        #print(f"checking library node {node}")
                        neighbours = self.get_all_input_nodes(state, node)
                        neighbours |= self.get_all_output_nodes(state, node)
                        for neighbour in neighbours:
                            self._find_intrastate_copies(sdfg, state, neighbour, node, gpu_access_nodes)

                    else:
                        raise RuntimeError(f"unknown schedule node {node} in find_and_insert_copies")


    def _find_intrastate_copies(self, sdfg:dace.SDFG, state:dace.SDFGState, node:nodes.Node, prev:nodes.Node, gpu_access_nodes:set):
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
                    self._find_intrastate_copies(sdfg, state, neighbour, node, gpu_access_nodes)

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

        

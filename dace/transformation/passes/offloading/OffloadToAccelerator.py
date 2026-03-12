import dace
from dace import dtypes, properties, data
from dace.sdfg import nodes, SDFG
from dace.sdfg.utils import get_last_view_node
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

    ### STEP 1 ###
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

    
    ### HELPERS ###

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
    

    def get_children(self, state, node):
        return {e.dst for e in state.out_edges(node)}
    
    def get_predecessors(self, state, node):
        return {e.src for e in state.in_edges(node)}

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
            print(state, node, e)
            root_dst = state.memlet_tree(e).root().edge.dst
            if not type or isinstance(root_dst, type):
                output_nodes.add(root_dst)
        return output_nodes
    
    ### STEP 2 ###

    def get_data_locations_of_map(self, sdfg: SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        """
        finds all arrays accessed by a map, i.e. arrays which are
            - part of the read/write set of an enclosed tasklet
            - data of an enclosed access node
            - accessed by a second, enclosed map
            - the original arrays behind an accessed view
        
        and decides whether their location should be on gpu or a cpu, i.e.
            - gpu if ANY parent map has a gpu schedule (even if the direct parent has cpu-schedule)
            - cpu else

        returns two sets (gpu_set, cpu_set) with the names of the respective arrays
        """
        def _add_data(data_name: str, gpu_set:set, cpu_set:set, is_gpu:bool):
            if data_name in gpu_set: # has already been accessed on GPU
                if not is_gpu: # is now accessed on CPU
                    raise RuntimeError("GPU->CPU within map. This should never happen. If outer map is GPU then inner data must also be on GPU (seq map runs as kernel)")

            elif data_name in cpu_set: # has already been accessed on CPU
                if is_gpu: # is now accessed on GPU
                    raise RuntimeError(f"CPU->GPU copy needed within map for {data_name}")

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        def _add_access_or_view_data(sdfg: SDFG, state: dace.SDFGState, node:nodes.AccessNode, gpu_set:set, cpu_set:set, is_gpu:bool):
            data_name = node.data
            _add_data(data_name, gpu_set, cpu_set, is_gpu) # add given node.data to sets
            
            if isinstance(sdfg.arrays[data_name], data.View): # check if its a view (alias of another data container) - if so, find original and add it too
                access_node = node
                if not isinstance(node, nodes.AccessNode): # node could e.g. be a Memlet. In that case, we need to find the corresponding view access node by iteration.
                    for n in state.data_nodes():
                        if n.data == data_name:
                            access_node = n
                            break
                   
                original = get_last_view_node(state, access_node) # once the view access node is known, its original access node can be found and it's data added
                #print(f"\t\tis_view of {access_node}")
                _add_data(original.data, gpu_set, cpu_set, is_gpu)
                
        def _recursive_helper(sdfg: SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry, gpu_set:Set, cpu_set:Set, is_gpu:bool):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES # Q: how not to hardcode?
            
            map_nodes = [n for n, parent in state.scope_dict().items() if parent is map_entry]
            #print(f"map {map_entry}\n\tis_gpu: {is_gpu}\n\tmap nodes: {map_nodes}")
            for node in map_nodes:
                #print(f"\tnode: {node}\n\t\tgpu:{gpu_set}, cpu:{cpu_set}")
                if isinstance(node, nodes.AccessNode): # internal access node -> add
                    _add_access_or_view_data(sdfg, state, node, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.Tasklet): # find original accessed array -> add
                    reads = {e.data for e in state.in_edges(node)}
                    writes = {e.data for e in state.out_edges(node)}
                    #print(f"\t\treads {reads} writes {writes}")
                    for node in reads | writes:
                        _add_access_or_view_data(sdfg, state, node, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.MapEntry): # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu)
        

        # function body
        gpu_set, cpu_set = set(), set()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set

    
    def get_data_locations_of_state(self): pass

    def get_data_locations_of_sdfg(self): pass



    """
    # first attempt at get_data_locations_of_state and
    # get_data_locations_of_sdfg
    # not functional, but I'll reuse pieces when implementing
    # the above functions properly

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
        #pre condition: node is nodes.AccessNode
        #post condition: node in gpu_set or neighbour in cpu_set
        
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
    """
import dace
from dace import dtypes, properties, data
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
import dace.data 

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

        # step 1: offload maps and library nodes -> heuristic only! document TODO
        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
        """
        library node -> GPU
        map cpu
            library node -> Seq
            map cpu
                library node -> Seq
                map gpu
                    library node -> Seq

        let user set the schedule to anything else than Device -> stays that way
        if user set to GPU device, then map cannot be offloaded (known limitation)
        """

        # step 2: copy analysis
        gpu_set, cpu_set = self.get_data_locations(sdfg)

        for name in gpu_set - cpu_set:
            sdfg.arrays[name].storage = dtypes.StorageType.GPU_GLOBAL

        for name in cpu_set - gpu_set:
            sdfg.arrays[name].storage = dtypes.StorageType.CPU_GLOBAL

        # step 3: insert copies
        both = gpu_set & cpu_set
        if both:
            raise NotImplemented(f"the following nodes are needed on GPU and CPU: {gpu_set & cpu_set}")
        

        """
        1) use Yakup's sfgds as unit test cases
        2) implement my own test suite, check in with Yakup
        3) implement copy pass

        next meeting monday 9.30
        
        heat3d everything on GPU
        assume inputs on CPU, then copy in beginning -> implement two options, all on GPU, all on CPU -> check non transient, copy if mismatch
        
        copying:
            access node connected to other access node is a copy

            node needs A on GPU
                    |
            access node A_gpu, GPU storage  <-- ADD TODO
                    |
            access node A_cpu, CPU storage
                    |
            node needs A on CPU

        copy in beginning, now there is A and A_gpu
        node by node, set to use A_gpu

        track before and after in linegraph
        """
        

    ### STEP 1 ###
    def set_toplevel_to_GPU(self, sdfg: SDFG, type:Type):
        assert type in (nodes.MapEntry, nodes.MapExit, nodes.LibraryNode)

        for state in sdfg.states():
            scope_dict = state.scope_dict() 

            for node in state.nodes():
                if not isinstance(node, type): # filter
                    continue
                    
                if scope_dict[node] is None: # toplevel node -> change schedule
                    self.set_schedule(node)

                else: # within nested scope -> must not have GPU schedule (defensive check)
                    if self.has_GPU_schedule(node):
                        raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass." \
                        "All maps must have default or CPU schedule before pass." \
                        f"Node {node} has schedule type {self.get_schedule(node)}" )

    
    ### generic HELPERS ###

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

    

    ### STEP 2 ###

    # helpers to get the set of arrays accessed / connected to nodes or edges #

    def get_arrays_used_by_access_node(self, sdfg:SDFG, state:SDFGState, access_node:nodes.AccessNode) -> set[str]:
        data_name = access_node.data
        container = sdfg.arrays[data_name]
        arrays : set[str] = {data_name}

        if isinstance(container, data.Array):
            return arrays

        elif isinstance(container, data.View):
            original = get_last_view_node(state, access_node) # once the view access node is known, its original access node can be found and it's data added
            return arrays | self.get_arrays_used_by_access_node(sdfg, state, original)
            
        else: # might be a scalar access of another array
            for pred in self.get_predecessors(state, access_node):
                if isinstance(pred, nodes.AccessNode):
                    arrays |= self.get_arrays_used_by_access_node(sdfg, state, pred)
            return arrays
       
    def get_arrays_used_by_edge(self, sdfg:SDFG, state:SDFGState, edge, is_out_edge:bool):
        if not edge.data.is_empty():

            data_name = edge.data.data
            container = sdfg.arrays[data_name]

            if isinstance(container, data.Array): # array access on edge
                return {data_name}

            elif isinstance(container, data.View): # view -> we need to find the corresponding view access node by iteration.
                for n in state.data_nodes(): 
                    if n.data == data_name:
                        return self.get_arrays_used_by_access_node(n)
                
            else: # might be a scalar access of an array slice
                if is_out_edge:
                    if isinstance(edge.dst, nodes.AccessNode):
                        return self.get_arrays_used_by_access_node(sdfg, state, edge.dst)
                else:
                    if isinstance(edge.src, nodes.AccessNode):
                        return self.get_arrays_used_by_access_node(sdfg, state, edge.src)
                
        return set()
    
    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays : set[str] = set()

        if isinstance(node, nodes.AccessNode):
            return self.get_arrays_used_by_access_node(sdfg, state, node)

        for e in state.in_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, False)

        for e in state.out_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, True)

        return arrays


    # find all accessed arrays and sort them into gpu and cpu sets #

    def get_data_locations_of_map(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
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

        # helper to validate data and add it to correct set
        def _add_data(data_name: str, gpu_set:set[str], cpu_set:set[str], is_gpu:bool) -> tuple[set[str], set[str]]:
            if data_name in gpu_set: # has already been accessed on GPU
                if not is_gpu: # is now accessed on CPU
                    raise RuntimeError("GPU->CPU within map. This should never happen. If outer map is GPU then inner data must also be on GPU (seq map runs as kernel)")

            elif data_name in cpu_set: # has already been accessed on CPU
                if is_gpu: # is now accessed on GPU
                    raise RuntimeError(f"CPU->GPU copy needed within map for {data_name}")

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        # main work horse, can recurse to nested maps
        def _recursive_helper(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, gpu_set:set[str], cpu_set:set[str], is_gpu:bool):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES # TODO Q: how not to hardcode?
            
            # get all nodes within this map's scope
            map_nodes = [n for n, parent in state.scope_dict().items() if parent is map_entry] 
            #print(f"map {map_entry}\n\tis_gpu: {is_gpu}\n\tmap nodes: {map_nodes}")

            for node in map_nodes:
                #print(f"\tnode: {node}\n\t\tgpu:{gpu_set}, cpu:{cpu_set}")
                if isinstance(node, nodes.MapEntry): # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu)
                
                elif isinstance(node, nodes.AccessNode): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_access_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.Tasklet): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, ControlFlowRegion):
                    g,c = self.get_data_locations_of_cfregion(sdfg, node)
                    if not is_gpu:
                        gpu_set |= g
                        cpu_set |= c
                    else:
                        gpu_set |= g | c

                elif isinstance(node, nodes.MapExit) or isinstance(node, nodes.LibraryNode):
                    pass # nothing to do

                else:
                    raise RuntimeError(f"inside map: unhandled node {node} of type {node.__class__.__name__} in state {state}")

             

        # function body, calls recursive helper
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set


    def get_data_locations_of_state(self, sdfg: SDFG, state: SDFGState) -> tuple[set[str], set[str]]:
        # iterate through all toplevel nodes of this state
        #  - map entry -> give to get_data_locations_of_map, which handles all nodes inside scope
        #  - control flow (nested) -> recurse
        #  - non-nested toplevel scopes -> add accessed data to cpu set
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()

        top_level_nodes = state.scope_children()[None]
        for node in top_level_nodes:

            g,c = set(), set()

            # process map and all nodes within -> may be on GPU
            if isinstance(node, nodes.MapEntry):
                g,c = self.get_data_locations_of_map(sdfg, state, node)
                #print(f"map: {node}, gpu: {g}, cpu: {c}")

            # library nodes are usually GPU, can be CPU
            elif isinstance(node, nodes.LibraryNode):
                if self.has_GPU_schedule(node):
                    g = self.get_arrays_used_by_node(sdfg, state, node)
                else:
                    c = self.get_arrays_used_by_node(sdfg, state, node)
                #print(f"lib: {node}, gpu: {g}, cpu: {c}")

            # recurse if nested
            elif isinstance(node, ControlFlowRegion):
                g,c = self.get_data_locations_of_cfregion(sdfg, node)
                #print(f"cfr: {node}, gpu: {g}, cpu: {c}")

            # all else is definitely on CPU
            elif isinstance(node, nodes.Tasklet): # outside a map scope (else handled by locations_of_map) -> cpu
                c = self.get_arrays_used_by_node(sdfg, state, node)
                #print(f"task: {node}, gpu: {g}, cpu: {c}")
                
            elif isinstance(node, nodes.MapExit) or isinstance(node, nodes.AccessNode):
                pass # nothing to do, access nodes are covered via tasklets

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set
    

    def get_data_locations_of_condblock(self, sdfg: SDFG, block:ConditionalBlock) -> tuple[set[str], set[str]]:
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()

        # get array accesses in condition
        for memlet in block.get_meta_read_memlets():
            if memlet.data in sdfg.arrays:
                cpu_set.add(memlet.data)

        # add array accesses in branches
        for _, branch in block.branches:
            g,c = self.get_data_locations_of_cfregion(sdfg, branch)
            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set
    

    def get_data_locations_of_loop(self, sdfg: SDFG, loop:LoopRegion) -> tuple[set[str], set[str]]:
        # get array accesses in init_statement, update_statement, and loop_condition
        cpu_set : set[str] = set()
        for memlet in loop.get_meta_read_memlets():
            if memlet.data in sdfg.arrays:
                cpu_set.add(memlet.data)
        
        # add array accesses in loop body
        gpu_set, c = self.get_data_locations_of_cfregion(sdfg, loop)
        cpu_set |= c

        return gpu_set, cpu_set
    

    def get_data_locations_of_cfregion(self, sdfg:SDFG, cf: ControlFlowRegion) -> tuple[set[str], set[str]]:
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()

        for block in cf.bfs_nodes():

            if isinstance(block, SDFGState):
                g,c = self.get_data_locations_of_state(sdfg, block)

            elif isinstance(block, ConditionalBlock):
                g,c = self.get_data_locations_of_condblock(sdfg, block)

            elif isinstance(block, LoopRegion):
                g,c = self.get_data_locations_of_loop(sdfg, block)

            elif isinstance(block, ControlFlowRegion):
                g,c = self.get_data_locations_of_cfregion(sdfg, block)

            elif isinstance(block, nodes.ReturnBlock, ContinueBlock, BreakBlock):
                pass

            else:
                raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")
        
            gpu_set |= g
            cpu_set |= c
            
        return gpu_set, cpu_set


    # wrapper
    def get_data_locations(self, sdfg:SDFG) -> tuple[set[str], set[str]]:
        return self.get_data_locations_of_cfregion(sdfg, sdfg)

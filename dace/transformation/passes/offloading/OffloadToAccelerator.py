
from dace import dtypes, properties, data
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
import dace

from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union

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
        make new branch
        organise test suite
        
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
        
    def get_arrays_used_by_incoming_access_nodes(self, sdfg:SDFG, state:SDFGState, node:nodes.Node) -> set[str]:
        arrays : set[str] = set()

        # if current node is access node
        #print(f"incoming of {node}")
        if isinstance(node, nodes.AccessNode): 
            data_name = node.data
            arrays = {data_name} # add current access node
            #print(f"accessing {data_name}")

            if isinstance(sdfg.arrays[data_name], data.View): # trace it if it is a view
                original = get_last_view_node(state, node) # once the view access node is known, its original access node can be found and it's data added
                #print(f"is view of {original}")
                arrays |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, original)
                
        # check if more access nodes UPstream
        for n in self.get_predecessors(state, node):
            if isinstance(n, nodes.AccessNode):
                #print(f"found pred access node {n.data}")
                arrays |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, n)

        #print(f"incoming of {node} are {arrays}") 
        return arrays
    
    def get_arrays_used_by_outgoing_access_nodes(self, sdfg:SDFG, state:SDFGState, node:nodes.Node) -> set[str]:
        arrays : set[str] = set()

        # if current node is access node
        if isinstance(node, nodes.AccessNode): 
            data_name = node.data
            arrays = {data_name} # add current access node

            if isinstance(sdfg.arrays[data_name], data.View): # trace it if it is a view
                original = get_last_view_node(state, node) # once the view access node is known, its original access node can be found and it's data added
                arrays |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, original)
                
        # check if more access nodes DOWNstream
        for n in self.get_children(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, n)
                 
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
                        if is_out_edge:
                            return self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, n)
                        return self.get_arrays_used_by_incoming_access_nodes(sdfg, state, n)
                
            else: # might be a scalar access of an array slice
                if is_out_edge:
                    if isinstance(edge.dst, nodes.AccessNode):
                        return self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, edge.dst)
                else:
                    if isinstance(edge.src, nodes.AccessNode):
                        return self.get_arrays_used_by_incoming_access_nodes(sdfg, state, edge.src)
                
        return set()
    
    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays : set[str] = set()

        # edges
        for e in state.in_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, False)
        
        for e in state.out_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, True)

        # neighbouring access nodes
        arrays |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, node)
        arrays |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, node)
       
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
                    gpu_set.add(data_name)
                    #raise RuntimeError("CPU->GPU copy needed within map for " + data_name)

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        # main work horse, can recurse to nested maps
        def _recursive_helper(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, gpu_set:set[str], cpu_set:set[str], is_gpu:bool):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES # TODO Q: how not to hardcode?
            
            # get all nodes within this map's scope
            map_nodes = [n for n, parent in state.scope_dict().items() if parent is map_entry]
            #print(f"map {map_entry}\n\tis_gpu: {is_gpu}\n\tmap nodes: {map_nodes}")

            # input & output nodes
            if is_gpu:
                gpu_set |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, map_entry)
                #print(f"map entry {map_entry} (gpu): {gpu_set}")
                gpu_set |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, state.exit_node(map_entry))
                #print(f"map exit {map_entry} (gpu): {gpu_set}")
            else:
                cpu_set |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, map_entry)
                gpu_set |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, state.exit_node(map_entry))
                #print(f"map entry {map_entry} (cpu): {gpu_set} & {cpu_set}")
                #print(f"map exit {map_entry} (cpu): {gpu_set} & {cpu_set}")

            # internal nodes
            for node in map_nodes:
                #print(f"\tnode: {node}\n\t\tgpu:{gpu_set}, cpu:{cpu_set}")
                if isinstance(node, nodes.MapEntry): # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu)
                
                elif isinstance(node, nodes.AccessNode): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.Tasklet): # find accessed arrays -> add
                    #print("map taSKLET:", node)
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, (ControlFlowRegion, nodes.NestedSDFG)):
                    g,c = self.get_data_locations_of_cfregion(sdfg, node) if isinstance(node, ControlFlowRegion) else self.get_data_locations(node.sdfg)
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
        #print(f"map {map_entry}, gpu {gpu_set}, cpu {cpu_set}\n")
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

            elif isinstance(node, nodes.MapExit):
                pass

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

            elif isinstance(node, nodes.AccessNode):
                pass # nothing to do; cannot be classified without context

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        print(f"state {state}, gpu {gpu_set}, cpu {cpu_set}")
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
    
    def get_data_locations_of_cfblock(self, sdfg:SDFG, block: ControlFlowBlock) -> tuple[set[str], set[str]]:
        if isinstance(block, SDFGState):
            return self.get_data_locations_of_state(sdfg, block)

        elif isinstance(block, ConditionalBlock):
            return self.get_data_locations_of_condblock(sdfg, block)

        elif isinstance(block, LoopRegion):
            return self.get_data_locations_of_loop(sdfg, block)

        elif isinstance(block, ControlFlowRegion):
            return self.get_data_locations_of_cfregion(sdfg, block)

        elif isinstance(block, nodes.NestedSDFG):
            return self.get_data_locations(block)

        elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
            return set(), set() # do nothing

        raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")
        

    def get_data_locations_of_cfregion(self, sdfg:SDFG, cfr: ControlFlowRegion) -> tuple[set[str], set[str]]:
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()

        # 1) iterate through all nodes
        #print(f"*** cfr {cfr}")
        for block in cfr.bfs_nodes():
            g,c = self.get_data_locations_of_cfblock(sdfg, block)
            gpu_set |= g
            cpu_set |= c
            #print("gpu:", g, "\ncpu:", c)

        # 2) iterate through all interstate edges
        for edge in cfr.edges():
            arrays = edge.data.used_arrays(sdfg.arrays)
            cpu_set |= arrays # conditions are always on cpu
            print("edge:", edge, "->", arrays)
        
        #print(f"***cfr {cfr}, gpu {gpu_set}, cpu {cpu_set}\n")
        return gpu_set, cpu_set


    # wrapper
    def get_data_locations(self, sdfg:SDFG) -> tuple[set[str], set[str]]:
        return self.get_data_locations_of_cfregion(sdfg, sdfg)
    
    # TODO: only add inputs and outputs of map to use set if they are actually used in map (not in nested map)
    ### copy insertion

    """
    pseudo code of copy logic

    func main():
        insert_copies(sdfg, None, set(), set())

    func insert_copies(graph, prev_state, prev_gpu_set, prev_cpu_set):
        for state in graph:
        
            if state is nested: # recurse
                gpu_set, cpu_set = insert_copies(state, prev_state, prev_gpu_set, prev_cpu_set)
                # returns sets of last node in nested state -> inputs for next state

            else: # basecase
                gpu_set,cpu_set = get_data_locations(state)

                # intrastate copy (later)
                if gpu_set & cpu_set:
                    gpu_set,cpu_set = intrastate_copy(state, gpu_set, cpu_set) # update sets

                # interstate copy
                if prev_state:
                    for gpu_copy in (gpu_set & prev_cpu_set):
                        create_interstate_copy(sdfg, prev_state, state, gpu_copy, to_gpu = True)

                    for cpu_copy in (cpu_set & prev_gpu_set):
                        create_interstate_copy(sdfg, prev_state, state, cpu_copy, to_gpu = False)
            
            prev_state = state
            prev_gpu_set = gpu_set
            prev_cpu_set = cpu_set

        return gpu_set, cpu_set
    """

    def insert_copies_state(self, sdfg, next_state, prev_state, prev_gpu_set, prev_cpu_set):
        pass

    def insert_copies_cfr(self, sdfg: SDFG, cfr:ControlFlowRegion, prev_state, prev_gpu_set, prev_cpu_set):
        """prev_gpu_set : set[str] = set()
        prev_cpu_set : set[str] = set()
        prev_state = None"""

        block : ControlFlowBlock
        for block in cfr.bfs_nodes():

            if isinstance(block, SDFGState): # insert to state
                prev_gpu_set, prev_cpu_set = self.insert_copies_state(sdfg, block, prev_state, prev_gpu_set, prev_cpu_set)
                prev_state = block
    
            elif isinstance(block, ConditionalBlock): # insert to both branches
                branch : ControlFlowRegion
                for _, branch in block.branches:
                    g,c = self.insert_copies_cfr(sdfg, branch, prev_state, prev_gpu_set, prev_cpu_set)
                # TODO: merge g,c, determine prev_state
                
            elif isinstance(block, LoopRegion):
                return self.get_data_locations_of_loop(sdfg, block)

            elif isinstance(block, ControlFlowRegion):
                return self.get_data_locations_of_cfregion(sdfg, block)

            elif isinstance(block, nodes.NestedSDFG):
                return self.get_data_locations(block)

            elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                return set(), set() # do nothing

            raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")
            


            g,c = self.get_data_locations_of_state(sdfg, state)

            for gpu_copy in (cpu_set & g):
                self.interstate_copy_to_gpu(gpu_copy, prev_state, state)
            
            for cpu_copy in (gpu_set & c):
                self.interstate_copy_to_gpu(cpu_copy, prev_state, state)

            gpu_set, cpu_set = g, c
            prev_state = state


    

    def _insert_state_on_interstate_edge(self, graph, edge, new_state):
        """Insert new_state on an interstate edge (between two control-flow blocks)."""
        graph.add_edge(edge.src, new_state, edge.data)              # src -> new_state (keep condition/assignments)
        graph.add_edge(new_state, edge.dst, dace.InterstateEdge())  # new_state -> dst (unconditional)
        graph.remove_edge(edge)                                     # del original edge
    

    def create_interstate_copy(self, sdfg, state1, state2, array_names, to_gpu:bool):
        assert state2 is not None, "state2 is None - but there is no use inserting a copy node without a successor state." # precondition

        # get name and storage location of new array based on to_gpu
        new_name = array_name + "_gpu" if to_gpu else array_name + "_cpu"
        location = dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default

        # create the empty copy state and connect it
        copy_state = sdfg.add_state(f"copy_{array_name}_to_{new_name}")
        
        if state1 is None: # copy state becomes new start block
            assert state2 is sdfg.start_block
            sdfg.start_block = sdfg.node_id(copy_state)
            sdfg.add_edge(copy_state, state2, dace.InterstateEdge())
        else:
            outedges = sdfg.out_edges(state1)
            assert len(outedges) == 1 # assert that it's a line graph
            assert outedges[0].dst == state2 # assert state1 & state2 are really connected
            self._insert_state_on_interstate_edge(sdfg, outedges[0], copy_state)

        # build the insides of the copy state
        sdfg.add_transient(new_name, [1], dace.float64, storage=location)
        a_copy_in = copy_state.add_access(array_name)
        a_gpu_copy_out = copy_state.add_access(new_name)
        copy_state.add_edge(a_copy_in, None, a_gpu_copy_out, None, dace.Memlet(f"{array_name} -> {new_name}"))
        
        # rename the access of A -> A_gpu in the next state
        for node in state2.data_nodes(): # nodes
            if node.data == array_name:
                node.data = new_name

        for edge in state2.edges(): # edges
            for e in state2.memlet_tree(edge):
                memlet = e.data
                if memlet is not None and not memlet.is_empty() and memlet.data == array_name:
                    memlet.data = new_name
        
        # NOTE: map connector names are purely symbolic and are thus not changed here

        # heuristic: set all other arrays used in state2 (used on GPU) to GPU
        # Avoid CUDA IllegalCopy: Register <-> CPU_Heap / CPU_ThreadLocal inside GPU scopes
        """
        names = {n.data for n in state2.data_nodes() if n.data in sdfg.arrays and isinstance(sdfg.arrays[n.data], data.Array)}
        for name in names:
            sdfg.arrays[name].storage = dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default
        
        if to_gpu:
            return set(), names
        else:
            return names, set()"""
        
            


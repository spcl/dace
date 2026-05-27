
from dace import dtypes, properties, data
from dace.memlet import Memlet
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
import dace

from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union

class OffloadingIRNode:

    STATE, COND, LOOP, NESTED, END, INIT = 0,1,2,3,4,5

    def __init__(self, block, cpu_set=set(), gpu_set=set(), next=set(), node_type = 0):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.block : ControlFlowBlock = block
        self.cpu_set : set[str] = cpu_set
        self.gpu_set : set[str] = gpu_set
        self.next : set[OffloadingIRNode] = next
        self.type : int = node_type

        self.body : OffloadingIRNode = None

        self.debug_name = f"{block}"

    def __repr__(self):
        return self._get_str(set(), -4)
    def __str__(self): 
        return self.__repr__()
    def _get_str(self, visited_set, len_before):
        s = f"{self.debug_name}:"
        spaces = 30 - (len_before + len(s))
        s += spaces * " " + f"cpu = {sorted(list(self.cpu_set))}, gpu = {sorted(list(self.gpu_set))}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        next_list = sorted(list(self.next), key=lambda x:x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s
    
    # utility functions
    def is_empty(self):
        return self.block is None
    
    def append_node(self, node):
        self.next.add(node)

    def get_all_leaves(self):
        def _find_all_leaves(node:OffloadingIRNode, result:set, visited_set:set):
            if node in visited_set:
                return
            visited_set.add(node)

            if node.is_leaf():
                result.add(node)
                return
            for next in node.next:
                _find_all_leaves(next, result, visited_set)

        result = set()
        _find_all_leaves(self, result, set())
        return result
    
    def get_single_leaf(node):
        leaves : set = node.get_all_leaves()
        assert len(leaves) == 1
        return leaves.pop()

    def is_leaf(self):
        return not self.next

    # static makers
    def make_empty(debug_name):
        node = OffloadingIRNode(None, set(), set(), set())
        node.debug_name = debug_name
        return node
    
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

    def __init__(self):
        super().__init__()
        self._copy_states = {}

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

        TODO: let user set the schedule to anything else than Device -> stays that way
        if user set to GPU device, then map cannot be offloaded (known limitation)
        """

        # step 2 & 3: copy analysis & create IR to store analysis results
        print("--- Analysis---")
        sdfgIR = self.sdfg_to_IR(sdfg)
        print(f"\n--- IR ---\n{sdfgIR}\n")

        # step 4: insert copies based on IR
        print("--- Copies ---")
        self.eval_IR(sdfg, sdfgIR, {array : (None, dtypes.StorageType.Default) for array in sdfg.arrays})
        print()

    # helper for testing, usually internal details are not exposed
    def get_IR(self, sdfg):
        self.set_toplevel_to_GPU(sdfg, nodes.MapEntry)
        self.set_toplevel_to_GPU(sdfg, nodes.LibraryNode)
        return self.sdfg_to_IR(sdfg)

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

    

    ### STEP 2: copy analysis ###

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

        print(f"state {state}: gpu {gpu_set if gpu_set else "{}"}, cpu {cpu_set if cpu_set else "{}"}")
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
    
    ### STEP 3: Intermediate Representation ###

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

    def sdfg_to_IR(self, sdfg:SDFG):
        all_arrays = {name for name in sdfg.arrays}
        # TODO: check if anything is already GPU or not default and create error
        
        # create inital node where all arrays must be on CPU
        IR = OffloadingIRNode.make_empty("_INIT")
        IR.cpu_set = all_arrays # all arrays are initially assumed to be on CPU
        IR.type = OffloadingIRNode.INIT

        # create IR
        self._parse_to_IR(sdfg, sdfg, IR)

        # all arrays which aren't used by this state retain their previous status
        # self._propagate_arrays(IR)

        # clean up
        #self._remove_empty_nodes(IR)

        # add final node where all arrays must be on CPU again
        final = OffloadingIRNode.make_empty("_END")
        final.cpu_set = all_arrays
        final.type = OffloadingIRNode.END

        print("final IR: {", [IR], "}")
        for leaf in IR.get_all_leaves():
            leaf.append_node(final)

        return IR

    def _parse_to_IR(self, sdfg:SDFG, cfr:ControlFlowRegion, curr_node:OffloadingIRNode) -> OffloadingIRNode:
        # edges
        # if there are edge accesses, use a new node to represent this controlflow region
        # if copies are later necessary, they will be added before the entire region
        if cfr.parent_graph is not None:
            arrays = set()
            edge : dace.sdfg.InterstateEdge
            for edge in cfr.parent_graph.edges():
                arrays |= set(edge.data.used_arrays(sdfg.arrays))

            if arrays:
                new_node = OffloadingIRNode(cfr, cpu_set=arrays, gpu_set=set(), next=set())
                curr_node.append_node(new_node)
                curr_node = new_node
            
        # nodes
        block : ControlFlowBlock
        for block in cfr.bfs_nodes():
            
            # non-nested state
            if isinstance(block, SDFGState):
                state : SDFGState = block
                gpu_set,cpu_set = self.get_data_locations_of_state(sdfg, state) # beating heart of this function
                new_node = OffloadingIRNode(state, cpu_set, gpu_set, set())
                curr_node.append_node(new_node)
                curr_node = new_node

            # if else
            elif isinstance(block, ConditionalBlock):
                # connect current node to new node reprenting the branching condition
                # find all array accesses in condition and add to cpu set
                # if condition necessitates copies, they will be added before the block
                branch_condition = OffloadingIRNode(block)
                for memlet in block.get_meta_read_memlets():
                    if memlet.data in sdfg.arrays:
                        branch_condition.cpu_set.add(memlet.data)
                curr_node.append_node(branch_condition)

                # parse branches and connect each branch head to branch condition
                tails = set()
                for _, branch in block.branches:
                    branch_head : OffloadingIRNode = self._parse_to_IR(sdfg, branch, branch_condition)
                    tails |= branch_head.get_all_leaves()

                # connect all tails to empty connector node (= new current node)
                curr_node = OffloadingIRNode.make_empty("_branch_connector")
                for tail in tails:
                    tail.append_node(curr_node)

            # loop
            elif isinstance(block, LoopRegion):
                loop : LoopRegion = block

                # get array accesses of init_statement, update_statement, and loop_condition add them to head's cpu_set
                loop_node = OffloadingIRNode(loop, node_type=OffloadingIRNode.LOOP)
                curr_node.next = [loop_node]
                
                loop_node.body = OffloadingIRNode.make_empty("_loop_head")
                for memlet in loop.get_meta_read_memlets():
                    if memlet.data in sdfg.arrays:
                        loop_node.body.cpu_set.add(memlet.data)

                # parse loop body and connect to head
                self._parse_to_IR(sdfg, loop, loop_node.body) # linked list representing all internal nodes of loop
                
                # connect all tails to empty connector node
                # connect connector to head again (-> loop)
                loop_tail = OffloadingIRNode.make_empty("_loop_tail")
                for leaf in loop_node.body.get_all_leaves():
                    leaf.append_node(loop_tail)

                self._propagate_arrays(loop_node.body)

                curr_node = loop_node

            # nested region -> flatten   
            elif isinstance(block, ControlFlowRegion):
                self._parse_to_IR(sdfg, block, curr_node)

            elif isinstance(block, nodes.NestedSDFG):
                self._parse_to_IR(block.sdfg, block.sdfg, curr_node)

            # do nothing
            elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                pass 

            else:
                raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

        return curr_node

    def _propagate_arrays(self, node):
        # all arrays which aren't used by this state retain their previous status
        # ASSUMPTION: arrays are either gpu or cpu within a state
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)
            
            for next in node.next:
                next_arrays = next.cpu_set | next.gpu_set
                
                for array in node.cpu_set:
                    if not array in next_arrays:
                        next.cpu_set.add(array)
                for array in node.gpu_set:
                    if not array in next_arrays:
                        next.gpu_set.add(array)
                recursion(next, visited_set)

        return recursion(node, set())
    
    def _remove_empty_nodes(self, node:OffloadingIRNode):
        # NOTE: if given node is empty, it won't be removed - only children are checked
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            for next in node.next:
                recursion(next, visited_set)

            empties = {next for next in node.next if next.is_empty()}
            
            for empty in empties:
                node.next.remove(empty)
                for nextnext in empty.next:
                    node.append_node(nextnext)

        return recursion(node, set())

    """def eval_IR(self, sdfg, IR:OffloadingIRNode):
        # modifies SDFG in place & inserts all necessary copies
        def recursion(sdfg, IR:OffloadingIRNode, visited_set:set):
            if IR in visited_set:
                return
            visited_set.add(IR)

            for next in IR.next:

                if IR.cpu_set & IR.gpu_set:
                    updated_c, updated_g = print(f"insert intrastate copy for {IR}")
                    IR.cpu_set = updated_c
                    IR.gpu_set = updated_g

                gpu_copies = IR.cpu_set & next.gpu_set
                if gpu_copies:
                    print(f"insert gpu copy for {gpu_copies} between {IR.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, IR.block, next.block, gpu_copies, to_gpu=True)
                    
                cpu_copies = IR.gpu_set & next.cpu_set
                if cpu_copies:
                    print(f"insert cpu copy for {cpu_copies} between {IR.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, IR.block, next.block, cpu_copies, to_gpu=False)
                    
            for next in IR.next:
                recursion(sdfg, next, visited_set)

        return recursion(sdfg, IR, set())"""


    def eval_IR(self, sdfg:SDFG, prev:OffloadingIRNode, data:Dict):
        print("eval IR ", prev.block.label if prev.block else None, ": ", prev.type)
        match prev.type:
            case OffloadingIRNode.STATE | OffloadingIRNode.INIT:
                self.eval_state(sdfg, prev, data)
            case OffloadingIRNode.COND:
                self.eval_cond(sdfg, prev, data)
            case OffloadingIRNode.LOOP:
                self.eval_loop(sdfg, prev, data)
            case OffloadingIRNode.NESTED:
                self.eval_nested(sdfg, prev, data)
            case OffloadingIRNode.END:
                self.eval_end(sdfg, prev, data)
                return

        for curr in prev.next:
            self.eval_IR(sdfg, curr, data)

    def eval_state(self, sdfg, prev, data):
        print("eval state ", prev.debug_name)
        if prev.cpu_set & prev.gpu_set:
            assert False, f"invalid: need to insert intrastate copy for {prev}"
        
        for curr in prev.next:
            # intrastate copies
            #gpu_copies, cpu_copies = self._get_neccessary_copies(curr, data) TODO
            for array in curr.gpu_set:
                last_access, curr_loc = data[array]
                if curr_loc != dtypes.StorageType.GPU_Global:
                    print(f"gpu copy of {array} with data {data[array]}")
                    if last_access is None: # make copy at very beginning
                        self.insert_initial_copy(sdfg, array, True)
                    else: # has been accessed in a previous state
                        self._add_copies_after_state(sdfg, last_access, [array], True)

            for array in curr.cpu_set:
                last_access, curr_loc = data[array]
                if curr_loc != dtypes.StorageType.Default:
                    print(f"cpu copy of {array} with data {data[array]}")
                    if last_access is None: # make copy at very beginning
                        self.insert_initial_copy(sdfg, array, False)
                    else: # has been accessed in a previous state
                        self._add_copies_after_state(sdfg, last_access, [array], False)

            # update data
            for array in curr.cpu_set:
                data[array] = (curr.block, dtypes.StorageType.Default)

            for array in curr.gpu_set:
                data[array] = (curr.block, dtypes.StorageType.GPU_Global)
                curr.block.replace_dict({array: self._gpu_name(array)})

            print("after ", curr.debug_name, " data is: ", data)

    def eval_cond(self, sdfg:SDFG, cf_block:ConditionalBlock, if_IR:OffloadingIRNode, else_IR:OffloadingIRNode, data:Dict):
        old_data = data.copy()
        if_data = data.copy()
        else_data = data.copy()

        # recursively ensure all inside copies are made
        self.eval_IR(if_IR, if_data)
        self.eval_IR(else_IR, else_data)

        # fuse branches into one consistent state, insert copies where needed
        if_accesses = self._get_accesses_by_comp(old_data, if_data)
        else_accesses = self._get_accesses_by_comp(old_data, else_data)
        next_used = self.next_used(sdfg, cf_block, if_accesses | else_accesses) # TODO

        for branch, accesses, a_data in [(if_IR, if_accesses, if_data), (else_IR, else_accesses, else_data)]:
            end_IR : OffloadingIRNode = branch.get_single_leaf()

            gpu_copies, cpu_copies = [], []
            for array in accesses:
                _, curr_loc = a_data[array]
                if curr_loc != next_used[array]:
                    (gpu_copies if curr_loc == dtypes.StorageType.Default else cpu_copies).append(array)

                # merge data from both branches into one consistent data dictionary
                # and close conditional block at the same time
                data[array] = (cf_block, curr_loc)

            self.create_interstate_copy(sdfg, end_IR.block, None, gpu_copies, True)
            self.create_interstate_copy(sdfg, end_IR.block, None, cpu_copies, False)


    def eval_loop(self, sdfg:SDFG, loop_node:OffloadingIRNode, data:Dict):
        old_data = data.copy()

        # recursively ensure all inside copies are made
        self.eval_IR(sdfg, loop_node.body, data)

        # compare beginning and ending of loop
        loop_end : OffloadingIRNode = loop_node.body.get_single_leaf()
        iterative_gpu_copies, iterative_cpu_copies = self._get_neccessary_copies_between_IRs(loop_node.body, loop_end)
        onetime_gpu_copies, onetime_cpu_copies = self._get_neccessary_copies(loop_node.body, old_data)
        
        # add one_time before & outside of loop
        self._add_copies_before_state(sdfg, loop_node.block, onetime_gpu_copies, True)
        self._add_copies_before_state(sdfg, loop_node.block, onetime_cpu_copies, False)
       
        # add iterative after last loop state within loop region
        self._add_copies_after_state(sdfg, loop_end, iterative_gpu_copies, True)
        self._add_copies_after_state(sdfg, loop_end, iterative_cpu_copies, False)
       
        # close block
        print(f"data before closing {loop_node.debug_name}: {data}")
        self.close_block(loop_node.block, data, old_data)
        print(f"data after closing  {loop_node.debug_name}: {data}\n")

    def eval_nested(self, cf_block:ControlFlowBlock, inside_IR:OffloadingIRNode, data:Dict):
        old_data = data.copy()
        
        # recursively ensure all inside copies are made
        self.eval_IR(inside_IR)

        # close block
        self.close_block(cf_block, data, old_data)

    def _get_neccessary_copies(self, IR, data):
        gpu_copies = {array for array in IR.gpu_set if data[array][0] != dtypes.StorageType.GPU_Global}
        cpu_copies = {array for array in IR.cpu_set if data[array][0] != dtypes.StorageType.Default}
        return gpu_copies, cpu_copies
    
    def _get_neccessary_copies_between_IRs(self, IR1, IR2):
        gpu_copies = IR1.cpu_set & IR2.gpu_set
        cpu_copies = IR1.gpu_set & IR2.cpu_set
        return gpu_copies, cpu_copies
    
    def close_block(self, block, data, old_data):
        # mark the arrays which were last accessed within this state simply as "accessed by this state"
        # any future copies will be added after this state rather than within it; for multiple nested states this creates a recursive wrapping
        # => follow the "insert copies as high-level as possible" principle
        # semantically: this state is finished and will be treated as a blackbox from here on out
        for array in self._get_accesses_by_comp(old_data, data):
           _, curr_loc = data[array]
           data[array] = (block, curr_loc)
            
    def _get_accesses_by_comp(self, old_data, new_data) -> set:
        changed_arrays = set()
        for array in new_data:
            previous_last_access, _ = old_data[array]
            last_access, _ = new_data[array]
            if previous_last_access != last_access:
                changed_arrays.add(array)
        return changed_arrays


    def eval_end(self, sdfg:SDFG, prev:OffloadingIRNode, data:Dict, end_loc=dtypes.StorageType.Default):
        copies = []

        for array in data:
            assert array in sdfg.arrays
            if not sdfg.arrays[array].transient: 
                continue # transient arrays don't need copies

            _, curr_loc = data[array] # state which last accessed array & whether its currently on GPU or CPU
            if curr_loc != end_loc:
                copies.append(array)

        print("copies:", copies)
        self._add_copies_after_state(sdfg, prev.block, copies, to_gpu = end_loc==dtypes.StorageType.GPU_Global)


    
    


    ### Step 4: Copy Insertion ###
    def insert_initial_copy(self, sdfg, array:str, to_gpu:bool):
        print("initial copy of", array)
        assert array in sdfg.arrays
        array_desc = sdfg.arrays[array]
        assert array_desc.storage == dtypes.StorageType.Default if to_gpu else dtypes.StorageType.GPU_Global
        
        if array_desc.transient: # allocate directly on device
            array_desc.storage = dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default
            if to_gpu: # TODO: better renaming scheme
                sdfg.replace(array, self._gpu_name(array))
        
        else: # copy needed
            self._add_copies_before_state(sdfg, sdfg.start_block, [array], to_gpu)

    """
    # TODO: ensure state1 and state2 can be ControlFlowBlocks
    def create_interstate_copy(self, sdfg, state1:ControlFlowBlock, state2:ControlFlowBlock, array_names, to_gpu:bool):
        assert state1 is not None or state2 is not None, "invalid: both states are None"
        assert isinstance(array_names, set), f"array_names must be a set[str], got {type(array_names).__name__}"
        if not array_names: return

        # get name and storage location of new arrays
        rename_map = {(name if to_gpu else self._gpu_name(name)): (self._gpu_name(name) if to_gpu else name) for name in array_names}

        # create ONE copy state for all arrays in array_names
        target_graph = state1.parent_graph if state1 is not None else state2.parent_graph
        assert target_graph is not None, "copy insertion requires a parent control-flow graph"

        joined = "_".join(sorted(array_names))
        copy_state = target_graph.add_state(f"copy_{joined}_{'to_gpu' if to_gpu else 'to_cpu'}")

        if state1 is None: # copy state becomes new start block, connect to head
            assert state2 is target_graph.start_block
            target_graph.start_block = target_graph.node_id(copy_state)
            target_graph.add_edge(copy_state, state2, dace.InterstateEdge())

        elif state2 is None: # connect copy state to last state of graph
            target_graph.add_edge(state1, copy_state, dace.InterstateEdge())

        else:
            outedges = target_graph.out_edges(state1)
            assert len(outedges) == 1 # assert that it's a line graph
            assert outedges[0].dst == state2 or state1 == state2, f"state1 {state1} has outedges {outedges[0].dst} and doesn't connect to state2 {state2}" # assert state1 & state2 are really connected
            self._insert_state_on_interstate_edge(target_graph, outedges[0], copy_state)

        # build all copies inside the same inserted state
        for name in array_names:
            self._add_copy_to_state(sdfg, copy_state, name, to_gpu)
        
        if state2:
            # rename data nodes in successor state
            for node in state2.data_nodes():
                if node.data in rename_map:
                    node.data = rename_map[node.data]

            # rename memlets in successor state
            for edge in state2.edges():
                for e in state2.memlet_tree(edge):
                    memlet = e.data
                    if memlet is not None and not memlet.is_empty() and memlet.data in rename_map:
                        memlet.data = rename_map[memlet.data]
            
            # NOTE: map connector names are purely symbolic and are thus do not need renaming
    """
            
    def _insert_state_on_interstate_edge(self, graph, edge, new_state):
        """Insert new_state on an interstate edge (between two control-flow blocks)."""
        graph.add_edge(edge.src, new_state, edge.data)              # src -> new_state (keep condition/assignments)
        graph.add_edge(new_state, edge.dst, dace.InterstateEdge())  # new_state -> dst (unconditional)
        graph.remove_edge(edge)  # del original edge


    def _add_copies_after_state(self, sdfg:SDFG, state:SDFGState, names:list[str], to_gpu:bool):
        self._add_copies(sdfg, state, names, to_gpu, True)
    
    def _add_copies_before_state(self, sdfg:SDFG, state:SDFGState, names:list[str], to_gpu:bool):
        self._add_copies(sdfg, state, names, to_gpu, False)

    def _add_copies(self, sdfg:SDFG, state:SDFGState, names:list[str], to_gpu:bool, after:bool=True):
        # figure out naming (TODO)
        if not names: return
        rename_map = {(name if to_gpu else self._gpu_name(name)): (self._gpu_name(name) if to_gpu else name) for name in names}
        print("rename_map: ", rename_map)

        for old_name, new_name in rename_map.items():
            if old_name not in sdfg.arrays:
                raise KeyError(f"Unknown array: {old_name}")
            
            if new_name not in sdfg.arrays:
                sdfg.add_transient(new_name, sdfg.arrays[old_name].shape, dace.float64, storage= dtypes.StorageType.GPU_Global if to_gpu else dtypes.StorageType.Default)
        

        # find copy state
        assert state is not None
        copy_state : SDFGState = None

        if after: # copy state is after given state; default
            if state in self._copy_states:
                copy_state = self._copy_states[state]
            else:
                copy_state = sdfg.add_state_after(state, "copy_state_after_" + state.label, False)
                self._copy_states[state] = copy_state

        else: # copy state is before given state -> must find predecessor
            pred_states = [e.src for e in state.parent_graph.in_edges(state)]
            if not pred_states: pred_states = [None]

            for pred_state in pred_states:
                if pred_state in self._copy_states:
                    copy_state = self._copy_states[pred_state]
                    break

            else:
                copy_state = sdfg.add_state_before(state, "copy_state_before_" + state.label, is_start_block=(state.parent_graph.start_block is state))
                for pred_state in pred_states:
                    self._copy_states[pred_state] = copy_state


        # add copies to copy state
        for old_name, new_name in rename_map.items():
            copy_in = copy_state.add_access(old_name)
            copy_out = copy_state.add_access(new_name)
            copy_state.add_edge(copy_in, None, copy_out, None, dace.Memlet(f"{old_name} -> {new_name}"))
                
    
            
    """def get_old_and_new_name(name, to_gpu):
        old_name = name           if to_gpu else  name + "_gpu"
        new_name = name + "_gpu"  if to_gpu else  name
        return old_name, new_name"""
    
    def _gpu_name(self, name):
        return name + "_gpu"
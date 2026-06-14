
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

    ID = 0

    def __init__(self, block, cpu_set, gpu_set, next):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.block : ControlFlowBlock = block
        self.cpu_set : set[str] = cpu_set
        self.gpu_set : set[str] = gpu_set
        self.next : list[OffloadingIRNode] = next

        self.set_debug_name(f"{block}")
        self.persistent_helper = False

    def set_debug_name(self, name):
        self.debug_name = name + str(OffloadingIRNode.ID) 
        OffloadingIRNode.ID += 1

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

        next_list = sorted(self.next, key=lambda x:x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s
    
    # utility functions
    def is_empty(self):
        return not self.cpu_set and not self.gpu_set and not self.persistent_helper
    
    def is_join(self):
        return self.debug_name == "_join"
    
    def append_node(self, node):
        self.next.append(node)

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

    def is_leaf(self):
        return not self.next

    # static makers
    def make_helper(debug_name, persistence):
        node = OffloadingIRNode(None, set(), set(), [])
        node.set_debug_name(debug_name)
        node.persistent_helper = persistence
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
        self.eval_IR(sdfg, sdfgIR)
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
        IR = OffloadingIRNode.make_helper("_SDFG_head", True)
        IR.cpu_set = all_arrays # all arrays are initially assumed to be on CPU

        # create IR
        end = self._parse_to_IR(sdfg, sdfg, IR)

        # add final node where all arrays must be on CPU again
        tail = OffloadingIRNode.make_helper("_SDFG_tail", True)
        end.append_node(tail)
        tail.gpu_set = end.gpu_set
        tail.cpu_set = end.cpu_set
        tail.block = list(sdfg.bfs_nodes())[-1] # final toplevel block

        copy_back = OffloadingIRNode.make_helper("_SDFG_copy_back", True)
        tail.append_node(copy_back)
        copy_back.cpu_set = all_arrays
        

        print("--- early IR ---\n", IR, "\n\n")

        # clean up: if a node has no accesses or is a join node, then delete it from IR
        # this aids performance but is also required for correctness, as join nodes must be connected directly to the next relevant node to synchronize activity properly
        self._remove_empty_nodes(IR)
        print("--- clean IR ---\n", IR, "\n\n")
        
        # all arrays which aren't used by this state retain their previous status
        self._propagate_arrays(IR)
        
        # join nodes are special nodes which can take multiple inputs and connect them to one output
        # the output assumes the sets of its next relevant node & thus ensures copy consistency of all the inputs at the join-point
        self._determine_join_sets(IR)
        
        # create & rename all GPU arrays
        self._create_gpu_arrays_from(sdfg, IR)

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
                new_node = OffloadingIRNode(cfr, cpu_set=arrays, gpu_set=set(), next=[])
                curr_node.append_node(new_node)
                curr_node = new_node
            
        # nodes
        print(f"{cfr} contains {", ".join([str(block) for block in cfr.bfs_nodes()])}")
        block : ControlFlowBlock
        for block in cfr.bfs_nodes():
            if str(cfr) == "for_224":
                print(f"about to process {block}. curr node is {curr_node.debug_name}")
            
            # non-nested state
            if isinstance(block, SDFGState):
                state : SDFGState = block
                gpu_set,cpu_set = self.get_data_locations_of_state(sdfg, state) # beating heart of this function
                new_node = OffloadingIRNode(state, cpu_set, gpu_set, [])
                curr_node.append_node(new_node)
                curr_node = new_node

            # if else
            elif isinstance(block, ConditionalBlock):
                # connect current node to new node reprenting the branching condition
                # find all array accesses in condition and add to cpu set
                # if condition necessitates copies, they will be added before the block
                branch_condition = OffloadingIRNode(block, cpu_set=set(), gpu_set=set(), next=[])
                for memlet in block.get_meta_read_memlets():
                    if memlet.data in sdfg.arrays:
                        branch_condition.cpu_set.add(memlet.data)
                curr_node.append_node(branch_condition)

                # parse branches and connect each branch head to branch condition
                tails = []
                for _, branch in block.branches:
                    branch_end : OffloadingIRNode = self._parse_to_IR(sdfg, branch, branch_condition)
                    tails.append(branch_end)

                # connect all tails to a join node (special functionality, ensures consistency)
                curr_node = OffloadingIRNode.make_helper("_join", True)
                for tail in tails:
                    tail.append_node(curr_node)

            # loop
            elif isinstance(block, LoopRegion):
                # head -> body -> tails -> tail -> head 
                loop : LoopRegion = block

                # HEAD: get array accesses of init_statement, update_statement, and loop_condition add them to head's cpu_set
                head = OffloadingIRNode.make_helper("_loop_head", False)
                
                for memlet in loop.get_meta_read_memlets():
                    if memlet.data in sdfg.arrays:
                        head.cpu_set.add(memlet.data)
                curr_node.append_node(head)

                # BODY: parse loop region and connect to head
                body = self._parse_to_IR(sdfg, loop, head) # linked list representing all internal nodes of loop
                
                # TAILS: connect all tails to a join node (special functionality, ensures consistency)
                check = list(head.get_all_leaves())
                assert len(check) <= 1, str(check)
                if len(check) == 1:
                    assert check[0] == body, f"{check[0]} != {body}"

                curr_node = OffloadingIRNode.make_helper("_join", False)
                body.append_node(curr_node)

                # CLOSE: connect connector to head again (-> loop)
                curr_node.append_node(head)

                print(f"just finished loop {block.label} ({head.debug_name} -> {curr_node.debug_name}): {head}\n")

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

        cfr_repr = OffloadingIRNode.make_helper(f"_{cfr}_rep", True)
        cfr_repr.gpu_set = curr_node.gpu_set
        cfr_repr.cpu_set = curr_node.cpu_set
        curr_node.append_node(cfr_repr)
        return cfr_repr
    
    def __traverse_IR(self, IR:OffloadingIRNode, method):
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            method(node)
            
            for next in node.next:
                recursion(next, visited_set)

        return recursion(IR, set())
    
    def __traverse_single_path(self, IR:OffloadingIRNode, method):
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            method(node)
            
            for next in node.next:
                recursion(next, visited_set)
                break # !!!

        return recursion(IR, set())
    
    def __traverse_IR_bottomup(self, IR:OffloadingIRNode, method):
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            for next in node.next:
                recursion(next, visited_set)

            method(node)

        return recursion(IR, set())

    def _propagate_arrays(self, IR:OffloadingIRNode):
        # all arrays which aren't used by this state retain their previous status
        # ASSUMPTION: arrays are either gpu or cpu within a state
        def propagate(node):        
            for next in node.next:
                next_arrays = next.cpu_set | next.gpu_set
                
                for array in node.cpu_set:
                    if not array in next_arrays:
                        next.cpu_set.add(array)
                for array in node.gpu_set:
                    if not array in next_arrays:
                        next.gpu_set.add(array)

        self.__traverse_IR(IR, propagate)
    
    def _remove_empty_nodes(self, IR:OffloadingIRNode):
        
        def remove_empty(node:OffloadingIRNode):
            empties = {next for next in node.next if next.is_empty()}
            
            for empty in empties:
                node.next.remove(empty)
                for nextnext in empty.next:
                    node.append_node(nextnext)

        self.__traverse_IR_bottomup(IR, remove_empty)
    
    # join nodes connect multiple inputs to a single (relevant) output
    # they take on the sets of the output and thus ensure that all
    # they are interm. helpers & are required because the set of inputs is often known before the next relevant node. The only way to move on without knowing the output, the inputs are connected to the join.
    def _determine_join_sets(self, IR:OffloadingIRNode):
        def join_sets(node:OffloadingIRNode):
            if node.is_join():
                next_node : OffloadingIRNode = node.next[-1] # HACK: only join nodes are cond & loop. Cond only has 1 next, loop has 2. For loop next[0] = loop head, next[1] = next node -> deterministic bc insertion order -> hence access both as next[-1]
                node.cpu_set = next_node.cpu_set
                node.gpu_set = next_node.gpu_set
                #print(f"\n\nBRANCH {node.debug_name} -> {next_node.debug_name}")
                #print(f"{node.debug_name}: cpu = {node.cpu_set}, gpu = {node.gpu_set} -> {next_node.debug_name}: cpu = {next_node.cpu_set}, gpu = {next_node.gpu_set}\n\n")

        self.__traverse_IR(IR, join_sets)

    def _create_before_node(self,IR:OffloadingIRNode):
        before = OffloadingIRNode.make_helper(f"before_{IR.block.label}", True)
        
        def get_befores(node:OffloadingIRNode):
            for array_name in node.gpu_set:
                seen_before = array_name in before.gpu_set or array_name in before.cpu_set
                if not seen_before:
                    before.gpu_set.add(array_name)

            for array_name in node.cpu_set:
                seen_before = array_name in before.gpu_set or array_name in before.cpu_set
                if not seen_before:
                    before.cpu_set.add(array_name)

        self.__traverse_single_path(IR, get_befores)
        return before
    
      
    
    
    def _create_gpu_arrays_from(self, sdfg, IR:OffloadingIRNode):
        def gpu_arrays(node:OffloadingIRNode):
            if node.block:
                rename_dict = {array : array + "_gpu" for array in node.gpu_set}
                self._rename_arrays_in_block(node.block, rename_dict)

                for cpu_name, gpu_name in rename_dict.items():
                    #assert cpu_name in sdfg.arrays, cpu_name
                    if not cpu_name in sdfg.arrays: # there is an internal nested sdfg -> do not rename
                        continue
                    if not gpu_name in sdfg.arrays: # create gpu-euivalents as transient on-GPU arrays
                        cpu_array = sdfg.arrays[cpu_name]
                        sdfg.add_array(gpu_name, cpu_array.shape, cpu_array.dtype, storage = dtypes.StorageType.GPU_Global, transient=True)
                    
        self.__traverse_IR(IR, gpu_arrays)
    
    def _rename_arrays_in_block(self, block:ControlFlowBlock, rename_dict:dict):
        if isinstance(block, SDFGState):
            self._rename_arrays_in_state(block, rename_dict)
        
        elif isinstance(block, ControlFlowBlock):
            # rename meta accesses (control-flow metadata like loop bounds or conditions)
            block.replace_meta_accesses(rename_dict)
            # NOTE: internal states / blocks have their own IRNode and don't need to be handled recursively here
        else:
            raise NotImplementedError(f"in _rename_all_gpu_arrays: IR.block unhandled type: {IR.block} is {IR.block.__class__.__name__}")

    def _rename_arrays_in_state(self, state : SDFGState, rename_dict:dict):
        # rename access nodes
        for access in state.data_nodes():
            if access.data in rename_dict:
                access.data = rename_dict[access.data]

        # rename edge conditions
        for edge in state.edges():
            for e in state.memlet_tree(edge):
                memlet = e.data
                if memlet is not None and not memlet.is_empty() and memlet.data in rename_dict:
                    memlet.data = rename_dict[memlet.data]
        

    def eval_IR(self, sdfg, IR:OffloadingIRNode):
        # modifies SDFG in place & inserts all necessary copies
        def eval(node:OffloadingIRNode):
            for next in node.next:

                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(f"This pass does not support copies within a single state. State {node.debug_name} uses arrays {node.cpu_set & node.gpu_set} on both cpu and gpu.")

                gpu_copies = node.cpu_set & next.gpu_set
                if gpu_copies:
                    print(f"insert gpu copy for {gpu_copies} between {node.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, node.block, next.block, gpu_copies, to_gpu=True)
                    
                cpu_copies = node.gpu_set & next.cpu_set
                if cpu_copies:
                    print(f"insert cpu copy for {cpu_copies} between {node.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, node.block, next.block, cpu_copies, to_gpu=False)

        self.__traverse_IR(IR, eval)

    """
    # This won't work. even if an array has never been used, it needs to be copies if its user given input
    # or it needs to be copies back after final use. Need actual read write set anaylsis, not sure its worth it.
    def eval_IR2(self, sdfg, IR:OffloadingIRNode):
        # modifies SDFG in place & inserts all necessary copies

        # USE ANALYSIS
        first_use : dict = {} # array -> IR node with first use
        final_use : dict = {} # array -> IR node with last use

        def first_use_analysis(node:OffloadingIRNode):
            for array in node.gpu_set | node.cpu_set:
                if not array in first_use:
                    first_use[array] = node
        self.__traverse_IR(IR, first_use_analysis)

        def final_use_analysis(node:OffloadingIRNode):
            for array in node.gpu_set | node.cpu_set:
                if not array in final_use:
                    final_use[array] = node
        self.__traverse_IR_bottomup(IR, final_use_analysis)


        print("first_use:", {array:node.debug_name for array,node in first_use.items()})
        print("final_use:", {array:node.debug_name for array,node in final_use.items()})

       # EVAL
        active_set = set() # contains all arrays which are after first and before final use
        
        def eval(node:OffloadingIRNode):
            # update active set (final)
            for array,final_node in final_use.items():
                if final_node == node:
                    active_set.remove(array)

            # insert copies
            for next in node.next:
                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(f"This pass does not support copies within a single state. State {node.debug_name} uses arrays {node.cpu_set & node.gpu_set} on both cpu and gpu.")

                gpu_copies = node.cpu_set & next.gpu_set & active_set
                if gpu_copies:
                    print(f"insert gpu copy for {gpu_copies} between {node.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, node.block, next.block, gpu_copies, to_gpu=True)
                    
                cpu_copies = node.gpu_set & next.cpu_set & active_set
                if cpu_copies:
                    print(f"insert cpu copy for {cpu_copies} between {node.debug_name} and {next.debug_name}")
                    self.create_interstate_copy(sdfg, node.block, next.block, cpu_copies, to_gpu=False)

            # update active set (first)
            for array,first_node in first_use.items():
                if first_node == node:
                    active_set.add(array)

        self.__traverse_IR(IR, eval)"""
    
    ### Step 4: Copy Insertion ###
    def create_interstate_copy(self, sdfg, state1, state2, array_names, to_gpu:bool):
        # create ONE copy state for all arrays in array_names

        assert state1 is not None or state2 is not None, "invalid: both states are None"
        if not array_names: return

        # do not copy transient arrays -> ASSUMPTION: NO REUSE OF TRANSIENTS WHICH REQUIRES COPIES
        clean_array_names = [name for name in array_names if not sdfg.arrays[name].transient]

        # insert new state
        copy_state : SDFGState
        label = f"copy_{"_".join(sorted(clean_array_names))}_{'to_gpu' if to_gpu else 'to_cpu'}"
        
        if state2:
            print("INSERT COPY BEFORE ", state2)
            target_graph = state2.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s2)"

            copy_state = target_graph.add_state_before(state2, label = label)
            if state2 is target_graph.start_block: 
                target_graph.start_block = target_graph.node_id(copy_state) # copy state becomes new start block
        
        else:
            print("INSERT COPY AFTER ", state1)
            target_graph = state1.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s1)"

            copy_state = target_graph.add_state_after(state1, label = label)
        
        # build all copies inside the new state (arrays were created and located previously)
        copy_map = {(name if to_gpu else name + "_gpu"): (name + "_gpu" if to_gpu else name) for name in clean_array_names}

        for old_name, new_name in copy_map.items():
            assert new_name in sdfg.arrays
            assert old_name in sdfg.arrays

            copy_in = copy_state.add_access(old_name)
            copy_out = copy_state.add_access(new_name)
            copy_state.add_edge(copy_in, None, copy_out, None, dace.Memlet(f"{old_name} -> {new_name}"))
                
        
# A -> A_gpu, A -> A_host

from dace import dtypes, properties, data
from dace.sdfg import nodes, SDFG, InterstateEdge
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
import dace

from typing import Any, Dict, Tuple, List, Optional, Set, Type, Union

class OffloadingIRNode:
    # INVARIANT: IR-trees are always DAGs
    CTR = 0

    STATE = -1
    OPEN = 0
    CLOSE = 1
    OPEN_LOOP = 2
    OPEN_COND = 3
    EDGE = 4 # interstate edge

    def __init__(self, type:int, block:ControlFlowBlock, cpu_set:set, gpu_set:set, next:list, close):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.type = type
        self.block : ControlFlowBlock = block
        self.cpu_set : set[str] = cpu_set
        self.gpu_set : set[str] = gpu_set
        self.next : list[OffloadingIRNode] = next
        self.close = close
        
        self.open = None
        self.debug_name = "Cpt. Nemo"
        self.copy_successor = False
        
        # there should be a reference to the corresponding close node IFF the current node is an open node
        assert (self.close is not None) == self.is_open_node(), f"node {self.debug_name} of type {self.type} has close {self.close}"
        

    def __repr__(self):
        return self._get_str(set(), -4)
    def __str__(self): 
        return self.__repr__()
    def _get_str(self, visited_set, len_before):
        s = f"{self.debug_name}:"
        spaces = 50 - (len_before + len(s))
        s += spaces * " " + f"cpu = {sorted([name for name in self.cpu_set if len(name) <= 5])}, gpu = {sorted([name for name in self.gpu_set if len(name) <= 5])}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        next_list = sorted(self.next, key=lambda x:x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s
    
    # utility functions
    def is_empty(self):
        return not self.cpu_set and not self.gpu_set
    
    def is_open_node(self):
        return self.type in [OffloadingIRNode.OPEN, OffloadingIRNode.OPEN_LOOP, OffloadingIRNode.OPEN_COND]

    def is_close_node(self):
        return self.type in [OffloadingIRNode.CLOSE]

    def append_node(self, node):
        self.next.append(node)

    def get_all_tails(self):
        assert self.is_open_node()

        def recursion(node, result:list):
            for next in node.next:
                if next == self.close: # definition of a tail: a node that points at this section's end (close-node)
                    result.append(node)
                    return
                recursion(next, result)

        result = []
        recursion(self, result)
        return result
    
    # static makers
    def new_open_node(block:ControlFlowBlock):
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, set(), set(), [], None)
        close.debug_name = f"_close_{block.label}"#{OffloadingIRNode.CTR}"

        type : int
        if isinstance(block, LoopRegion):
            type = OffloadingIRNode.OPEN_LOOP
        elif isinstance(block, ConditionalBlock):
            type = OffloadingIRNode.OPEN_COND
        else:
            type = OffloadingIRNode.OPEN

        open = OffloadingIRNode(type, block, set(), set(), [], close)
        open.debug_name = f"_{OffloadingIRNode.get_type_as_str(type)}_{block.label}"#{OffloadingIRNode.CTR}"
        close.open = open

        OffloadingIRNode.CTR += 1
        return open
    
    def new_state_node(block:ControlFlowBlock, cpu_set:set, gpu_set:set):
        state = OffloadingIRNode(OffloadingIRNode.STATE, block, cpu_set, gpu_set, [], None)
        state.debug_name = f"_state_{block.label}"
        return state
    
    def new_edge_node(edge:InterstateEdge, cpu_set:set):
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, edge, cpu_set, set(), [], None)
        edge_node.debug_name = f"_edge_{edge.label}"
        return edge_node
    
    def get_type_as_str(type:int):
        match type:
            case OffloadingIRNode.STATE: return "state"
            case OffloadingIRNode.OPEN: return "open"
            case OffloadingIRNode.OPEN: return "close"
            case OffloadingIRNode.OPEN_LOOP: return "loop"
            case OffloadingIRNode.OPEN_COND: return "cond"
        raise ValueError(f"Invalid IR type to convert to string: {type}")
    
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
                    if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
                        node.map.schedule = dtypes.ScheduleType.GPU_Device
                    elif isinstance(node, nodes.LibraryNode):
                        node.schedule = dtypes.ScheduleType.GPU_Device
                    else:
                        assert False

                else: # within nested scope -> must not have GPU schedule (defensive check)
                    if self.has_GPU_schedule(node):
                        raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass." \
                        "All maps must have default or CPU schedule before pass." \
                        f"Node {node} has schedule type {self.get_schedule(node)}" )

    
    ### generic HELPERS ###

    def get_schedule(self, node):
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            return node.map.schedule
        elif isinstance(node, nodes.LibraryNode):
            return node.schedule
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
        if isinstance(node, nodes.AccessNode): 
            data_name = node.data
            arrays = {data_name} # add current access node

            if isinstance(sdfg.arrays[data_name], data.View): # trace it if it is a view
                original = get_last_view_node(state, node) # once the view access node is known, its original access node can be found and it's data added
                arrays |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, original)
                
        # check if more access nodes UPstream
        for n in self.get_predecessors(state, node):
            if isinstance(n, nodes.AccessNode):
                arrays |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, n)

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
           
            # input & output nodes
            if is_gpu:
                gpu_set |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, map_entry)
                gpu_set |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, state.exit_node(map_entry))
            else:
                cpu_set |= self.get_arrays_used_by_incoming_access_nodes(sdfg, state, map_entry)
                cpu_set |= self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, state.exit_node(map_entry))

            # internal nodes
            for node in map_nodes:
                if isinstance(node, nodes.MapEntry): # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu)
                
                elif isinstance(node, nodes.AccessNode): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_outgoing_access_nodes(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.Tasklet): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, (ControlFlowRegion, nodes.NestedSDFG)):
                    g,c = self.get_data_locations_of_cfregion(sdfg, node) if isinstance(node, ControlFlowRegion) else self.get_data_locations(node.sdfg)
                    if not is_gpu:
                        gpu_set |= g
                        cpu_set |= c
                    else:
                        gpu_set |= g | c

                elif isinstance(node, nodes.LibraryNode):
                    # nothing to do but change the Library Schedule
                    node.schedule = dtypes.ScheduleType.GPU_Device

                elif isinstance(node, nodes.MapExit):
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

            elif isinstance(node, nodes.MapExit):
                pass

            # library nodes are usually GPU, can be CPU
            elif isinstance(node, nodes.LibraryNode):
                if self.has_GPU_schedule(node):
                    g = self.get_arrays_used_by_node(sdfg, state, node)
                else:
                    c = self.get_arrays_used_by_node(sdfg, state, node)

            # recurse if nested
            elif isinstance(node, ControlFlowRegion):
                g,c = self.get_data_locations_of_cfregion(sdfg, node)

            # all else is definitely on CPU
            elif isinstance(node, nodes.Tasklet): # outside a map scope (else handled by locations_of_map) -> cpu
                c = self.get_arrays_used_by_node(sdfg, state, node)

            elif isinstance(node, nodes.AccessNode):
                pass # nothing to do; cannot be classified without context

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        # Check for invalid state configurations, where arrays are accessed on both CPU and GPU. 
        overlap = gpu_set & cpu_set
        if overlap: 

            def set_to_gpu(type, nodes):
                has_gpu_node = False
                for node in nodes:
                    if isinstance(node, type) and self.has_GPU_schedule(node): #second conditions prevents infinite recursion on unfixable cases
                        node.schedule = dtypes.ScheduleType.Default
                        has_gpu_node = True
                return has_gpu_node
                
            # This is most often caused by lib nodes which were offloaded but weren't supposed to (initial heuristic failed).
            # Try to keep lib nodes on CPU & analyse state again.
            resolved = False
            change_made = set_to_gpu(nodes.LibraryNode, top_level_nodes)
            
            if change_made:
                gpu_set, cpu_set = self.get_data_locations_of_state(sdfg, state)
                if not (gpu_set & cpu_set):
                    resolved = True

            # If that didn't work, set all maps to CPU (Sequential).
            # This should remove all sources of conflict by completely running the state on CPU
            # In case the conflict still isn't resolved (shouldn't happen), raise an error.
            # Else, raise a Warning to the user that this state couldn't be offloaded properly and continue.
            if not resolved:
                change_made = set_to_gpu(nodes.MapEntry, top_level_nodes)
                
                if change_made:
                    gpu_set, cpu_set = self.get_data_locations_of_state(sdfg, state)
                    if not (gpu_set & cpu_set):
                        resolved = True
                        
            if not resolved:
                raise NotImplementedError(f"(This should never happen.) This pass does not support copies within a single state. State {state} uses arrays {overlap} on both cpu and gpu.")
            
            print(Warning(f"The sdfg {sdfg} could not be fully offloaded. The state {state} accesses the arrays {overlap} both with GPU-capable nodes (maps & library nodes) AND with CPU-bound nodes (e.g. top-level tasklets). This pass only supports copies between states, copies within states are not supported. As a consequence, all maps & library nodes in {state} remain sequential (no offloading). To enable optimization, ensure arrays are EITHER accessed by maps OR by top-level tasklets within the same state."))


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

        for block in cfr.bfs_nodes():
            g,c = self.get_data_locations_of_cfblock(sdfg, block)
            gpu_set |= g
            cpu_set |= c
        
        return gpu_set, cpu_set


    # wrapper
    def get_data_locations(self, sdfg:SDFG) -> tuple[set[str], set[str]]:
        return self.get_data_locations_of_cfregion(sdfg, sdfg)


    ### STEP 3: Intermediate Representation ###

    def sdfg_to_IR(self, sdfg:SDFG):
        for array in sdfg.arrays:
            if sdfg.arrays[array].storage == dtypes.StorageType.GPU_Global:
                raise RuntimeError(f"Graph cannot be offloaded because {array} is already set to GPU location. All arrays must first be in Default location.")
        
        non_transients = {name for name in sdfg.arrays if not sdfg.arrays[name].transient}

        # create inital node where all arrays must be on CPU
        IR = OffloadingIRNode.new_open_node(sdfg)
        IR.cpu_set = non_transients # all arrays are initially assumed to be on CPU
        
        # parse entire graph
        end = self._parse_to_IR(sdfg, sdfg, IR)

        # set final close node where all arrays must be on CPU again
        end.append_node(IR.close)
        IR.close.cpu_set = non_transients
    
        print(f"--- early IR ---\n{IR}\n")

        self._propagate_arrays(IR)
        
        print(f"--- full IR ---\n{IR}\n\n")

        # create & rename all GPU arrays
        self._create_gpu_arrays_from(sdfg, IR)

        return IR

  

    def _parse_to_IR(self, sdfg:SDFG, cfr:ControlFlowRegion, curr_node:OffloadingIRNode) -> OffloadingIRNode:
        # NOTE to self: ControlFlowRegion inherits from ControlFlowBlock
        
        block : ControlFlowBlock
        for block in cfr.bfs_nodes():

            # iterate through all (incoming) interstate edges
            in_edge_arrays = set()
            for edge in cfr.in_edges(block):
                arrays = edge.data.used_arrays(sdfg.arrays)
                in_edge_arrays |= arrays
                print("EGDE:", cfr, " ", edge, "->", arrays)

            if in_edge_arrays:
                edge_node = OffloadingIRNode.new_edge_node(block, in_edge_arrays)
                curr_node.append_node(edge_node)
                curr_node = edge_node

            # iterate through all nodes
            # non-nested state
            if isinstance(block, SDFGState):
                state : SDFGState = block
                gpu_set,cpu_set = self.get_data_locations_of_state(sdfg, state) # beating heart of this entire function
                state_node = OffloadingIRNode.new_state_node(state, cpu_set, gpu_set)
                curr_node.append_node(state_node)
                curr_node = state_node

            # do nothing
            elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                pass 
            
            # container node with outer wrapper
            else:
                # outer node
                outer_node = OffloadingIRNode.new_open_node(block)
                curr_node.append_node(outer_node)
                curr_node = outer_node

                # if else
                if isinstance(block, ConditionalBlock):
                    cond_block : LoopRegion = block

                    # branch condition
                    meta_data_node : OffloadingIRNode = None
                    meta_data = {memlet.data for memlet in cond_block.get_meta_read_memlets() if memlet.data in sdfg.arrays}
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=set())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse branches and connect each branch to close node
                    for _, branch in cond_block.branches:
                        branch_end : OffloadingIRNode = self._parse_to_IR(sdfg, branch, curr_node)
                        # TODO: FIND ALL TAILS
                        branch_end.append_node(outer_node.close)
                    
                # loop
                elif isinstance(block, LoopRegion):
                    loop : LoopRegion = block

                    # add meta data node if needed
                    meta_data_node : OffloadingIRNode = None
                    meta_data = {memlet.data for memlet in loop.get_meta_read_memlets() if memlet.data in sdfg.arrays}
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=set())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse body and connect to loop close node
                    # TODO: FIND ALL TAILS
                    curr_node = self._parse_to_IR(sdfg, loop, curr_node) # linked list representing all internal nodes of loop
                    curr_node.append_node(outer_node.close)
                    
                # nested region -> flatten   
                elif isinstance(block, ControlFlowRegion):
                    curr_node = self._parse_to_IR(sdfg, block, curr_node)
                    curr_node.append_node(outer_node.close)

                elif isinstance(block, nodes.NestedSDFG):
                    curr_node = self._parse_to_IR(block.sdfg, block.sdfg, curr_node)
                    curr_node.append_node(outer_node.close)
                
                else:
                    raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

                # finish container
                self._populate_container_node_sets(outer_node)
                curr_node = outer_node.close

        # TODO: FIND ALL TAILS?
        return curr_node
    

    def __traverse_IR(self, IR:OffloadingIRNode, method):
        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            method(node)
            
            for next in node.next:
                recursion(next, visited_set)

        return recursion(IR, set())
    

    def __traverse_same_level(self, IR:OffloadingIRNode, method): #DFS
        queue = IR.next.copy()
        while queue:
            curr = queue.pop()
            if curr.type == OffloadingIRNode.STATE or curr.type == OffloadingIRNode.EDGE: # data node
                method(curr)
                queue += curr.next

            elif curr.is_open_node():
                method(curr)
                queue += curr.close.next

            elif curr.type == OffloadingIRNode.CLOSE:
                break

            else:
                assert False


    def _populate_container_node_sets(self, IR:OffloadingIRNode):
        self.__populate_open_node_sets(IR)
        self.__populate_close_node_sets(IR)
        # TODO for both: deal with FIND ALL TAILS

    def __populate_open_node_sets(self, IR:OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        # Behavior 1: 
        # if there are no or multiple direct children, leave the sets empty & simply propagate later
        # there is no good heuristic to choose from here, which copies to make and which not
        # (not without significantly more analysis)
        children = IR.next
        if len(children) != 1:
            return
        
        # Behavior 2: 
        # if there is a single direct child, then analyse the section & find first known location of each used array
        # if the graph splits later, the first of all possible paths is chosen for analysis
        # this can lead to unnecessary copies in the other paths
        location_on_gpu = {}
        def gather_data(node:OffloadingIRNode):
            for array_name in node.gpu_set:
                if not array_name in location_on_gpu:
                    location_on_gpu[array_name] = True

            for array_name in node.cpu_set:
                if not array_name in location_on_gpu:
                    location_on_gpu[array_name] = False

        # traverse graph
        self.__traverse_same_level(IR, gather_data)

        # populate IR sets
        IR.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
        IR.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}


    def __populate_close_node_sets(self,IR:OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        tails = IR.get_all_tails()
        assert tails, f"{IR.debug_name} doesn't have any tails! {IR}"

        # Behavior 1: 
        # if there is a single tail node (node that leads to this section's close node),
        # then analyse the section & find last known location of each used array
        if len(tails) == 1:
            # define data gathering function
            location_on_gpu = {}
            def gather_data(node:OffloadingIRNode):
                for array_name in node.gpu_set:
                    location_on_gpu[array_name] = True

                for array_name in node.cpu_set:
                    location_on_gpu[array_name] = False
            
            # traverse graph
            self.__traverse_same_level(IR, gather_data)

            # populate IR sets
            IR.close.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
            IR.close.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}


        # Behaviour 2:
        # if there are multiple tail nodes, then mark this node for later.
        # In a second pass, it will assume the gpu&cpu set of its next successor.
        # This means that each branch will have to insert copies individually, usually leading to the least amount of necessary copies.
        # There is however a risk that this introduces copies within a loop unnecessarily.
        else:
            print(f"copy_successor: {IR.debug_name} has tails: {[n.debug_name for n in tails]}")
            IR.copy_successor = True
        
    
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

    def _fill_in_successor_copies(self, IR):
        def fill_in_successor_copies(node):
            if node.copy_successor and node.next:
                node.cpu_set = node.next[0].cpu_set
                node.gpu_set = node.next[0].gpu_set
        self.__traverse_IR(IR, fill_in_successor_copies)


    def _create_gpu_arrays_from(self, sdfg, IR:OffloadingIRNode):
        def gpu_arrays(node:OffloadingIRNode):
            if node.block:
                rename_dict = {array : self.get_gpu_name(array) for array in node.gpu_set}
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
        def insert_copies(node, next, node_block, next_block):
            gpu_copies = node.cpu_set & next.gpu_set
            if gpu_copies:
                #print(f"insert gpu copy for {gpu_copies} between {node.debug_name} and {next.debug_name}")
                self.create_interstate_copy(sdfg, node_block, next_block, gpu_copies, to_gpu=True)
                
            cpu_copies = node.gpu_set & next.cpu_set
            if cpu_copies:
                #print(f"insert cpu copy for {cpu_copies} between {node.debug_name} and {next.debug_name}")
                self.create_interstate_copy(sdfg, node_block, next_block, cpu_copies, to_gpu=False)


        def eval(node:OffloadingIRNode):
            for next in node.next:

                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(f"This pass does not support copies within a single state. State {node.debug_name} uses arrays {node.cpu_set & node.gpu_set} on both cpu and gpu.")

                # edge case: if this condition is true, both blocks are None, can't insert
                if node.type == OffloadingIRNode.CLOSE and next.type == OffloadingIRNode.CLOSE:
                    insert_copies(node, next, node.open.block, None)

                elif next.type == OffloadingIRNode.EDGE: # then I want the copy AFTER the node, not before
                    insert_copies(node, next, node.block, None)

                else: # the usual: copies between node -> next
                    insert_copies(node, next, node.block, next.block)
                    

            # loop copies if applicable
            if node.type == OffloadingIRNode.OPEN_LOOP:
                top = node # INV: top.type == OffloadingIRNode.OPEN_LOOP
                bottom = node.close # INV: bottom.type == OffloadingIRNode.CLOSE
                tails = OffloadingIRNode.get_all_tails(top) # INV: all are STATE or CLOSE if there's a nested loop

                gpu_copies = bottom.cpu_set & top.gpu_set
                #print(f"LOOP COPIES for {node.debug_name}: {gpu_copies}")

                if gpu_copies:
                    for tail in tails:
                        #print(f"insert LOOP gpu copy for {gpu_copies} between {tail.debug_name} and {bottom.debug_name}")
                        
                        if tail.type == OffloadingIRNode.CLOSE: # and bottom.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, gpu_copies, to_gpu=True)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, gpu_copies, to_gpu=True)

                cpu_copies = bottom.gpu_set & top.cpu_set
                if cpu_copies:
                    for tail in tails:
                        #print(f"insert LOOP cpu copy for {cpu_copies} between {tail.debug_name} ({tail.type}) and {bottom.debug_name} ({bottom.type})")
                        if tail.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, cpu_copies, to_gpu=False)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, cpu_copies, to_gpu=False)


        self.__traverse_IR(IR, eval)


    ### Step 4: Copy Insertion ###
    # create ONE copy state for all arrays in array_names

    def create_interstate_copy(self, sdfg, state1, state2, array_names, to_gpu:bool):
        assert state1 is not None or state2 is not None, "invalid: both states are None"

        # insert new state
        copy_state : SDFGState
        label = f"copy_{"_".join(sorted(array_names))}_{'to_gpu' if to_gpu else 'to_cpu'}"
        
        if state2 is not None:
            print("INSERT COPY BEFORE ", state2)
            target_graph = state2.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s2)"

            copy_state = target_graph.add_state_before(state2, label = label)
            if state2 is target_graph.start_block: 
                target_graph.start_block = target_graph.node_id(copy_state) # copy state becomes new start block
        
        elif state1 is not None:
            print("INSERT COPY AFTER ", state1)
            target_graph = state1.parent_graph if state1.parent_graph else state1
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s1)"

            #copy_state = self.add_state_after(target_graph, state1, label)
            copy_state = target_graph.add_state_after(state1, label = label)

            
        # build all copies inside the new state (arrays were created and located previously)
        copy_map = {(name if to_gpu else self.get_gpu_name(name)): (self.get_gpu_name(name) if to_gpu else name) for name in array_names}
        
        for old_name, new_name in copy_map.items():
            assert new_name in sdfg.arrays
            assert old_name in sdfg.arrays

            copy_in = copy_state.add_access(old_name)
            copy_out = copy_state.add_access(new_name)
            copy_state.add_edge(copy_in, None, copy_out, None, dace.Memlet(f"{old_name} -> {new_name}"))
                
        
    def get_gpu_name(self, cpu_name):
        if cpu_name.startswith("__return"):
            return "return" + cpu_name[8:] + "_gpu"
        return cpu_name + "_gpu"
    
    def add_state_after(self, graph, state, label):
        # adds new state after the given state
        # difference to sdfg.add_state_after:
        #   sdfg.add_state_after copies e.data of outgoing edges to the new connecting edge
        #   this method does not, it inserts a brand new edge between the state and the new state & moves e.data to the outgoing edge of the new state
        #   this is the desired behaviour for edge copies & makes no difference for other types of cpi

        pass
    
    # TODO: A -> A_gpu, A -> A_host
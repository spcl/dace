from dace import dtypes
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock, BreakBlock, ControlFlowBlock

import dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers as helpers
from dace.transformation.passes.offloading.OffloadingIRNode import OffloadingIRNode

class CopyAnalysisPhase():

    def apply(self, sdfg:SDFG, track_hybrid_states:set=None, sdfg_scope_dict:dict=None, verbose=False):
        self.verbose = verbose
        self.hybrid_states = track_hybrid_states # results passed back by values as long as track_hybrid_states is not None

        if sdfg_scope_dict:
            self.sdfg_scope_dict = sdfg_scope_dict
        else:
            self.sdfg_scope_dict = helpers.get_sdfg_scope_dict(sdfg)

        return self.sdfg_to_IR(sdfg)


    def sdfg_to_IR(self, sdfg:SDFG):
        # remember initial non-transient array locations
        non_transients = {name for name in sdfg.arrays if not sdfg.arrays[name].transient and not helpers.is_scalar(name, sdfg)}
        initially_on_gpu = set()
        initially_on_cpu = set()

        for array_name in non_transients:
            if helpers.is_array_stored_on_GPU(sdfg, array_name):
                initially_on_gpu.add(array_name)
            else:
                initially_on_cpu.add(array_name)

        # create inital node (open node)
        IR = OffloadingIRNode.new_open_node(sdfg)
        IR.gpu_set = initially_on_gpu.copy()
        IR.cpu_set = initially_on_cpu.copy() # no copy -> may cause sideeffects
        
        # parse entire graph
        end = self._parse_to_IR(sdfg, sdfg, IR)
        if self.verbose: print(f"Phase2: early IR\n{IR}\n")

        # finish graph: tie the final node together with the inital close node
        end.append_node(IR.close)
        IR.close.gpu_set = initially_on_gpu
        IR.close.cpu_set = initially_on_cpu # arrays end up where they started      

        self._propagate_arrays(IR)

        if self.verbose: print(f"Phase2: full IR \n{IR}\n\n")
        return IR






    #############################
    ###       create IR       ###
    #############################

    def _parse_to_IR(self, sdfg:SDFG, cfr:ControlFlowRegion, curr_node:OffloadingIRNode) -> OffloadingIRNode:
        block : ControlFlowBlock
        for block in cfr.bfs_nodes():

            # iterate through all (incoming) interstate edges
            in_edge_arrays = set()
            for edge in cfr.in_edges(block):
                arrays = {data_name for data_name in edge.data.used_arrays(sdfg.arrays) if helpers.is_array(data_name, sdfg)}
                in_edge_arrays |= arrays
                
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
                    cond_block : ConditionalBlock = block

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
                self.__populate_open_node_sets(outer_node)
                self.__populate_close_node_sets(outer_node)
                curr_node = outer_node.close

        return curr_node

    def __populate_open_node_sets(self, IR:OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        # Behavior 1: 
        # if there are no or multiple direct children, leave the sets empty & propagate later
        # there is no good heuristic which child to choose here, which copies to make and which not
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
            if isinstance(node.block, nodes.NestedSDFG): # Nested SDFGs do not share namespace, array names should not leak to outer scope
                return 
            
            for array_name in node.gpu_set:
                if not array_name in location_on_gpu:
                    location_on_gpu[array_name] = True

            for array_name in node.cpu_set:
                if not array_name in location_on_gpu:
                    location_on_gpu[array_name] = False

        # traverse graph
        helpers.traverse_same_level(IR, gather_data)

        # populate IR sets
        IR.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
        IR.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}


    def __populate_close_node_sets(self,IR:OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        tails = IR.get_all_tails()
        assert tails, f"{IR.debug_name} doesn't have any tails! {IR}"

        # Behavior 1: 
        # if there are no or multiple direct children, leave the sets empty & propagate later
        # there is no good heuristic which child to choose here, which copies to make and which not
        # (not without significantly more analysis)
        if len(tails) > 1:
            return
        
        # Behavior 2: 
        # if there is a single tail node (node that leads to this section's close node),
        # then analyse the section & find last known location of each used array
        
        # define data gathering function
        location_on_gpu = {}
        def gather_data(node:OffloadingIRNode):
            if isinstance(node.block, nodes.NestedSDFG): # Nested SDFGs do not share namespace, array names should not leak to outer scope
                return
            
            for array_name in node.gpu_set:
                location_on_gpu[array_name] = True

            for array_name in node.cpu_set:
                location_on_gpu[array_name] = False
        
        # traverse graph
        helpers.traverse_same_level(IR, gather_data)

        # populate IR sets
        IR.close.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
        IR.close.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}


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

        helpers.traverse_IR(IR, propagate)



    #######################################################
    ###  Helpers get Arrays Used by Edges & Nodes ###
    #######################################################
        
    def get_arrays_used_by_edge(self, sdfg:SDFG, state:SDFGState, edge, is_out_edge:bool):
        if edge.data and not edge.data.is_empty():
            data_name = edge.data.data
            
            if helpers.is_array(data_name, sdfg): # array access on edge
                return {data_name}

            elif helpers.is_view(data_name, sdfg): # view -> we need to find the corresponding view access node by iteration.
                for n in state.data_nodes(): 
                    if n.data == data_name:
                        if is_out_edge:
                            return helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, n)
                        return helpers.get_data_used_by_incoming_access_nodes(sdfg, state, n)
                
            elif helpers.is_scalar(data_name, sdfg): # might be a scalar access of an array slice
                if is_out_edge:
                    if isinstance(edge.dst, nodes.AccessNode):
                        return helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, edge.dst)
                else:
                    if isinstance(edge.src, nodes.AccessNode):
                        return helpers.get_data_used_by_incoming_access_nodes(sdfg, state, edge.src)
                
            else:
                raise RuntimeError(f"Unknown data type (not array, scalar or view) in get_arrays_used_by_edge. edge:{edge}, data:{edge.data}")

        return set()
    
    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays : set[str] = set()

        # edges
        for e in state.in_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, False)
        
        for e in state.out_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, True)

        # neighbouring access nodes
        arrays |= helpers.get_data_used_by_incoming_access_nodes(sdfg, state, node)
        arrays |= helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, node)
       
        return arrays



    
    ################################################################
    ###  Recursive Analysis: Each SDFG Node has dedicated method ###
    ################################################################
            

    def get_data_locations_of_map(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        # helper to validate data and add it to correct set
        def _add_data(data_name: str, gpu_set:set[str], cpu_set:set[str], is_gpu:bool) -> tuple[set[str], set[str]]:
            if data_name in gpu_set: # has already been accessed on GPU
                if not is_gpu: # is now accessed on CPU
                    raise RuntimeError("GPU->CPU within map. This should never happen. If outer map is GPU then inner data must also be on GPU (seq map runs as kernel)")

            elif data_name in cpu_set: # has already been accessed on CPU
                if is_gpu: # is now accessed on GPU
                    gpu_set.add(data_name)

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        # main work horse, can recurse to nested maps
        def _recursive_helper(sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry, gpu_set:set[str], cpu_set:set[str], is_gpu:bool):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES # TODO Q: how not to hardcode?
            
            # get all nodes within this map's scope
            map_nodes = [n for n, parent in self.sdfg_scope_dict[state].items() if parent is map_entry]
           
            # input & output nodes
            input_and_output = helpers.get_data_used_by_incoming_access_nodes(sdfg, state, map_entry) | helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, state.exit_node(map_entry))
            if is_gpu:
                gpu_set |= input_and_output
            else:
                cpu_set |= input_and_output
                
            # internal nodes
            for node in map_nodes:
                if isinstance(node, nodes.MapEntry): # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu)
                
                elif isinstance(node, nodes.AccessNode): # find accessed arrays -> add
                    for name in helpers.get_data_used_by_outgoing_access_nodes(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, nodes.Tasklet): # find accessed arrays -> add
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu)

                elif isinstance(node, (ControlFlowRegion)):
                    g,c = self.get_data_locations_of_cfregion(sdfg, node)
                    if not is_gpu:
                        gpu_set |= g
                        cpu_set |= c
                    else:
                        gpu_set |= g | c

                elif isinstance(node, (nodes.NestedSDFG, nodes.MapExit, nodes.LibraryNode)):
                    pass # nothing to do (do not analyse the arrays within a nested sdfg)

                else:
                    raise RuntimeError(f"Error: Unknown node type inside map {map_entry}: {node.label} of type {node.__class__.__name__} in state {state}")

        # function body, calls recursive helper
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set


    def get_data_locations_of_state(self, sdfg: SDFG, state: SDFGState, recursive_call=False) -> tuple[set[str], set[str]]:
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
                if helpers.has_GPU_schedule(node):
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
                raise RuntimeError(f"Unknown node {node} of type {node.__class__.__name__} in state {state}.")

            gpu_set |= g
            cpu_set |= c

        # Check for hybrid state configurations, where arrays are accessed on both CPU and GPU -> mark for future resolution
        overlap = gpu_set & cpu_set
        if overlap:
            if self.hybrid_states is not None:
                self.hybrid_states.add(state)
            gpu_set |= cpu_set
            cpu_set = set()
 
        return gpu_set, cpu_set
    

    def get_data_locations_of_condblock(self, sdfg: SDFG, block:ConditionalBlock) -> tuple[set[str], set[str]]:
        gpu_set : set[str] = set()
        cpu_set : set[str] = set()

        # get array accesses in condition
        for memlet in block.get_meta_read_memlets():
            if not memlet: continue
            data_name = memlet.data
            if memlet.data in sdfg.arrays and helpers.is_array(data_name, sdfg):
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
            if not memlet: continue
            data_name = memlet.data
            if data_name in sdfg.arrays and helpers.is_array(data_name, sdfg):
                cpu_set.add(data_name)
        
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

        elif isinstance(block, (nodes.NestedSDFG, ReturnBlock, ContinueBlock, BreakBlock)):
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

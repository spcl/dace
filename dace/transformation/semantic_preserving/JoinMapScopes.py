"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the JoinMapScopes transformation.

"""

import dace

class JoinMapScopes():
    """
    This class implements the JoinMapScopes transformation. It searches for two map scopes in the SDFG and joins them. The transformation is only applied if the map scopes are joinable.
    """

    def __init__(self):
        self.__name = 'JoinMapScopes'
        self.checked = False
    
    @property
    def name(self):
        return self.__name

    def find_entries(self, state):
        result = []
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                result.append(node)
        return result
    
    def find(self, sdfg):
        """
        This method searches for two map scopes in the SDFG and returns a list of tuples containing the two map scopes.

        Two map scopes are joinable if the following conditions are met:
            - The two map scopes have the same entry node
            - The two maps have the same map range
            - The two maps have the same map parameters
            - The second map scope comes directly after the first map scope and is the only map scope that follows the first map scope
            - Nodes that are output of the first map scope and input of the second map scope use the same indexing
            - If an output node of is involved in a WCR (reduction), the shape of the output node is equal to the range of the first map scope

        Args:
            sdfg (dace.SDFG): The SDFG to search for joinable map scopes.

        Returns:
            list: A list of tuples, where each tuple contains two joinable map scopes and the state they are in.
        """
        candidates = []
        for state in sdfg.nodes():
            entries = self.find_entries(state)
            for i in range(len(entries)):
                for j in range(len(entries)):
                    if i == j:
                        continue
                    entry1, entry2 = entries[i], entries[j]
                    if not state.entry_node(entry1) == state.entry_node(entry2):
                        continue
                    if not entry1.map.range == entry2.map.range:
                        continue
                    #TODO: allow but rename the map params ?!
                    if not entry1.map.params == entry2.map.params:
                        continue
                    out_nodes = [edge.dst for edge in state.out_edges(state.exit_node(entry1))]
                    next_maps = {edge.dst for node in out_nodes for edge in state.out_edges(node) if isinstance(edge.dst, dace.nodes.MapEntry)}
                    if not entry2 in next_maps or len(next_maps) > 1:
                        continue
                    candidates.append((state, entry1, entry2))

        result = []
        for (state, entry1, entry2) in candidates:
            if not self.check(state, entry1, entry2):
                continue
            result.append((state, entry1, entry2))
        self.checked = True
        return result
    
    def check(self, state, entry1, entry2):
        reject = False
        entry1_out = [edge.dst for edge in state.out_edges(state.exit_node(entry1))]
        entry2_in = [edge.src for edge in state.in_edges(entry2)]
        entry1_params = entry1.map.params
        entry2_params = entry2.map.params
        for elem in set([node.label for node in entry1_out]) & set([node.label for node in entry2_in]):
            nodes_1 = [node for node in entry1_out if node.label == elem]
            nodes_2 = [node for node in entry2_in if node.label == elem]
            idxs_1 = [edge.data for edge in state.in_edges(nodes_1[0])]
            idxs_2 = [edge.data for edge in state.out_edges(nodes_2[0])]
            for idx1 in idxs_1:
                for idx2 in idxs_2:
                    if not idx1.subset == idx2.subset:
                        reject = True
                    
            for node in nodes_1:
                in_edge = state.in_edges(node)[0]
                if in_edge.data.wcr:
                    above_maps = self.get_entry_maps(state, entry1)
                    above_idxs = [map_entry.map.params[0] for map_entry in above_maps] if above_maps else []
                    above_idxs.append(entry1.map.params[0])
                    write_op = self.get_write_op(state, node)
                    reduce_indices = set()
                    for in_edge in state.in_edges(write_op):
                        for subset in in_edge.data.subset:
                            reduce_indices.add(str(subset[0]))
                    for out_edge in state.out_edges(write_op):
                        for subset in out_edge.data.subset:
                            reduce_indices.discard(str(subset[0]))
                    #dst_shape = in_edge.data.dst_subset.size()
                    #print(f'above_idxs: {above_idxs}, reduce_indices: {reduce_indices}')
                    for idxs in reduce_indices:
                        if idxs in above_idxs:
                            reject = True
                            break
                    """
                    if not dst_shape == entry1.map.range.size():
                        reject = True
                    """
        if not reject:
            return True
        return False

    def get_write_op(self, state, node):
        data_name = node.data

        def traverse(node, visited=None):
            if visited is None:
                visited = set()
            visited.add(node)
            if isinstance(node, dace.nodes.Tasklet):
                if state.out_edges(node)[0].data.data == data_name:
                    return node
            for edge in state.in_edges(node):
                if edge.src in visited:
                    continue
                res = traverse(edge.src)
                if res:
                    return res
            return None
        
        return traverse(node)


    def get_entry_maps(self, state, node):
        dims = []
        entry = state.entry_node(node)
        if not entry:
            return dims
        while True:
            dims.append(entry)
            entry = state.entry_node(entry)
            if not entry:
                break
        return dims

    # not used currently
    def find_2(self, sdfg):
        transform = dace.transformation.dataflow.map_fusion.MapFusion()
        candidates = []
        for state in sdfg.nodes():
            entries = self.find_entries(state)
            for i in range(len(entries)):
                next = False
                for j in range(i+1, len(entries)):
                    node1, node2 = entries[i], entries[j]
                    if not state.entry_node(node1) == state.entry_node(node2):
                        continue
                    if next:
                        break
                    next = True
                    if not node1.map.range == node2.map.range:
                        continue
                    if not len(node1.map.params) == len(node2.map.params):
                        continue
                    transform.first_map_exit = state.exit_node(node1)
                    transform.second_map_entry = node2
                    if transform.can_be_applied(state, 0, sdfg):
                        candidates.append((state, (node1, node2)))
        result = []
        for state,(entry1, entry2) in candidates:
            reject = False
            entry1_out = [edge.dst for edge in state.out_edges(state.exit_node(entry1))]
            entry2_in = [edge.src for edge in state.in_edges(entry2)]
            entry1_params = entry1.map.params
            entry2_params = entry2.map.params
            for elem in set([node.label for node in entry1_out]) & set([node.label for node in entry2_in]):
                nodes_1 = [node for node in entry1_out if node.label == elem]
                nodes_2 = [node for node in entry2_in if node.label == elem]
                idxs_1 = [edge.data for edge in state.in_edges(nodes_1[0])]
                idxs_2 = [edge.data for edge in state.out_edges(nodes_2[0])]
                for idx1 in idxs_1:
                    for idx2 in idxs_2:
                        if not idx1.subset == idx2.subset:
                            reject = True
                            break
                
                for node in nodes_1:
                    in_edge = state.in_edges(node)[0]
                    if in_edge.data.wcr:
                        dst_shape = in_edge.data.dst_subset.size()
                        if not dst_shape == entry1.map.range.size():
                            reject = True
                            break
        if not reject:
            result.append((state, entry1, entry2))
        print(result)
        return []
        
    def apply(self, sdfg, state, entry1, entry2):
        """
        This method applies the JoinMapScopes transformation to the given SDFG. It joins the two given map scopes.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            state (dace.SDFGState): The state the map scopes are in.
            entry1 (dace.MapEntry): The first map scope to join.
            entry2 (dace.MapEntry): The second map scope to join.
        """
        if not self.checked and not (state, entry1, entry2) in self.find(sdfg):
            return
        self.checked = False
        exit1 = state.exit_node(entry1)
        exit2 = state.exit_node(entry2)
        input_nodes = set()
        input_nodes = {edge.src for edge in state.in_edges(entry1)}
        input_labels = {node.label for node in input_nodes}
        output_nodes = {edge.dst for edge in state.out_edges(exit1)}
        inbetweeners = set()
        fuse_nodes = set()
        input_prop = set()
        # redirect nodes that are inbetween the first exit and second entry
        # exit1
        for in_edge in state.in_edges(exit1):
            for out_edge in state.out_edges(exit1):
                if not in_edge.data.data == out_edge.data.data:
                    continue
                if not in_edge.dst_conn[3:] == out_edge.src_conn[4:]:
                    continue
                # output nodes of first and second map are same node
                if out_edge.dst in {edge.src for edge in state.in_edges(entry2)}:
                    inbetweeners.add(out_edge.dst)  
                    state.add_memlet_path(in_edge.src, out_edge.dst, memlet=in_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                # output is used elsewhere
                else:
                    state.add_memlet_path(in_edge.src, exit2, out_edge.dst, memlet=in_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                state.remove_edge(in_edge)
                state.remove_edge(out_edge)
             
        # entry2
        for in_edge in state.in_edges(entry2):
            for out_edge in state.out_edges(entry2):
                if not in_edge.data.data == out_edge.data.data:
                    continue
                if in_edge.dst_conn is None or out_edge.src_conn is None:
                    continue
                if not in_edge.dst_conn[3:] == out_edge.src_conn[4:]:
                    continue
                """
                Case 1: Input Node of Entry1 is input node of Entry2
                 => Redirect the edge through Entry1
                Case 2: Same data input for Entry1 and Entry2, but two separate nodes
                 => Remove one input node and redirect the edge through Entry1
                Case 3: Output of Entry1 is input of Entry2
                 => Redirect the edge
                Case 4: Input Node of Entry2 is not input entry2 and is transient
                 4.1 if written to and read in entry1 scope (happens if you use split transformation)
                 => fuse the node and redirect the edge
                Case 5: Else: Input of Entry2
                 => Redirect the edge through Entry1
                """
                # input nodes of first and second map are same node
                if in_edge.src in input_nodes:
                    # moved this to further in the code since the where errors with leftover nodes
                    input_prop.add((in_edge.src, out_edge.dst, out_edge.data, out_edge.dst_conn, in_edge.src_conn))
                    state.remove_edge(in_edge)
                    state.remove_edge(out_edge)
                    #state.add_memlet_path(src, entry1, dst, memlet=mem, dst_conn=dst_conn)
                # input nodes of first and second map are different nodes, but same data potentialy
                elif in_edge.src.label in input_labels and not state.in_edges(in_edge.src):
                    target = [node for node in input_nodes if node.label == in_edge.src.label][0]
                    if not target:
                        state.add_memlet_path(in_edge.src, entry1, out_edge.dst, memlet=out_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                        state.remove_edge(in_edge)
                        state.remove_edge(out_edge)
                    else:
                        state.add_memlet_path(target, entry1, out_edge.dst, memlet=out_edge.data, dst_conn=out_edge.dst_conn, src_conn=in_edge.src_conn)
                        state.remove_edge(in_edge)
                        # remove second identical input node
                        state.remove_node(in_edge.src)
                        state.remove_edge(out_edge)
                # output of first map is input of second map
                elif state.in_edges(in_edge.src) and in_edge.src in inbetweeners:
                    state.add_memlet_path(in_edge.src, out_edge.dst, memlet=out_edge.data, dst_conn=out_edge.dst_conn, src_conn=in_edge.src_conn)
                    state.remove_edge(out_edge)
                    state.remove_edge(in_edge)
                # input of second map, but produced and used in first map too
                elif not state.in_edges(in_edge.src) and sdfg.arrays[in_edge.data.data].transient:
                    t = self.find_node(state, entry1, in_edge.src)
                    #if entry1.label == 'map_1_inv_idx' and entry2.label == 'map_1_max_idx':
                    #    print(f'edge: {in_edge.data.data}, src: {in_edge.src}, dst: {out_edge.dst}, in_Edge_src: {in_edge.src_conn}, out_edge_dst: {out_edge.dst_conn} ({t}))')
                    if t:
                        # happens especially when using split transformation, some nodes get created because redrection is not possible
                        fuse_nodes.add(in_edge.src)
                        state.add_memlet_path(in_edge.src, out_edge.dst, memlet=out_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                    else:
                        state.add_memlet_path(in_edge.src, entry1, out_edge.dst, memlet=in_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                    state.remove_edge(in_edge)
                    state.remove_edge(out_edge)
                # input of second map, but not used in first map
                else:
                    state.add_memlet_path(in_edge.src, entry1, out_edge.dst, memlet=out_edge.data, src_conn=in_edge.src_conn, dst_conn=out_edge.dst_conn)
                    state.remove_edge(in_edge)
                    state.remove_edge(out_edge)
        
        #fuse the nodes
        for node in fuse_nodes:
            target = self.find_node(state, entry1, node)
            if target:
                for edge in state.out_edges(node):
                    state.add_memlet_path(target, edge.dst, memlet=edge.data, dst_conn=edge.dst_conn)
                state.remove_node(node)
            
        # propagate input nodes
        for src, dst, mem, dst_conn, src_conn in input_prop:
            state.add_memlet_path(src, entry1, dst, memlet=mem, dst_conn=dst_conn, src_conn=src_conn)
        
        # replace entry2 with entry1
        entry2.in_connectors = entry1.in_connectors
        entry2.out_connectors = entry1.out_connectors
        for in_edge in state.in_edges(entry1):
            dace.transformation.helpers.redirect_edge(state, in_edge, new_dst = entry2, new_dst_conn = in_edge.dst_conn)
        for out_edge in state.out_edges(entry1):
            dace.transformation.helpers.redirect_edge(state, out_edge, new_src = entry2, new_src_conn = out_edge.src_conn)

        # remove entry1 and exit1 
        state.remove_node(entry1)
        state.remove_node(exit1)
    
    def find_node(self, state, cur_node, target, traversed=None):
        if traversed is None:
            traversed = set()
        traversed.add(cur_node)
        for edge in state.out_edges(cur_node):
            if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.label == target.label and not edge.dst == target:
                return edge.dst
            elif isinstance(edge.dst, dace.nodes.MapExit):
                continue
            elif not edge.dst in traversed:
                tmp = self.find_node(state, edge.dst, target, traversed)
                if tmp:
                    return tmp
                else:
                    continue
        return None
"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the removeArrayDims transformation.

"""

import dace

class removeArrayDims():
    """
    This class implements the removeArrayDims transformation. It searches for arrays in the SDFG and removes a dimension. The transformation is only applied if the array is extendable.
    This class is supposed to inverse the addArrayDims transformation.
    Similarly, there are two modes for this transformation:
        Mode 0: read to write access, read is target
        Mode 1: write to read access, write is target
    """

    def __init__(self):
        self.__name = 'removeArrayDims'
        self.checked = False
    
    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG and returns a list of tuples containing the array name and the mode of the transformation.
        Mode 0: read to write access, read is target
        Mode 1: write to read access, write is target

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of tuples, where each tuple contains the array name and the mode of the transformation
        """
        result = []
        for name, data in sdfg.arrays.items():
            if not data.transient:
                continue
            discovered_rtw, discovered_wtr = self.deleteable_dim_candidates(sdfg, name)
            if discovered_rtw is not None:
                map_dim = discovered_rtw.map.range.size()[0]
                result.append((name, map_dim, 1))
            if discovered_wtr is not None:
                map_dim = discovered_wtr.map.range.size()[0]
                result.append((name, map_dim, 0))
        self.checked = True
        return result

    def deleteable_dim_candidates(self, sdfg, array_name):
        write_tasklets = self.find_write_tasklet(sdfg, array_name)
        read_tasklets = self.find_read_tasklet(sdfg, array_name)

        discovered_rtw = None
        discovered_wtr = None
        
        for w_state, w_node in write_tasklets:
            if not w_state.label == 'state_0':
                continue
            map_scopes = self.get_entry_maps(w_state, w_node)
            subset = w_state.out_edges(w_node)[0].data.subset
            for write_loc, widx in reversed(list(enumerate(subset))):
                idx = str(widx[0])
                if len(idx) != 2:
                    continue
                candidate_write_scope = [entry for entry in map_scopes if entry.map.params[0] == idx][0]
                candidate_dim = candidate_write_scope.map.range.size()[0]
                candidate_rejected = False
                read_scope_indices = []
                for r_state, r_node in read_tasklets:
                    if array_name == r_state.out_edges(r_node)[0].data.data:
                        continue
                    candidate_read_scope = None
                    index_dim_map = {entry.map.params[0]: entry.map.range.size()[0] for entry in self.get_entry_maps(r_state, r_node)}
                    for edge in r_state.in_edges(r_node):
                        if not edge.data.data == array_name:
                            continue
                        r_subset = edge.data.subset
                        for read_loc, ridx in reversed(list(enumerate(r_subset))):
                            read_idx = str(ridx[0])
                            if not index_dim_map[read_idx] == candidate_dim:
                                continue
                            #if not idx in read_idx:
                            #    continue
                            if len(read_idx) != 2:
                                candidate_rejected = True
                                break
                            if read_loc != write_loc:
                                candidate_rejected = True
                                break
                            cur_read_scope = [entry for entry in self.get_entry_maps(r_state, r_node) if entry.map.params[0] == read_idx][0]
                            if candidate_read_scope is None:
                                candidate_read_scope = cur_read_scope
                                read_scope_indices.append(candidate_read_scope.map.params[0])
                            elif candidate_read_scope != cur_read_scope:
                                candidate_rejected = True
                                break
                            read_dim = cur_read_scope.map.range.size()[0]
                            if read_dim != candidate_dim:
                                candidate_rejected = True
                                break
                    if candidate_read_scope is None:
                        candidate_rejected = True
                if candidate_rejected:
                    continue
                
                if discovered_rtw is None:
                    rtw_rejected = False
                    for edge in w_state.in_edges(w_node):
                        if edge.data.data == array_name:
                            continue
                        edge_subset = [str(idx[0]) for idx in list(edge.data.subset)] if edge.data.subset is not None else []
                        if candidate_write_scope.map.params[0] in edge_subset:
                            rtw_rejected = True
                    if not rtw_rejected:
                        discovered_rtw = candidate_write_scope
                
                if discovered_wtr is None:
                    wtr_rejected = False
                    for (rstate, rop), read_map_idx in zip(read_tasklets, read_scope_indices):
                        r_map_scopes = self.get_entry_maps(rstate, rop)
                        read_scope_idx = [entry.map.params[0] for entry in r_map_scopes if entry.map.params[0] == read_map_idx][0]
                        out_subset = [str(idx[0]) for idx in rstate.out_edges(rop)[0].data.subset]
                        if read_scope_idx in out_subset:
                            wtr_rejected = True
                    
                    common_read_reduce = None
                    for rstate, rop in read_tasklets:
                        out_edge = rstate.out_edges(rop)[0]
                        if out_edge.data.wcr:
                            if common_read_reduce is None:
                                common_read_reduce = rop.label
                            elif common_read_reduce != rop.label:
                                wtr_rejected = True

                    if not (w_node.label == 'move' or w_node.label == 'copy_temp'):
                        if common_read_reduce != w_node.label:
                            wtr_rejected = True
                    if not wtr_rejected:
                        discovered_wtr = candidate_write_scope

        return discovered_rtw, discovered_wtr

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

    def apply(self, sdfg, name, dim, mode):
        """
        This method applies the removeArrayDims transformation to the given SDFG. It removes the given dimension from the array.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            name (str): The array to remove the dimension from.
            dim (int): The dimension to remove.
            mode (int): The mode of the transformation. 0: read to write access, read is target. 1: write to read access, write is target.
        """
        if not self.checked and not (name, dim, mode) in self.find(sdfg):
            return
        self.checked = False

        self.adjust_data_array(sdfg, name, dim)
        self.remove_read_dim(sdfg, name, dim)
        self.remove_write_dim(sdfg, name, dim)

        w_tasklet = self.find_write_tasklet(sdfg, name)
        if mode == 0:
            r_tasklet = self.find_read_tasklet(sdfg, name)
            reducation_type = r_tasklet[0][0].out_edges(r_tasklet[0][1])[0].data.wcr
            for state, node in w_tasklet:
                if state.label == 'state_0':
                    self.add_reduction(state, node, reducation_type)
            
            for state, node in r_tasklet:
                target_entry = None
                entries = self.get_entry_maps(state, node)
                for i, tmp_e in enumerate(entries):
                    if tmp_e.map.range.size()[0] == dim:
                        target_entry = tmp_e
                        break
                exit = state.exit_node(target_entry)
                self.fix_memlet_volume(state, node, name, target_entry, exit, dim)
                self.remove_map(state, node, target_entry, exit)
                self.delete_reduction(state, node, name)
        else:
            for state, node in w_tasklet:
                target_entry = None
                entries = self.get_entry_maps(state, node)
                for i, tmp_e in enumerate(entries):
                    if tmp_e.map.range.size()[0] == dim:
                        target_entry = tmp_e
                        break
                exit = state.exit_node(target_entry)
                self.fix_memlet_volume(state, node, name, target_entry, exit, dim)

                self.remove_map(state, node, target_entry, exit)
    
    def add_reduction(self, state, node, wcr, name):
        out_edges = state.out_edges(node)
        if isinstance(node, dace.nodes.AccessNode) and node.data == name:
            return
        for edge in out_edges:
            edge.data.wcr = wcr
            self.add_reduction(state, edge.dst, wcr)
    
    def delete_reduction(self, state, node, name):
        out_edges = state.out_edges(node)
        if isinstance(node, dace.nodes.AccessNode):
            return
        for edge in out_edges:
            edge.data.wcr = None
            self.delete_reduction(state, edge.dst, name)
    
    def remove_map(self, state, node, entry, exit):
        for out_edge in state.out_edges(entry):
                    for in_edge in state.in_edges(entry):
                        if in_edge.data.data == out_edge.data.data:
                            dace.transformation.helpers.redirect_edge(state, out_edge, new_src = in_edge.src)
                
        for in_edge in state.in_edges(exit):
            for out_edge in state.out_edges(exit):
                if in_edge.data.data == out_edge.data.data:
                    dace.transformation.helpers.redirect_edge(state, in_edge, new_dst = out_edge.dst)
        
        state.remove_node(exit)
        state.remove_node(entry)

    def fix_memlet_volume(self, state, node, name, entry, exit, dim):

        def traverse(node, del_map, backwards=False, traversed=None, passed=False):
            if traversed is None:
               traversed = set()
            next_edges = state.in_edges(node) if backwards else state.out_edges(node)
            for edge in next_edges:
                if edge.data.data == name and passed:
                    volume = edge.data.volume / dim
                    edge.data.volume = volume
                elif not edge.data.data == name:
                    continue
                next_node = edge.src if backwards else edge.dst
                if not next_node in traversed and not (isinstance(next_node, dace.nodes.AccessNode) and next_node.data == name):
                    traversed.add(next_node)
                    t = passed
                    if next_node == del_map:
                        t = True
                    traverse(next_node, del_map, backwards, traversed, t)
        traverse(node, entry, backwards=True, traversed=None, passed=False)
        traverse(node, exit, backwards=False, traversed=None, passed=False)
        

    def remove_read_dim(self, sdfg, target, dim):
        r_tasklet = self.find_read_tasklet(sdfg, target)
        for state, node in r_tasklet:
            entries = self.get_entry_maps(state, node)
            idx = [entry.map.params[0] for entry in entries if entry.map.range.size()[0] == dim][0]
            self.traverse_and_replace(state, node, target, dim, idx, backwards=True)
    
    def remove_write_dim(self, sdfg, target, dim):
        w_tasklet = self.find_write_tasklet(sdfg, target)
        for state, node in w_tasklet:
            entries = self.get_entry_maps(state, node)
            idx = [entry.map.params[0] for entry in entries if entry.map.range.size()[0] == dim][0]
            
            self.traverse_and_replace(state, node, target, dim, idx)
    
    def traverse_and_replace(self, state, node, name, dim, idx, backwards=False, traversed=None):
        if traversed is None:
            traversed = set()
        
        if isinstance(node, dace.nodes.AccessNode) and node.data == name:
            return

        next_edges = None
        if backwards:
            next_edges = state.in_edges(node)
        else:
            next_edges = state.out_edges(node)
        for edge in next_edges:
            if edge.data.data == name:
                subset = list(edge.data.subset.ranges)
                new_subset = []
                replaced = False
                for sub_idx in reversed(subset):
                    if (str(sub_idx[0]) == str(idx)) and not replaced:
                        replaced = True
                        continue
                    new_subset.append(sub_idx)
                if not replaced:
                    new_subset = []
                    for sub_idx in reversed(subset):
                        if sub_idx[1]+1 == dim and not replaced:
                            replaced = True
                            continue
                        new_subset.append(sub_idx)
                new_subset = dace.subsets.Range(reversed(new_subset))
                edge.data.subset = new_subset
            
            if not node in traversed:
                traversed.add(node)
                if backwards:
                    self.traverse_and_replace(state, edge.src, name, dim, idx, backwards, traversed)
                else:
                    self.traverse_and_replace(state, edge.dst, name, dim, idx, backwards, traversed)

    def adjust_data_array(self, sdfg, target, dim):
        data_array = sdfg.arrays[target]
        data_array.total_size //= dim
        shape = list(data_array.shape)
        new_shape = []
        removed = False
        pos = -1
        length = len(shape)
        for i,s_dim in reversed(list(enumerate(shape))):
            if s_dim == dim and not removed:
                removed = True
                pos = i
            else:
                new_shape.append(s_dim)   
        data_array.shape = tuple(new_shape)
        strides = list(data_array.strides)
        new_strides = []
        for j in range(len(strides)-1):
            if j == pos:
                continue
            new_strides.append(strides[j]//dim)
        data_array.strides = tuple(new_strides)
        offset = list(data_array.offset)
        new_offset = []
        for k in range(len(offset)-1):
            if k == pos:
                continue
            new_offset.append(offset[k])
        data_array.offset = tuple(new_offset)

    def find_write_tasklet(self, sdfg, target):
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if edge.data.data == target:
                            result.append((state, node))
        return result
    
    def find_read_tasklet(self, sdfg, target):  
        result = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if edge.data.data == target:
                            result.append((state, node))
                            break
        return result
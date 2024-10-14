"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the addArrayDims transformation.

"""
import dace

class addArrayDims():
    """
    This class implements the addArrayDims transformation. It searches for arrays in the SDFG and adds an additional scope of iteration to the write operation. The transformation is only applied if the array is extendable.
    
    Transformation example (read to write, write is target):
    Before:                    After:
    M   x[m]=z[m]              M N x[m,n]=z[m]
    M N y[m,n]=x[m]            M N y[m,n]=x[m,n]
    
    Transformation example (write to read with reduction propagation, read is target):
    Before:                    After:
    M N x[m]+=y[m,n]           M N x[m,n]=y[m,n]
    M   z[m]=x[m]              M N z[m]+=x[m,n]

    Read to write:
        - The write operation is the target of the transformation (new scope added for write scope)
        -> mode 0
    
    Write to read:
        - The read operations are the target of the transformation (new scope added for read scopes)
        -> mode 1

    Assumptions:
        - No reusing of arrays
        - No joined scopes, every scope contains exactly one scope or tasklet

    Requirements:
        - All reads should be enclosed in the scope equivalent to the referenced scope.
    """

    def __init__(self):
        self.__name = 'addArrayDims'
        self.checked = False
   
    @property 
    def name(self):
        return self.__name
    
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

    def only_one_child(self, state, entry):
        out_edges = state.out_edges(entry)
        outs = set()
        for edge in out_edges:
            outs.add(edge.dst)
        if not len(outs) == 1:
            return False

        out = list(outs)[0]
        if isinstance(out, dace.nodes.MapEntry):
            return True
        elif isinstance(out, dace.nodes.Tasklet):
            out_edge = state.out_edges(out)[0]
            if isinstance(out_edge.dst, dace.nodes.MapExit):
                return True
        else:
            return False   

    def single_map_scope(self,state, node):
        if not state.entry_node(node):
            return True
        
        for edge in state.in_edges(node):
            if isinstance(edge.src, dace.nodes.AccessNode):
                return False
        for out_edge in state.out_edges(node):
            if isinstance(out_edge.dst, dace.nodes.AccessNode):
                return False
        
        return True
                      

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG and returns a list of tuples containing the tasklet and the mode of the transformation.
        Mode 0: read to write access, write is target
        Mode 1: write to read access, read is target

        This method is applicable if the following conditions are met:
            - The array is transient
            - All reads should be enclosed in the scope equivalent to the referenced scope
            - The write and read tasklets are the only nodes in their scope
        Write to read:
            - The write operation is a reduction operation

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.

        Returns:
            list: A list of tuples, where each tuple contains a the tasklet and the mode of the transformation
        """

        result = []
        # --- read to write access, write is target
        read_dims = {}
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        data_name = edge.data.data
                        if not data_name in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[data_name]
                        if not data_node.transient:
                            continue
                        if not self.single_map_scope(state, node):
                            continue
                        tmp = set(edge.data.subset)
                        subset = set()
                        for subs in tmp:
                            for idx in subs:
                                subset.add(str(idx))

                        entry_maps = self.get_entry_maps(state, node)
                        map_params = {m.map.params[0] for m in entry_maps}
                        unused_params = map_params - subset
                        unused_dims = set()

                        for entry in entry_maps:
                           
                            for param in unused_params:
                                if param in entry.map.params:
                                    unused_dims.add(entry.map.range.size()[0])
                    
                        if data_name in read_dims:
                            read_dims[data_name] &= unused_dims
                        else:
                            read_dims[data_name] = unused_dims

        #print(f'read_dims {read_dims}')
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not read_dims.get(edge.data.data, set()):
                            continue
                        result.append((node, 0))

        # --- write to read access, read is target
        write_dims = {}
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not edge.data.wcr:
                            continue
                        if not self.single_map_scope(state, node):
                            continue
                        tmp = set(edge.data.subset)
                        subset = set()
                        for subs in tmp:
                            for idx in subs:
                                subset.add(str(idx))
                        entry_maps = self.get_entry_maps(state, node)
                        map_params = {m.map.params[0] for m in entry_maps}
                        unused_params = map_params - subset
                        unused_dims = set()
                        for entry in entry_maps:
                            for param in unused_params:
                                if param in entry.map.params:
                                    unused_dims.add(entry.map.range.size()[0])

                        write_dims[edge.data.data] = (unused_dims, node)

        #print(f'write_dims {write_dims}')
        candidates = set()
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if not edge.data.data in sdfg.arrays:
                            continue
                        data_node = sdfg.arrays[edge.data.data]
                        if not data_node.transient:
                            continue
                        if not edge.data.data in write_dims:
                            continue
                        unused_dims, write_node = write_dims[edge.data.data]
                        if not "reduce" in node.label or ("reduce" in node.label and not node.label == write_node.label) or not node.label in ["move", "copy_temp"]:
                            continue
                        candidates.add(edge.data.data)

        #print(f'candidates {candidates}')
        for state in sdfg.nodes():
            if not state.label == 'state_0':
                continue
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    in_edges = state.in_edges(node)
                    for edge in in_edges:
                        if edge.data.data in candidates:
                            result.append((node, 1))
                            candidates.remove(edge.data.data)

        self.checked = True
        return result

    def apply(self, sdfg, op, mode):
        """
        This method applies the addArrayDims transformation to the given SDFG. It adds an additional scope of iteration to the write operation.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            op (dace.nodes.Tasklet): The write tasklet to apply the transformation to.
            mode (int): The mode of the transformation. 0: read to write access, write is target. 1: write to read access, read is target.
        """
        if not self.checked and not (op, mode) in self.find(sdfg):
            return
        self.checked = False

        # mode 0: read to write access
        if mode == 0:
            target = state_0(sdfg).out_edges(op)[0].data.data

            read_tasklet = self.find_read_tasklet(sdfg, target)
            write_tasklet = self.find_write_tasklet(sdfg, target)
           
            common_unused = None
            for i_state, node in read_tasklet:
                if not i_state.out_edges(node)[0].data.data == target:
                    continue
                in_edge = None
                for edge in i_state.in_edges(node):
                    if edge.data.data == target:
                        in_edge = edge
                        break
                map_scopes = self.get_entry_maps(i_state, node)
                edge_subset = [str(idx[0]) for idx in in_edge.data.subset]
                map_params = sorted({m.map.params[0] for m in map_scopes} - set(edge_subset))

                unused_dims = []
                for entry in map_scopes:
                    for param in map_params:
                        if param in entry.map.params:
                            unused_dims.append(entry.map.range.size()[0])

                if common_unused is None:
                    common_unused = unused_dims
                else:
                    tmp = common_unused & unused_dims
                    common_unused = [dim for dim in common_unused if dim in tmp]
            
            unused_dim = common_unused[0]

            self.adjust_data_array(sdfg, target, unused_dim)
      
            for i_state, op_node in write_tasklet:
                self.add_new_map(i_state, op_node, unused_dim)

            for i,(i_state, node) in enumerate(write_tasklet):
                self.replace_write_memlet(i_state, node, target, unused_dim)
                
            for i_state, node in read_tasklet:
                self.replace_read_memlet(i_state, node, target, unused_dim)
                
        # mode 1: write to read access
        elif mode == 1:
            target = state_0(sdfg).out_edges(op)[0].data.data
            wcr = state_0(sdfg).out_edges(op)[0].data.wcr
            unused_dims = set()
            write_tasklet = self.find_write_tasklet(sdfg, target)
            unused_dims = self.find_unused_dims(sdfg, write_tasklet, write=True)
            unused_dim = unused_dims[0]

            self.adjust_data_array(sdfg, target, unused_dim)

            for i_state, tasklet_node in write_tasklet:
                self.replace_write_memlet(i_state, tasklet_node, target, unused_dim)
            
            for i_state, tasklet_node in read_tasklet:
                if not i_state.out_edges(tasklet_node)[0].data.data == target:
                    self.add_new_map(i_state, tasklet_node, unused_dim)
            
            for i_state, tasklet_node in read_tasklet:
                self.replace_read_memlet(i_state, tasklet_node, target, unused_dim)
                self.add_reduction(i_state, tasklet_node, target, wcr)
            
            for i_state, tasklet_node in write_tasklet:
                self.delete_reduction(i_state, tasklet_node)

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
        return result
    
    def find_unused_dims(self, sdfg, tasklets, write=True):
        result = []
        for state, node in tasklets:
            edges = []
            if write:
                edges = state.out_edges(node)
            else:
                edges = state.in_edges(node)
            for edge in edges:
                entry_maps = self.get_entry_maps(state, node)
                unacc_indices = {entry.map.params[0] for entry in entry_maps}
                unacc_indices -= set(edge.data.subset)
                for entry in entry_maps:
                    if entry.map.params[0] in unacc_indices:
                        result.append(entry.map.range.size()[0])
        return result

    def adjust_data_array(self, sdfg, target, unused_dim):
        # shape adjustment
        shape = list(sdfg.arrays[target].shape)
        shape.append(unused_dim)
        sdfg.arrays[target].shape = tuple(shape)
        # total size adjustment
        sdfg.arrays[target].total_size *= unused_dim
        offset = list(sdfg.arrays[target].offset)
        offset.append(0)
        # offset adjustment
        sdfg.arrays[target].offset = tuple(offset)
        # strides adjustment
        strides = sdfg.arrays[target].strides
        new_strides = []
        for stride in strides:
            new_strides.append(stride*unused_dim)
        new_strides.append(1)
        sdfg.arrays[target].strides = tuple(new_strides)

    def add_new_map(self, state, op_node, unused_dim):
        entries = self.get_entry_maps(state, op_node)
        new_idx = f'i{len(entries)}'
        #idxs.append(new_idx)
        new_entry, new_exit = state.add_map(f'map_{len(entries)}_{op_node.label}', {new_idx: f'0:{unused_dim}'})
        
        # incorporate new map
        for edge in state.in_edges(op_node):
            #memlet = copy.deepcopy(edge.data)
            memlet = None
            if edge.data.data:
                memlet = edge.data.__deepcopy__({})
                state.add_memlet_path(edge.src, new_entry, op_node, memlet=memlet, dst_conn=edge.dst_conn, src_conn=edge.src_conn)
            else:
                state.add_nedge(edge.src, new_entry, data=dace.Memlet())
                state.add_nedge(new_entry, op_node, data=dace.Memlet())
            state.remove_edge(edge)
        
        for edge in state.out_edges(op_node):
            memlet = edge.data.__deepcopy__({})
            state.add_memlet_path(op_node, new_exit, edge.dst, memlet=memlet, src_conn=edge.src_conn, dst_conn=edge.dst_conn)
            state.remove_edge(edge)

    def replace_write_memlet(self, state, node, target, unused_dim):
        entries = self.get_entry_maps(state, node)
        idx = None
        for entry in entries:
            if str(unused_dim) == str(entry.map.range.size()[0]):
                idx = entry.map.params[0]
                break

        write_edges = []
        self.get_write_memlet(state, node, target, write_edges)
        subset = list(write_edges[0].data.subset.ranges)
        subset.append((idx, idx, 1))
        new_subset = dace.subsets.Range(subset)

        path_nodes = [edge.src for edge in write_edges]
        path_nodes.append(write_edges[-1].dst)
        out_conn = write_edges[0].src_conn
        wcr = write_edges[0].data.wcr
        for edge in write_edges:
            src_conn = edge.src_conn
            if src_conn:
                edge.src.remove_out_connector(src_conn)
            dst_conn = edge.dst_conn
            if dst_conn:
                edge.dst.remove_in_connector(dst_conn)
            state.remove_edge(edge)
        path_nodes[0].add_out_connector(out_conn)
        state.add_memlet_path(*path_nodes, memlet=dace.Memlet(data=target, subset= new_subset, wcr=wcr), src_conn=out_conn)

    def get_write_memlet(self, state, node, dst, result, traverse=None):
        if traverse is None:
            traverse = set()
        for edge in state.out_edges(node):
            if edge.data.data == dst:
                #subset = list(edge.data.subset.ranges)
                #subset.append((new_idx, new_idx, 1))
                #edge.data.subset = dace.subsets.Range(subset)
                result.append(edge)
            if not edge.dst in traverse:
                if not (isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == dst):
                    traverse.add(edge.dst)
                    self.get_write_memlet(state, edge.dst, dst, result, traverse)
    
    def delete_reduction(self, state, node, visited=None):
        if visited is None:
            visited = set()
        out_edges = state.out_edges(node)
        for edge in out_edges:
            edge.data.wcr = None
            if not edge.dst in visited and not isinstance(edge.dst, dace.nodes.AccessNode):
                visited.add(edge.dst)
                self.delete_reduction(state, edge.dst, visited)
    
    def add_reduction(self, state, node, wcr, visited=None):
        if visited is None:
            visited = set()
        out_edges = state.out_edges(node)
        for edge in out_edges:
           edge.data.wcr = wcr
           if not edge.dst in visited and not isinstance(edge.dst, dace.nodes.AccessNode):
                visited.add(edge.dst)
                self.add_reduction(state, edge.dst, wcr, visited)

    def replace_read_memlet(self, state, node, target, unused_dim):
        entries = self.get_entry_maps(state, node)
        idx = None
        for entry in entries:
            if str(unused_dim) == str(entry.map.range.size()[0]):
                idx = entry.map.params[0]
                break
        
        read_edges = []
        self.get_read_memlet(state, node, target, read_edges)
        subset = list(read_edges[0].data.subset.ranges)
        subset.append((idx, idx, 1))
        #subset.append(idx)
        new_subset = dace.subsets.Range(subset)
        
        path_nodes = [edge.src for edge in reversed(read_edges)]
        path_nodes.append(read_edges[0].dst)
        out_conn = read_edges[0].dst_conn
        for edge in read_edges:
            src_conn = edge.src_conn
            if src_conn:
                edge.src.remove_out_connector(src_conn)
            dst_conn = edge.dst_conn
            if dst_conn:
                edge.dst.remove_in_connector(dst_conn)
            state.remove_edge(edge)

        path_nodes[-1].add_in_connector(out_conn)
        state.add_memlet_path(*path_nodes, memlet=dace.Memlet(data=target, subset= new_subset), dst_conn=out_conn)

    def get_read_memlet(self, state, node, target, result, traverse=None):
        if traverse is None:
            traverse = set()
        for edge in state.in_edges(node):
            if edge.data.data == target:
                result.append(edge)
            if not edge.src in traverse:
                traverse.add(edge.src)
                if not (isinstance(edge.src, dace.nodes.AccessNode) and edge.src.data == target):
                    self.get_read_memlet(state, edge.src, target, result, traverse)

def state_0(sdfg):
    for state in sdfg.nodes():
        if state.label == 'state_0':
            return state

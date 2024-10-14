"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the reuseArrayLocations transformation.

"""

import dace

class reuseArrayLocations():
    """
    This class implements the reuseArrayLocations transformation. It avoids materializing dimensions.

    Attributes:
        race_cond_check (bool): If True, all the map involved in the scope of the array to be reused are set to sequential schedule to avoid race condtions.
        keep_sizes (bool): If True, the sizes of the array are kept. If False, the sizes are adjusted to the new array size.
    """

    def __init__(self):
        self.__name = 'reuseArrayLocations'
        self.checked = False
        # reusing a dimension of an array can lead to incorrect results when using OMP for parallelization (race conditions)
        self.race_cond_check = True
        self.keep_sizes = True

    @property
    def name(self):
        return self.__name

    def get_read_ops(self, state, target):
        ops = set()
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                in_edges = state.in_edges(node)
                for edge in in_edges:
                    if edge.data.data == target.data:
                        ops.add(node)
                        continue
        return ops
    
    def get_write_ops(self, state, target):
        ops = set()
        for node in state.nodes():
            if isinstance(node, dace.nodes.Tasklet):
                out_edges = state.out_edges(node)
                for edge in out_edges:
                    if edge.data.data == target.data:
                        ops.add(node)
        return ops
    
    def get_entry_maps(self, state, node):
        dims = []
        entry = state.entry_node(node)

        while True:
            dims.append(entry)
            entry = state.entry_node(entry)
            if not entry:
                break

        return dims
    
    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that can be reused. It returns a list of tuples containing the array names and the dimensions that can be reused.
        
        This transformation is applicable to the following cases:
            - all accesses to the dimension of the array for reuse refer to the same scope s and only to this scope
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of tuples containing the array names and the dimensions that can be reused.
        """
        reusable_dims = set()
        for state in sdfg.nodes():
            #source, sink = state.source_nodes(), state.sink_nodes()
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    data_node = sdfg.arrays[node.data]
                    if not data_node.transient:
                        continue
                    if data_node.shape is [1]:
                        continue
                    if not state.entry_node(node):
                        continue
                    
                    # get all read and write tasklets
                    node_entry = state.entry_node(node)
                    read_ops = self.get_read_ops(state, node)
                    write_ops = self.get_write_ops(state, node)
                    
                    # applic. when all accesses to the dimension of the array targeted for reuse refer to the same scope s and only to this scope
                    reusable = True
                    for tasklet in read_ops:
                        if not state.entry_node(tasklet) == node_entry:
                            reusable = False
                            break
                    if not reusable:
                        continue
                    for tasklet in write_ops:
                        if not state.entry_node(tasklet) == node_entry:
                            reusable = False
                            break
                    if not reusable:
                        continue
                    
                    # get the dimension that is not used anymore
                    shape = data_node.shape
                    tmp = set()
                    shape_and_stride = {shape[i]: data_node.strides[i] for i in range(len(shape))}
                    for op in write_ops:
                        maps = self.get_entry_maps(state, op)
                        tmp = {map_entry for map_entry in maps if map_entry.range.size()[0] in shape and not shape_and_stride[map_entry.range.size()[0]] == 0}
                    for op in read_ops:
                        entry_dims = self.get_entry_maps(state, op)
                        if set(tmp).issubset(set(entry_dims)):
                            for maps in tmp:
                                reusable_dims.add((node, maps.range.size()[0]))
                        else:
                            diff = set(tmp) - set(entry_dims)
                            for maps in diff:
                                reusable_dims.add((node, maps.range.size()[0]))
        self.checked = True
        return sorted(list(reusable_dims), key=lambda x: x[0].data)
                                    
                    
    def apply(self, sdfg, node, dim):
        """
        This method applies the reuseArrayLocations transformation to the given SDFG. It avoids materializing the given dimension of the array by setting the stride to 0.
        If keep_sizes is set to True, the sizes of the array are kept. If False, the sizes are adjusted to the new array size.
        If race_cond_check is set to True, all the map involved in the scope of the array to be reused are set to sequential schedule

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            node (dace.nodes.AccessNode): The array to apply the transformation to.
            dim (Str): The dimension to remove.
        """

        if not self.checked and not (node,dim) in self.find(sdfg):
            return
        self.checked = False

        data_node = sdfg.arrays[node.data]
        shape = data_node.shape
        strides = list(data_node.strides)
        dim_pos = -1
        for i,dims in reversed(list(enumerate(shape))):
            if dims == dim:
                dim_pos = i
                if self.keep_sizes:
                    strides[i] = 0
                    break
        data_node.strides = tuple(strides)

        # applying this optimization and using OMP for parallel can lead to race condtions, this always sees that it keeps first shape
        if self.race_cond_check:
            entry_maps = self.get_entry_maps(find_state(sdfg, node), node)
            for entry in entry_maps:
                entry.schedule = dace.ScheduleType.Sequential
        
        state = find_state(sdfg, node)
        # remove the dimension that is not used anymore (lose of information of total size before optimization)
        if not self.keep_sizes:
            new_shape = []
            new_offset = []
            offset = list(data_node.offset)
            strides = list(data_node.strides)
            new_strides = []
            for i,dims in reversed(list(enumerate(shape))):
                if dims == dim:
                    continue
                new_shape.append(dims)
                new_offset.append(offset[i])
                new_strides.append(strides[i])
            if not new_shape:
                new_shape.append(1)
                new_offset.append(0)
                new_strides.append(1)
            data_node.shape = tuple(new_shape)
            data_node.offset = tuple(new_offset)
            data_node.strides = tuple(new_strides)
            data_node.total_size = data_node.total_size // dim
            #print(f'{data_node} and {data_node.shape} and {data_node.total_size} and {data_node.strides} ({dim})')
            if not dim_pos == -1:
                self.adjust_memlets(sdfg, state, node, dim_pos)
            

    def adjust_memlets(self, sdfg, state, node, dim_pos):
        #source_nodes = state.source_nodes()
        #visited = set()
        #self.traverse_subsets(state, state.entry_node(node), node.data, dim_pos, visited, scope_only=True)
        for other_states in sdfg.nodes():
            visited = set()
            sources = other_states.source_nodes()
            for source in sources:
                self.traverse_subsets(other_states, source, node.data, dim_pos, visited, scope_only=False)

    def traverse_subsets(self, state, node, data, dim_pos, visited = None, scope_only = False):
        if visited is None:
            visited = set()
        visited.add(node)
        if scope_only and isinstance(node, dace.nodes.MapExit):
            return
        out_edges = state.out_edges(node)
        for edge in out_edges:
            if edge.data.data == data:
                #print(f'{data} {dim_pos} {edge.data.subset} and {type(edge.data.subset)}')
                self.update_subset(state, edge, dim_pos)
            if not edge.dst in visited:
                self.traverse_subsets(state, edge.dst, data, dim_pos, visited, scope_only)

    def update_subset(self, state, edge, dim_pos):
        subset =  list(edge.data.subset.ranges)
        new_subset = []
        for i,sub in enumerate(subset):
            if i == dim_pos:
                continue
            new_subset.append(sub)
        if not new_subset:
            new_subset.append((0,0,1))
        edge.data.subset = dace.subsets.Range(new_subset)
        #print(f'new subset {edge.data.subset} and {type(edge.data.subset)}\n')

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
        
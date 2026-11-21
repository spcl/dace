"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the reuseArray transformation.

"""

import dace

class reuseArray():
    """
    This class implements the reuseArray transformation. It searches for arrays in the SDFG that have the same shape and data type and are transient. The transformation is only applied if the read access is before the write access.

    Attributes:
        remove_reused (bool): If True, the array that is reused is removed from the SDFG. Doesn't preserve the original information about the sdfg.
    """
    def __init__(self):
        self.__name = 'reuseArray'
        self.checked = False
        self.remove_reused = True
    
    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that can be reused. It returns a list of tuples containing the array names that can be reused.

        This transformation is applicable to the following cases:
            - Two arrays have the same shape and data type
            - The arrays have no overlapping accesses / live ranges
            - The array to be replaced has to be a transient array, i.e. now input or output to the kernel

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.

        Returns:
            list: A list of tuples containing two array names that can be reused.
        """
        same_data = set()
        data = sdfg.arrays
        all_nodes = set()
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    all_nodes.add(node.data)
        
        state = None
        for tmpstate in sdfg.nodes():
            if tmpstate.label == 'state_0':
                state = tmpstate
        inputs = state.source_nodes()
        input_names = [node.data for node in inputs if isinstance(node, dace.nodes.AccessNode)]

        for arr_name1, data1 in data.items():
            for arr_name2, data2 in data.items():
                if arr_name1 == arr_name2:
                    continue
                if not type(data1) == type(data2):
                    continue
                if not data1.transient: #or not data2.transient: # later check is not necessaryily needed, but safer
                    continue
                # if second array has a reduction (i.e. the output node is given also as input to tasklet), could overwritte initialisation value (0, min_val, max_val)
                if (arr_name2 in input_names and data2.transient):
                    continue
        
                if isinstance(data1, dace.data.Array):
                    if not arr_name1 in all_nodes or not arr_name2 in all_nodes:
                        continue
                    if data1.shape == data2.shape and data1.dtype == data2.dtype and data1.strides == data2.strides:
                        same_data.add((arr_name1, arr_name2)) #if arr_name1 < arr_name2 else same_data.add((arr_name2, arr_name1))
                if isinstance(data1, dace.data.Scalar):
                    if not arr_name1 in all_nodes or not arr_name2 in all_nodes:
                        continue
                    if data1.dtype == data2.dtype:
                        same_data.add((arr_name1, arr_name2)) #if arr_name1 < arr_name2 else same_data.add((arr_name2, arr_name1))
            
        reusable_arrays = []

        for arr1, arr2 in same_data:
            last_read1 = self.get_last_read(state, arr1)
            #last_read2 = self.get_last_read(state, arr2)
            #first_write1 = self.get_first_write(state, arr1)
            first_write2 = self.get_first_write(state, arr2)
            is_reusable = True
            
            # last read arr1 <= first write arr2
            if last_read1 is None or first_write2 is None:
                is_reusable = False
            else:
                r1w2 = self.trav_check(state, last_read1, [first_write2])
                if r1w2:
                    is_reusable = False
            if is_reusable:
                reusable_arrays.append((arr1, arr2))

        #TODO: check if arr1 used as input and arr2 as output in same map scope, that their index are matching => no index mismatch
        self.checked = True
        return sorted((reusable_arrays))
    
    def get_last_read(self, state, arr):
        sink = state.sink_nodes()
        reads = set()
        visited_reads = set()
        for node in sink:
            res = self.traverse(node, state, arr, visited=visited_reads,reverse=True)
            if res:
                reads = reads | res
        if len(reads) == 0:
            return None
        elif len(reads) == 1:
            return list(reads)[0]
        else:
            new_reads = []
            for read in reads:
                b = self.trav_check(state, read, reads-{read})
                if b:
                    new_reads.append(read)
            return new_reads[0] if len(new_reads) == 1 else None

    def get_first_write(self, state, arr):
        source = state.source_nodes()
        visited = set()
        writes = set()
        for node in source:
            res = self.traverse(node, state, arr, visited)
            if res:
                writes = writes | res
        if len(writes) == 0:
            return None
        elif len(writes) == 1:
            return list(writes)[0]
        else:
            new_writes = []
            for write in writes:
                b = self.trav_check(state, write, writes-{write}, reverse=True)
                if b:
                    new_writes.append(write)
        
            return new_writes[0] if len(new_writes) == 1 else None
    
    def read_tasklet(self, state, node, arr_name, res=None):
        if res is None:
            res = set()
        out_edges = state.out_edges(node)
        for edge in out_edges:
            if not edge.data.data == arr_name:
                continue
            if isinstance(edge.dst, (dace.nodes.AccessNode, dace.nodes.MapExit, dace.nodes.MapEntry)):
                self.read_tasklet(state, edge.dst, arr_name, res)
            elif isinstance(edge.dst, dace.nodes.Tasklet):
                res.add(edge.dst)

    def write_tasklet(self, state, node, arr_name, res=None):
        if res is None:
            res = set()
        in_edges = state.in_edges(node)
        for edge in in_edges:
            if not edge.data.data == arr_name:
                continue
            if isinstance(edge.src, (dace.nodes.AccessNode, dace.nodes.MapEntry, dace.nodes.MapExit)):
                self.write_tasklet(state, edge.src, arr_name, res)
            elif isinstance(edge.src, dace.nodes.Tasklet):
                res.add(edge.src)

    def trav_check(self, state, cur_node, check_set, visited = None, reverse = False):
        if not visited:
            visited = set()
        if cur_node in check_set:
            return False
        visited.add(cur_node)
        next_edge = state.out_edges(cur_node) if not reverse else state.in_edges(cur_node)
        for edge in next_edge:
            next_node = edge.dst if not reverse else edge.src
            if not next_node in visited:
                tmp = self.trav_check(state, next_node, check_set, visited, reverse=reverse)
                if not tmp:
                    return False
    
        return True
        
    def traverse(self, cur_node, state, search_item, visited = None, reverse = False):
        if not visited:
            visited = set()
        if isinstance(cur_node, dace.nodes.AccessNode):
            if cur_node.data == search_item:
                res = set()
                if reverse:
                    self.read_tasklet(state, cur_node, search_item, res)
                else:
                    self.write_tasklet(state, cur_node, search_item, res)
                return res
        
        visited.add(cur_node)
        next_edges = state.in_edges(cur_node) if reverse else state.out_edges(cur_node)
        tmp_ret = set()
        for edge in next_edges:
            next_node = edge.src if reverse else edge.dst
            if not next_node in visited:
                ret = self.traverse(next_node, state, search_item, visited, reverse=reverse)
                if ret:
                    tmp_ret = tmp_ret | ret
        if tmp_ret:
            return tmp_ret
        return None
    
    def traverser_check(self, state, arr1, arr2, node, write, visited = None):

        if visited is None:
            visited = set()
        visited.add(node)

        # last read arr1 < first write arr2
        if isinstance(node, dace.nodes.Tasklet):
            in_edges = state.in_edges(node)
            out_edges = state.out_edges(node)
            for edge in in_edges:
                if edge.data.data == arr1:
                    if write:
                        return False
            for edge in out_edges:
                if edge.data.data == arr2:
                    write = True
        
        out_edges = state.out_edges(node)
        for edge in out_edges:
            if not edge.dst in visited:
                ret = self.traverser_check(state, arr1, arr2, edge.dst, write, visited)
                if not ret:
                    return False
                
        return True
    
    def apply(self, sdfg, array1, array2):
        """
        This method applies the reuseArray transformation to the given SDFG. It replaces all accesses to the first array with accesses to the second array.
        If the remove_reused attribute is set to True, the first array is removed from the SDFG.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            array1 (str): The array to replace.
            array2 (str): The array to replace with.
        """
        if not self.checked and not (array1, array2) in self.find(sdfg):
            return
        self.checked = False

        for state in sdfg.nodes():
            state.replace(array1, array2)

        data1 = sdfg.arrays[array1]
        if not data1.transient:
            data2 = sdfg.arrays[array2]
            data2.transient = False

        if self.remove_reused:
            sdfg.remove_data(array1)
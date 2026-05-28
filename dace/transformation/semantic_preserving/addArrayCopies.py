"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the addArrayCopies transformation.
"""

import dace

class addArrayCopies():
    """
    This class implements the addArrayCopies transformation. It searches for arrays in the SDFG and adds copies of them. The transformation is only applied if the array is copyable.
    """

    def __init__(self):
        self.__name = 'addArrayCopies'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG and returns a list of copyable arrays.

        This method is applicable to all arrays that are used (as Nodes) in the SDFG.

        Args:
            sdfg (dace.SDFG): The SDFG to search for copyable arrays.
        
        Returns:
            list: A list of copyable arrays.
        """
        copyable_arrays = set()
        for state in sdfg.nodes():
            if not state.label == 'state_0':
               continue
            for node in state.nodes():
                # TODO: Not nodes that are also used as loop bounds, e.g. N
                if isinstance(node, dace.nodes.AccessNode):
                    # copy source nodes only once, otherwise gets messy with states
                    if node in state.source_nodes():
                        if 'copy' in node.data:
                            continue
                    copyable_arrays.add(node.data)
        self.checked = True
        return sorted(list(copyable_arrays))
    
    def apply(self, sdfg, arr_name):
        """
        This method applies the addArrayCopies transformation to the given SDFG. It adds a copy of the given array.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            arr_name (str): The array to add a copy of.
        """
        if not self.checked and not arr_name in self.find(sdfg):
            return
        self.checked = False

        state = state_0(sdfg)
        data = sdfg.arrays[arr_name]
        input_nodes = state.source_nodes()
        adj_state = state
        
        # check if array is input of main state, if so, create a new state to copy it
        if arr_name in [node.data for node in input_nodes]:
            s = len(sdfg.nodes())
            new_state = sdfg.add_state(f'state_{s}')
            sdfg.add_edge(new_state, sdfg.nodes()[s-1], dace.InterstateEdge())
            adj_state = new_state
        
        # get a new name for the array
        new_name = self.new_name(sdfg, arr_name)

        # add the array copy
        if data.shape:
            sdfg.add_array(name=new_name, shape=data.shape, dtype=data.dtype, storage=data.storage, transient=True, lifetime=data.lifetime)
        else:
            sdfg.add_scalar(name=new_name, dtype=data.dtype, transient=True, lifetime=data.lifetime)
        # the move instruction
        tasklet = adj_state.add_tasklet("move", {"in_1"}, {"out"}, "out = in_1")
       
       # binding all the elements together
        output_node = adj_state.add_access(new_name)
        if arr_name in [node.data for node in input_nodes]:
            # input nodes
            new_input_node = new_state.add_read(arr_name)
            loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
            adj_state.add_memlet_path(new_input_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(new_input_node, idxs))
            adj_state.add_memlet_path(tasklet, *loop_exit, output_node, src_conn='out', memlet=dace.Memlet.simple(output_node, idxs))
            state.replace(arr_name, new_name)
        else:
            input_node = None
            for edge in state.edges():
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    input_node = edge.dst
                    break 
            out_edges = state.out_edges(input_node)
            entry = state.entry_node(input_node)
            if not entry and out_edges:
                # map exit nodes
                loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
                state.add_memlet_path(input_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(input_node, idxs))
                state.add_memlet_path(tasklet, *loop_exit, output_node, src_conn='out', memlet=dace.Memlet.simple(output_node, idxs))
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name)
            elif not entry:
                #output node => to keep naming consistent, we switch temporary and original array
                in_edges = state.in_edges(input_node)
                loop_entry, loop_exit, idxs = self.add_maps(adj_state, data.shape)
                state.add_memlet_path(output_node, *loop_entry, tasklet, dst_conn='in_1', memlet=dace.Memlet.simple(output_node, idxs))
                state.add_memlet_path(tasklet, *loop_exit, input_node, src_conn='out', memlet=dace.Memlet.simple(input_node, idxs))
                for edge in in_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_dst = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name, backwards=True)
            else:
                # nodes that are only used inside a scope
                entries = [entry]
                while True:
                    tmp = entries[-1]
                    e = state.entry_node(tmp)
                    if not e:
                        break
                    entries.append(e)
                dims = list(reversed([entry.map.range.size()[0] for entry in entries]))
                idxs = ''
                for shape in data.shape:
                    if not shape in dims:
                        #not implemented: if map operates not over all of the shape dimension
                        assert False
                    else:
                        idx = dims.index(shape)
                        idxs += f'i{idx},'
                idxs = idxs[:-1]
                state.add_edge(input_node, None, tasklet, 'in_1', dace.Memlet.simple(input_node, out_edges[0].data.subset))
                state.add_edge(tasklet, 'out', output_node, None, dace.Memlet.simple(output_node, idxs))
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = output_node)
                self.traverse_n_replace(state, output_node, arr_name, new_name)

    def new_name(self, sdfg, arr_name):   
        new_name = ''
        i = 0
        while True:
            new_name = f'{arr_name}_copy_{i}'
            if not new_name in sdfg.arrays:
                break
            i += 1
        return new_name

    def add_maps(self, state, shape):
        loop_entry, loop_exit = [], []
        idxs = ''
        for i,s in enumerate(shape):
            dim = str(s)
            index = f'i{i}'
            idxs += f'{index},'
            entry, exit = state.add_map(f'copy_map_{i}', {index: f'0:{dim}'})
            loop_entry.append(entry)
            loop_exit.append(exit)
        idxs = idxs[:-1]
        return loop_entry, loop_exit, idxs

    def traverse_n_replace(self, state, node, old_name, new_name, start=True, backwards=False):
        if not start:
            if not isinstance(node, dace.nodes.MapEntry) and not isinstance(node, dace.nodes.MapExit):
                return
      
        next_edges = None
        if backwards:
            next_edges = state.in_edges(node)
        else:
            next_edges = state.out_edges(node)

        for edge in next_edges:
            if edge.data.data == old_name:
                edge.data.data = new_name
            if backwards:
                self.traverse_n_replace(state, edge.src, old_name, new_name, False, True)
            else:
                self.traverse_n_replace(state, edge.dst, old_name, new_name, False)

def state_0(sdfg):
    for state in sdfg.nodes():
        if state.label == 'state_0':
            return state
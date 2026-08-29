"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the removeArrayCopies transformation.

"""

import dace

class removeArrayCopies():
    """
    This class implements the removeArrayCopies transformation. It searches for array copies in the SDFG and removes them. The transformation is only applied if the array copy is removable.
    This class reverses the addArrayCopies transformation.
    """

    def __init__(self):
        self.__name = 'removeArrayCopies'
        self.checked = False

    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for array copies in the SDFG and returns a list of removable array copies.
        
        This method is applicable to arrays/scalars in the sdfg that
            - are not output nodes of a state
            - are coppied/moved to another array/scalar in the same state

        Args:
            sdfg (dace.SDFG): The SDFG to search for removable array copies.

        Returns:
            list: A list of removable array copies.
        """
        deleteable_arrays = set()
        for state in sdfg.nodes():
            out = [node.data for node in state.sink_nodes()]
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    if node.label.startswith('copy_temp') or node.label.startswith('move'):
                        out_node = state.out_edges(node)[0].data
                        if out_node.data in out:
                            continue
                        in_node = state.in_edges(node)[0].data
                        in_data = sdfg.arrays[in_node.data]
                        out_data = sdfg.arrays[out_node.data]
                        if in_node.subset == out_node.subset and in_data.shape == out_data.shape and in_data.dtype == out_data.dtype:
                            deleteable_arrays.add(out_node.data)
        self.checked = True
        return sorted(list(deleteable_arrays))
    
    def apply(self, sdfg, arr_name):
        """
        This method applies the removeArrayCopies transformation to the given SDFG. It removes the given array copy.
        Reverses the addArrayCopies transformation.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            arr_name (str): The array copy to remove.
        """
        if not self.checked and not arr_name in self.find(sdfg):
            return
        self.checked = False

        data = sdfg.arrays[arr_name]
        arr_node = None
        state = None
        for tstate in sdfg.nodes():
            for edge in tstate.edges():
                if isinstance(edge.dst, dace.nodes.AccessNode) and edge.dst.data == arr_name:
                    arr_node = edge.dst
                    state = tstate
                    break

        out_edges = state.out_edges(arr_node)
        entry = state.entry_node(arr_node)

        if entry:
            in_edge = state.in_edges(arr_node)[0]
            tasklet = in_edge.src
            orig_node = state.in_edges(tasklet)[0].src
            orig_name = orig_node.data
            for edge in out_edges:
                dace.transformation.helpers.redirect_edge(state, edge, new_src = orig_node)
            self.delete_copy(state, arr_node, orig_node)
            self.traverse_n_replace(state, orig_node, arr_name, orig_name)
        else:
            tasklet = None
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet) and(node.label.startswith('copy_temp') or node.label.startswith('move')):
                    if state.out_edges(node)[0].data.data == arr_name:
                        tasklet = node
                        break
            entry = state.entry_node(tasklet)
            while True:
                t = state.entry_node(entry)
                if t:
                    entry = t
                else:
                    break
            orig_node = state.in_edges(entry)[0].src
            orig_name = orig_node.data
            if orig_node in state.source_nodes() and not out_edges:
                # orig_node is input node
                sdfg.remove_node(state)
                state_0(sdfg).replace(arr_name, orig_name)
            else:
                #rmv_edge = state.out_edges(orig_node)[0]
                for edge in out_edges:
                    dace.transformation.helpers.redirect_edge(state, edge, new_src = orig_node)
                self.traverse_n_replace(state, orig_node, arr_name, orig_name)
                self.delete_copy(state, arr_node, orig_node)
    
        sdfg.remove_data(arr_name)

    def delete_copy(self, state, node, stop_node):
        if isinstance(node, dace.nodes.AccessNode) and node == stop_node:
            return
        in_edge = state.in_edges(node)
        state.remove_node(node)
        for edge in in_edge:
            src = edge.src
            #state.remove_edge(edge)
            self.delete_copy(state, src, stop_node)

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
"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the SwapMapScope transformation.

"""

import dace

class SwapMapScope():
    """
    This class implements the SwapMapScope transformation. It searches for two map scopes in the SDFG and swaps them, i.e. inducing a order between two map scopes. The transformation is only applied if the map scopes are swappable.
    """
     
    def __init__(self):
        self.__name = 'SwapMapScope'
        self.checked = False
    
    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for two map scopes in the SDFG and returns a list of tuples containing the two map scopes.
        
        Two map scopes are swappable if the following conditions are met:
            - The two maps scopes have the same entry node (None if they are the outtermost maps)
            - The map scopes are in the same state
            - The input of the second map scope is not used in the output of the first map scope, i.e. the scopes are independent/parallel
            - Map Scopes should be consecutive
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for swappable map scopes.
        
        Returns:
            list: A list of tuples, where each tuple contains two swappable map scopes.
        """
        result = []
        map_scopes = self.find_map_scopes(sdfg)
        for i in range(len(map_scopes)):
            entry1, exit1, state1 = map_scopes[i]
            next = False
            for j in range(i+1, len(map_scopes)):
                entry2, exit2, state2 = map_scopes[j]
                if not state1.entry_node(entry1) == state2.entry_node(entry2):
                    continue
                if not state1 == state2:
                    continue
                if next:
                    break
                next = True
                output_1 = {edge.data.data for edge in state1.out_edges(exit1)}
                input_2 = {edge.data.data for edge in state2.in_edges(entry2)}
                if input_2 & output_1:
                    continue
                result.append((entry1, entry2))
        self.checked = True
        return result
    
    def find_map_scopes(self, sdfg):
        map_scopes = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    #map_scopes.append(dace.sdfg.scope._scope_subgraph(state, node, True, True))
                    map_scopes.append((node, state.exit_node(node), state))
                    
        return map_scopes
                    
    def apply(self, sdfg, entry1, entry2):
        """
        This method applies the SwapMapScope transformation to the given SDFG. It adds an empty edge from the output of the second map scope to the MapEntry of the first map scope.
        
        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            entry1 (dace.MapEntry): The first map scope to swap.
            entry2 (dace.MapEntry): The second map scope to swap.
        """
        if not self.checked and not (entry1, exit2) in self.find(sdfg):
            return
        self.checked = False
        state = find_state(sdfg, entry2)
        exit2 = state.exit_node(entry2)
        dst_node = [edge.dst for edge in state.out_edges(exit2)][0]
        state.add_nedge(dst_node, entry1, dace.memlet.Memlet())

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
        
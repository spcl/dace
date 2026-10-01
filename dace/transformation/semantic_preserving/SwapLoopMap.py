"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the SwapLoopMap transformation.

"""

import dace

class SwapLoopMap():
    """
    This class implements the SwapLoopMap transformation. It searches for two consecutive maps in the SDFG and swaps them. The transformation is only applied if the maps are swappable.
    """
    def __init__(self):
        self.__name = 'SwapLoopMap'
        self.checked = False

    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for two consecutive maps in the SDFG and returns a list of tuples containing the two maps.
        
        Two maps are swappable if the following conditions are met:
            - The maps are consecutive
            - The outer map contains exactly one child: the inner map
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for swappable maps.

        
        Returns:
            set: A list of tuples, where each tuple contains two consecutive swappable maps.
        """
        swappable_maps = set()
        for state in sdfg.nodes():
            for edges in state.edges():
                if isinstance(edges.src, dace.nodes.MapEntry) and isinstance(edges.dst, dace.nodes.MapEntry):
                    if (edges.src, edges.dst) in swappable_maps:
                        continue
                    out_edges = state.out_edges(edges.src)
                    dst_entry = edges.dst
                    not_swapable = False
                    for edge in out_edges:
                        if not isinstance(edge.dst, dace.nodes.MapEntry):
                            not_swapable = True
                            break
                        if isinstance(edge.dst, dace.nodes.MapEntry) and not dst_entry == edge.dst:
                            not_swapable = True
                            break
                    if not_swapable:
                        continue
                    src = edges.src.map
                    dst = edges.dst.map
                    for i, param in enumerate(src.params):
                        if param in dst.params:
                            continue
                    swappable_maps.add((edges.src, edges.dst))
        self.checked = True
        return list(swappable_maps)
    
    def find_exit_maps(self, sdfg, map1, map2):
        map1 = map1.map
        map2 = map2.map
        exit1 = None
        exit2 = None
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapExit):
                    if node.map == map1:
                        exit1 = node
                    if node.map == map2:
                        exit2 = node
        return exit1, exit2

    def apply(self, sdfg, map_entry1, map_entry2):
        """
        This method applies the SwapLoopMap transformation to the given SDFG. It swaps the two given maps.
        
        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            map_entry1 (dace.MapEntry): The first (outer) map to swap.
            map_entry2 (dace.MapEntry): The second (inner) map to swap.
        """
        assert isinstance(map_entry1, dace.sdfg.nodes.MapEntry)
        assert isinstance(map_entry2, dace.sdfg.nodes.MapEntry)
        if not self.checked and not (map_entry1, map_entry2) in self.find(sdfg):
            return
        self.checked = False

        exit1, exit2 = self.find_exit_maps(sdfg, map_entry1, map_entry2)
        assert exit1 is not None
        assert exit2 is not None

        state = find_state(sdfg, map_entry2)
        for edge in state.in_edges(map_entry1):
            dace.transformation.helpers.redirect_edge(state, edge, new_dst = map_entry2)
        for edge in state.out_edges(map_entry2):
            dace.transformation.helpers.redirect_edge(state, edge, new_src = map_entry1)
        for edge in state.out_edges(map_entry1):
            if edge.dst == map_entry2:
                dace.transformation.helpers.redirect_edge(state, edge, new_src = map_entry2, new_dst = map_entry1)
        for edge in state.in_edges(exit2):
            dace.transformation.helpers.redirect_edge(state, edge, new_dst = exit1)
        for edge in state.out_edges(exit1):
            dace.transformation.helpers.redirect_edge(state, edge, new_src = exit2)
        for edge in state.out_edges(exit2):
            if edge.dst == exit1:
                dace.transformation.helpers.redirect_edge(state, edge, new_src = exit1, new_dst = exit2)

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
        
"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

Transformation Classes:
    SwapLoopMap
    SwapMapScope
    SwapOperations
    TileMapScope
    JoinMapScopes
    SplitMapScopes
    UnrollMapScope
    addArrayCopies
    removeArrayCopies
    addArrayDims
    removeArrayDims
    reuseArray
    reuseArrayLocation
    StackToHeap
    HeapToStack
    changeMemoryLayout

Every class contains a find and apply method. The find method searches for possible transformations in the given SDFG and the apply method applies the transformation to the SDFG.
"""

###* Imports *###

import sys
import copy
#from dace.transformation import transformation
#from dace.transformation.helpers import redirect_edge
#import pdb
import traceback
import random
import dace
import numpy as np
import math
import sympy
import argparse

###* Transformation Classes *###

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
            map_entry1 (dace.MapEntry): The first map to swap.
            map_entry2 (dace.MapEntry): The second map to swap.
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
        dst_node = [edge.dst for edge in state(sdfg).out_edges(exit2)][0]
        state.add_nedge(dst_node, entry1, dace.memlet.Memlet())

class SwapOpertions():
    """ 
    This class implements the SwapOperations transformation. It searches for two tasklets in the SDFG and swaps their execution order. The transformation is only applied if the tasklets are swappable.
    """
    
    def __init__(self):
        self.__name = 'SwapOperations'
        self.checked = False
    
    @property
    def name(self):
        return self.__name
    
    def find(self, sdfg):
        """
        This method searches for two tasklets in the SDFG and returns a list of tuples containing the two tasklets.
        
        Two tasklets are swappable if the following conditions are met:
            - The two tasklets are in the same state
            - The two tasklets are consecutive
            - The output of the first tasklet is not used in the input of the second tasklet
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for swappable tasklets.
        
        Returns:
            list: A list of tuples, where each tuple contains two swappable tasklets.
        """
        result = []
        tasklets = self.find_tasklets(sdfg)
        for i in range(len(tasklets)):
            tasklet1, state1 = tasklets[i]
            next = False
            for j in range(i+1, len(tasklets)):
                tasklet2, state2 = tasklets[j]
                if not state1 == state2:
                    continue
                if not state1.entry_node(tasklet1) == state2.entry_node(tasklet2):
                    continue
                if next:
                    break
                next = True
                output_1 = {edge.data.data for edge in state1.out_edges(tasklet1)}
                input_2 = {edge.data.data for edge in state2.in_edges(tasklet2)}
                if input_2 & output_1:
                    continue
                result.append((tasklet1, tasklet2))
        self.checked = True
        return result
    
    def find_tasklets(self, sdfg):
        result = []
        for state in sdfg.nodes():
                for node in state.nodes():
                    if isinstance(node, dace.nodes.Tasklet):
                        result.append((node, state))
        return result
    
    def apply(self, sdfg, op1, op2):
        """
        This method applies the SwapOperations transformation to the given SDFG. It swaps the order of execution of the two given tasklets by adding an empty edge from the output of the second tasklet to the first tasklet.
        """
        if not self.checked and not (op1, op2) in self.find(sdfg):
            return
        self.checked = False
        state = find_state(sdfg, op2)
        out_node = [edge.dst for edge in state.out_edges(op2)][0]
        state.add_nedge(out_node, op1, dace.memlet.Memlet())

class TileMapScope():
    """
    This class implements the TileMapScope transformation. It searches for map scopes in the SDFG and tiles them. The transformation is only applied if the map scope is tileable.
    
    Attributes:
        name (str): The name of the transformation.
        checked (bool): A flag to indicate whether the transformation has been checked.
        put_last (bool): If set, the tiled map is put as innermost map.
        only_innermost (bool): If set, only the innermost map is tiled.
        max_tile (int): The maximum tile size.
        tile_sizes (list): A list of tile sizes. Can be None.
    """

    def __init__(self):
        self.__name = 'TileMapScope'
        self.checked = False
        # if set, the tiled map is put as innermost map
        self.put_last = True
        # if set, only the innermost map is tiled
        self.only_innermost = False
        # maximum tile size
        self.max_tile = 1024
        # for symbolic loop bounds
        self.tile_sizes = [64] #None
    
    @property
    def name(self):
        return self.__name
    
    def isInnermost(self, state, map_entry, traversed=None):
        if traversed is None:
            traversed = set()
        for edge in state.out_edges(map_entry):
            if isinstance(edge.dst, dace.nodes.MapEntry):
                return False
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            if not edge.dst in traversed:
                if not self.isInnermost(state, edge.dst, traversed):
                    return False
        return True
    
    def find(self, sdfg):
        """
        This method searches for map scopes in the SDFG and returns a list of tuples containing the map scope and the tile size.
        
        The map scope is tileable if the following conditions are met:
            - The map scope has a size greater than 1 if it is a constant loop bound
            - The tile size is a divisor of the map scope size

        Args:
            sdfg (dace.SDFG): The SDFG to search for tileable map scopes.

        Returns:
            list: A list of tuples, where each tuple contains a tileable map scope and a tile size.
        """
        tileable_maps = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if self.only_innermost and not self.isInnermost(state, node):
                        continue
                    map = node.map
                    dim = map.range.size()[0]
                    if isinstance(dim, sympy.core.numbers.Integer) and int(dim) > 1:
                        dim = int(dim)
                        if self.tile_sizes:
                            divisors = [i for i in self.tile_sizes if dim % i == 0 ]
                            tileable_maps.extend([(node, tile_size) for tile_size in divisors])
                        else:
                            divisors = [i for i in range(2, min(math.floor(math.sqrt(dim))+1, self.max_tile)) if dim % i == 0]
                            tileable_maps.extend([(node, tile_size) for tile_size in divisors])
                    elif isinstance(dim, sympy.core.symbol.Symbol):
                        tileable_maps.extend([(node, tile_size) for tile_size in self.tile_sizes])
        self.checked = True
        return tileable_maps
    
    def apply(self, sdfg, map_entry, tile_size):
        """
        This method applies the TileMapScope transformation to the given SDFG. It tiles the given map scope with the given tile size.
        This method uses the dace.transformation.helpers.tile method to tile the map scope. If the map scope has a symbolic loop bound, the method uses the dace.transformation.helpers.tile method with the divides_evenly set to False.
        If the put_last attribute is set, the tiled map is put as the innermost map.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            map_entry (dace.MapEntry): The map scope to tile.
            tile_size (int): The tile size.
        """
        if not self.checked and not (map_entry, tile_size) in self.find(sdfg):
            return
        self.checked = False
        assert isinstance(map_entry, dace.sdfg.nodes.MapEntry)
        assert len(map_entry.map.params) == 1
        paras = map_entry.map.params[0]
        tile_dict = {paras: tile_size}
        if isinstance(map_entry.range.size()[0], sympy.core.symbol.Symbol):
            dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=False, skew=True, **tile_dict)
        else:
            dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=True, skew=True, **tile_dict)
        
        if self.put_last:
            prop_map = map_entry
            state = find_state(sdfg, map_entry)
         
            if prop_map:
                while True:
                    cant_swap = False
                    swap_map = None
                    for out_edge in state.out_edges(prop_map):
                        if not isinstance(out_edge.dst, dace.nodes.MapEntry):
                            cant_swap = True
                            break
                        else:
                            if swap_map is None:
                                swap_map = out_edge.dst
                            else:
                                if not swap_map == out_edge.dst:
                                    cant_swap = True
                                    break
                    if cant_swap or swap_map is None:
                        break
                    opt = SwapLoopMap()
                    opt.apply(sdfg, prop_map, swap_map)

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
                    dst_shape = in_edge.data.dst_subset.size()
                    if not dst_shape == entry1.map.range.size():
                        reject = True
        if not reject:
            return True
        return False

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

class SplitMapScopes():
    """
    This class implements the SplitMapScopes transformation. It searches for map scopes in the SDFG and splits them. The transformation is only applied if the map scope is splittable.        
    """

    def __init__(self):
        self.__name = 'SplitMapScopes'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def isInnermost(self, state, map_entry, traversed=None):
        if traversed is None:
            traversed = set()
        for edge in state.out_edges(map_entry):
            if isinstance(edge.dst, dace.nodes.MapEntry):
                return False
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            if not edge.dst in traversed:
                if not self.isInnermost(state, edge.dst, traversed):
                    return False
        return True

    def traverse(self, state, map_entry, tasklets=None, visited=None):
        if tasklets is None:
            tasklets = set()
        if visited is None:
            visited = set()

        visited.add(map_entry)
        next_edges = state.out_edges(map_entry)
        for edge in next_edges:
            if isinstance(edge.dst, dace.nodes.Tasklet) and not edge.dst in tasklets:
                tasklets.add((state, edge.dst))
            elif isinstance(edge.dst, dace.nodes.MapExit) or isinstance(edge.dst, dace.nodes.Tasklet):
                continue
            if not edge.dst in visited:
                self.traverse(state, edge.dst, tasklets, visited)
        
        return list(tasklets)

    def find_2(self, sdfg):
        splitable_targets = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if not self.isInnermost(state, node):
                        continue
                    splitable_targets.extend(self.traverse(state, node))
        result = []
        for state, target_tasklet in splitable_targets:
            for edge in state.in_edges(target_tasklet):
                if not isinstance(edge.src, dace.nodes.MapEntry):
                    result.append((state, target_tasklet))
                    break
        self.checked = True
        return result
    
    def find(self, sdfg):
        """
        This method searches for map scopes in the SDFG and returns a list of tuples containing the map scope and the tasklet to split.

        The map scope is splittable if the following conditions are met:
            - The map scope is the innermost map scope
            - The map scope contains more than one tasklet

        Args:
            sdfg (dace.SDFG): The SDFG to search for splittable map scopes.
        
        Returns:
            list: A list of tuples, where each tuple contains a the state and the tasklet to split.
        """

        splitable_targets = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if not self.isInnermost(state, node):
                        continue
                    splitable_targets.extend(self.traverse(state, node))
        
        result = []
    
        # for tigthening the application space
        for state, node in splitable_targets:
            scope_nodes = []
            reject = False
            self.find_scope_nodes(state, node, scope_nodes)
            for scope_node in scope_nodes[1:]:
                inputs = [edge.src for edge in state.in_edges(scope_node)]
                for ins in inputs:
                    if isinstance(ins, dace.nodes.MapEntry):
                        reject = True
                        break 
                    if ins not in scope_nodes:
                        reject = True 
                        break
                    
            if not reject:
                result.append(node)
        self.checked = True
        return result
                                            
    def apply(self, sdfg, state, target_tasklet):
        """
        This method applies the SplitMapScopes transformation to the given SDFG. It splits the given map scope at the given tasklet.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            state (dace.SDFGState): The state the map scope is in.
            target_tasklet (dace.nodes.Tasklet): The tasklet to split the map scope at.
        """
        if not self.checked and not (state, target_tasklet) in self.find(sdfg):
            return
        self.checked = False

        map_entry = state.entry_node(target_tasklet)
        map1_input1 = []
        map1_input2 = []
        map1_input1 = [edge for edge in state.in_edges(map_entry)]
        map1_input2 = [edge for edge in state.out_edges(map_entry)]

        #add a new map
        new_map_entry, new_map_exit = state.add_map(f'{map_entry.label}', {map_entry.map.params[0]: f'0:{map_entry.map.range.size()[0]}'})
        
        # reconnect old map entry edges to new one
        for edge in map1_input1:
            in_conn = edge.dst_conn
            if in_conn:
                map_entry.remove_in_connector(in_conn)
            new_map_entry.add_in_connector(in_conn)
            dace.transformation.helpers.redirect_edge(state, edge, new_dst = new_map_entry, new_dst_conn = in_conn)
        for edge in map1_input2:
            out_conn = edge.src_conn
            if out_conn:
                map_entry.remove_out_connector(out_conn)
            new_map_entry.add_out_connector(out_conn)
            dace.transformation.helpers.redirect_edge(state, edge, new_src = new_map_entry, new_src_conn = out_conn)
        
        # find the paths that have to be redirected because of the new map
        inbetween_nodes = []
        inbetween_nodes = list(set([edge.src for edge in state.in_edges(target_tasklet)]))
        path_nodes = []
        for node in inbetween_nodes:
            if isinstance(node, dace.nodes.AccessNode):
                in_edges = state.in_edges(node)
                out_edges = state.out_edges(node)
                for edge in in_edges:
                    for edge2 in out_edges:
                        if not edge2.dst == target_tasklet:
                            continue
                        path_nodes.append((edge, node, edge2))
            if isinstance(node, dace.nodes.MapEntry):
                in_edges = {edge:False for edge in state.in_edges(node)}
                out_edges = {edge:False for edge in state.out_edges(node)}
                for edge, used in in_edges.items():
                    if used:
                        continue
                    for edge2, used2 in out_edges.items():
                        if not edge2.dst == target_tasklet:
                            continue
                        if used2:
                            continue
                        if edge.dst_conn[3:] == edge2.src_conn[4:]:
                            in_edges[edge] = True
                            out_edges[edge2] = True
                            path_nodes.append((edge, node, edge2))
                            break

        # redirect the edges of the input nodes of the tasklet
        # make a copy if input is used somewhere else aswell
        already_mapped = set()
        keep = set()
        for (edge1, btw_node, edge2) in path_nodes:
            if isinstance(btw_node, dace.nodes.AccessNode):
                out_edges = {edge.dst for edge in state.out_edges(btw_node) if not isinstance(edge.dst, dace.nodes.MapEntry)}
                if len(out_edges) == 1:
                    memlet = edge1.data
                    memlet2 = copy.deepcopy(edge1.data)
                    src = edge1.src
                    dst = edge2.dst
                    if isinstance(src, dace.nodes.Tasklet) and not src in already_mapped:
                        already_mapped.add(src)
                        state.add_memlet_path(src, new_map_exit, btw_node, memlet=memlet, src_conn=edge1.src_conn, dst_conn=edge1.dst_conn)
                    state.add_memlet_path(btw_node, map_entry, dst, memlet=memlet2, dst_conn=edge2.dst_conn, src_conn=edge2.src_conn)
                else:
                    new_node = copy.deepcopy(btw_node)
                    memlet = copy.deepcopy(edge1.data)
                    keep.add(edge1)
                    state.add_memlet_path(new_node, map_entry, edge2.dst, memlet=memlet, dst_conn=edge2.dst_conn)
                    
                    #add in and out connector manually if not already doen by add_memlet_path
                    out_edge = state.out_edges(new_node)[0]
                    if not out_edge.dst_conn in map_entry.in_connectors:
                        map_entry.add_in_connector(out_edge.dst_conn)
                    in_edges = state.in_edges(new_node)
                    in_edge = None
                    for edge in in_edges:
                        if edge.src_conn[4:] == out_edge.dst_conn[3:]:
                            in_edge = edge
                            break
                    if in_edge and not in_edge.src_conn in map_entry.out_connectors:
                        new_node.add_out_connector(in_edge.src_conn)

            elif isinstance(btw_node, dace.nodes.MapEntry):
                memlet = edge1.data
                tmp = edge1.dst_conn
                if tmp:
                    btw_node.remove_in_connector(tmp)
                tmp = edge2.src_conn
                if tmp:
                    map_entry.remove_out_connector(tmp)
                in_conn = edge1.src_conn
                dst_conn = edge2.dst_conn
                new_map_entry.remove_out_connector(edge2.src_conn)
                new_map_entry.remove_in_connector(edge1.dst_conn)
                state.add_memlet_path(edge1.src, map_entry, edge2.dst, memlet=memlet, src_conn=in_conn, dst_conn=dst_conn)

        # remove the edges that are not needed anymore
        for edge1, btw_node, edge2 in path_nodes:
            try:
                if not edge1 in keep:
                    state.remove_edge(edge1)
            except:
                pass
            try:
                state.remove_edge(edge2)
            except:
                pass

        # add empty edge if first map exit is not connected anymore
        if len(state.in_edges(new_map_exit)) == 0:
            if keep:
                dangeling_node = keep.pop().dst
                state.add_nedge(dangeling_node, new_map_exit, dace.memlet.Memlet())
                # enforce first map scope is executed before the second one
                if len(state.out_edges(map_entry)) == 0:
                    in_nodes = [edge.src for edge in state.in_edges(map_entry)]
                    for node in in_nodes:
                        if node.label == dangeling_node.label:
                            state.add_nedge(new_map_exit, node, dace.memlet.Memlet())
                            break

        # find paths in current sdfg that have to be redirected
        scope_nodes = []
        self.find_scope_nodes(state, map_entry, scope_nodes)
        fixable_paths = []
        redirection_maps = []
        for node in scope_nodes[1:]:
            in_edges = state.in_edges(node)
            for edge in in_edges:
                if edge.src not in scope_nodes:
                    if isinstance(edge.src, dace.nodes.MapEntry):
                        redirection_maps.append((edge, node))
                    else:
                        fixable_paths.append((edge, node))
        
        fixable_maps = []
        for edge, node in redirection_maps:
            in_edges = state.in_edges(edge.src)
            for edge2 in in_edges:
                if edge2.dst_conn[3:] == edge.src_conn[4:]:
                    fixable_maps.append((edge2, edge, node))
        
        # tmp variables that are produced in first map scope and used in second map scope have to be redirected
        # if node has to outgoing edges, node is copied, else redirected
        for edge, node in fixable_paths:
            if isinstance(edge.src, dace.nodes.AccessNode):
                input_edge = state.in_edges(edge.src)[0]
                src_conn = input_edge.src_conn
                dst_conn = edge.dst_conn
                new_node = copy.deepcopy(edge.src)
                out_set = {edge.dst for edge in state.out_edges(edge.src)}
                if len(out_set) == 1:
                    state.add_memlet_path(input_edge.src, new_map_exit, new_node, memlet=input_edge.data, src_conn=src_conn, dst_conn=dst_conn)
                    state.remove_node(edge.src)
                state.add_memlet_path(new_node, map_entry, node, memlet=edge.data, src_conn=edge.src_conn, dst_conn=edge.dst_conn)
                try:
                    state.remove_edge(edge)
                except:
                    pass
            elif isinstance(edge.src, dace.nodes.Tasklet):
                # should not happen
                pass
        
        # output edges of previous map scope that are only used in first map scope have to be redirected to new map exit
        exit_node = state.exit_node(map_entry)
        exit_input = [(edge, edge.src) for edge in state.in_edges(exit_node)]
        for edge, node in exit_input:
            if not node in scope_nodes:
                memlet = edge.data
                dst_edge = None
                for edge2 in state.out_edges(exit_node):
                    if edge2.src_conn[4:] == edge.dst_conn[3:]:
                        dst_edge = edge2
                        break
                if dst_edge:
                    exit_node.remove_in_connector(edge.dst_conn)
                    exit_node.remove_out_connector(dst_edge.src_conn)
                    state.add_memlet_path(node, new_map_exit, dst_edge.dst, memlet=memlet, src_conn=edge.src_conn, dst_conn=dst_edge.dst_conn)
                    state.remove_edge(edge)
                    state.remove_edge(dst_edge)
                
        # input of previous map scope that are only used in second new map scope can directly be redirected to the new map scope
        for edge2, edge, node in fixable_maps:
            new_map_entry.remove_in_connector(edge2.dst_conn)
            new_map_entry.remove_out_connector(edge.src_conn)
            state.add_memlet_path(edge2.src, map_entry, node, memlet=edge.data, src_conn=edge2.src_conn, dst_conn=edge.dst_conn)
            state.remove_edge(edge2)
            state.remove_edge(edge)
        
        # maybe fuse input nodes of second map if the are the same
        input_nodes = [edge.src for edge in state.in_edges(map_entry)]
        in_map = {}
        for node1 in input_nodes:
            if node1.label not in in_map:
                in_map[node1.label] = set()
            in_map[node1.label].add(node1)
        for key, value in in_map.items():
            if len(value) > 1:
                chosen_node = None
                for node in value:
                    if state.in_edges(node):
                        chosen_node = node
                        break
                if not chosen_node:
                    chosen_node = value.pop()
                for node in value:
                    if node == chosen_node:
                        continue
                    out_edges = state.out_edges(node)
                    for edge in out_edges:
                        dace.transformation.helpers.redirect_edge(state, edge, new_src = chosen_node)
                    state.remove_node(node)
        
        # some nodes from the first map scope that are hanging without any connection can be reconnect potentially
        scope_1_nodes = []
        self.find_scope_nodes(state, new_map_entry, scope_1_nodes)
        for node in scope_1_nodes[1:]:
            if isinstance(node, dace.nodes.AccessNode):
                out_edges = state.out_edges(node)
                if not out_edges or (out_edges[0] and not out_edges[0].data.data):
                    dst_node = [edge.src for edge in state.in_edges(map_entry) if edge.src.label == node.label][0]
                    if dst_node and state.in_edges(dst_node) and state.in_edges(dst_node)[0].data.data:
                        continue
                    elif dst_node and (not state.in_edges(dst_node) or (state.in_edges(dst_node)[0] and not state.in_edges(dst_node)[0].data.data)):
                        src_edge = state.in_edges(node)[0]
                        state.add_memlet_path(src_edge.src, new_map_exit, dst_node, memlet=src_edge.data, src_conn=src_edge.src_conn, dst_conn=src_edge.dst_conn)
                        state.remove_node(node)
                        if state.in_edges(dst_node)[0] and not state.in_edges(dst_node)[0].data.data:
                            state.remove_edge(state.in_edges(dst_node)[0])
                        continue

    def find_scope_nodes(self, state, node, scope_nodes, visited=None):
        if visited is None:
            visited = set()

        visited.add(node)
        scope_nodes.append(node)
        next_edges = [edge for edge in state.out_edges(node)]
        for edge in next_edges:
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            elif not edge.dst in visited:
                self.find_scope_nodes(state, edge.dst, scope_nodes, visited)
        return    

class UnrollMapScope():
    """
    This class implements the UnrollMapScope transformation. It searches for map scopes in the SDFG and unrolls them. The transformation is only applied if the map scope is unrollable.
    Only applicable to constant number of iterations.

    Attributes:
        max_unroll_factor (int): The maximum unroll factor.
        explicit_factor (bool): A flag that indicates whether the unroll factor is explicitly given.

    Args:
        unroll_factor (int): The maximum unroll factor (default is 2).
    """

    def __init__(self, unroll_factor = 2):
        self.__name = 'UnrollMapScope'
        self.checked = False
        self.max_unroll_factor = unroll_factor
        self.explicit_factor = True
    
    @property
    def name(self):
        return self.__name

    def isInnermost(self, state, map_entry, traversed=None):
        if traversed is None:
            traversed = set()
        for edge in state.out_edges(map_entry):
            if isinstance(edge.dst, dace.nodes.MapEntry):
                return False
            if isinstance(edge.dst, dace.nodes.MapExit):
                continue
            if not edge.dst in traversed:
                if not self.isInnermost(state, edge.dst, traversed):
                    return False
        return True

    def find(self, sdfg):
        """
        This method searches for map scopes in the SDFG and returns a list of tuples containing the map scope and the unroll factor.

        The map scope is unrollable if the following conditions are met:
            - The loop bounds are constant
            - The map scope is the innermost map scope
            - The unroll factor is less than the maximum unroll factor
            - The unroll factor is a divisor of the map range size
            - The map is not already unrolled
        
        Args:
            sdfg (dace.SDFG): The SDFG to search for unrollable map scopes.

        Returns:
            list: A list of tuples, where each tuple contains a the map scope and the unroll factor.
        """
        unrollable_maps = []
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if node.map.schedule == dace.ScheduleType.Unrolled or node.map.unroll:
                        continue
                    if not self.isInnermost(state, node):
                        continue
                    dim = node.map.range.size()[0]
                    if isinstance(dim, sympy.core.numbers.Integer) and int(dim) > 1:
                        if self.explicit_factor:
                            divisors = [self.max_unroll_factor] if dim % self.max_unroll_factor == 0 else [i for i in range(2, min(self.max_unroll_factor+1, int(dim))) if dim % i == 0]
                            unrollable_maps.extend([(node, div) for div in divisors])
                        else:
                            divisors = [i for i in range(2, min(self.max_unroll_factor+1, int(dim))) if dim % i == 0]
                            unrollable_maps.extend([(node, div) for div in divisors])
        self.checked = True
        return unrollable_maps
    
    def apply(self, sdfg, map_entry, unroll_factor):
        """
        This method applies the UnrollMapScope transformation to the given SDFG. It unrolls the given map scope with the given unroll factor.

        It uses the dace.transformation.helpers.tile method to unroll the map scope. It creates a new map scope with the given unroll factor and sets the schedule type to unrolled.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            map_entry (dace.nodes.MapEntry): The map scope to unroll.
            unroll_factor (int): The unroll factor.
        """
        if not self.checked and not (map_entry, unroll_factor) in self.find(sdfg):
            return
        self.checked = False
        assert isinstance(map_entry, dace.sdfg.nodes.MapEntry)
        para = map_entry.map.params[0]
        label = map_entry.map.label
        unroll_dict = {para: unroll_factor}
        dace.transformation.helpers.tile(sdfg=sdfg, map_entry=map_entry, divides_evenly=True, skew=True, **unroll_dict)
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and node.map.label == label:
                    if node.map.params[0] == para and node.map.range.size()[0] == unroll_factor:
                        new_map = node.map
                        new_map.schedule = dace.ScheduleType.Unrolled
                        new_map.unroll = True
                        new_map.unroll_factor = unroll_factor
                        break

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
           for node in state.nodes():
               if isinstance(node, dace.nodes.AccessNode):
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
        output_nodes = state.sink_nodes()
        adj_state = state
        if arr_name in [node.data for node in input_nodes]:
            s = len(sdfg.nodes())
            new_state = sdfg.add_state(f'state_{s}')
            sdfg.add_edge(new_state, sdfg.nodes()[s-1], dace.InterstateEdge())
            adj_state = new_state
        
        new_name = ''
        i = 0
        while True:
            new_name = f'{arr_name}_copy_{i}'
            if not new_name in sdfg.arrays:
                break
            i += 1
   
        sdfg.add_array(name=new_name, shape=data.shape, dtype=data.dtype, storage=data.storage, transient=True, lifetime=data.lifetime)
        
        tasklet = adj_state.add_tasklet("move", {"in_1"}, {"out"}, "out = in_1")
        
       
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
                        edge_subset = [str(idx[0]) for idx in list(edge.data.subset)]
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
                    if data1.shape == data2.shape and data1.dtype == data2.dtype:
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
          
class StackToHeap():
    """
    This class implements the StackToHeap transformation. It changes the storage type of the array to the heap.
    """
    def __init__(self):
        self.__name = 'StackToHeap'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that are not yet stored on the heap (CPU_HEAP).

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.

        Returns:
            list: A list of data arrays that are not stored on the heap.
        """
        result = []
        for name, data in sdfg.arrays.items():
            if not data.storage == dace.StorageType.CPU_Heap:
                result.append(data)
        self.checked = True
        return result
    
    def apply(self, sdfg, data):
        """
        This method applies the StackToHeap transformation to the given SDFG. It changes the storage type of the array to the heap.
        """
        if not self.checked and not data in self.find(sdfg):
            return 
        self.checked = False

        data.storage = dace.StorageType.CPU_Heap

class HeapToStack():
    """
    This class implements the HeapToStack transformation. It changes the storage type of the array to the stack.
    
    Inputs:
        - hard_limit (int): The hard limit of the stack size. If the stack size exceeds this limit, the transformation is not applied. Default is None (no hard limit).
        - stack_size_factor (int): The factor to calculate the stack size. Default is 1. Which means the stack size is 1 * 1MB (minus a small amount). This is just a heuristic.


    Attributes:
        check_limit (bool): If True, the transformation checks if the stack size is exceeded.
        stack_size (int): The size of the stack. Default is 1MB. This can be changed by the user. This is just a heuristic.
        hard_limit (int): The hard limit of the stack size. If the stack size exceeds this limit, the transformation is not applied. Default is None (no hard limit).
    """

    def __init__(self, check_limit = True, hard_limit=None, stack_size_factor=1.0):
        self.__name = 'HeapToStack'
        self.checked = False
        # check if stack size is exceeded
        self.check_limit = check_limit
        # Default assumption: stack size is 1MB = 1 * 1'048'576 bytes
        self.stack_size = stack_size_factor * 1048576 * 0.95 #1MB * 0.95 
        self.hard_limit = None #65536 #(64KB)

    @property
    def name(self):
        return self.__name

    def get_stack_size(self, sdfg, limit='soft'):
        import os
        if os.name == 'nt': #windows
            self.stack_size = None
        else: #unix like systems
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_STACK)
            if limit == 'soft':
                self.stack_size = soft_limit
            else:
                self.stack_size = hard_limit

    def current_stack_size(self, sdfg):
        current_size = 0
        all_nodes = set()
        # we want only the arrays that are represented in the SDFG
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    all_nodes.add(node.label)
    
        for name, data in sdfg.arrays.items():
            if not name in all_nodes:
                continue
            if data.storage == dace.StorageType.Register:
                current_size += data.total_size * data.dtype.bytes
        return current_size

    def check_current_stack_usage(self):
        """Check the current stack size usage in bytes."""
        import sys
        frame = sys._getframe()
        stack_usage = 0
        while frame is not None:
            # Each frame has its own local variables and other information
            stack_usage += sum(sys.getsizeof(v) for v in frame.f_locals.values())
            frame = frame.f_back  # Move to the previous frame
        return stack_usage
    
    def stack_size_exceeded(self, sdfg, data, safety_factor=1.0):
        size = data.total_size * data.dtype.bytes
        available_stack = self.stack_size - self.current_stack_size(sdfg)
        if safety_factor*size > available_stack:
            return True
        return False

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG that are not yet stored on the stack (Register).
        
        This transformation is only applied if the stack size is not exceeded.
        
        We try not to exceed the stack size.
        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of data arrays that are not stored on the stack.
        """
        result = []
        for name, data in sdfg.arrays.items():
            if data.storage == dace.StorageType.Register:
                continue
            if self.hard_limit:
                if self.hard_limit < data.total_size * data.dtype.bytes:
                    continue
            if self.check_limit and self.stack_size_exceeded(sdfg, data):
                continue
            result.append(data)
        self.checked = True
        return result
    
    def apply(self, sdfg, data):
        if not self.checked and not data in self.find(sdfg):
            return
        self.checked = False
        data.storage = dace.StorageType.Register

class changeMemoryLayout():
    """
    This class implements the changeMemoryLayout transformation. It changes the memory layout of the array.
    """

    def __init__(self):
        self.__name = 'changeMemoryLayout'
        self.checked = False

    @property
    def name(self):
        return self.__name

    def find(self, sdfg):
        """
        This method searches for arrays in the SDFG on which the changeMemeoryLayout Transformation can be applied.

        Appicable to all data arrays which are not used as input or output to the kernel.

        Args:
            sdfg (dace.SDFG): The SDFG to search for arrays.
        
        Returns:
            list: A list of data arrays on which the transformation can be applied.
        """
        changeable_arrays = set()
        for state in sdfg.nodes():
            source, sink = state.source_nodes(), state.sink_nodes()
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode):
                    if not node in source and not node in sink:
                        changeable_arrays.add(node)
        self.checked = True
        return (list(changeable_arrays))
    
    def apply(self, sdfg, node, dim_order):
        """
        This method applies the changeMemoryLayout transformation to the given SDFG. It changes the memory layout of the array by chaning the strides of the array.

        Args:
            sdfg (dace.SDFG): The SDFG to apply the transformation to.
            node (dace.nodes.AccessNode): The array to apply the transformation to.
            dim_order (list): The new order of the dimensions.
        """
        if not self.checked and not node in self.find(sdfg):
            return
        self.checked = False

        data = sdfg.arrays[node.data]
        shape = data.shape

        # check if order of dimension is correct
        assert len(shape) == len(dim_order)
        assert max(dim_order) == len(shape) - 1
        assert min(dim_order) == 0

        m = dict(zip(shape, dim_order))
        new_strides = list()
        for dim in shape:
            stride = 1
            for dim2 in shape:
                if m[dim2] > m[dim]:
                    stride *= dim2
            new_strides.append(stride)

        # set new stride
        data.strides = tuple(new_strides)
        return

###* Helper functions *###
# NOT USED
def get_source_node(state, node, data, visited = []):
    if isinstance(node, dace.nodes.AccessNode):
        if node.data == data:
            return node
    visited.append(node)

    for edge in state.in_edges(node):
        if not edge.src in visited:
            get_source_node(state, edge.src, data)
    
    return None

# NOT USED
def get_sink_node(state, node, data, visited = []):
    if isinstance(node, dace.nodes.AccessNode):
        if node.data == data:
            return node
    visited.append(node)

    for edge in state.out_edges(node):
        if not edge.dst in visited:
            get_sink_node(state, edge.dst, data)
    
    return None

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
        
def find_state_by_name(sdfg, name):
    for state in sdfg.nodes():
        for node in state.nodes():
          if isinstance(node, dace.nodes.AccessNode):
              if node.data == name:
                return state

# NOT USED       
def start_nodes(state):
    incoming_edges = {node: 0 for node in state.nodes()}
    for edge in state.edges():
        incoming_edges[edge.dst] += 1
    starting_nodes = [node for node, count in incoming_edges.items() if count == 0]
    return starting_nodes

def order_sdfg(state):
    incoming_edges = {node: 0 for node in state.nodes()}
    for edge in state.edges():
        incoming_edges[edge.dst] += 1
    starting_nodes = [node for node, count in incoming_edges.items() if count == 0]

    order = []
    visited = set()

    # Stack to manage nested structures
    stack = [order]

    def traverse(node):
        if node in visited:
            return
        visited.add(node)

        # Check if node is a MapEntry, if so, create a new sublist
        if isinstance(node, dace.nodes.MapEntry):
            current_list = []
            stack[-1].append(node)
            stack[-1].append(current_list)
            stack.append(current_list)
        else:
            stack[-1].append(node)

        for edge in state.edges():
            if edge.src == node:
                traverse(edge.dst)

        # If node is a MapExit, pop the last list from the stack
        if isinstance(node, dace.nodes.MapExit):
            stack.pop()

    for node in starting_nodes:
        traverse(node)

    return order

def state_0(sdfg):
    for state in sdfg.nodes():
        if state.label == 'state_0':
            return state

###* Optimizing SDFG *###

def optimize_sdfg(sdfg, optimization):
    """
    This method applies the given optimization to the SDFG.
    
    Args:
        sdfg (dace.SDFG): The SDFG to apply the optimization to.
        optimization: The optimization to apply.
    """
    # Apply
    transformation = optimization()
    locations = transformation.find()
    argument = random.choice(locations)
    if isinstance(argument,(tuple,list)):
        transformation.apply(sdfg, *argument)
    else:
        transformation.apply(sdfg, argument)

###* Kernel Testing *###

def validate(sdfg):
    """
    This method tests the SDFG by comparing the output of the SDFG with the expected output.
    Only works for the gemm and softmax kernel (args.kernel).
    """
    size = 32

    inputs = set()
    for state in sdfg.nodes():
        source = state.source_nodes()
        for src in source:
            if isinstance(src, dace.nodes.AccessNode):
                if not 'copy' in src.data:
                    inputs.add(src)

    input_names = [node.data for node in inputs if not node.desc(sdfg).transient]
    outputs = [node for node in state_0(sdfg).sink_nodes() if isinstance(node, dace.nodes.AccessNode)]
    output_names = [node.data for node in outputs]
    data = sdfg.arrays

    loop_index = {name: size for name, data in data.items() if isinstance(data, dace.data.Scalar) and not name in [n.data for n in state_0(sdfg).nodes() if isinstance(n, dace.nodes.AccessNode)]}
    input_data = {}
    for output_name in output_names:
        if output_name in input_names:
            continue
        data_container = data[output_name]
        if isinstance(data_container, dace.data.Scalar):
            input_data[output_name] = np.zeros(1)
        else:
            shape = data_container.shape
            np_shape = []
            for dim in shape:
                d = str(dim)
                if d in loop_index:
                    np_shape.append(loop_index[d])
            input_data[output_name] = np.zeros(shape=np_shape)
        
    for input_name in input_names:
        data_container = data[input_name]
        if isinstance(data_container, dace.data.Scalar):
            input_data[input_name] = np.random.rand()
        else:
            shape = data_container.shape
            np_shape = []
            for dim in shape:
                d = str(dim)
                if d in loop_index:
                    np_shape.append(loop_index[d])
            input_data[input_name] = np.random.rand(*np_shape)


    expected = np.array([])
    output = np.array([])
    if args.kernel == "gemm":
        expected = input_data['alpha'] * input_data['A'] @ input_data['B'] + input_data['beta']*input_data['C']
        output = input_data['D']
    elif args.kernel == "softmax":
        argsrc = np.copy(input_data['src'])
        argsrc -= np.max(argsrc, axis=1, keepdims=True)
        np.exp(argsrc, out=argsrc)
        argsrc /= np.sum(argsrc, axis=1, keepdims=True)
        expected = argsrc
        output = input_data['dst']
    
    input_data.update(loop_index)
    try:
        sdfg(**input_data)
    except Exception as e:
        print(e)     

    diff = np.linalg.norm(output - expected)
    print("Difference:", diff)
    if diff >= 1e-4:
        print("Validation failed")
    else:
        print("Validation successful")
    print("==== Program end ====")


def main(args):
    """
    This method is the main method of the optimize_sdfg.py script. It applies the given optimization to the SDFG and validates the output.
    Currently only the gemm and softmax kernel are supported to test in this script. Refer to mk_kernel_tests.py for other kernels.

    Call the script with the following arguments:
    - kernel: The kernel to test (gemm, softmax) (--kernel gemm)
    - opt: The optimization to apply (tile, swap, unroll, split, join, reuseArray, addCopies, removeCopies, reuseLoc) (--opt tile)
    - const: If True, the kernel with constant values is used (gemm_const, softmax_const) (--const)
    - validate: If True, the output of the SDFG is validated (--validate)
    """
    
    print("==== Program start ====")

    source = ''
    sdfg_name = ''
    if args.kernel == "gemm":
        postfix = ""
        if not args.const:
            source = "test_kernel/test_gemm.sdfg"
            #source = ".dacecache/affine_grid_2d/program.sdfg"
        else: 
            source  = "test_kernel/test_gemm_const.sdfg"
            postfix = "_const"
        sdfg_name = "gemm_" + args.opt + postfix
    elif args.kernel == "softmax":
        postfix = ""
        if not args.const:
            source = "test_kernel/test_softmax.sdfg"
        else:
            source = "test_kernel/test_softmax_const.sdfg"
            postfix = "_const"
        sdfg_name = "softmax_" + args.opt + postfix
    else:
        sys.exit("Invalid kernel")
    
    sdfg = dace.SDFG.from_file(source)
    sdfg.name = sdfg_name

    if args.opt == "tile":
        transformation = TileMapScope()
        locations = transformation.find(sdfg)
        if locations:
            print("tile locations", locations)
            loc = locations[0]
            print("tile location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied loop tile")

    if args.opt == "swap":
        transformation = SwapLoopMap()
        locations = transformation.find(sdfg)
        if locations:
            loc = locations[0]
            print("swap location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied swap scope")

    if args.opt == "unroll":
        print("unroll")
        transformation = UnrollMapScope()
        locations = transformation.find(sdfg)
        if locations:
            loc = locations[0]
            print("unroll location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied unroll scope")

    if args.opt == "split":
        transformation = SplitMapScopes()
        locations = transformation.find(sdfg)
        print(f'splitable locations {locations}')
        if locations:
            #loc = locations[0]
            loc = None
            for t in locations:
                if t[1].label == 'mul':
                    if isinstance(t[0].out_edges(t[1])[0].dst, dace.nodes.AccessNode):
                        if t[0].out_edges(t[1])[0].dst.data == 'inv_sum_val_div_u':#'f32_copyfrom_i32':
                            loc = t
                            break
        
            print("split location", loc[1])
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied split scope")

    if args.opt == "join":
        """
        # first do a split
        transformation = SplitMapScopes()
        locations = transformation.find(sdfg)
        print(f'splitable locations {locations}')
        if locations:
            loc = None
            for t in locations:
                if t[1].label == 'fnmsub':
                    if isinstance(t[0].out_edges(t[1])[0].dst, dace.nodes.AccessNode):
                        if t[0].out_edges(t[1])[0].dst.data == 'inv_sum_val_inv_th_2':#'f32_copyfrom_i32':
                            loc = t
                            break
            transformation.apply(sdfg, loc[0], loc[1])
        sdfg.view()
        """
        #join
        transformation = JoinMapScopes()
        locations = transformation.find(sdfg)
        print(f'joinable locations {locations}')
        if locations:
            loc = locations[-1]
            print("join location", loc[0], loc[1], loc[2])
            transformation.apply(sdfg, loc[0], loc[1], loc[2])
            print("applied join")
    
    if args.opt == "reuseArray":
        transformation = reuseArray()
        locations = transformation.find(sdfg)
        if locations:
            print("reuse array locations", locations)
            loc = locations[0]
            print("reuse array location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied reuse array")
    
    if args.opt == "addCopies":
        transformation = addArrayCopies()
        locations = transformation.find(sdfg)
        if locations:
            loc = "dst"
            print("add copies location", loc)
            transformation.apply(sdfg, loc)
            print("applied add copies")
    
    if args.opt == "removeCopies":
        transformation = addArrayCopies()
        loc = "dst"
        transformation.apply(sdfg, loc)
        transformation = removeArrayCopies()
        locations = transformation.find(sdfg)
        print(f'applicable locations {locations}')
        if locations:
            loc = loc + "_copy_0"
            print("remove copies location", loc)
            transformation.apply(sdfg, loc)
            print("applied remove copies")
        
    if args.opt == "reuseLoc":  
        transformation = reuseArrayLocations()
        locations = transformation.find(sdfg)
        print(f'reusable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("reuse location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied reuse location")

    if args.opt == "changeLayout":
        transformation = changeMemoryLayout()
        locations = transformation.find(sdfg)
        print(f'changeable arrays {locations}')
        if locations:
            loc = locations[0]
            print("change layout location", loc)
            transformation.apply(sdfg, loc, [1,0])
            print("applied change layout")
        
    if args.opt == "addDimension":
        transformation = addArrayDims()
        locations = transformation.find(sdfg)
        print(f'createable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("add dimension location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied add dimension")
        
    if args.opt == "removeDimension":
        transformation = addArrayDims()
        locations = transformation.find(sdfg)
        if locations:
            loc = locations[0]
            transformation.apply(sdfg, loc[0], loc[1])
        transformation = removeArrayDims()
        locations = transformation.find(sdfg)
        print(f'deleteable dimensions {locations}')
        if locations:
            loc = locations[0]
            print("remove dimension location", loc)
            print(f'{loc[0]} {loc[1]} {loc[2]}')
            transformation.apply(sdfg, loc[0], loc[1], loc[2])
            print("applied remove dimension")
    
    if args.opt == "swapScopes":
        transformation = SwapMapScope()
        locations = transformation.find(sdfg)
        if locations:
            loc = locations[0]
            print("swap operation location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied swap operation")
        
    if args.opt == "swapOp":
        transformation = SwapOpertions()
        locations = transformation.find(sdfg)
        print(f'swappable operations {locations}')
        if locations:
            loc = locations[0]
            print("swap operation location", loc)
            transformation.apply(sdfg, loc[0], loc[1])
            print("applied swap operation")
    
    if args.opt == "test_split_join":
        transformation1 = SplitMapScopes()
        locations = transformation1.find(sdfg)
        for loc in locations:
            copy_sdfg = copy.deepcopy(sdfg)
            locations1 = transformation1.find(copy_sdfg)
            loc1 = None
            for locs in locations1:
                if not locs[1].label == loc[1].label:
                    continue
                out_edge1 = [edge.data.data for edge in locs[0].out_edges(locs[1])]
                out_edge0 = [edge.data.data for edge in loc[0].out_edges(loc[1])]
                if not out_edge1 == out_edge0:
                    continue
                else:
                    loc1 = locs
                    break
            if loc1:
                try:
                    transformation1.apply(copy_sdfg, loc1[0], loc1[1])
                    print(f"applied split scope on {loc1[1]} ({loc1[0].out_edges(loc1[1])[0].data.data})")
                    transformation2 = JoinMapScopes()
                    locations2 = transformation2.find(copy_sdfg)
                    if locations2:
                        loc2 = locations2[-1]
                        try:
                            transformation2.apply(copy_sdfg, loc2[0], loc2[1], loc2[2])
                            print(f"applied join scope on {loc2[1]} and {loc2[2]}")
                            try:
                                copy_sdfg.validate()
                                print(f'==> validated {loc1[1]}\n')
                            except Exception as e:
                                print(e)
                                sdfg.view()
                                print(f'==> invalid {loc1[1]}\n')
                        except Exception as e:
                            print(f'invalid join {loc2[1]}, abort\n')

                except Exception as e:
                    print(f'invalid split {loc1[1]}, abort\n')
        
    if args.opt == "test_split_join_2":
     trans_split = SplitMapScopes()
     trans_join = JoinMapScopes()
     while True:
        split_locs = trans_split.find(sdfg)
        if not split_locs:
            break
        loc1 = random.choice(split_locs)
        try:
            trans_split.apply(sdfg, loc1[0], loc1[1])
            print(f'applied split {loc1[1]} ({loc1[0].out_edges(loc1[1])[0].data.data})')
            join_locs = trans_join.find(sdfg)
            if not join_locs:
                break
            loc = random.choice(join_locs)
            try:
                trans_join.apply(sdfg, loc[0], loc[1], loc[2])
                print(f'applied join {loc[1]} and {loc[2]}')
                try:
                    sdfg.validate()
                    print(f'==> validated {loc[1]} and {loc[2]}\n')
                except Exception as e:
                    traceback.print_exc()
                    print(f'==> invalid {loc[1]} and {loc[2]}\n')
            except Exception as e:
                traceback.print_exc()
                print(f'failed to join {loc[1]}\n')
                break
        except Exception as e:
            traceback.print_exc()
            print(f'failed to split {loc1[1]}\n')
            break

    if args.opt == "test_split_join_3":
        trans_join = JoinMapScopes()
        trans_split = SplitMapScopes()
        violation = False
        its = 10
        while True:
            split_locs = trans_split.find(sdfg)
            if not split_locs or its == 0:
                break
            its -= 1
            loc1 = random.choice(split_locs)
            try:
                trans_split.apply(sdfg, loc1[0], loc1[1])
                print(f'applied split {loc1[1]} ({loc1[0].out_edges(loc1[1])[0].data.data})')
                try:
                    sdfg.validate()
                except Exception as e:
                    violation = True
                    print(f'==> invalid {loc1[1]}\n')
                    break
            except Exception as e:
                violation = True
                print(f'failed to split {loc1[1]} ({loc1[0].out_edges(loc1[1])[0].data.data})\n')
                break
        its = 10
        while not violation:
            join_locs = trans_join.find(sdfg)
            if not join_locs or its == 0:
                break
            its -= 1
            loc = random.choice(join_locs)
            try:
                trans_join.apply(sdfg, loc[0], loc[1], loc[2])
                print(f'applied join {loc[1]} and {loc[2]}')
                try:
                    sdfg.validate()
                except Exception as e:
                    print(f'==> invalid {loc[1]} and {loc[2]}\n')
                    break
            except Exception as e:
                print(f'failed to join {loc[1]}\n')
                break
            
    sdfg.view()
    csdfg = sdfg.compile()
    if args.validate:
        validate(sdfg)

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize SDFGs.')
    parser.add_argument('--validate', action='store_true', help='see if the generated code is correct')
    parser.add_argument('--print', action='store_true', help='show printed code')
    parser.add_argument('--opt', type=str, help='optimization to apply', default='')
    parser.add_argument('--kernel', type=str, help='kernel to apply the optimization to', default='gemm')
    parser.add_argument('--const', action='store_true', help='Symbolic or Constant Loop Sizes', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
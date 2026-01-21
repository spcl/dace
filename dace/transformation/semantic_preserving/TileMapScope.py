"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the TileMapScope transformation.

"""

import dace
import sympy
import sp_opts.SwapLoopMap as SwapLoopMap

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

    def __init__(self, put_last = True, tile_sizes = [16]):
        self.__name = 'TileMapScope'
        self.checked = False
        # if set, the tiled map is put as innermost map
        self.put_last = put_last
        # if set, only the innermost map is tiled
        self.only_innermost = False
        self.not_innermost = False
        # maximum tile size
        self.max_tile = 1024
        # for symbolic loop bounds
        self.tile_sizes = tile_sizes
        # dont tile maps where
    
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
                    if self.not_innermost and self.isInnermost(state, node):
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
            # all idx are in the form 'i0', 'i1', ... (or 'tiled_i0', 'tiled_i1', ...)
            prop_map_idx = int(prop_map.map.params[0][-1])
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
                    # try to have the same order of maps as before
                    print(f'swap_map: {swap_map.map.params[0]}/{int(swap_map.map.params[0][-1])} and prop_map: {prop_map.map.params[0]}')
                    if not swap_map.range.size()[2] > 1:
                        if int(swap_map.map.params[0][-1]) > prop_map_idx:
                            break
                    opt = SwapLoopMap.SwapLoopMap()
                    opt.apply(sdfg, prop_map, swap_map)

def find_state(sdfg, node):
    for sdfg_state in sdfg.nodes():
        if node in sdfg_state.nodes():
            return sdfg_state
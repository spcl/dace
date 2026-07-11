"""
***ETH Zurich SPCL - Bachelors Thesis: BSc thesis: Deep learning transformations with semantic-preservation in DaCe by Severin Huber***

This module contains several classes that implement various transformations and optimizations for SDFGs (Stateful DataFlow Graphs) using the DaCe (Data-Centric Parallel Programming) framework.

This is part of the bachelor thesis of Severin Huber at SPCL. The topic is semantic-preserving transformations.

This optimization transformations are implemented not for general sdfgs, but for those generated 
from Andrei Ivanovs microkernel IR with a converter implemented as a part of this thesis. 
General usage of these transformations is not guaranteed.

This class implements the UnrollMapScope transformation.

"""

import sympy
import dace

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

    def __init__(self, unroll_factor = 4):
        self.__name = 'UnrollMapScope'
        self.checked = False
        self.max_unroll_factor = unroll_factor
        self.explicit_factor = True
        self.no_reduction = False
    
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
                    if self.no_reduction:
                        br_cond = False
                        exit = state.exit_node(node)
                        in_edges = state.in_edges(exit)
                        for edge in in_edges:
                            if edge.data.wcr:
                                br_cond = True
                                break
                        if br_cond:
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

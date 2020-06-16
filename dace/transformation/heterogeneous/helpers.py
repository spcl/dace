from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg

from copy import deepcopy as dcpy
from typing import List, Union, Dict, Tuple

import dace.libraries.standard as stdlib



# ****************
# Helper functions

def common_map_base_ranges(maps: List[nodes.Map]) -> List:
    """ Finds a maximal set of ranges that can be found
        in every instance of the maps in the given list
    """
    if len(maps) == 0:
        return None
    # first pass: find maximal set
    map_base = [rng for rng in maps[0].range]
    for map in maps:
        tmp = [rng for rng in map.range]

        map_base_new = []
        for element in tmp:
            if element in map_base:
                map_base_new.append(element)
                map_base.remove(element)

        map_base = map_base_new


    return map_base


def find_reassignment(maps: List[nodes.Map], map_base_ranges) -> Dict[nodes.Map, List]:
    """ Provided a list of maps and their common base ranges
        (found via common_map_base_ranges()),
        for each map greedily assign each loop to an index so that
        a base range has the same index in every loop.
        If a loop range of a certain map does not correspond to
        a common base range, no index is assigned (=-1)


        :param maps:            List of maps
        :param map_base_ranges: Common ranges extracted via
                                common_map_base_ranges()

        :returns: Dict that maps each map to a vector with
                  the same length as number of map loops.
                  The vector contains, in order, an index
                  for each map loop that maps it to a
                  common base range or '-1' if it does not.
    """
    result = {map: None for map in maps}
    outer_ranges_dict = dict(enumerate(map_base_ranges))

    for map in maps:
        result_map = []
        for current_range in map.range:
            found = False
            for j, outer_range in outer_ranges_dict.items():
                if current_range == outer_range and j not in result_map:
                    result_map.append(j)
                    found = True
                    break
            if not found:
                result_map.append(-1)
        result[map] = result_map

    return result

########################################################################

def toplevel_scope_subgraph(graph, subgraph, scope_dict = None):
    """
    returns the toplevel scope of a *connected* subgraph
    """
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for element in subgraph:
        scopes.add(scope_dict[element])
    for scope in scopes:
        # search the one whose parent is not in scopes
        # that must be the top level one
        if scope_dict[scope] not in scopes:
            return scope

    raise RuntimeError("Subgraph is not sound (must be connected)")

def toplevel_scope_maps(graph, maps, scope_dict = None):
    """
    returns the toplevel scope of a set of given *connected* maps
    """
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for map in maps:
        scopes.add(scope_dict[map])
    for scope in scopes:
        if scope_dict[scope] not in scopes:
            return scope

    raise RuntimeError("Map structure is not sound (underlying subgraph must be connected")

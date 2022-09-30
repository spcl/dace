# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
''' Subgraph Transformation Helper API '''
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView

from collections import defaultdict
import copy
from typing import List, Union, Dict, Tuple, Set

import dace.libraries.standard as stdlib

import itertools

# ****************
# Helper functions


def common_map_base_ranges(ranges: List[subsets.Range]) -> List[subsets.Range]:
    """ Finds a maximal set of ranges that can be found
        in every instance of the ranges in the given list
    """
    if len(ranges) == 0:
        return None
    # first pass: find maximal set
    range_base = [rng for rng in ranges[0]]
    for current_range in ranges:
        tmp = [rng for rng in current_range]

        range_base_new = []
        for element in tmp:
            if element in range_base:
                range_base_new.append(element)
                range_base.remove(element)

        range_base = range_base_new

    return range_base


def find_reassignment(maps: List[nodes.Map], common_ranges, offset=False) -> Dict[nodes.Map, List]:
    """ Provided a list of maps and their common base ranges
        (found via common_map_base_ranges()),
        for each map greedily assign each loop to an index so that
        a base range has the same index in every loop.
        If a loop range of a certain map does not correspond to
        a common base range, no index is assigned (=-1)


        :param maps:            List of maps
        :param common_ranges: Common ranges extracted via
                                common_map_base_ranges()
        :param offset: If true, offsets each range to 0  
                       before checking 

        :returns: Dict that maps each map to a vector with
                  the same length as number of map loops.
                  The vector contains, in order, an index
                  for each map loop that maps it to a
                  common base range or '-1' if it does not.
    """
    result = {m: None for m in maps}
    outer_ranges_dict = dict(enumerate(common_ranges))

    for m in maps:
        result_map = []
        map_range = copy.deepcopy(m.range)
        if offset:
            map_range.offset(map_range.min_element(), negative=True)
        for current_range in map_range:
            found = False
            for j, outer_range in outer_ranges_dict.items():
                if current_range == outer_range and j not in result_map:
                    result_map.append(j)
                    found = True
                    break
            if not found:
                result_map.append(-1)
        result[m] = result_map

    return result


def outermost_scope_from_subgraph(graph, subgraph, scope_dict=None):
    """
    Returns the outermost scope of a subgraph.
    If the subgraph is not connected, there might be several
    scopes that are locally outermost. In this case, it
    throws an Exception.

    """
    if scope_dict is None:
        scope_dict = graph.scope_dict()
    scopes = set()
    for element in subgraph:
        scopes.add(scope_dict[element])
    # usual case: Root of scope tree is in subgraph,
    # return None (toplevel scope)
    if None in scopes:
        return None

    toplevel_candidates = set()
    for scope in scopes:
        # search the one whose parent is not in scopes
        # that must be the top level one
        current_scope = scope_dict[scope]
        while current_scope and current_scope not in scopes:
            current_scope = scope_dict[current_scope]
        if current_scope is None:
            toplevel_candidates.add(scope)

    if len(toplevel_candidates) != 1:
        raise TypeError("There are several locally top-level nodes. "
                        "Please check your subgraph and see to it "
                        "being connected.")
    else:
        return toplevel_candidates.pop()


def outermost_scope_from_maps(graph, maps, scope_dict=None):
    """
    Returns the outermost scope of a set of given maps.
    If the underlying maps are not topologically connected
    to each other, there might be several scopes that are
    locally outermost. In this case it throws an Exception
    """
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for map in maps:
        scopes.add(scope_dict[map])
    # usual case: Root of scope tree is in subgraph,
    # return None (toplevel scope)
    if None in scopes:
        return None

    toplevel_candidates = set()
    for scope in scopes:
        current_scope = scope_dict[scope]
        while current_scope and current_scope not in scopes:
            current_scope = scope_dict[current_scope]
        if current_scope is None:
            toplevel_candidates.add(scope)

    if len(toplevel_candidates) != 1:
        raise TypeError("There are several locally top-level nodes. "
                        "Please check your subgraph and see to it "
                        "being connected.")
    else:
        return toplevel_candidates.pop()


def get_outermost_scope_maps(sdfg, graph, subgraph=None, scope_dict=None):
    """
    Returns all Map Entries inside of a given subgraph
    that have the outermost scope.
    If the underlying subgraph is not connected, there
    might be multiple locally outermost scopes. In this
    ambiguous case, the method returns an empty list.
    If subgraph == None, the whole graph is taken
    for analysis.
    """
    subgraph = graph if subgraph is None else subgraph
    if scope_dict is None:
        scope_dict = graph.scope_dict()

    # first, get the toplevel scope of the underlying subgraph
    # if not found, return empty list (ambiguous)
    try:
        outermost_scope = outermost_scope_from_subgraph(graph, subgraph, scope_dict)
    except TypeError:
        return []

    maps = [
        node for node in subgraph.nodes() if isinstance(node, nodes.MapEntry) and scope_dict[node] == outermost_scope
    ]

    return maps


def subgraph_from_maps(sdfg, graph, map_entries, scope_children=None):
    """
    Given a list of map entries in a single graph,
    return a subgraph view that includes all nodes
    inside these maps as well as map entries and exits
    as well as adjacent nodes.
    """
    if not scope_children:
        scope_children = graph.scope_children()
    node_set = set()
    for map_entry in map_entries:
        node_set |= set(scope_children[map_entry])
        node_set |= set(e.dst for e in graph.out_edges(graph.exit_node(map_entry))
                        if isinstance(e.dst, nodes.AccessNode))
        node_set |= set(e.src for e in graph.in_edges(map_entry) if isinstance(e.src, nodes.AccessNode))

        node_set.add(map_entry)

    return SubgraphView(graph, list(node_set))

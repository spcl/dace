# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg

from collections import defaultdict
import copy
from typing import List, Union, Dict, Tuple, Set

import dace.libraries.standard as stdlib

import itertools

# ****************
# Helper functions

def common_map_base_ranges(maps: List[nodes.Map]) -> List[subsets.Range]:
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
    returns the toplevel scope of a subgraph
    """
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for element in subgraph:
        scopes.add(scope_dict[element])
    for scope in scopes:
        # search the one whose parent is not in scopes
        # that must be the top level one
        current_scope = scope_dict[scope]
        while current_scope and current_scope not in scopes:
            current_scope = scope_dict[current_scope]
        if current_scope is None:
            return scope


    raise RuntimeError("Subgraph is not sound")

def toplevel_scope_maps(graph, maps, scope_dict = None):
    """
    returns the toplevel scope of a set of given maps
    """
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for map in maps:
        scopes.add(scope_dict[map])
    for scope in scopes:
        current_scope = scope_dict[scope]
        while current_scope and current_scope not in scopes:
            current_scope = scope_dict[current_scope]
        if current_scope is None:
            return scope

    raise RuntimeError("Map structure is not sound (underlying subgraph must be connected)")


def get_highest_scope_maps(sdfg, graph, subgraph = None):
    """
    returns the Map Entries of the highest scope maps
    that reside inside a given subgraph.
    If subgraph = None, the whole graph is taken
    """
    subgraph = graph if not subgraph else subgraph
    scope_dict = graph.scope_dict()

    def is_lowest_scope(node):
        while scope_dict[node]:
            if scope_dict[node] in subgraph.nodes():
                return False
            node = scope_dict[node]

        return True

    maps = [node for node in subgraph.nodes() if isinstance(node, nodes.MapEntry)
                                              and is_lowest_scope(node)]

    return maps

def are_subsets_contiguous(subset_a: subsets.Subset,
                           subset_b: subsets.Subset) -> bool:
    ''' If subsets are contiguous, return True '''
    bbunion = subsets.bounding_box_union(subset_a, subset_b)

    return all([bbsz == asz + bsz for (bbsz, asz, bsz) \
                    in zip(bbunion.size(), subset_a.bounding_box_size(), subset_b.bounding_box_size())])
    '''
    return bbunion.num_elements() == (subset_a.num_elements() +
                                      subset_b.num_elements())
    '''

def find_contiguous_subsets(
        subset_list: List[subsets.Subset]) -> Set[subsets.Subset]:
    """
    Finds the set of largest contiguous subsets in a list of subsets.
    :param subsets: Iterable of subset objects.
    :return: A list of contiguous subsets.
    """
    # Currently O(n^2) worst case.
    subset_set = set(subset_list)
    while True:
        for sa, sb in itertools.product(subset_set, subset_set):
            if sa is sb:
                continue
            if sa.covers(sb):
                subset_set.remove(sb)
                break
            elif sb.covers(sa):
                subset_set.remove(sa)
                break
            elif are_subsets_contiguous(sa, sb):
                subset_set.remove(sa)
                subset_set.remove(sb)
                subset_set.add(subsets.bounding_box_union(sa, sb))
                break
        else:  # No modification performed
            break
    return subset_set
def deduplicate(sdfg, graph, map_entry, out_connector, edges):
    ''' applies Deduplication to ALL edges coming from the same
        out_connector specified in out_connector.
        Suitable after consolidating edges at the entry node.
        WARNING: This is not guaranteed to be deterministic
    '''
    # Steps:
    # 1. Find unique subsets
    # 2. Find sets of contiguous subsets
    # 3. Create transients for subsets
    # 4. Redirect edges through new transients

    # only connector we are interested in
    conn = out_connector

    # Get original data descriptor
    edge0 = next(iter(edges))
    dname = edge0.data.data
    desc = sdfg.arrays[edge0.data.data]

    # Get unique subsets
    unique_subsets = set(e.data.subset for e in edges)

    # Find largest contiguous subsets
    contiguous_subsets = find_contiguous_subsets(unique_subsets)
    #print("Subsets:", contiguous_subsets)

    # Map original edges to subsets
    edge_mapping = defaultdict(list)
    for e in edges:
        for ind, subset in enumerate(contiguous_subsets):
            if subset.covers(e.data.subset):
                edge_mapping[ind].append(e)
                break
        else:
            raise ValueError(
                "Failed to find contiguous subset for edge %s" % e.data)

    # Create transients for subsets and redirect edges
    for ind, subset in enumerate(contiguous_subsets):
        name, _ = sdfg.add_temp_transient(subset.size(), desc.dtype)
        anode = graph.add_access(name)
        graph.add_edge(map_entry, conn, anode, None,
                       Memlet(data=dname, subset=subset))
        for e in edge_mapping[ind]:
            graph.remove_edge(e)
            new_memlet = copy.deepcopy(e.data)
            # Offset memlet to match new transient
            new_memlet.subset.offset(subset, True)
            new_edge = graph.add_edge(anode, None, e.dst, e.dst_conn,
                                      new_memlet)
            # Rename data on memlet
            for pe in graph.memlet_tree(new_edge):
                pe.data.data = name
                pe.data.subset.offset(subset, True)

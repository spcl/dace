# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Subgraph Transformation Helper API """
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView
import dace

from collections import defaultdict
import copy
from typing import List, Union, Dict, Tuple, Set, Optional
from sympy import S

import dace.libraries.standard as stdlib

import itertools

# ****************
# Helper functions


def range_eq_with_difference(this_range: Tuple[int, int, int],
                             other_range: Tuple[int, int, int],
                             max_difference_start: int = 0,
                             max_difference_end: int = 0) -> bool:
    """
    Checks if the two given ranges are equal, except for some given allowed difference in start and end values.

    :param this_range: This range, given by tuple of start, end and step
    :type this_range: Tuple[int, int, int]
    :param other_range: Other range, given by tuple of start, end and step
    :type other_range: Tuple[int, int, int]
    :param max_difference_start: Max value start values are allowed to differ, defaults to 0
    :type max_difference_start: int, optional
    :param max_difference_end: Max value end values are allowed to differ, defaults to 0
    :type max_difference_end: int, optional
    :return: If both ranges are "equal"
    :rtype: bool
    """
    if any(dace.symbolic.issymbolic(trng) != dace.symbolic.issymbolic(orng) for trng, orng in zip(this_range, other_range)):
        return False
    return abs(this_range[0] - other_range[0]) <= max_difference_start \
        and abs(this_range[1] - other_range[1]) <= max_difference_end \
        and this_range[2] == other_range[2]


def range_in_ranges_with_difference(
    this_range: Tuple[int, int, int],
    other_ranges: List[Tuple[int, int, int]],
    max_difference_start: int = 0,
    max_difference_end: int = 0
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    Checks if the given range, is inside the given list of ranges while allowing for some slight difference in start and
    end value

    :param this_range: This range, given by tuple of start, end and step
    :type this_range: Tuple[int, int, int]
    :param other_ranges: List of other ranges given by tuples of start, end and step
    :type other_ranges: List[Tuple[int, int, int]]
    :param max_difference_start: Max value start values are allowed to differ, defaults to 0
    :type max_difference_start: int, optional
    :param max_difference_end: Max value end values are allowed to differ, defaults to 0
    :type max_difference_end: int, optional
    :return: If this_range is found in other_ranges, returns range (again as tuple) with start and end values covering
    both ranges.
    :rtype: Tuple[int, int, int]
    """
    for other_range in other_ranges:
        if range_eq_with_difference(this_range, other_range, max_difference_start, max_difference_end):
            return ((min(this_range[0], other_range[0]), max(this_range[1], other_range[1]), this_range[2]),
                    other_range)
    return None


def common_map_base_ranges(ranges: List[subsets.Range],
                           max_difference_start: int = 0,
                           max_difference_end: int = 0) -> Tuple[List[subsets.Range], List[List[int]]]:
    """
    Finds a maximal set of ranges that can be found in every instance of the ranges in the given list. If start/end
    values can differ will return the range which covers all.

    :param ranges: The list of ranges
    :type ranges: List[subsets.Range]
    :param max_difference_start: Max allowed difference in start value of ranges, defaults to 0
    :type max_difference_start: int, optional
    :param max_difference_end: Max allowed difference in end value of ranges, defaults to 0
    :type max_difference_end: int, optional
    :return: List of the maximal ranges and list of the indices of the ranges included. list at position i is the
    indices for range at position i in ranges.
    :rtype: Tuple[List[subsets.Range], List[List[int]]]
    """
    indices = []
    if len(ranges) == 0:
        return ([], [])
    # first pass: find maximal set
    range_base = [rng for rng in ranges[0]]
    for current_range in ranges:
        indices.append([])
        tmp = [rng for rng in current_range]

        range_base_new = []
        for index, element in enumerate(tmp):
            # if element in range_base:
            range_in_ranges = range_in_ranges_with_difference(element, range_base, max_difference_start, max_difference_end)
            if range_in_ranges is not None:
                max_element, other_element = range_in_ranges
                range_base_new.append(max_element)
                range_base.remove(other_element)
                indices[-1].append(index)

        range_base = range_base_new

    return range_base, indices


def find_reassignment(maps: List[nodes.Map],
                      common_ranges,
                      offset=False,
                      max_difference_start: int = 0,
                      max_difference_end: int = 0) -> Dict[nodes.Map, List]:
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
        :param max_difference_start: Max value start values are allowed to differ, defaults to 0
        :type max_difference_start: int, optional
        :param max_difference_end: Max value end values are allowed to differ, defaults to 0
        :type max_difference_end: int, optional
        :return: Dict that maps each map to a vector with
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
                if range_eq_with_difference(current_range, outer_range, max_difference_start, max_difference_end) \
                        and j not in result_map:
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


def add_modulo_to_all_memlets(graph: dace.sdfg.SDFGState, data_name: str, data_shape: Tuple[dace.symbolic.symbol],
                              offsets: Optional[Tuple[dace.symbolic.symbol]] = None):
    """
    Add modulos to all memlet subset ranges in the given state and all nested SDFGs for the given data. The modulos are
    added to allow the array to shrink and be used in a cirular buffer manner. Deals with nested SDFGs by calling itself
    recursively

    :param graph: The state containing the memlets to alter
    :type graph: dace.sdfg.SDFGState
    :param data_name: Name of the array whose memlets need to be alterex
    :type data_name: str
    :param data_shape: Shape of the new array
    :type data_shape: Tuple[dace.symbolic.symbol]
    :param offsets: Offsets for each dimension, if None will be zero. Used when dealing with nested SDFGs, defaults to None
    :type offsets: Optional[Tuple[dace.symbolic.symbol]], optional
    """
    if offsets is None:
        offsets = [S.Zero] * len(data_shape)
    # Need to change each edge only once
    changed_edges = set()
    # Need to recusively call each state only once
    changed_states = set()
    for node in graph.nodes():
        if isinstance(node, nodes.AccessNode):
            for io_edge in [*graph.in_edges(node), *graph.out_edges(node)]:
                for edge in graph.memlet_tree(io_edge):
                    if edge.data.data == data_name and graph.edge_id(edge) not in changed_edges:
                        for index, (dim_size, offset) in enumerate(zip(data_shape, offsets)):

                            if isinstance(edge.dst, nodes.NestedSDFG):
                                for state in edge.dst.sdfg.states():
                                    if state not in changed_states:
                                        changed_states.add(state)
                                        add_modulo_to_all_memlets(state, data_name, data_shape,
                                                                  [start for start, _, _ in edge.data.subset.ranges])
                            if edge.data.data == data_name:
                                rng = edge.data.subset.ranges[index]
                            else:
                                rng = edge.data.other_subset.ranges[index]
                            if rng[0] == rng[1]:
                                # case that in this dimension we only need one index
                                new_range = ((offset + rng[0]) % dim_size, (offset + rng[1]) % dim_size, *rng[2:])
                            elif ((rng[1] - rng[0] + S.One).evalf(subs=graph.parent.constants) ==
                                  dim_size.evalf(subs=graph.parent.constants)):
                                # memlet goes over the whole new shape
                                new_range = (0, dim_size - S.One, *rng[2:])
                            else:
                                # Should not reach that state
                                # TODO: Add warning/error: Can not shrink size with modulo for array if there is a
                                # memlet which has a range where one index is neither the full size of the dimension
                                # or just one index
                                import sys
                                print(
                                    f"ERROR: Can not reduce size of {data_name} to {data_shape} while being used as a"
                                    f" circular buffer as edge {edge.src} -> {edge.dst} ({edge.data}) has a dimension"
                                    f" where neither the full size of dimension or just one index is used ({index}-th "
                                    f"dimension)", file=sys.stderr)
                                print(f"rng[1]-rng[0] + 1={(rng[1]-rng[0]+S.One).evalf(subs=graph.parent.constants)}, "
                                      f"dim_size: {dim_size.evalf(subs=graph.parent.constants)}")
                                assert False
                            if edge.data.data == data_name:
                                edge.data.subset.ranges[index] = new_range
                            else:
                                edge.data.other_subset.ranges[index] = new_range
                        changed_edges.add(graph.edge_id(edge))

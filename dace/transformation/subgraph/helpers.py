# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Subgraph Transformation Helper API """
import ast
from dace import dtypes, registry, symbolic, subsets, data
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView
from dace.frontend.python import astutils
import dace
import logging
from numbers import Number

from collections import defaultdict
import copy
from typing import List, Union, Dict, Tuple, Set, Optional
from sympy import S
import sympy

import dace.libraries.standard as stdlib

import itertools

logger = logging.getLogger(__name__)

# ****************
# Helper functions


def sympy_min(a: Union[Number, symbolic.SymExpr], b: Union[Number, symbolic.SymExpr]):
    if symbolic.simplify(a > b):
        return b
    else:
        return a


def sympy_max(a: Union[Number, symbolic.SymExpr], b: Union[Number, symbolic.SymExpr]):
    if symbolic.simplify(a > b):
        return a
    else:
        return b


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
    if any(symbolic.issymbolic(trng) != symbolic.issymbolic(orng) for trng, orng in zip(this_range, other_range)):
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
        try:
            if range_eq_with_difference(this_range, other_range, max_difference_start, max_difference_end):
                start = this_range[0] if this_range[0] == other_range[0] else sympy_min(this_range[0], other_range[0])
                end = this_range[1] if this_range[1] == other_range[1] else sympy_max(this_range[1], other_range[1])
                return ((start, end, this_range[2]), other_range)
        except TypeError as e:
            logger.error(e)
            logger.error("this_range: %s, other_range: %s, max_diff_start: %s, max_diff_end: %s", this_range,
                         other_range, max_difference_start, max_difference_end)


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


def adjust_data_shape_if_needed(nsdfg: SDFG, new_desc: data.Array, nname: str, memlet: Memlet):
    """
    Changes the data shape and stride in the given sdfg to match the given array desc. Only changes something if the
    shape length differs. Is used to propagate change in shape into nested SDFG.

    :param nsdfg: The (nested) SDFG to change in data in
    :type nsdfg: SDFG
    :param new_desc: The array description to change it to
    :type new_desc: data.Array
    :param nname: The name of the array in the given (nested) SDFG
    :type nname: str
    :param memlet: Memlet adjacent to the nested SDFG that leads to the  access node with the corresponding data name
    :type memlet: Memlet
    """
    # check whether array needs to change
    if len(new_desc.shape) != len(nsdfg.data(nname).shape):
        # Case where nsdfg array as fewer dimensions
        subset_copy = copy.deepcopy(memlet.subset)
        non_ones = subset_copy.squeeze()
        strides = []
        shape = []
        total_size = 1

        if non_ones:
            strides = []
            total_size = 1
            for (i, (sh, st)) in enumerate(zip(new_desc.shape, new_desc.strides)):
                if i in non_ones:
                    shape.append(sh)
                    strides.append(st)
                    total_size *= sh
                else:
                    shape.append(1)
                    strides.append(1)
        else:
            strides = [1]
            total_size = 1
            shape = [1]

        if isinstance(nsdfg.data(nname), data.Array):
            nsdfg.data(nname).strides = tuple(strides)
            nsdfg.data(nname).total_size = total_size
            nsdfg.data(nname).shape = tuple(shape)

    else:
        # Otherwise can just copy
        if isinstance(nsdfg.data(nname), data.Array):
            nsdfg.data(nname).strides = new_desc.strides
            nsdfg.data(nname).total_size = new_desc.total_size
            nsdfg.data(nname).shape = new_desc.shape


def add_modulo_to_all_memlets(graph: dace.sdfg.SDFGState, data_name: str, data_shape: Tuple[symbolic.symbol],
                              offsets: Optional[Tuple[symbolic.symbol]] = None):
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
    logger.debug("%s: graph: %s, data_shape: %s, offsets: %s", data_name, graph, data_shape, offsets)
    if offsets is None:
        offsets = [S.Zero] * len(data_shape)
    # Need to change each edge only once
    changed_edges = set()
    # Need to recusively call each state only once
    changed_states = set()
    changed_nsdfg = set()
    for node in graph.nodes():
        if isinstance(node, nodes.AccessNode):
            for io_edge in [*graph.in_edges(node), *graph.out_edges(node)]:
                for edge in graph.memlet_tree(io_edge):
                    if edge.data.data == data_name and graph.edge_id(edge) not in changed_edges:
                        if edge.data.data == data_name:
                            subset = copy.deepcopy(edge.data.subset)
                        else:
                            subset = copy.deepcopy(edge.data.other_subset)
                        remove_min_max(subset)

                        new_ranges = None
                        try:
                            new_ranges = get_range_with_modulo(subset, data_shape, offsets, graph.parent.constants)
                            logger.debug("%s, change subset from %s to %s for edge %s -> %s", data_name, subset,
                                         new_ranges, edge.src, edge.dst)

                            nsdfg = None
                            # assume that there are no direct memlets between two nsdfg, there is at least one access
                            # node between
                            if isinstance(edge.dst, nodes.NestedSDFG):
                                nsdfg = edge.dst
                            if isinstance(edge.src, nodes.NestedSDFG):
                                nsdfg = edge.src

                            if nsdfg is not None and nsdfg not in changed_nsdfg:
                                logger.debug("Found nsdf: %s", nsdfg)
                                changed_nsdfg.add(nsdfg)
                                # Make sure to add the symbols into the nested sdfg symbol map if there are any
                                # in the offset
                                symbols = []
                                for offset in offsets:
                                    symbols.extend(offset.free_symbols)
                                for symbol in symbols:
                                    if str(symbol) not in nsdfg.sdfg.symbols:
                                        nsdfg.sdfg.add_symbol(str(symbol), int)
                                    if str(symbol) not in nsdfg.symbol_mapping:
                                        nsdfg.symbol_mapping[str(symbol)] = symbol

                                adjust_data_shape_if_needed(nsdfg.sdfg, graph.parent.data(data_name), data_name,
                                                            edge.data)

                                # Compute the offset for the nested SDFG based on the memlet going into the nested SDFG
                                new_offsets = [S.Zero] * len(data_shape)
                                for idx, new_rng in enumerate(new_ranges.ranges):
                                    if new_rng[0] != new_rng[1]:
                                        new_offsets[idx] = subset.ranges[idx][0] + offsets[idx]
                                # Adjust interstate edges inside the nested SDFG
                                add_modulo_to_interstate_edges(nsdfg.sdfg, data_name, data_shape,
                                                               copy.deepcopy(new_offsets))

                                # Adjust memlets inside each state in the nested SDFG
                                for state in nsdfg.sdfg.states():
                                    if state not in changed_states:
                                        changed_states.add(state)
                                        add_modulo_to_all_memlets(state, data_name, data_shape,
                                                                  copy.deepcopy(new_offsets))

                            if edge.data.data == data_name:
                                edge.data.subset.ranges = new_ranges
                            else:
                                edge.data.other_subset.ranges = new_ranges
                            changed_edges.add(graph.edge_id(edge))

                        except AssertionError:
                            # Should not reach that state
                            logger.error(
                                f"Can not reduce size of {data_name} to {data_shape} while being used as a"
                                f" circular buffer as edge {edge.src} -> {edge.dst} ({edge.data}) has a dimension"
                                f" where neither the full size of dimension or just one index is used")


def get_range_with_modulo(rng: subsets.Range, data_shape: Tuple[symbolic.symbol], offsets: Tuple[symbolic.symbol],
                          symbols: Dict[str, int]) -> subsets.Subset:
    """
    Returns a new range where modulo operations have been added in order to shrink the array to a circular buffer. Does
    return a new range.

    :param rng: The range to change
    :type rng: subsets.Range
    :param data_shape: Shape of the new array
    :type data_shape: Tuple[dace.symbolic.symbol]
    :param offsets: Offsets for each dimension, if None will be zero. Used when dealing with nested SDFGs, defaults to None
    :type offsets: Optional[Tuple[dace.symbolic.symbol]], optional
    :param symbols: Mapping of symbols to their values used when evaluating a range
    :type symbols: Dict[str, int]
    :return: The new subset
    :rtype: subsets.Subset
    """
    new_ranges = []
    logger.debug("rng: %s", rng)
    for index, (dim_size, offset) in enumerate(zip(data_shape, offsets)):
        start, end, step = rng.ranges[index]
        if start == end:
            # case that in this dimension we only need one index
            new_range = ((offset + start) % dim_size, (offset + end) % dim_size, step)
        elif ((end - start + S.One).evalf(subs=symbols) ==
              dim_size.evalf(subs=symbols)):
            # memlet goes over the whole new shape
            new_range = (0, dim_size - S.One, step)
        else:
            logger.error(
                f"end - start + 1={(end - start + S.One).evalf(subs=symbols)}, "
                f"dim_size: {dim_size.evalf(subs=symbols)}, index: {index}"
            )
            raise AssertionError
        new_ranges.append(new_range)
    return subsets.Range(new_ranges)


class ASTModuloAdder(ast.NodeTransformer):
    def __init__(self, data_name: str, sdfg: SDFG, offsets: Tuple[symbolic.symbol], indices: Set[int] = None) -> None:
        self.data_name = data_name
        self.sdfg = sdfg
        self.offsets = offsets
        self.indices = indices

    def visit_Subscript(self, node: ast.Subscript):
        if astutils.rname(node.value) == self.data_name:
            old_rng = subsets.Range(astutils.subscript_to_slice(node, self.sdfg.arrays, without_array=True))
            data_shape = self.sdfg.arrays[self.data_name].shape
            new_rng = get_range_with_modulo(old_rng, data_shape, self.offsets, self.sdfg.constants)
            logger.debug("%s: old_rng: %s, new_rng: %s", self.data_name, old_rng, new_rng)
            return ast.copy_location(astutils.slice_to_subscript(self.data_name, new_rng), node)

        return self.generic_visit(node)


def add_modulo_to_interstate_edges(sdfg: SDFG, data_name: str, data_shape: Tuple[symbolic.symbol],
                                   offsets: Optional[Tuple[symbolic.symbol]] = None):
    """
    Add modulo operations to interstate edges so that the array can be used as a circular buffer. Does not go into
    nested SDFG. It expects that the array has already been reshaped in the SDFG given.

    :param sdfg: The SDFG containing all interstate edges to check
    :type sdfg: SDFG
    :param data_name: Name of the array to be used as a circular buffer
    :type data_name: str
    :param data_shape: The new shape of the array as a circular buffer
    :type data_shape: Tuple[symbolic.symbol]
    :param offsets: offset to add, defaults to None
    :type offsets: Optional[Tuple[symbolic.symbol]], optional
    """
    refiner = ASTModuloAdder(data_name, sdfg, offsets)
    for edge in sdfg.edges():
        if edge.data.condition.language is dtypes.Language.Python:
            for i, stmt in enumerate(edge.data.condition.code):
                edge.data.condition.code[i] = refiner.visit(stmt)
        else:
            logger.warning("Encountered edge to add modulo to whose condition is not in python. Can't change memory"
                           "access. Edge: %s", edge)


def remove_min_max(rng: subsets.Range):
    """
    Removes the min or max operation. Assumes that we only want the 2nd argument of the operation.

    :param rng: The ranges where min/max will be removed in the start/end values
    :type rng: subsets.Range
    """
    def extract(elem: dace.symbolic.SymbolicType):
        if isinstance(elem, sympy.Min) or isinstance(elem, sympy.Max):
            # we assume that the start/end value of the loop is always the first argument
            return elem.args[1]
        elif elem is not None and len(elem.args) > 0:
            new_args = []
            for e in elem.args:
                new_args.append(extract(e))
            # elem.args = tuple(new_args)
            elem_class = type(elem)
            return elem_class(*new_args)
        else:
            return elem

    while has_min_max(rng):
        for index, (start, end, step) in enumerate(rng):
            rng[index] = (extract(start), extract(end), step)


def has_min_max(rng: subsets.Range):
    """
    Returns True if there is a min/max element inside the given subset

    :param rng: The subset to check
    :type rng: subsets.Range
    """
    def elem_has_min_max(elem):
        if isinstance(elem, sympy.Max) or isinstance(elem, sympy.Min):
            return True
        elif len(elem.args) > 0:
            return any([elem_has_min_max(e) for e in elem.args])
        else:
            return False

    elems = [s for s, _, _ in rng]
    elems.extend([e for _, e, _ in rng])
    return any(elem_has_min_max(elem) for elem in elems)


def is_map_init(state: SDFGState, map_exit: nodes.MapEntry,
                array_shape: List[Union[int, dace.symbol]],
                symbols: Dict[str, int],
                found_shape_idx: List[int] = []
                ) -> Tuple[bool, List[Tuple[nodes.Map, int]]]:
    """
    Check if the given map_exit is part of the initialisation (setting to 0) of an array. Recurses into itself if there
    are several stacked maps.

    :param state: The state containing the given map
    :type state: SDFGState
    :param map_exit: The map exit to start from
    :type map_exit: nodes.MapEntry
    :param array_shape: The shape of the array initialised
    :type array_shape: List[Union[int, dace.symbol]]
    :return: True if the maps initialises, False otherwise
    :param symbols: Symbol mapping used to evaluate shape and ranges if required
    :type symobls: Dict[str, int]
    :param found_shape_idx: List of already found indices of the array_shape. Used when recursing into iself, optional,
    defaults to []
    :type found_shape_idx: List[int]
    :rtype: bool
    """

    # Init maps can only be stacked if there are no other edges, except one going to the next map
    if len(state.in_edges(map_exit)) > 1:
        return (False, [])

    # All map ranges need to correspond to an array dimension
    for rng, itervar in zip(map_exit.map.range, map_exit.map.params):
        # step must be 1
        if rng[2] != 1:
            return (False, [])
        memlet = state.in_edges(map_exit)[0].data

        # Get the dimension indices where the iteration variable is used
        memlet_idx_itervar = []
        for idx, (memlet_start, memlet_end, memlet_step) in enumerate(memlet.subset):
            if memlet_step != 1:
                return (False, [])
            if str(itervar) in str(memlet_start):
                memlet_idx_itervar.append(idx)

        # Expect that only one memlet dimension uses the itervar
        if len(memlet_idx_itervar) != 1:
            return (False, [])

        # The map range needs to correspond to the array shape at the dimension found before using the memlet
        if (dace.symbolic.evaluate_if_possible(rng[1] - rng[0] + 1, symbols) !=
                dace.symbolic.evaluate_if_possible(array_shape[memlet_idx_itervar[0]], symbols)):
            return (False, [])
        else:
            found_shape_idx.append(memlet_idx_itervar[0])

    # Go to the next node
    next_node = state.in_edges(map_exit)[0].src
    if isinstance(next_node, nodes.MapExit):
        is_init, maps = is_map_init(state, next_node, array_shape, symbols)
        maps.append((map_exit.map, memlet_idx_itervar[0]))
        return (is_init, maps)
    elif isinstance(next_node, nodes.Tasklet):
        # Check that all dimensions which are bigger than one have been initialised
        found_all_dimensions = True
        for idx in range(len(array_shape)):
            if (idx not in found_shape_idx and dace.symbolic.evaluate_if_possible(array_shape[idx], symbols) != 1.0):
                found_all_dimensions = False

        # For it to be an init map, the tasklet needs to set to 0 and all dimensions need to be covered
        return ((next_node.code.as_string.split('= ')[1] == '0.0' and found_all_dimensions), [(map_exit.map,
                                                                                               memlet_idx_itervar[0])])
    return (False, [])

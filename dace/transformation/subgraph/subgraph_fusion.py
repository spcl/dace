# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement subgraph fusion.    """
import dace
import networkx as nx

from dace import dtypes, registry, symbolic, subsets, data
from dace.sdfg import nodes, utils, replace, SDFG, scope_contains_scope
from dace.sdfg.state import SDFGState
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.graph import SubgraphView
from dace.sdfg.scope import ScopeTree
from dace.memlet import Memlet
from dace.transformation import transformation
from dace.properties import EnumProperty, ListProperty, make_properties, Property, CodeBlock
from dace.symbolic import overapproximate
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlet, propagate_memlets_scope, _propagate_node, \
                                  propagate_memlets_state
from dace.transformation.subgraph import helpers
from dace.transformation.dataflow import RedundantArray
from dace.sdfg.utils import consolidate_edges_scope, get_view_node
from dace.transformation.helpers import find_contiguous_subsets, nest_state_subgraph

from copy import deepcopy as dcpy
from typing import List, Union, Tuple, Optional, Callable, Dict
import warnings
from sympy import S
import sympy
import logging

import dace.libraries.standard as stdlib

from collections import defaultdict
from itertools import chain

logger = logging.getLogger(__name__)


@make_properties
class SubgraphFusion(transformation.SubgraphTransformation):
    """
    Implements the SubgraphFusion transformation.
    Fuses together the maps contained in the subgraph and pushes inner nodes
    into a global outer map, creating transients and new connections
    where necessary.

    SubgraphFusion requires all lowest scope level maps in the subgraph
    to have the same indices and parameter range in every dimension.
    This can be achieved using the MultiExpansion transformation first.
    Reductions can also be expanded using ReduceExpansion as a
    preprocessing step.
    """

    debug = Property(desc="Show debug info", dtype=bool, default=False)

    transient_allocation = EnumProperty(dtype=dtypes.StorageType,
                                        desc="Storage Location to push transients to that are "
                                        "fully contained within the subgraph.",
                                        default=dtypes.StorageType.Default)

    schedule_innermaps = Property(desc="Schedule of inner maps. If none, "
                                  "keeps schedule.",
                                  dtype=dtypes.ScheduleType,
                                  default=None,
                                  allow_none=True)
    consolidate = Property(desc="Consolidate edges that enter and exit the fused map.", dtype=bool, default=False)

    propagate = Property(desc="Propagate memlets of edges that enter and exit the fused map."
                         "Disable if this causes problems (e.g., if memlet propagation does"
                         "not work correctly).",
                         dtype=bool,
                         default=True)

    disjoint_subsets = Property(desc="Check for disjoint subsets in can_be_applied. If multiple"
                                "access nodes pointing to the same data appear within a subgraph"
                                "to be fused, this check confirms that their access sets are"
                                "independent per iteration space to avoid race conditions.",
                                dtype=bool,
                                default=True)

    keep_global = ListProperty(
        str,
        desc="A list of array names to treat as non-transients and not compress",
    )

    max_difference_start = Property(dtype=int, desc="Max difference between start of ranges of maps", default=0)
    max_difference_end = Property(dtype=int, desc="Max difference between end of ranges of maps", default=1)
    change_init_outside = Property(dtype=int,
                                   desc="Changes arraysizes even if it is initialised outside the current state. "
                                        "Experimental",
                                   default=False)
    is_map_sequential = Property(
        dtype=Callable,
        desc="Function to call which returns if the map should be treated as a sequential map",
        default=lambda _: False)

    fixed_new_shapes = Property(
            dtype=Dict,
            desc="Dictionary with array names and their new shapes, if they are compressible. If an array is not listed "
            "here will compute the new shape automatically.",
            default={})

    blacklisted_arrays = ['ZPFPLSX', 'ZLIQFRAC', 'ZPFPLSX', 'ZFOEALFA', 'ZAORIG', 'ZQSLIQ', 'ZLNEG', 'ZFOEEW', 'ZFOEEWMT', 'ZQX0', 'ZA',
                          'ZQX', 'ZQSICE', 'ZICEFRAC', 'ZQXN2D']

    def _map_ranges_compatible(self, this_map: nodes.Map, other_map: nodes.Map) -> bool:
        for rng, orng in zip(this_map.range, other_map.range):
            if abs(rng[0] - orng[0]) > self.max_difference_start or \
                    abs(rng[1] - orng[1]) > self.max_difference_end or \
                    rng[2] != orng[2]:
                return False
        return True

    def _check_memlet_sizes_for_circular_buffer(self, graph: SDFGState, data_name: str, shape:
                                                Tuple[dace.symbolic.symbol]) -> bool:
        """
        Check that for all memlets with the given data, the subset range is for each dimension either one index or the
        whole size of the dimension as given by the shape.

        :param graph: The state to look for all edges
        :type graph: SDFGState
        :param data_name: The name of the data
        :type data_name: str
        :param shape: The shape of the dara array
        :type shape: Tuple[dace.symbolic.symbol]
        :return: True if condition is met, otherwise false
        :rtype: bool
        """
        check = True
        for edge in graph.edges():
            if isinstance(edge, Memlet):
                if edge.data.data == data_name:
                    if edge.data.data == data_name:
                        relevant_subset = edge.subset.ranges
                    else:
                        relevant_subset = edge.other_subset.ranges

                    if isinstance(edge.dst, nodes.NestedSDFG):
                        for state in edge.dst.sdfg.states():
                            nsdfg_shape = dcpy(shape)
                            # Remove dimensions from shape where just one index is given
                            for idx in len(shape):
                                if relevant_subset[0] == relevant_subset[1]:
                                    nsdfg_shape.remove(nsdfg_shape[idx])
                            check = check or self._check_memlet_sizes_for_circular_buffer(state, data_name,
                                                                                          nsdfg_shape)
                    check = check or all(rng[1] - rng[0] == 0 or rng[1] - rng[0] + S.One == size for rng,
                                         size in zip(relevant_subset, shape))
        return check

    def _can_intermediate_array_be_transformed(
            self, sdfg: SDFG, graph: SDFGState, node: nodes.AccessNode,
        lower_subset: subsets.Range, union_upper: subsets.Range,
        lower_map_ranges: subsets.Range
    ) -> Optional[List[dace.symbolic.symbol]]:
        """
        Checks whether the given AccessNode can be transformed to a smaller array eventhough incoming subset does not
        cover outgoing. This by buffering intermediate results. For this the map ranges need to match the difference in
        the subsets.

        :param graph: The state the AccessNode is in
        :type graph: SDFGState
        :param node: The AccessNode
        :type node: nodes.AccessNode
        :param lower_subset: The range subsets going away from the AccessNode
        :type lower_subset: subsets.Range
        :param union_upper: The union of all incoming range subsets
        :type union_upper: subsets.Range
        :return: None if not possible to transform, other list the sizes the array needs to be bigger than 1 for
        buffering. Index in list gives dimension index.
        :rtype: Optional[List[dace.symbolic.symbol]]
        """
        # Assume lower and upper subset are a range with possible multiple dimensions.
        incoming_maps = set([e.src for e in graph.in_edges(node)])
        # outgoing_maps = set([e.dst for e in graph.out_edges(node)])

        # Somehow compute the cover/min/max of in/out ranges
        in_range = incoming_maps.pop().map.range
        for map in incoming_maps:
            in_range = subsets.union(in_range, map.map.range)

        differences = [0] * len(lower_subset.ranges)

        for index, (lower_rng, upper_rng, lower_map_range) in enumerate(zip(lower_subset, union_upper, lower_map_ranges)):
            # Compute difference, in what is not covered
            diff_start_rng = upper_rng[0] - lower_rng[0]
            if dace.symbolic.issymbolic(diff_start_rng):
                diff_start_rng = diff_start_rng.evalf(subs=sdfg.constants)
            # diff_end_rng = lower_rng[1] - upper_rng[1]
            diff_end_rng = upper_rng[1] - lower_rng[1]
            if dace.symbolic.issymbolic(diff_end_rng):
                diff_end_rng = diff_end_rng.evalf(subs=sdfg.constants)

            # Compute the difference in the map ranges
            # diff_start_map = out_range.ranges[index][0] - in_range.ranges[index][0]
            diff_start_map = lower_map_range[0] - in_range.ranges[index][0]
            if dace.symbolic.issymbolic(diff_start_map):
                diff_start_map = diff_start_map.evalf(subs=sdfg.constants)
            diff_end_map = in_range.ranges[index][1] - lower_map_range[1]
            if dace.symbolic.issymbolic(diff_end_map):
                diff_end_map = diff_end_map.evalf(subs=sdfg.constants)
            try:
                # Check that upper range start has higher values than lowers start and same for end
                # Also check that the map ends cover each other and the start difference must be bigger or equal than
                # the start difference of the ranges
                if diff_start_rng < 0 or diff_end_rng < 0 or diff_start_map < diff_start_rng or diff_end_map < 0:
                    # Check that the difference in maps covers the difference in the memlet ranges
                    logger.debug("Can't transform %s: diff_start_rng: %s, diff_end_rng: %s, diff_start_map: %s, "
                                 "diff_end_map: %s", node.data, diff_start_rng, diff_end_rng, diff_start_map,
                                 diff_end_map)
                    logger.debug("For lower_rng: %s, upper_rng: %s, lower_map_range: %s, in_range: %s", lower_rng,
                                 upper_rng, lower_map_range, in_range)
                    return None
            except TypeError as e:
                logger.warning("Got error when comparing: %s. Most likely due to symbols. Give up on computing the "
                               "difference", e)
                logger.warning("diff_start_map: %s, diff_start_rng: %s, diff_end_map: %s, diff_end_rng: %s",
                               diff_start_map, diff_start_rng, diff_end_map, diff_end_rng)
                return None

            # Size of circular buffer needs to accommodate the difference in the ranges between upper and lower
            differences[index] = diff_start_rng + upper_rng[1] - upper_rng[0]
            logger.debug("Set differences[%i] to %s = %s + %s - %s", index, differences[index], diff_start_rng,
                         upper_rng[1], upper_rng[0])

        return differences

    def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
        """
        Fusible if

            1. Maps have the same access sets and ranges in order
            2. Any nodes in between two maps are AccessNodes only, without WCR
               There is at most one AccessNode only on a path between two maps,
               no other nodes are allowed
            3. The exiting memlets' subsets to an intermediate edge must cover
               the respective incoming memlets' subset into the next map.
               Also, as a limitation, the union of all exiting memlets'
               subsets must be contiguous.
            4. Check for any disjoint accesses of arrays.
        """
        # get graph
        graph = subgraph.graph
        for node in subgraph.nodes():
            if node not in graph.nodes():
                return False

        # next, get all the maps
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        maps = [map_entry.map for map_entry in map_entries]

        # 1. basic checks:
        # 1.1 we need to have at least two maps
        if len(maps) <= 1:
            return False

        logger.debug("Check for maps %s", maps)
        # 1.2 check whether all maps are the same
        base_map = maps[0]
        for map in maps:
            if map.get_param_num() != base_map.get_param_num():
                logger.debug("Rejected: Number of parameters does not match")
                return False
            if not all([p1 == p2 for (p1, p2) in zip(map.params, base_map.params)]):
                logger.debug("Rejected: Map parameters are name differently")
                return False
            if not self._map_ranges_compatible(map, base_map):
                logger.debug("Rejected: Map ranges are not compatible")
                return False

        # 1.3 check whether all map entries have the same schedule
        schedule = map_entries[0].schedule
        if not all([entry.schedule == schedule for entry in map_entries]):
            logger.debug("Rejected: Map schedules differ. Schedules: %s", {entry.map: entry.schedule for entry in
                                                                           map_entries})
            return False

        # 2. check intermediate feasiblility
        # see map_fusion.py for similar checks
        # with the restrictions below being more relaxed

        # 2.1 do some preparation work first:
        # calculate node topology (see apply for definition)
        try:
            node_config = SubgraphFusion.get_adjacent_nodes(sdfg, graph, map_entries)
        except NotImplementedError:
            return False
        in_nodes, intermediate_nodes, out_nodes = node_config

        # 2.2 topological feasibility:
        if not SubgraphFusion.check_topo_feasibility(sdfg, graph, map_entries, intermediate_nodes, out_nodes):
            return False

        # 2.3 memlet feasibility
        # For each intermediate node, look at whether inner adjacent
        # memlets of the exiting map cover inner adjacent memlets
        # of the next entering map.
        # We also check for any WCRs on the fly.
        try:
            invariant_dimensions = self.determine_invariant_dimensions(sdfg, graph, intermediate_nodes, map_entries,
                                                                       map_exits)
        except NotImplementedError:
            return False

        logger.debug("Invariant dimensions: %s", invariant_dimensions)

        # Dict with arrays which can not be completely removed but need to be shrank
        self.arrays_as_circular_buffer = {}
        for node in intermediate_nodes:
            upper_subsets = set()
            lower_subsets = set()
            # First, determine which dimensions of the memlet ranges
            # change with the map, we do not need to care about the other dimensions.
            dims_to_discard = invariant_dimensions[node.data]
            # find upper_subsets and corresponding map ranges
            for in_edge in graph.in_edges(node):
                # first check for WCRs
                if in_edge.data.wcr:
                    # check whether the WCR is actually produced at
                    # this edge or further up in the memlet path. If not,
                    # we can still fuse!
                    in_in_edge = graph.memlet_path(in_edge)[-2]
                    subset_params = set([str(s) for s in in_in_edge.data.subset.free_symbols])
                    if any([p not in subset_params for p in in_edge.src.map.params]):
                        logger.debug("Rejected: Free symbol not defined in map parameters")
                        return False
                if in_edge.src in map_exits:
                    for iedge in graph.in_edges(in_edge.src):
                        if iedge.dst_conn[2:] == in_edge.src_conn[3:]:
                            subset_to_add = dcpy(iedge.data.subset if iedge.data.data ==
                                                 node.data else iedge.data.other_subset)

                            subset_to_add.pop(dims_to_discard)
                            logger.debug("Add to upper subsets %s, node: %s, oedge: %s", subset_to_add, node, iedge)
                            upper_subsets.add(subset_to_add)
                else:
                    warnings.warn("SubgraphFusion::Nodes between two maps to be"
                                  "fused with incoming edges"
                                  "from outside the maps are not"
                                  "allowed yet.")
                    return False

            # find lower_subsets
            for out_edge in graph.out_edges(node):
                if out_edge.dst in map_entries:
                    for oedge in graph.out_edges(out_edge.dst):
                        if oedge.src_conn and oedge.src_conn[3:] == out_edge.dst_conn[2:]:
                            subset_to_add = dcpy(oedge.data.subset if oedge.data.data ==
                                                 node.data else oedge.data.other_subset)
                            subset_to_add.pop(dims_to_discard)
                            # Also want to know the map range to which the lower subset is associated
                            map_ranges = subsets.Range(out_edge.dst.map.range.ranges)
                            # Would probably have to figure out which dimensions of the memlet the maps covers
                            # Maybe it will never cover an ignored dimensions???
                            # map_ranges.pop(dims_to_discard)
                            logger.debug("Add to lower subset %s, node: %s, oedge: %s with map range: %s and "
                                         "discarded dimensions: %s",
                                         subset_to_add.ranges, node, oedge, map_ranges, dims_to_discard)
                            lower_subsets.add((subset_to_add, map_ranges))

            # We assume that upper_subsets are contiguous
            # Check for this.
            try:
                contiguous_upper = find_contiguous_subsets(upper_subsets)
                if len(contiguous_upper) > 1:
                    return False
            except TypeError:
                warnings.warn('SubgraphFusion::Could not determine whether subset is continuous.'
                              'Exiting Check with False.')
                return False

            # now take union of upper subsets
            logger.debug("Take union of upper subsets: %s", upper_subsets)
            upper_iter = iter(upper_subsets)
            union_upper = next(upper_iter)
            for subs in upper_iter:
                union_upper = subsets.union(union_upper, subs)
                if not union_upper:
                    # something went wrong using union -- we'd rather abort
                    logger.error("Something went wrong when taking tha union. union_upper: %s, subs: %s", union_upper,
                                 subs)
                    return False

            # finally check coverage
            # every lower subset must be completely covered by union_upper
            logger.debug("going through the lower subsets: %s", lower_subsets)
            for lower_subset, lower_map_ranges in lower_subsets:
                if not union_upper.covers(lower_subset):
                    logger.debug("Check if intermediate node %s can be transformed with lower subset %s and upper %s",
                            node, lower_subset.ranges, union_upper.ranges)
                    differences = self._can_intermediate_array_be_transformed(sdfg, graph, node, lower_subset,
                                                                              union_upper, lower_map_ranges)
                    if differences is not None:
                        logger.debug("%s: Use as circular buffer with difference %s", node.data, differences)
                        if node.data in self.arrays_as_circular_buffer:
                            for index, diff in differences:
                                self.arrays_as_circular_buffer[
                                    node.data][index] = max(
                                        self.arrays_as_circular_buffer[
                                            node.data][index], diff)
                        else:
                            # TODO: Cover case where a memlet has a subset of a dimension volume >1
                            # add ignored dimensions to differences
                            for idx in dims_to_discard:
                                differences.insert(idx, 0)
                            self.arrays_as_circular_buffer[node.data] = differences
                    else:
                        logger.debug("Rejected: subsets don't cover each other. Union upper: %s, lower_subset: %s",
                                union_upper, lower_subset)
                        return False
            if node.data in self.arrays_as_circular_buffer:
                if not self._check_memlet_sizes_for_circular_buffer(graph, node.data, self.arrays_as_circular_buffer):
                    logger.debug("Rejected: Memlet sizes wrong for circular buffer for node datra: %s")
                    return False

        logger.debug("Check 2.4")
        # 2.4 Check for WCRs in out nodes: If there is one, the corresponding
        # data must never be accessed anywhere else
        intermediate_data = set([n.data for n in intermediate_nodes])
        in_data = set([n.data for n in in_nodes if isinstance(n, nodes.AccessNode)])
        out_data = set([n.data for n in out_nodes if isinstance(n, nodes.AccessNode)])

        view_nodes = set()
        for node in chain(in_nodes, out_nodes, intermediate_nodes):
            if isinstance(node, nodes.AccessNode):
                is_view = isinstance(sdfg.data(node.data), dace.data.View)
                for edge in chain(graph.in_edges(node), graph.out_edges(node)):
                    for e in graph.memlet_tree(edge):
                        if isinstance(e.dst, nodes.AccessNode) and (is_view or isinstance(
                                sdfg.data(e.dst.data), dace.data.View)):
                            view_nodes.add(e.dst)
                        if isinstance(e.src, nodes.AccessNode) and (is_view or isinstance(
                                sdfg.data(e.src.data), dace.data.View)):
                            view_nodes.add(e.src)

        view_data = set([n.data for n in view_nodes])

        for out_node in out_nodes:
            for in_edge in graph.in_edges(out_node):
                if in_edge.src in map_exits and in_edge.data.wcr:
                    if in_edge.data.data in in_data or in_edge.data.data in intermediate_data or in_edge.data.data in view_data:
                        logger.debug("Rejected: WCR in out nodes")
                        return False

        # Check compressibility for each intermediate node -- this is needed in the following checks
        is_compressible, _ = SubgraphFusion.determine_compressible_nodes(sdfg, graph, intermediate_nodes, map_entries,
                                                                         map_exits)

        logger.debug("Check 2.5")
        # 2.5 Intermediate Arrays must not connect to ArrayViews
        for n in intermediate_nodes:
            if is_compressible[n.data]:
                for out_edge in graph.out_edges(n):
                    for e in graph.memlet_tree(out_edge):
                        if isinstance(e.dst, nodes.AccessNode) and isinstance(sdfg.data(e.dst.data), dace.data.View):
                            warnings.warn("SubgraphFusion::View Node Compression not supported!")
                            return False
                for in_edge in graph.in_edges(n):
                    for e in graph.memlet_tree(in_edge):
                        if isinstance(e.src, nodes.AccessNode) and isinstance(sdfg.data(e.src.data), dace.data.View):
                            warnings.warn("SubgraphFusion::View Node Compression not supported")
                            return False

        logger.debug("Check 2.6: %s", self.disjoint_subsets)
        # 2.6 Check for disjoint accesses for arrays that cannot be compressed
        if self.disjoint_subsets == True:
            container_dict = defaultdict(list)
            for node in chain(in_nodes, intermediate_nodes, out_nodes):
                if isinstance(node, nodes.AccessNode):
                    container_dict[node.data].append(node)

            # Check for read/write dependencies between input and output nodes
            # NOTE: Is it valid for a MapExit to be an output node? (empty memlet maybe?)
            outputs = set(n.data for n in out_nodes if isinstance(n, nodes.AccessNode))
            from dace.transformation.interstate import StateFusion
            # Ignore this for sequential maps.
            if not all(self.is_map_sequential(map) for m in maps):
                for node in in_nodes:
                    if isinstance(node, nodes.AccessNode) and node.data in outputs:
                        matching_outputs = [n for n in out_nodes if n.data == node.data]
                        # Overall ranges overlap: potential data race
                        if StateFusion.memlets_intersect(graph, [node], True, graph, matching_outputs, False):
                            # Check memlet leaves in more detail
                            in_leaves = [l for e in graph.out_edges(node) for l in graph.memlet_tree(e).leaves()]
                            out_leaves = [
                                l for n in matching_outputs for e in graph.in_edges(n)
                                for l in graph.memlet_tree(e).leaves()
                            ]
                            # All-pairs check. If memlets are equal then there are no races.
                            # If they are not, and we cannot know whether they intersect or they do, we do not match.
                            for ea in in_leaves:
                                for eb in out_leaves:
                                    if ea.data.src_subset == eb.data.dst_subset:  # Equal - no data race
                                        continue
                                    logger.debug("Rejected: Potential data race")
                                    return False  # Otherwise - potential data race

            for (node_data, compressible) in is_compressible.items():
                # we only care about disjoint subsets...
                # 1. if the array is not compressible
                if not compressible:
                    # 2. if there are multiple containers appearing pointing to the same data
                    if len(container_dict[node_data]) > 1:
                        # retrieve map inner access sets of all access nodes appearing within the subgraph

                        access_set = None
                        for node in container_dict[node_data]:
                            for e in graph.out_edges(node):
                                if e.dst in map_entries:
                                    # get corresponding inner memlet and join its subset to our access set
                                    for oe in graph.out_edges(e.dst):
                                        if oe.src_conn[3:] == e.dst_conn[2:]:
                                            current_subset = dcpy(oe.data.subset)
                                            current_subset.pop(invariant_dimensions[node_data])

                                            access_set = subsets.union(access_set, current_subset)
                                            if access_set is None:
                                                warnings.warn("SubgraphFusion::Disjoint Access found")
                                                logger.warn("SubgraphFusion::Disjoint Access found v1 on node %s with "
                                                            "edges %s and %s", node, e, oe)
                                                return False
                            for e in graph.in_edges(node):
                                if e.src in map_exits:
                                    for ie in graph.in_edges(e.src):
                                        # get corresponding inner memlet and join its subset to our access set
                                        if ie.dst_conn[2:] == e.src_conn[3:]:
                                            current_subset = dcpy(ie.data.subset)
                                            current_subset.pop(invariant_dimensions[node_data])

                                            access_set = subsets.union(access_set, current_subset)
                                            if access_set is None:
                                                warnings.warn("SubgraphFusion::Disjoint Access found")
                                                logger.warn("SubgraphFusion::Disjoint Access found v2 on node %s with "
                                                            "edges %s and %s", node, e, ie)
                                                return False

                        # compare iteration space i_d and i_d-1 in each dimension,
                        # where i_d is the iteration variable of the respective dimension
                        # if there is any intersection in any dimension, return False
                        subset_plus = dcpy(access_set)
                        subset_minus = dcpy(access_set)
                        repl_dict = {
                            symbolic.pystr_to_symbolic(f'{param}'): symbolic.pystr_to_symbolic(f'{param}-1')
                            for param in map_entries[0].params
                        }  # e.g., ['i' -> 'i-1']
                        subset_minus.replace(repl_dict)

                        for (rng, orng) in zip(subset_plus, subset_minus):
                            rng_1dim = subsets.Range((rng, ))
                            orng_1dim = subsets.Range((orng, ))
                            try:
                                intersection = rng_1dim.intersects(orng_1dim)
                            except TypeError as e:
                                logger.error("type error in intersetction in 2.6: %s", e)
                                logger.error("rng_1dim: %s, orng_1dim: %s", rng_1dim, orng_1dim)
                                return False
                            if intersection is None or intersection == True:
                                warnings.warn("SubgraphFusion::Disjoint Accesses found!")
                                logger.warn("SubgraphFusion::Disjoint Accesses found v3 on node %s with ranges %s and "
                                            "%s", node, rng, orng)
                                return False

        logger.debug("Can be applied successful")
        logger.debug("Maps end: %s", maps)
        return True

    @staticmethod
    def get_adjacent_nodes(
            sdfg, graph, map_entries) -> Tuple[List[nodes.AccessNode], List[nodes.AccessNode], List[nodes.AccessNode]]:
        """ 
        For given map entries, finds a set of in, out and intermediate nodes as defined below

        :param sdfg: SDFG
        :param graph: State of interest
        :param map_entries: List of all outermost scoped maps that induce the subgraph 
        :return: Tuple of (in_nodes, intermediate_nodes, out_nodes)
        
        - In_nodes are nodes that serve as pure input nodes for the map entries 
        - Out nodes are nodes that serve as pure output nodes for the map entries
        - Interemdiate nodes are nodes that serve as buffer storage between outermost scoped map entries and exits 
          of the induced subgraph 

        -> in_nodes are trivially disjoint from the other two types of access nodes
        -> Intermediate_nodes and out_nodes are not necessarily disjoint

        """

        # Nodes that flow into one or several maps but no data is flowed to them from any map
        in_nodes = set()

        # Nodes into which data is flowed but that no data flows into any map from them
        out_nodes = set()

        # Nodes that act as intermediate node - data flows from a map into them and then there
        # is an outgoing path into another map
        intermediate_nodes = set()

        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        for map_entry, map_exit in zip(map_entries, map_exits):
            for edge in graph.in_edges(map_entry):
                in_nodes.add(edge.src)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if len(graph.out_edges(current_node)) == 0:
                    out_nodes.add(current_node)
                else:
                    for dst_edge in graph.out_edges(current_node):
                        if dst_edge.dst in map_entries:
                            # add to intermediate_nodes
                            intermediate_nodes.add(current_node)

                        else:
                            # add to out_nodes
                            out_nodes.add(current_node)

        # any intermediate_nodes currently in in_nodes shouldn't be there
        in_nodes -= intermediate_nodes

        for node in intermediate_nodes:
            for e in graph.in_edges(node):
                if e.src not in map_exits:
                    warnings.warn("SubgraphFusion::Nodes between two maps to be"
                                  "fused with *incoming* edges"
                                  "from outside the maps are not"
                                  "allowed yet.")
                    raise NotImplementedError()

        return (in_nodes, intermediate_nodes, out_nodes)

    @staticmethod
    def check_topo_feasibility(sdfg, graph, map_entries, intermediate_nodes, out_nodes):
        """ 
        Checks whether given outermost scoped map entries have topological structure apt for fusion

        :param sdfg: SDFG 
        :param graph: State 
        :param map_entries: List of outermost scoped map entries induced by subgraph 
        :param intermediate_nodes: List of intermediate access nodes 
        :param out_nodes: List of outgoing access nodes 
        :return: Boolean value indicating fusibility 
        """
        # For each intermediate and out node: must never reach any map
        # entry if it is not connected to map entry immediately

        # for memoization purposes
        visited = set()

        def visit_descendants(graph, node, visited, map_entries):
            # check whether the node has already been processed once
            if node in visited:
                return True
            # check whether the node is in our map entries.
            if node in map_entries:
                return False
            # for every out edge, continue exploring whether
            # we and up at another map entry that is in our set
            for oedge in graph.out_edges(node):
                if not visit_descendants(graph, oedge.dst, visited, map_entries):
                    return False

            # this node does not lead to any other map entries, add to visited
            visited.add(node)
            return True

        for node in intermediate_nodes | out_nodes:
            # these nodes must not lead to a map entry
            nodes_to_check = set()
            for oedge in graph.out_edges(node):
                if oedge.dst not in map_entries:
                    nodes_to_check.add(oedge.dst)

            for forbidden_node in nodes_to_check:
                if not visit_descendants(graph, forbidden_node, visited, map_entries):
                    return False

        return True

    def get_invariant_dimensions(self, sdfg: dace.sdfg.SDFG, graph: dace.sdfg.SDFGState,
                                 map_entries: List[nodes.MapEntry], map_exits: List[nodes.MapExit],
                                 node: nodes.AccessNode):
        """
        For a given intermediate access node, return a set of indices that correspond to array / subset dimensions in which no change is observed 
        upon propagation through the corresponding map nodes in map_entries / map_exits.

        :param map_entries: List of outermost scoped map entries 
        :param map_exits: List of corresponding exit nodes to map_entries, in order 
        :param node: Intermediate access node of interest 
        :return: Set of invariant integer dimensions 
        """
        variant_dimensions = set()
        subset_length = -1

        for in_edge in graph.in_edges(node):
            if in_edge.src in map_exits:
                other_edge = graph.memlet_path(in_edge)[-2]
                other_subset = other_edge.data.subset \
                               if other_edge.data.data == node.data \
                               else other_edge.data.other_subset

                for (idx, (ssbs1, ssbs2)) \
                    in enumerate(zip(in_edge.data.subset, other_subset)):
                    if ssbs1 != ssbs2:
                        variant_dimensions.add(idx)
            else:
                warnings.warn("SubgraphFusion::Nodes between two maps to be"
                              "fused with *incoming* edges"
                              "from outside the maps are not"
                              "allowed yet.")

            if subset_length < 0:
                subset_length = other_subset.dims()
            else:
                assert other_subset.dims() == subset_length

        for out_edge in graph.out_edges(node):
            if out_edge.dst in map_entries:
                for other_edge in graph.out_edges(out_edge.dst):
                    if other_edge.src_conn and other_edge.src_conn[3:] == out_edge.dst_conn[2:]:
                        other_subset = other_edge.data.subset \
                                       if other_edge.data.data == node.data \
                                       else other_edge.data.other_subset
                        for (idx, (ssbs1, ssbs2)) in enumerate(zip(out_edge.data.subset, other_subset)):
                            if ssbs1 != ssbs2:
                                variant_dimensions.add(idx)
                        assert other_subset.dims() == subset_length

        invariant_dimensions = set([i for i in range(subset_length)]) - variant_dimensions
        return invariant_dimensions

    def copy_edge(self,
                  graph,
                  edge,
                  new_src=None,
                  new_src_conn=None,
                  new_dst=None,
                  new_dst_conn=None,
                  new_data=None,
                  remove_old=False):
        """
        Copies an edge going from source to dst.
        If no destination is specified, the edge is copied with the same
        destination and port as the original edge, else the edge is copied
        with the new destination and the new port.
        If no source is specified, the edge is copied with the same
        source and port as the original edge, else the edge is copied
        with the new source and the new port
        If remove_old is specified, the old edge is removed immediately
        If new_data is specified, inserts new_data as a memlet, else
        else makes a deepcopy of the current edges memlet
        """
        data = new_data if new_data else dcpy(edge.data)
        src = edge.src if new_src is None else new_src
        src_conn = edge.src_conn if new_src is None else new_src_conn
        dst = edge.dst if new_dst is None else new_dst
        dst_conn = edge.dst_conn if new_dst is None else new_dst_conn

        ret = graph.add_edge(src, src_conn, dst, dst_conn, data)

        if remove_old:
            graph.remove_edge(edge)
        return ret

    def adjust_arrays_nsdfg(self, sdfg: dace.sdfg.SDFG, nsdfg: nodes.NestedSDFG, name: str, nname: str, memlet: Memlet):
        """
        DFS to replace strides and volumes of data that exhibits nested SDFGs 
        adjacent to its corresponding access nodes, applied during post-processing 
        of a fused graph. Operates in-place.

        :param sdfg: SDFG 
        :param nsdfg: The Nested SDFG of interest 
        :param name: Name of the array in the SDFG 
        :param nname: Name of the array in the nested SDFG 
        :param memlet: Memlet adjacent to the nested SDFG that leads to the 
                       access node with the corresponding data name
        """
        # check whether array needs to change
        helpers.adjust_data_shape_if_needed(nsdfg, sdfg.data(name), nname, memlet)

        # traverse the whole graph and search for arrays
        for ngraph in nsdfg.nodes():
            for nnode in ngraph.nodes():
                if isinstance(nnode, nodes.AccessNode) and nnode.label == nname:
                    # trace and recurse if necessary
                    for e in chain(ngraph.out_edges(nnode), ngraph.in_edges(nnode)):
                        for te in ngraph.memlet_tree(e):
                            # if te.data.data == nnode.data:
                            #     te.data.subset.offset(min_offset, True)
                            # elif te.data.other_subset:
                            #     te.data.other_subset.offset(min_offset, True)
                            if isinstance(te.dst, nodes.NestedSDFG):
                                self.adjust_arrays_nsdfg(nsdfg, te.dst.sdfg, nname, te.dst_conn, te.data)
                            if isinstance(te.src, nodes.NestedSDFG):
                                self.adjust_arrays_nsdfg(nsdfg, te.src.sdfg, nname, te.src_conn, te.data)

    @staticmethod
    def determine_compressible_nodes(sdfg: dace.sdfg.SDFG,
                                     graph: dace.sdfg.SDFGState,
                                     intermediate_nodes: List[nodes.AccessNode],
                                     map_entries: List[nodes.MapEntry],
                                     map_exits: List[nodes.MapExit],
                                     do_not_override: List[str] = [],
                                     change_init_outside: bool = False
                                     ) -> Tuple[Dict[str, bool], List[Tuple[SDFGState, nodes.Map]]]:
        """
        Checks for all intermediate nodes whether they appear
        only within the induced fusible subgraph my map_entries and map_exits.
        This is returned as a dict that contains a boolean value for each
        intermediate node as a key.

        :param sdfg: SDFG
        :param state: State of interest
        :param intermediate_nodes: List of intermediate nodes appearing in a fusible subgraph
        :param map_entries: List of outermost scoped map entries in the subgraph
        :param map_exits: List of map exits corresponding to map_entries in order
        :param do_not_override: List of data array names not to be compressed
        :param change_init_outside: Set to True if arrays which are initialised outside the subgraph should be
        compressible
        :param return: A dictionary indicating for each data string whether its array can be compressed and a list of
        initialisation states and maps not inside the given graph for any intermediate data
        """

        # search whether intermediate_nodes appear outside of subgraph
        # and store it in dict
        data_counter = defaultdict(int)
        data_counter_subgraph = defaultdict(int)

        data_intermediate = set([node.data for node in intermediate_nodes])
        init_maps = {}

        # do a full global search and count each data from each intermediate node
        scope_dict = graph.scope_dict()
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in data_intermediate:
                    # add them to the counter set in all cases
                    data_counter[node.data] += 1
                    # see whether we are inside the subgraph scope
                    # if so, add to data_counter_subgraph
                    # do not add if it is in out_nodes / in_nodes
                    if state == graph and \
                        (node in intermediate_nodes or scope_dict[node] in map_entries):
                        data_counter_subgraph[node.data] += 1
                    elif node.data not in init_maps and len(state.in_edges(node)) == 1:
                        logger.debug("%s: Found potential init access in %s", node.data, state)
                        if isinstance(state.in_edges(node)[0].src, nodes.MapExit):
                            init_map_candidate = state.in_edges(node)[0].src
                            is_init, maps = helpers.is_map_init(state, init_map_candidate, list(sdfg.arrays[node.data].shape),
                                                                sdfg.constants)
                            if is_init:
                                init_maps[node.data] = (state, maps)
                                # if the found access is the initialisation we can ignore or for the global counter
                                logger.debug("%s: Found init access in %s", node.data, state)
                                if change_init_outside:
                                    data_counter[node.data] -= 1
                    else:
                        logger.debug("%s: Found access outside of subgraph in %s", node.data, state)


        # next up: If intermediate_counter and global counter match and if the array
        # is declared transient, it is fully contained by the subgraph

        subgraph_contains_data = {data: data_counter[data] == data_counter_subgraph[data] \
                                        and sdfg.data(data).transient \
                                        and data not in do_not_override \
                                  for data in data_intermediate}
        return (subgraph_contains_data, init_maps)

    @staticmethod
    def data_initialised_somewhere_else(sdfg: dace.sdfg.SDFG,
                                        this_state: SDFGState,
                                        data_name: str) -> Optional[Tuple[SDFGState, List[Tuple[nodes.MapExit, int]]]]:
        """
        Check if the data of the given array name is initialised outside of the given state

        :param sdfg: [TODO:description]
        :type sdfg: dace.sdfg.SDFG
        :param this_state: [TODO:description]
        :type this_state: SDFGState
        :param data_name: [TODO:description]
        :type data_name: str
        :return: If an init 
        :rtype: Optional[Tuple[SDFGState, List[Tuple[nodes.MapExit, int]]]]
        """
        init_state = None
        for state in sdfg.nodes():
            if state != this_state:
                read_set, write_set = state.read_and_write_sets()
                if data_name in write_set or data_name in read_set:
                    # There can only be one init state and it can only write
                    if init_state is None and data_name not in read_set:
                        init_state = state
                    elif init_state is not None:
                        # Can't have other states writing to it
                        return None

        init_maps = [None] * len(sdfg.arrays[data_name].shape)
        if init_state is not None:
            for node in init_state.nodes():
                # Expect one or multiple map, with a tasklet inside
                if isinstance(node, nodes.AccessNode) and node.data == data_name and len(init_state.in_edges(node)) == 1:
                    map_exit = init_state.in_edges(node)[0].src
                    found_map = False
                    while isinstance(map_exit, nodes.MapExit) and len(init_state.in_edges(map_exit)) == 1:
                        map_range_lengths = [int((rng[1] - rng[0] + S.One).evalf(subs=sdfg.constants)) for rng in
                                             map_exit.map.range]
                        array_dims = [int(dim_size.evalf(subs=sdfg.constants)) for dim_size in sdfg.arrays[data_name].shape]
                        for map_idx, map_length in enumerate(map_range_lengths):
                            # Can lead to problems if map as two dimension of same size
                            if map_length in array_dims:
                                init_maps[array_dims.index(map_length)] = (map_exit, map_idx)
                            else:
                                return None
                        edge = init_state.in_edges(map_exit)[0]
                        map_exit = edge.src
                        found_map = True

                    tasklet = map_exit
                    if (isinstance(tasklet, nodes.Tasklet) and len(init_state.in_edges(tasklet)) == 1 and
                       init_state.in_edges(tasklet)[0].data.data is None and found_map):
                        return (init_state, init_maps)
        return None


    def clone_intermediate_nodes(self, sdfg: dace.sdfg.SDFG, graph: dace.sdfg.SDFGState,
                                 intermediate_nodes: List[nodes.AccessNode], out_nodes: List[nodes.AccessNode],
                                 map_entries: List[nodes.MapEntry], map_exits: List[nodes.MapExit]):
        """ 
        Creates cloned access nodes and data arrays for nodes that are both in intermediate nodes 
        and out nodes, redirecting output from the original node to the cloned node. Operates in-place.

        :param sdfg: SDFG 
        :param state: State of interest 
        :param intermediate_nodes: List of intermediate nodes appearing in a fusible subgraph 
        :param out_nodes: List of out nodes appearing in a fusible subgraph
        :param map_entries: List of outermost scoped map entries in the subgraph 
        :param map_exits: List of map exits corresponding to map_entries in order 
        :return: A dict that maps each intermediate node that also functions as an out node 
                       to the respective cloned transient node 
        """

        transients_created = {}
        for node in intermediate_nodes & out_nodes:
            # create new transient at exit replacing the array
            # and redirect all traffic
            data_ref = sdfg.data(node.data)

            out_trans_data_name = node.data + '_OUT'
            out_trans_data_name = sdfg._find_new_name(out_trans_data_name)
            data_trans = sdfg.add_transient(name=out_trans_data_name,
                                            shape=dcpy(data_ref.shape),
                                            dtype=dcpy(data_ref.dtype),
                                            storage=dcpy(data_ref.storage),
                                            offset=dcpy(data_ref.offset))
            node_trans = graph.add_access(out_trans_data_name)
            if node.setzero:
                node_trans.setzero = True

            # redirect all relevant traffic from node_trans to node
            edges = list(graph.out_edges(node))
            for edge in edges:
                if edge.dst not in map_entries:
                    self.copy_edge(graph, edge, new_src=node_trans, remove_old=True)

            graph.add_edge(node, None, node_trans, None, Memlet())

            transients_created[node] = node_trans

        return transients_created

    def determine_invariant_dimensions(self, sdfg: dace.sdfg.SDFG, graph: dace.sdfg.SDFGState,
                                       intermediate_nodes: List[nodes.AccessNode], map_entries: List[nodes.MapEntry],
                                       map_exits: List[nodes.MapExit]):
        """
        Determines the invariant dimensions for each node -- dimensions in 
        which the access set of the memlets propagated through map entries and 
        exits does not change.
        
        :param sdfg: SDFG 
        :param state: State of interest 
        :param intermediate_nodes: List of intermediate nodes appearing in a fusible subgraph 
        :param map_entries: List of outermost scoped map entries in the subgraph 
        :param map_exits: List of map exits corresponding to map_entries in order 
        :return: A dict mapping each intermediate node (nodes.AccessNode) to a list of integer dimensions
        """
        # create dict for every array that for which
        # subgraph_contains_data is true that lists invariant axes.
        invariant_dimensions = {}
        for node in intermediate_nodes:
            data = node.data
            inv_dims = self.get_invariant_dimensions(sdfg, graph, map_entries, map_exits, node)
            if node in invariant_dimensions:
                # do a check -- we want the same result for each
                # node containing the same data
                if not inv_dims == invariant_dimensions[node]:
                    warnings.warn(f"SubgraphFusion::Data dimensions that are not propagated through differ"
                                  "across multiple instances of access nodes for data {node.data}"
                                  "Please check whether all memlets to AccessNodes containing"
                                  "this data are sound.")
                    invariant_dimensions[data] |= inv_dims

            else:
                invariant_dimensions[data] = inv_dims

        return invariant_dimensions

    def prepare_intermediate_nodes(self,
                                   sdfg: dace.sdfg.SDFG,
                                   graph: dace.sdfg.SDFGState,
                                   in_nodes: List[nodes.AccessNode],
                                   out_nodes: List[nodes.AccessNode],
                                   intermediate_nodes: List[nodes.AccessNode],
                                   map_entries: List[nodes.MapEntry],
                                   map_exits: List[nodes.MapExit],
                                   do_not_override: List[str] = []):
        """
        Helper function that computes the following information:
        1. Determine whether intermediate nodes only appear within the induced fusible subgraph, with the possible
        exception of initialisation (if self.change_init_outside is true). This is equivalent to checking for
        compressibility.
        2. Determine whether any intermediate transients are also out nodes, if so they have to be cloned
        3. Determine invariant dimensions for any intermediate transients (that are compressible).

        :return: A tuple (subgraph_contains_data, init_maps, transients_created, invariant_dimensions)
                 of dictionaries containing the necessary information
        """

        # 1. Compressibility
        subgraph_contains_data, init_maps = SubgraphFusion.determine_compressible_nodes(
            sdfg,
            graph,
            intermediate_nodes,
            map_entries,
            map_exits,
            do_not_override,
            change_init_outside=self.change_init_outside)

        # 2. Clone intermediate & out transients
        transients_created = self.clone_intermediate_nodes(sdfg, graph, intermediate_nodes, out_nodes, map_entries,
                                                           map_exits)
        # 3. Gather invariant dimensions
        invariant_dimensions = self.determine_invariant_dimensions(sdfg, graph, intermediate_nodes, map_entries,
                                                                   map_exits)

        return (subgraph_contains_data, init_maps, transients_created, invariant_dimensions)

    def apply(self, sdfg, do_not_override=None, **kwargs):
        """ Apply the SubgraphFusion Transformation. See @fuse for more details """
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph

        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        self.fuse(sdfg, graph, map_entries, do_not_override, **kwargs)

    def add_condition_to_map(
            self,
            graph: SDFGState,
            sdfg: SDFG,
            map_entry: nodes.MapEntry,
            condition_edge: InterstateEdge,
            map_start_value: Optional[Union[int, dace.symbol]] = None,
            map_end_value: Optional[Union[int, dace.symbol]] = None):
        start_nodes = set(e.dst for e in graph.out_edges(map_entry))

        start_node = start_nodes.pop()
        if len(start_nodes) == 0 and isinstance(start_node, nodes.NestedSDFG):
            nsdfg = start_node
        else:
            contained_graph_view = graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)
            nsdfg = nest_state_subgraph(sdfg, graph, contained_graph_view, full_data=False)
            # Add symbols from outer nsdfg if existing
            if graph.parent and graph.parent.parent_nsdfg_node:
                nsdfg.symbol_mapping = dcpy(graph.parent.parent_nsdfg_node.symbol_mapping)

        for itervar in map_entry.map.params:
            if itervar not in nsdfg.sdfg.symbols:
                nsdfg.sdfg.add_symbol(itervar, int)
            if itervar not in nsdfg.symbol_mapping:
                nsdfg.symbol_mapping[itervar] = itervar

            # Change memlets going into/out of the nsdfg, to reflect the access pattern by the guard state before
            for edge in [*graph.in_edges(nsdfg), *graph.out_edges(nsdfg)]:
                if edge.data.data is not None:
                    rng = edge.data.subset
                    new_rng = []
                    for start, end, step in rng:
                        one_elem = False
                        if start==end:
                            one_elem = True
                        # Check if the itervar of the map is in the memlet range
                        if itervar in str(start) and map_start_value is not None:
                            start = sympy.Max(0, start)
                            if one_elem:
                                end = start
                        if itervar in str(end) and map_end_value is not None:
                            end = sympy.Min(map_end_value, end)
                            if one_elem:
                                start = end
                        new_rng.append((start, end, step))
                    edge.data.subset = subsets.Range(new_rng)

        # If the condition edge uses a symbol which uses a different name inside the nsdfg, change the name
        # accordingly
        for edge_symbol in condition_edge.free_symbols:
            mapping_values = [str(sym) for sym in nsdfg.symbol_mapping.values()]
            if edge_symbol in mapping_values:
                nsdfg_symbol = [key for key, value in nsdfg.symbol_mapping.items() if str(value) == edge_symbol][0]
                condition_edge.condition = CodeBlock(condition_edge.condition.as_string.replace(edge_symbol,
                                                                                                nsdfg_symbol))

        old_start_state = nsdfg.sdfg.start_state
        guard_state = nsdfg.sdfg.add_state(f"guard_{map_entry.map.label}", is_start_state=True)
        nsdfg.sdfg.add_edge(guard_state, old_start_state, condition_edge)

    def fuse(self,
             sdfg: dace.sdfg.SDFG,
             graph: dace.sdfg.SDFGState,
             map_entries: List[nodes.MapEntry],
             do_not_override=None,
             **kwargs):
        """ takes the map_entries specified and tries to fuse maps.

            all maps have to be extended into outer and inner map
            (use MapExpansion as a pre-pass)

            Arrays that don't exist outside the subgraph get pushed
            into the map and their data dimension gets cropped.
            Otherwise the original array is taken.

            For every output respective connections are crated automatically.

            :param sdfg: SDFG
            :param graph: State
            :param map_entries: Map Entries (class MapEntry) of the outer maps
                                which we want to fuse
            :param do_not_override: List of data names whose corresponding nodes
                                    are fully contained within the subgraph
                                    but should not be compressed
                                    nevertheless.
        """

        # if there are no maps, return immediately
        if len(map_entries) == 0:
            return

        do_not_override = do_not_override or []
        do_not_override.extend(self.keep_global)

        # get maps and map exits
        maps = [map_entry.map for map_entry in map_entries]
        logger.debug("Maps: %s", maps)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]
        map_base_ranges, _ = helpers.common_map_base_ranges([map.range for map in maps], self.max_difference_start, self.max_difference_end)

        # See function documentation for an explanation of these variables
        node_config = SubgraphFusion.get_adjacent_nodes(sdfg, graph, map_entries)
        (in_nodes, intermediate_nodes, out_nodes) = node_config

        if self.debug:
            print("SubgraphFusion::In_nodes", in_nodes)
            print("SubgraphFusion::Out_nodes", out_nodes)
            print("SubgraphFusion::Intermediate_nodes", intermediate_nodes)
        logger.debug("in nodes: %s, out_nodes: %s, intermediate nodes: %s, circular buffers: %s", in_nodes, out_nodes,
                     intermediate_nodes, self.arrays_as_circular_buffer)

        # all maps are assumed to have the same params and range in order
        global_map = nodes.Map(label="outer_fused", params=maps[0].params, ndrange=subsets.Range(map_base_ranges))
        logger.debug("Create global maps with params: %s and range: %s", maps[0].params, map_base_ranges)
        global_map_entry = nodes.MapEntry(global_map)
        global_map_exit = nodes.MapExit(global_map)

        for map_entry in map_entries:
            # as maps are split into outer and inner, indices should only have one entry as we are fusing one range
            # dimension only
            for idx in range(len(map_entry.map.range)):
                # can_be_applied checks that all params of all maps to merge are the same -> we can use the params from
                # the current map
                map_start = map_entry.map.range.ranges[idx][0]
                map_end = map_entry.map.range.ranges[idx][1]

                edge = InterstateEdge()
                map_param = map_entry.map.params[idx]
                add_args = {}
                if map_start != global_map.range.ranges[idx][0]:
                    edge.condition = CodeBlock(f"{map_param} >= {map_start}")
                    add_args['map_start_value'] = map_start
                if map_end != global_map.range.ranges[idx][1]:
                    condition_str = f"{map_param} < {map_end}"
                    add_args['map_end_value'] = map_end
                    if edge.condition.as_string != '1':
                        edge.condition = CodeBlock(f"({edge.condition.as_string}) and {condition_str}")
                    else:
                        edge.condition = CodeBlock(condition_str)
                if edge.condition.as_string != '1':
                    self.add_condition_to_map(graph, sdfg, map_entry, edge, **add_args)

        schedule = map_entries[0].schedule
        global_map_entry.schedule = schedule
        graph.add_node(global_map_entry)
        graph.add_node(global_map_exit)

        # next up, for any intermediate node, find whether it only appears
        # in the subgraph or also somewhere else / as an input
        # create new transients for nodes that are in out_nodes and
        # intermediate_nodes simultaneously
        # also check which dimensions of each transient data element correspond
        # to map axes and write this information into a dict.
        node_info = self.prepare_intermediate_nodes(sdfg, graph, in_nodes, out_nodes, \
                                                    intermediate_nodes,\
                                                    map_entries, map_exits, \
                                                    do_not_override)
        (subgraph_contains_data, init_maps, transients_created, invariant_dimensions) = node_info

        if self.debug:
            print("SubgraphFusion:: {Intermediate_node: subgraph_contains_data} dict")
            print(subgraph_contains_data)
        logger.debug("subgraph contains data of: %s", subgraph_contains_data)

        inconnectors_dict = {}
        # Dict for saving incoming nodes and their assigned connectors
        # Format: {access_node: (edge, in_conn, out_conn)}

        for map_entry, map_exit in zip(map_entries, map_exits):
            # handle inputs
            for edge in graph.in_edges(map_entry):
                src = edge.src
                out_edges = [
                    e for e in graph.out_edges(map_entry) if (e.src_conn and e.src_conn[3:] == edge.dst_conn[2:])
                ]
                is_dynamic = False
                if not edge.data.is_empty() and edge.dst_conn[:3] != "IN_":
                    is_dynamic = True
                    dyn_in_conn = edge.dst_conn

                if not edge.data.is_empty() and src in in_nodes:
                    in_conn = None
                    out_conn = None
                    if not is_dynamic and src in inconnectors_dict:
                        # for access nodes only
                        in_conn = inconnectors_dict[src][1]
                        out_conn = inconnectors_dict[src][2]

                    else:
                        if is_dynamic:
                            if dyn_in_conn not in global_map_entry.in_connectors:
                                global_map_entry.add_in_connector(dyn_in_conn)
                            in_conn = dyn_in_conn
                            if out_edges:
                                if dyn_in_conn not in global_map_entry.out_connectors:
                                    global_map_entry.add_out_connector(dyn_in_conn)
                                out_conn = dyn_in_conn
                        else:
                            next_conn = global_map_entry.next_connector()
                            in_conn = 'IN_' + next_conn
                            out_conn = 'OUT_' + next_conn
                            global_map_entry.add_in_connector(in_conn)
                            global_map_entry.add_out_connector(out_conn)

                            if isinstance(src, nodes.AccessNode):
                                inconnectors_dict[src] = (edge, in_conn, out_conn)

                        # reroute in edge via global_map_entry
                        if not (is_dynamic and list(graph.in_edges_by_connector(global_map_entry, in_conn))):
                            # self.copy_edge(graph, edge, new_dst = global_map_entry, new_dst_conn = in_conn)
                            graph.add_edge(edge.src, edge.src_conn, global_map_entry, in_conn, dcpy(edge.data))

                    # map out edges to new map
                    for out_edge in out_edges:
                        # self.copy_edge(graph, out_edge, new_src = global_map_entry, new_src_conn = out_conn)
                        graph.add_edge(global_map_entry, out_conn, out_edge.dst, out_edge.dst_conn, dcpy(out_edge.data))

                else:
                    # connect directly
                    for out_edge in out_edges:
                        mm = dcpy(out_edge.data)
                        # self.copy_edge(graph, out_edge, new_src=src, new_src_conn=None, new_data=mm)
                        graph.add_edge(edge.src, edge.src_conn, out_edge.dst, out_edge.dst_conn, mm)

            for edge in graph.out_edges(map_entry):
                # special case: for nodes that have no data connections
                if not edge.src_conn:
                    # self.copy_edge(graph, edge, new_src=global_map_entry)
                    graph.add_edge(global_map_entry, None, edge.dst, edge.dst_conn, dcpy(edge.data))

            ######################################

            for edge in graph.in_edges(map_exit):
                if not edge.dst_conn:
                    # no destination connector, path ends here.
                    # self.copy_edge(graph, edge, new_dst=global_map_exit)
                    graph.add_edge(edge.src, edge.src_conn, global_map_exit, None, dcpy(edge.data))
                    continue

                # # NOTE: Duplicate edges coming out of a previously fused MapExit
                # if not isinstance(edge.src, nodes.AccessNode):
                #     continue

                # find corresponding out_edges for current edge
                out_edges = [oedge for oedge in graph.out_edges(map_exit) if oedge.src_conn[3:] == edge.dst_conn[2:]]

                # Tuple to store in/out connector port that might be created
                port_created = None

                for out_edge in out_edges:
                    dst = out_edge.dst

                    if dst in intermediate_nodes & out_nodes:

                        # create connection through global map from
                        # dst to dst_transient that was created
                        dst_transient = transients_created[dst]
                        next_conn = global_map_exit.next_connector()
                        in_conn = 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_exit.add_in_connector(in_conn)
                        global_map_exit.add_out_connector(out_conn)

                        # for each transient created, create a union
                        # of outgoing memlets' subsets. this is
                        # a cheap fix to override assignments in invariant
                        # dimensions
                        union = None
                        for oe in graph.out_edges(transients_created[dst]):
                            subset = oe.data.get_src_subset(oe, graph)
                            if subset is None:
                                subset = oe.data.subset
                            union = subsets.union(union, subset)
                        if isinstance(union, subsets.Indices):
                            union = subsets.Range.from_indices(union)
                        inner_memlet = dcpy(edge.data)
                        for i, s in enumerate(edge.data.subset):
                            if i in invariant_dimensions[dst.label]:
                                inner_memlet.subset[i] = union[i]

                        inner_memlet.other_subset = dcpy(inner_memlet.subset)

                        e_inner = graph.add_edge(dst, None, global_map_exit, in_conn, inner_memlet)

                        outer_memlet = dcpy(out_edge.data)
                        e_outer = graph.add_edge(global_map_exit, out_conn, dst_transient, None, outer_memlet)

                        # remove edge from dst to dst_transient that was created
                        # in intermediate preparation.
                        for e in graph.out_edges(dst):
                            if e.dst == dst_transient:
                                graph.remove_edge(e)
                                break

                    # handle separately: intermediate_nodes and pure out nodes
                    # case 1: intermediate_nodes: can just redirect edge
                    if dst in intermediate_nodes:
                        # self.copy_edge(graph,
                        #                out_edge,
                        #                new_src=edge.src,
                        #                new_src_conn=edge.src_conn,
                        #                new_data=dcpy(edge.data))
                        graph.add_edge(edge.src, edge.src_conn, out_edge.dst, out_edge.dst_conn, dcpy(edge.data))

                    # case 2: pure out node: connect to outer array node
                    if dst in (out_nodes - intermediate_nodes):
                        if edge.dst != global_map_exit:
                            next_conn = global_map_exit.next_connector()

                            in_conn = 'IN_' + next_conn
                            out_conn = 'OUT_' + next_conn
                            global_map_exit.add_in_connector(in_conn)
                            global_map_exit.add_out_connector(out_conn)
                            # self.copy_edge(graph, edge, new_dst=global_map_exit, new_dst_conn=in_conn)
                            graph.add_edge(edge.src, edge.src_conn, global_map_exit, in_conn, dcpy(edge.data))
                            port_created = (in_conn, out_conn)

                        else:
                            in_conn = port_created.st
                            out_conn = port_created.nd

                        # map
                        graph.add_edge(global_map_exit, out_conn, dst, out_edge.dst_conn, dcpy(out_edge.data))

            # maps are now ready to be discarded
            # all connected edges will be finally removed as well
            graph.remove_node(map_entry)
            graph.remove_node(map_exit)

        # create a mapping from data arrays to offsets
        # for later memlet adjustments later
        min_offsets = dict()

        # do one pass to compress all transient arrays
        def change_data(data_name, sdfg, shape, strides, total_size, offset, lifetime, storage):
            if data_name in sdfg.arrays:
                transient_array = sdfg.arrays[data_name]
                logger.debug("Change shape of %s from %s to %s on sdfg: %s", data_name, transient_array.shape, shape, sdfg.label)
                if shape is not None:
                    transient_array.shape = shape
                if strides is not None:
                    transient_array.strides = strides
                if total_size is not None:
                    transient_array.total_size = total_size
                if offset is not None:
                    transient_array.offset = offset
                if lifetime is not None:
                    transient_array.lifetime = lifetime
                if storage is not None:
                    transient_array.storage = storage

                # recurse into all nsdfg
                # for state in sdfg.states():
                #     for node in state.nodes():
                #         if isinstance(node, nodes.NestedSDFG):
                #             # Will ignore if an array has a different shape inside a nsdfg
                #             change_data(data_name, node.sdfg, shape, strides, total_size, offset, lifetime, storage)

        intermediate_nodes = [n for n in intermediate_nodes if n.data not in self.blacklisted_arrays]
        data_intermediate = set([node.data for node in intermediate_nodes])
        data_intermediate -= set(self.blacklisted_arrays)
        logger.debug("Intermediate data: %s, intermediate nodes: %s, blacklisted arrays: %s", data_intermediate,
                     intermediate_nodes, self.blacklisted_arrays)

        for data_name in data_intermediate:
            desc = sdfg.data(data_name)
            if (subgraph_contains_data[data_name] and isinstance(desc, dace.data.Array) 
                and data_name not in self.arrays_as_circular_buffer):
                all_nodes = [n for n in intermediate_nodes if n.data == data_name]
                in_edges = list(chain(*(graph.in_edges(n) for n in all_nodes)))

                in_edges_iter = iter(in_edges)
                in_edge = next(in_edges_iter)
                target_subset = dcpy(in_edge.data.subset)
                logger.debug("%s, taget subset from edge: %s", data_name, target_subset)
                # Make 0-based
                target_subset.offset(desc.offset, False)
                logger.debug("%s, taget subset after offset by %s: %s", data_name, desc.offset, target_subset)
                target_subset.pop(invariant_dimensions[data_name])
                logger.debug("%s, taget subset after popping invariant dimensions: %s", data_name, target_subset)
                helpers.remove_min_max(target_subset)
                logger.debug("%s, taget subset after removing min max: %s", data_name, target_subset)
                while True:
                    try:  # executed if there are multiple in_edges
                        in_edge = next(in_edges_iter)
                        target_subset_curr = dcpy(in_edge.data.subset)
                        # Make 0-based
                        target_subset_curr.offset(desc.offset, False)
                        target_subset_curr.pop(invariant_dimensions[data_name])
                        helpers.remove_min_max(target_subset_curr)
                        target_subset = subsets.union(target_subset, \
                                                      target_subset_curr)
                        logger.debug("%s, target subset after unionizing with '%s': %s", data_name, target_subset_curr, target_subset)
                    except StopIteration:
                        break

                min_offsets_cropped = target_subset.min_element_approx()
                # calculate the new transient array size.
                target_subset.offset(min_offsets_cropped, True)
                logger.debug("%s, taget subset after offset by %s: %s", data_name, min_offsets_cropped, target_subset)

                # re-add invariant dimensions with the corresponding offset and save to min_offsets
                min_offset = []
                index = 0
                for i in range(len(desc.shape)):
                    if i in invariant_dimensions[data_name]:
                        min_offset.append(0)
                    else:
                        if data_name in self.arrays_as_circular_buffer:
                            min_offset.append(min_offsets_cropped[index] - self.arrays_as_circular_buffer[data_name][index])
                        else:
                            min_offset.append(min_offsets_cropped[index])
                        index += 1

                min_offsets[data_name] = min_offset

                # determine the shape of the new array.
                if data_name in self.fixed_new_shapes:
                    logger.debug("%s has a fixed new shape", data_name)
                    new_data_shape = self.fixed_new_shapes[data_name]
                else:
                    new_data_shape = []
                    index = 0
                    for i, sz in enumerate(desc.shape):
                        if i in invariant_dimensions[data_name]:
                            new_data_shape.append(sz)
                        else:
                            new_data_shape.append(target_subset.size()[index])
                            if target_subset.size()[index] != 1 and data_name not in self.arrays_as_circular_buffer:
                                # Empty list as the shape has already been adjusted
                                self.arrays_as_circular_buffer[data_name] = []
                            index += 1

                    if data_name in self.arrays_as_circular_buffer:
                        for index, diff in enumerate(self.arrays_as_circular_buffer[data_name]):
                            new_data_shape[index] += diff

                new_data_strides = [data._prod(new_data_shape[i + 1:]) for i in range(len(new_data_shape))]

                new_data_totalsize = data._prod(new_data_shape)
                # new_data_offset = [0] * len(new_data_shape)
                new_data_offset = desc.offset

                # compress original shape
                change_data(data_name,
                            sdfg,
                            shape=new_data_shape,
                            strides=new_data_strides,
                            total_size=new_data_totalsize,
                            offset=new_data_offset,
                            lifetime=dtypes.AllocationLifetime.Scope,
                            storage=self.transient_allocation)
                # If the compressed array is also initialised somewhere, change that map to the new size
                if data_name in init_maps:
                    state, map_infos = init_maps[data_name]
                    for map, shape_idx in map_infos:
                            # We assume that map are expanded and thus have only one range
                            map.range[0] = (
                                map.range[0][0], map.range[0][0] +
                                sdfg.arrays[data_name].shape[shape_idx] -
                                S.One, map.range[0][2])
                    propagate_memlets_state(sdfg, state)
            else:
                # don't modify data container - array is needed outside
                # of subgraph.

                # hack: set lifetime to State if allocation has only been
                # scope so far to avoid allocation issues
                if sdfg.data(data_name).lifetime == dtypes.AllocationLifetime.Scope:
                    sdfg.data(data_name).lifetime = dtypes.AllocationLifetime.State

        # Add modulo operations to the memlets with the data which is shrank but needs to be used as a circular buffer
        # for data_name in self.arrays_as_circular_buffer:
        #     if data_name in subgraph_contains_data and subgraph_contains_data[data_name]:
        #         logger.debug("Add modulo to memlets of %s", data_name)
        #         helpers.add_modulo_to_all_memlets(graph, data_name, sdfg.data(data_name).shape)

        for node in intermediate_nodes:
            # memlets of created transient:
            # correct data names
            if node in transients_created:
                transient_in_edges = graph.in_edges(transients_created[node])
                transient_out_edges = graph.out_edges(transients_created[node])
                for edge in chain(transient_in_edges, transient_out_edges):
                    for e in graph.memlet_tree(edge):
                        if e.data.data == node.data:
                            e.data.data = transients_created[node].data

            # intermediate_nodes is a set -> each data node only appears only once. Need to find all the other access
            # nodes to the same data
            # for access_node in graph.data_nodes():
            #     # circular buffers are treated differently, they need to add modulos
            # if access_node.data == node.data and node.data not in self.arrays_as_circular_buffer:
            if node.data not in self.arrays_as_circular_buffer:
                access_node = node

                # all incoming edges to node
                in_edges = graph.in_edges(access_node)
                # outgoing edges going to another fused part
                out_edges = graph.out_edges(access_node)

                # memlets of all in between transients:
                # offset memlets if array has been compressed
                if subgraph_contains_data[node.data] and isinstance(sdfg.data(node.data), dace.data.Array):
                    # get min_offset
                    min_offset = min_offsets[node.data]
                    # re-add invariant dimensions with offset 0
                    for iedge in in_edges:
                        for edge in graph.memlet_tree(iedge):
                            # nested SDFG: adjust arrays connected
                            if isinstance(edge.src, nodes.NestedSDFG):
                                nsdfg = edge.src.sdfg
                                nested_data_name = edge.src_conn
                                self.adjust_arrays_nsdfg(sdfg, nsdfg, node.data, nested_data_name, edge.data)
                            if edge.data.data == node.data:
                                edge.data.subset.offset(min_offset, True)
                            elif edge.data.other_subset:
                                edge.data.other_subset.offset(min_offset, True)

                    for cedge in out_edges:
                        for edge in graph.memlet_tree(cedge):
                            # nested SDFG: adjust arrays connected
                            if isinstance(edge.dst, nodes.NestedSDFG):
                                nsdfg = edge.dst.sdfg
                                nested_data_name = edge.dst_conn
                                self.adjust_arrays_nsdfg(sdfg, nsdfg, node.data, nested_data_name, edge.data)
                            if edge.data.data == node.data:
                                edge.data.subset.offset(min_offset, True)
                            elif edge.data.other_subset:
                                edge.data.other_subset.offset(min_offset, True)

                    # if in_edges has several entries:
                    # put other_subset into out_edges for correctness
                    if len(in_edges) > 1:
                        for oedge in out_edges:
                            if oedge.dst == global_map_exit and \
                                                oedge.data.other_subset is None:
                                oedge.data.other_subset = dcpy(oedge.data.subset)
                                oedge.data.other_subset.offset(min_offset, False)
        # consolidate edges if desired
        if self.consolidate:
            consolidate_edges_scope(graph, global_map_entry)
            consolidate_edges_scope(graph, global_map_exit)

        # propagate edges adjacent to global map entry and exit
        # if desired
        if self.propagate:
            _propagate_node(graph, global_map_entry)
            _propagate_node(graph, global_map_exit)

        # create a hook for outside access to global_map
        self._global_map_entry = global_map_entry
        if self.schedule_innermaps is not None:
            for node in graph.scope_children()[global_map_entry]:
                if isinstance(node, nodes.MapEntry):
                    node.map.schedule = self.schedule_innermaps

        # Try to remove intermediate nodes that are not contained in the subgraph
        # by reconnecting their adjacent edges to nodes outside the subgraph.
        # NOTE: Currently limited to cases where there is a single source and sink
        # if there are multiple intermediate accesses for the same data.
        # NOTE: Currently limited to intermediate data that do not have a separate output node

        # Filter out outputs
        output_data = set([n.data for n in out_nodes])
        true_intermediate_nodes = set([n for n in intermediate_nodes if n.data not in output_data])

        # Sort intermediate nodes by data name
        intermediate_data = dict()
        for acc in true_intermediate_nodes:
            if acc.data in intermediate_data:
                intermediate_data[acc.data].append(acc)
            else:
                intermediate_data[acc.data] = [acc]

        filtered_intermediate_data = dict()
        intermediate_sources = dict()
        intermediate_sinks = dict()
        for dname, accesses in intermediate_data.items():

            sources = set(accesses)
            sinks = set(accesses)

            # Find sinks
            for acc0 in accesses:
                for acc1 in set(sinks):
                    if acc0 is acc1:
                        continue
                    if nx.has_path(graph.nx, acc0, acc1):
                        sinks.remove(acc0)
                        break
            if len(sinks) > 1:
                continue
            # Find sources
            for acc0 in accesses:
                for acc1 in set(sources):
                    if acc0 is acc1:
                        continue
                    if nx.has_path(graph.nx, acc1, acc0):
                        sources.remove(acc0)
                        break
            if len(sources) > 1:
                continue

            filtered_intermediate_data[dname] = accesses
            intermediate_sources[dname] = sources
            intermediate_sinks[dname] = sinks

        edges_to_remove = set()

        for dname, accesses in filtered_intermediate_data.items():

            # Checking if data are contained in the subgraph
            if not subgraph_contains_data[dname] and False:
                # Find existing outer access nodes
                inode, onode = None, None
                for e in graph.in_edges(global_map_entry):
                    if isinstance(e.src, nodes.AccessNode) and dname == e.src.data:
                        inode = e.src
                        break
                for e in graph.out_edges(global_map_exit):
                    if isinstance(e.dst, nodes.AccessNode) and dname == e.dst.data:
                        onode = e.dst
                        break

                # Compute the union of all incoming subsets.
                # TODO: Do we expect this operation to ever fail?
                in_subset: subsets.Subset = None
                first_subset: subsets.Subset = None
                for acc in accesses:
                    for ie in graph.in_edges(acc):
                        if in_subset:
                            in_subset = subsets.union(in_subset, ie.data.dst_subset)
                        else:
                            in_subset = ie.data.dst_subset
                            first_subset = ie.data.dst_subset

                can_proceed = True
                for oe in graph.out_edges(node):
                    if not in_subset.covers(oe.data.src_subset):
                        try:
                            intersects = in_subset.intersects(oe.data.src_subset)
                        except TypeError:
                            intersects = False
                        if intersects:
                            can_proceed = False
                            break

                if not can_proceed:
                    continue

                # Create transient data corresponding to the union of the incoming subsets.
                desc = sdfg.arrays[dname]
                new_name, _ = sdfg.add_temp_transient(in_subset.bounding_box_size(), desc.dtype, desc.storage)

                for acc in accesses:

                    acc.data = new_name

                    # Reconnect incoming edges through the transient data.
                    for ie in graph.in_edges(acc):
                        mem = Memlet(data=new_name,
                                     subset=ie.data.dst_subset.offset_new(in_subset, True),
                                     other_subset=ie.data.src_subset)
                        # new_edge = graph.add_edge(ie.src, ie.src_conn, new_node, None, mem)
                        ie.data = mem
                        # Update memlet paths.
                        for e in graph.memlet_path(ie):
                            if e.data.data == dname:
                                e.data.data = new_name
                                e.data.dst_subset.offset(in_subset, True)

                    # Reconnect outgoing edges through the transient data.
                    for oe in graph.out_edges(acc):
                        if in_subset.covers(oe.data.src_subset):
                            mem = Memlet(data=new_name,
                                         subset=oe.data.src_subset.offset_new(in_subset, True),
                                         other_subset=oe.data.dst_subset)
                            # new_edge = graph.add_edge(new_node, None, oe.dst, oe.dst_conn, mem)
                            oe.data = mem
                            # Update memlet paths.
                            for e in graph.memlet_path(oe):
                                if e.data.data == dname:
                                    e.data.data = new_name
                                    e.data.src_subset.offset(in_subset, True)
                        else:
                            # NOTE: For debugging purposes
                            intersect = subsets.intersects(in_subset, oe.data.src_subset)
                            if intersect is None:
                                warnings.warn(f'{dname}[{in_subset}] may intersect with {dname}[{oe.data.src_subset}]')
                            elif intersect:
                                raise ValueError(f'{dname}[{in_subset}] intersects with {dname}[{oe.data.src_subset}]')
                            # If the outgoing subset is not covered by the transient data, connect to the outer input node.
                            if not inode:
                                inode = graph.add_access(dname)
                            graph.add_memlet_path(inode, global_map_entry, oe.dst, memlet=oe.data, dst_conn=oe.dst_conn)
                            edges_to_remove.add(oe)

                    # Connect transient data to the outer output node.
                    if acc in intermediate_sinks[dname]:
                        if not onode:
                            onode = graph.add_access(dname)
                        graph.add_memlet_path(acc,
                                              global_map_exit,
                                              onode,
                                              memlet=Memlet(data=dname, subset=in_subset),
                                              src_conn=None)

        for e in edges_to_remove:
            graph.remove_edge(e)

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the orthogonal
    stencil tiling transformation. """

import math

import dace
from dace import dtypes, registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes
from dace.transformation import transformation
from dace.sdfg.propagation import _propagate_node

from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.strip_mining import StripMining
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.subgraph import SubgraphFusion

from copy import deepcopy as dcpy

import dace.subsets as subsets
import dace.symbolic as symbolic

import itertools
import warnings

from collections import defaultdict

from dace.transformation.subgraph import helpers


@registry.autoregister_params(singlestate=True)
@make_properties
class StencilTiling(transformation.SubgraphTransformation):
    """ Operates on top level maps of the given subgraph.
        Applies orthogonal tiling to each of the maps with
        the given strides and extends the newly created
        inner tiles to account for data dependencies
        due to stencil patterns. For each map all outgoing
        memlets to an array must cover the memlets that
        are incoming into a following child map.

        All maps must have the same map parameters in
        the same order.
    """

    # Properties
    debug = Property(desc="Debug mode", dtype=bool, default=False)

    prefix = Property(dtype=str,
                      default="stencil",
                      desc="Prefix for new inner tiled range symbols")

    strides = ShapeProperty(dtype=tuple, default=(1, ), desc="Tile stride")

    schedule = Property(dtype=dace.dtypes.ScheduleType,
                        default=dace.dtypes.ScheduleType.Default,
                        desc="Dace.Dtypes.ScheduleType of Inner Maps")

    unroll_loops = Property(desc="Unroll Inner Loops if they have Size > 1",
                            dtype=bool,
                            default=False)

    @staticmethod
    def coverage_dicts(sdfg, graph, map_entry, outer_range=True):
        '''
        returns a tuple of two dicts:
        the first dict has as a key all data entering the map
        and its associated access range
        the second dict has as a key all data exiting the map
        and its associated access range
        if outer_range = True, substitutes outer ranges
        into min/max of inner access range
        '''
        map_exit = graph.exit_node(map_entry)
        map = map_entry.map

        entry_coverage = {}
        exit_coverage = {}
        # create dicts with which we can replace all iteration
        # variable_mapping by their range
        map_min = {
            dace.symbol(param): e
            for param, e in zip(map.params, map.range.min_element())
        }
        map_max = {
            dace.symbol(param): e
            for param, e in zip(map.params, map.range.max_element())
        }

        # look at inner memlets at map entry
        for e in graph.out_edges(map_entry):
            if not e.data.subset:
                continue
            if outer_range:
                # get subset
                min_element = [
                    m.subs(map_min) for m in e.data.subset.min_element()
                ]
                max_element = [
                    m.subs(map_max) for m in e.data.subset.max_element()
                ]
                # create range
                rng = subsets.Range(
                    (min_e, max_e, 1)
                    for min_e, max_e in zip(min_element, max_element))
            else:
                rng = dcpy(e.data.subset)

            if e.data.data not in entry_coverage:
                entry_coverage[e.data.data] = rng
            else:
                old_coverage = entry_coverage[e.data.data]
                entry_coverage[e.data.data] = subsets.union(old_coverage, rng)

        # look at inner memlets at map exit
        for e in graph.in_edges(map_exit):
            if outer_range:
                # get subset
                min_element = [
                    m.subs(map_min) for m in e.data.subset.min_element()
                ]
                max_element = [
                    m.subs(map_max) for m in e.data.subset.max_element()
                ]
                # craete range
                rng = subsets.Range(
                    (min_e, max_e, 1)
                    for min_e, max_e in zip(min_element, max_element))
            else:
                rng = dcpy(e.data.subset)

            if e.data.data not in exit_coverage:
                exit_coverage[e.data.data] = rng
            else:
                old_coverage = exit_coverage[e.data]
                exit_coverage[e.data.data] = subsets.union(old_coverage, rng)

        # return both coverages as a tuple
        return (entry_coverage, exit_coverage)

    @staticmethod
    def topology(sdfg, graph, map_entries):
        # first get dicts of parents and children for each map_entry
        # get source maps as a starting point for BFS
        # these are all map entries reachable from source nodes
        sink_maps = set()
        children_dict = defaultdict(set)
        parent_dict = defaultdict(set)
        map_exits = {graph.exit_node(entry): entry for entry in map_entries}

        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for e in graph.in_edges(map_entry):
                if isinstance(e.src, nodes.AccessNode):
                    for ie in graph.in_edges(e.src):
                        if ie.src in map_exits:
                            other_entry = map_exits[ie.src]
                            children_dict[other_entry].add(map_entry)
                            parent_dict[map_entry].add(other_entry)
            out_counter = 0
            for e in graph.out_edges(map_exit):
                if isinstance(e.dst, nodes.AccessNode):
                    for oe in graph.out_edges(e.dst):
                        if oe.dst in map_entries:
                            other_entry = oe.dst
                            children_dict[map_entry].add(other_entry)
                            parent_dict[other_entry].add(map_entry)
                            out_counter += 1
            if out_counter == 0:
                sink_maps.add(map_entry)

        return (children_dict, parent_dict, sink_maps)

    @staticmethod
    def can_be_applied(sdfg, subgraph) -> bool:
        # get highest scope maps
        graph = subgraph.graph
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(entry) for entry in map_entries]

        # 1.1: There has to be more than one outermost scope map entry
        if len(map_entries) <= 1:
            return False

        # 1.2: check basic constraints:
        # - all parameters have to be the same (this implies same length)
        # - no parameter permutations here as ambiguity is very high then
        # - same strides everywhere
        first_map = next(iter(map_entries))
        params = dcpy(first_map.map.params)
        strides = first_map.map.range.strides()
        schedule = first_map.map.schedule

        for map_entry in map_entries:
            if map_entry.map.params != params:
                return False
            if map_entry.map.range.strides() != strides:
                return False
            if map_entry.map.schedule != schedule:
                return False

        # 1.3: check whether all map entries only differ by a const amount
        max_amount = 0
        first_entry = next(iter(map_entries))
        for map_entry in map_entries:
            for r1, r2 in zip(map_entry.map.range, first_entry.map.range):
                if len((r1[0] - r2[0]).free_symbols) > 0:
                    return False
                else:
                    max_amount = max(max_amount, abs(r1[0] - r2[0]))
                if len((r1[1] - r2[1]).free_symbols) > 0:
                    return False
                else:
                    max_amount = max(max_amount, abs(r1[1] - r2[1]))

        # in case there is nothing to tile
        if max_amount == 0:
            return False

        # get intermediate_nodes, out_nodes from SubgraphFusion Transformation
        try:
            node_config = SubgraphFusion.get_adjacent_nodes(sdfg, graph,
                                                            map_entries)
            (_, intermediate_nodes, out_nodes) = node_config
        except NotImplementedError:
            return False

        # 1.4: check topological feasibility
        if not SubgraphFusion.check_topo_feasibility(
                sdfg, graph, map_entries, intermediate_nodes, out_nodes):
            return False
        # 1.5 nodes that are both intermediate and out nodes
        # are not supported in StencilTiling
        if len(intermediate_nodes & out_nodes) > 0:
            return False

        # 1.6 check that we only deal with compressible transients

        subgraph_contains_data = SubgraphFusion.determine_compressible_nodes(
            sdfg, graph, intermediate_nodes, map_entries, map_exits)
        if any([s == False for s in subgraph_contains_data.values()]):
            return False

        # get coverages for every map entry
        coverages = {}
        memlets = {}
        for map_entry in map_entries:
            coverages[map_entry] = StencilTiling.coverage_dicts(
                sdfg, graph, map_entry)
            memlets[map_entry] = StencilTiling.coverage_dicts(sdfg,
                                                              graph,
                                                              map_entry,
                                                              outer_range=False)

        # get DAG neighbours for each map
        dag_neighbors = StencilTiling.topology(sdfg, graph, map_entries)
        (children_dict, _, sink_maps) = dag_neighbors

        # 1.7: we now check coverage:
        # each outgoing coverage for a data memlet has to
        # be exactly equal to the union of incoming coverages
        # of all chidlren map memlets of this data

        # important:
        # 1. it has to be equal and not only cover it in order to
        #    account for ranges too long
        # 2. we check coverages by map parameter and not by
        #    array, this way it is even more general
        # 3. map parameter coverages are checked for each
        #    (map_entry, children of this map_entry) - pair
        for map_entry in map_entries:
            # get coverage from current map_entry
            map_coverage = coverages[map_entry][1]

            # final mapping map_parameter -> coverage will be stored here
            param_parent_coverage = {p: None for p in map_entry.params}
            param_children_coverage = {p: None for p in map_entry.params}
            for child_entry in children_dict[map_entry]:
                # get mapping data_name -> coverage
                for (data_name, cov) in map_coverage.items():
                    parent_coverage = cov
                    children_coverage = None
                    if data_name in coverages[child_entry][0]:
                        children_coverage = subsets.union(
                            children_coverage,
                            coverages[child_entry][0][data_name])

                    # extend mapping map_parameter -> coverage
                    # by the previous mapping

                    for i, (p_subset, c_subset) in enumerate(
                            zip(parent_coverage, children_coverage)):

                        # transform into subset
                        p_subset = subsets.Range((p_subset, ))
                        c_subset = subsets.Range((c_subset, ))

                        # get associated parameter in memlet
                        params1 = symbolic.symlist(
                            memlets[map_entry][1][data_name][i]).keys()
                        params2 = symbolic.symlist(
                            memlets[child_entry][0][data_name][i]).keys()
                        if params1 != params2:
                            return False
                        params = params1
                        if len(params) > 1:
                            # this is not supported
                            return False
                        try:
                            symbol = next(iter(params))
                            param_parent_coverage[symbol] = subsets.union(
                                param_parent_coverage[symbol], p_subset)
                            param_children_coverage[symbol] = subsets.union(
                                param_children_coverage[symbol], c_subset)

                        except StopIteration:
                            # current dim has no symbol associated.
                            # ignore and continue
                            warnings.warn(
                                f"StencilTiling::In map {map_entry}, there is a "
                                "dimension belonging to {data_name} "
                                "that has no map parameter associated.")
                            pass

                        except KeyError:
                            return False

            #parameter mapping must be the same
            if param_parent_coverage != param_children_coverage:
                return False

        # 1.8: we want all sink maps to have the same range size
        assert len(sink_maps) > 0
        first_sink_map = next(iter(sink_maps))
        if not all([
                map.range.size() == first_sink_map.range.size()
                for map in sink_maps
        ]):
            return False

        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        subgraph = self.subgraph_view(sdfg)
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)

        result = StencilTiling.topology(sdfg, graph, map_entries)
        (children_dict, parent_dict, sink_maps) = result

        # next up, calculate inferred ranges for each map
        # for each map entry, this contains a tuple of dicts:
        # each of those maps from data_name of the array to
        # inferred outer ranges. An inferred outer range is created
        # by taking the union of ranges of inner subsets corresponding
        # to that data and substituting this subset by the min / max of the
        # parametrized map boundaries
        # finally, from these outer ranges we can easily calculate
        # strides and tile sizes required for every map
        inferred_ranges = defaultdict(dict)

        # create array of reverse topologically sorted map entries
        # to iterate over
        topo_reversed = []
        queue = set(sink_maps.copy())
        while len(queue) > 0:
            element = next(e for e in queue
                           if not children_dict[e] - set(topo_reversed))
            topo_reversed.append(element)
            queue.remove(element)
            for parent in parent_dict[element]:
                queue.add(parent)

        # main loop
        # first get coverage dicts for each map entry
        # for each map, contains a tuple of two dicts
        # each of those two maps from data name to outer range
        coverage = {}
        for map_entry in map_entries:
            coverage[map_entry] = StencilTiling.coverage_dicts(sdfg,
                                                               graph,
                                                               map_entry,
                                                               outer_range=True)

        # we have a mapping from data name to outer range
        # however we want a mapping from map parameters to outer ranges
        # for this we need to find out how all array dimensions map to
        # outer ranges

        variable_mapping = defaultdict(list)
        for map_entry in topo_reversed:
            map = map_entry.map

            # first find out variable mapping
            for e in itertools.chain(graph.out_edges(map_entry),
                                     graph.in_edges(
                                         graph.exit_node(map_entry))):
                mapping = []
                for dim in e.data.subset:
                    syms = set()
                    for d in dim:
                        syms |= symbolic.symlist(d).keys()
                    if len(syms) > 1:
                        raise NotImplementedError(
                            "One incoming or outgoing stencil subset is indexed "
                            "by multiple map parameters. "
                            "This is not supported yet.")
                    try:
                        mapping.append(syms.pop())
                    except KeyError:
                        # just append None if there is no map symbol in it.
                        # we don't care for now.
                        mapping.append(None)

                if e.data in variable_mapping:
                    # assert that this is the same everywhere.
                    # else we might run into problems
                    assert variable_mapping[e.data.data] == mapping
                else:
                    variable_mapping[e.data.data] = mapping

            # now do mapping data name -> outer range
            # and from that infer mapping variable -> outer range
            local_ranges = {dn: None for dn in coverage[map_entry][1].keys()}
            for data_name, cov in coverage[map_entry][1].items():
                local_ranges[data_name] = subsets.union(local_ranges[data_name],
                                                        cov)
                # now look at proceeding maps
                # and union those subsets -> could be larger with stencil indent
                for child_map in children_dict[map_entry]:
                    if data_name in coverage[child_map][0]:
                        local_ranges[data_name] = subsets.union(
                            local_ranges[data_name],
                            coverage[child_map][0][data_name])

            # final assignent: combine local_ranges and variable_mapping
            # together into inferred_ranges
            inferred_ranges[map_entry] = {p: None for p in map.params}
            for data_name, ranges in local_ranges.items():
                for param, r in zip(variable_mapping[data_name], ranges):
                    # create new range from this subset and assign
                    rng = subsets.Range((r, ))
                    if param:
                        inferred_ranges[map_entry][param] = subsets.union(
                            inferred_ranges[map_entry][param], rng)

        # get parameters -- should all be the same
        params = next(iter(map_entries)).map.params.copy()
        # define reference range as inferred range of one of the sink maps
        self.reference_range = inferred_ranges[next(iter(sink_maps))]
        if self.debug:
            print("StencilTiling::Reference Range", self.reference_range)
        # next up, search for the ranges that don't change
        invariant_dims = []
        for idx, p in enumerate(params):
            different = False
            if self.reference_range[p] is None:
                invariant_dims.append(idx)
                warnings.warn(
                    f"StencilTiling::No Stencil pattern detected for parameter {p}"
                )
                continue
            for m in map_entries:
                if inferred_ranges[m][p] != self.reference_range[p]:
                    different = True
                    break
            if not different:
                invariant_dims.append(idx)
                warnings.warn(
                    f"StencilTiling::No Stencil pattern detected for parameter {p}"
                )

        # during stripmining, we will create new outer map entries
        # for easy access
        self._outer_entries = set()
        # with inferred_ranges constructed, we can begin to strip mine
        for map_entry in map_entries:
            # Retrieve map entry and exit nodes.
            map = map_entry.map

            stripmine_subgraph = {
                StripMining._map_entry: graph.nodes().index(map_entry)
            }

            sdfg_id = sdfg.sdfg_id
            last_map_entry = None
            original_schedule = map_entry.schedule
            self.tile_sizes = []
            self.tile_offset_lower = []
            self.tile_offset_upper = []

            # strip mining each dimension where necessary
            removed_maps = 0
            for dim_idx, param in enumerate(map_entry.map.params):
                # get current_node tile size
                if dim_idx >= len(self.strides):
                    tile_stride = symbolic.pystr_to_symbolic(self.strides[-1])
                else:
                    tile_stride = symbolic.pystr_to_symbolic(
                        self.strides[dim_idx])

                trivial = False

                if dim_idx in invariant_dims:
                    self.tile_sizes.append(tile_stride)
                    self.tile_offset_lower.append(0)
                    self.tile_offset_upper.append(0)
                else:
                    target_range_current = inferred_ranges[map_entry][param]
                    reference_range_current = self.reference_range[param]

                    min_diff = symbolic.SymExpr(reference_range_current.min_element()[0] \
                                    - target_range_current.min_element()[0])
                    max_diff = symbolic.SymExpr(target_range_current.max_element()[0] \
                                    - reference_range_current.max_element()[0])

                    try:
                        min_diff = symbolic.evaluate(min_diff, {})
                        max_diff = symbolic.evaluate(max_diff, {})
                    except TypeError:
                        raise RuntimeError("Symbolic evaluation of map "
                                           "ranges failed. Please check "
                                           "your parameters and match.")

                    self.tile_sizes.append(tile_stride + max_diff + min_diff)
                    self.tile_offset_lower.append(
                        symbolic.pystr_to_symbolic(str(min_diff)))
                    self.tile_offset_upper.append(
                        symbolic.pystr_to_symbolic(str(max_diff)))

                # get calculated parameters
                tile_size = self.tile_sizes[-1]
                dim_idx -= removed_maps
                # If map or tile sizes are trivial, skip strip-mining map dimension
                # special cases:
                # if tile size is trivial AND we have an invariant dimension, skip
                if tile_size == map.range.size()[dim_idx] and (
                        dim_idx + removed_maps) in invariant_dims:
                    continue

                # trivial map: we just continue
                if map.range.size()[dim_idx] in [0, 1]:
                    continue

                if tile_size == 1 and tile_stride == 1 and (
                        dim_idx + removed_maps) in invariant_dims:
                    trivial = True
                    removed_maps += 1

                # indent all map ranges accordingly and then perform
                # strip mining on these. Offset inner maps accordingly afterwards

                range_tuple = (map.range[dim_idx][0] +
                               self.tile_offset_lower[-1],
                               map.range[dim_idx][1] -
                               self.tile_offset_upper[-1],
                               map.range[dim_idx][2])
                map.range[dim_idx] = range_tuple
                stripmine = StripMining(sdfg_id, self.state_id,
                                        stripmine_subgraph, 0)

                stripmine.tiling_type = dtypes.TilingType.CeilRange
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix if not trivial else ''
                # use tile_stride for both -- we will extend
                # the inner tiles later
                stripmine.tile_size = str(tile_stride)
                stripmine.tile_stride = str(tile_stride)
                outer_map = stripmine.apply(sdfg)
                outer_map.schedule = original_schedule

                # if tile stride is 1, we can make a nice simplification by just
                # taking the overapproximated inner range as inner range
                # this eliminates the min/max in the range which
                # enables loop unrolling
                if not trivial:
                    if tile_stride == 1:
                        map_entry.map.range[dim_idx] = tuple(
                            symbolic.SymExpr(el._approx_expr) if isinstance(
                                el, symbolic.SymExpr) else el
                            for el in map_entry.map.range[dim_idx])

                    # in map_entry: enlarge tiles by upper and lower offset
                    # doing it this way and not via stripmine strides ensures
                    # that the max gets changed as well
                    old_range = map_entry.map.range[dim_idx]
                    map_entry.map.range[dim_idx] = (
                        (old_range[0] - self.tile_offset_lower[-1]),
                        (old_range[1] + self.tile_offset_upper[-1]),
                        old_range[2])

                # We have to propagate here for correct outer volume and subset sizes
                _propagate_node(graph, map_entry)
                _propagate_node(graph, graph.exit_node(map_entry))

                # usual tiling pipeline
                if last_map_entry:
                    new_map_entry = graph.in_edges(map_entry)[0].src
                    mapcollapse_subgraph = {
                        MapCollapse._outer_map_entry:
                        graph.node_id(last_map_entry),
                        MapCollapse._inner_map_entry:
                        graph.node_id(new_map_entry)
                    }
                    mapcollapse = MapCollapse(sdfg_id, self.state_id,
                                              mapcollapse_subgraph, 0)
                    mapcollapse.apply(sdfg)
                last_map_entry = graph.in_edges(map_entry)[0].src
            # add last instance of map entries to _outer_entries
            if last_map_entry:
                self._outer_entries.add(last_map_entry)

            # apply to the new map the schedule of the original one
            map_entry.map.schedule = self.schedule

            # Map Unroll Feature: only unroll if conditions are met:
            # Only unroll if at least one of the inner map ranges is strictly larger than 1
            # Only unroll if strides all are one
            if self.unroll_loops and all(s == 1 for s in self.strides) and any(
                    s not in [0, 1] for s in map_entry.range.size()):
                l = len(map_entry.params)
                if l > 1:
                    subgraph = {
                        MapExpansion.map_entry: graph.nodes().index(map_entry)
                    }
                    trafo_expansion = MapExpansion(sdfg.sdfg_id,
                                                   sdfg.nodes().index(graph),
                                                   subgraph, 0)
                    trafo_expansion.apply(sdfg)
                maps = [map_entry]
                for _ in range(l - 1):
                    map_entry = graph.out_edges(map_entry)[0].dst
                    maps.append(map_entry)

                for map in reversed(maps):
                    # MapToForLoop
                    subgraph = {
                        MapToForLoop._map_entry: graph.nodes().index(map)
                    }
                    trafo_for_loop = MapToForLoop(sdfg.sdfg_id,
                                                  sdfg.nodes().index(graph),
                                                  subgraph, 0)
                    trafo_for_loop.apply(sdfg)
                    nsdfg = trafo_for_loop.nsdfg

                    # LoopUnroll

                    guard = trafo_for_loop.guard
                    end = trafo_for_loop.after_state
                    begin = next(e.dst for e in nsdfg.out_edges(guard)
                                 if e.dst != end)

                    subgraph = {
                        DetectLoop._loop_guard: nsdfg.nodes().index(guard),
                        DetectLoop._loop_begin: nsdfg.nodes().index(begin),
                        DetectLoop._exit_state: nsdfg.nodes().index(end)
                    }
                    transformation = LoopUnroll(0, 0, subgraph, 0)
                    transformation.apply(nsdfg)

            elif self.unroll_loops:
                warnings.warn(
                    "StencilTiling::Did not unroll loops. Either all ranges are equal to "
                    "one or range difference is symbolic.")

        self._outer_entries = list(self._outer_entries)

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-collapse transformation. """

import copy
from dace import dtypes
from dace import symbolic
from dace.memlet import Memlet
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.symbolic import symlist
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties
from typing import Tuple


@make_properties
class MapCollapse(transformation.SingleStateTransformation):
    """ Implements the Map Collapse pattern.

        Map-collapse takes two nested maps with M and N dimensions respectively,
        and collapses them to a single M+N dimensional map.
    """

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Check the edges between the entries of the two maps.
        outer_map_entry: nodes.MapEntry = self.outer_map_entry
        inner_map_entry: nodes.MapEntry = self.inner_map_entry

        # Check that inner map range is independent of outer range
        map_deps = set()
        for s in inner_map_entry.map.range:
            map_deps |= set(map(str, symlist(s)))
        if any(dep in outer_map_entry.map.params for dep in map_deps):
            return False

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for _src, _, dest, _, _ in graph.out_edges(outer_map_entry):
            if dest != inner_map_entry:
                return False

        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for src, _, _, dst_conn, memlet in graph.in_edges(inner_map_entry):
            if src != outer_map_entry:
                return False

            # Check that dynamic input range memlets are independent of
            # first map range
            if dst_conn is not None and not dst_conn.startswith('IN_'):
                memlet_deps = set()
                for s in memlet.subset:
                    memlet_deps |= set(map(str, symlist(s)))
                if any(dep in outer_map_entry.map.params for dep in memlet_deps):
                    return False

        # Check the edges between the exits of the two maps.
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for _src, _, dest, _, _ in graph.out_edges(inner_map_exit):
            if dest != outer_map_exit:
                return False

        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for src, _, _dest, _, _ in graph.in_edges(outer_map_exit):
            if src != inner_map_exit:
                return False

        if not permissive:
            if inner_map_entry.map.schedule != outer_map_entry.map.schedule:
                return False

        return True

    def match_to_str(self, graph):
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [outer_map_entry, inner_map_entry])

    def apply(self, graph: SDFGState, sdfg: SDFG) -> Tuple[nodes.MapEntry, nodes.MapExit]:
        """
        Collapses two maps into one.
        
        :param sdfg: The SDFG to apply the transformation to.
        :return: A 2-tuple of the new map entry and exit nodes.
        """
        # Extract the parameters and ranges of the inner/outer maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        return sdutil.merge_maps(graph, outer_map_entry, outer_map_exit, inner_map_entry, inner_map_exit)


@make_properties
class DoacrossMapCollapse(MapCollapse):
    """
        TODO
    """

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, True):
            return False

        # Check the edges between the entries of the two maps.
        outer_map_entry: nodes.MapEntry = self.outer_map_entry
        inner_map_entry: nodes.MapEntry = self.inner_map_entry

        # Make sure the schedules are correct.
        if outer_map_entry.map.schedule != dtypes.ScheduleType.CPU_Multicore_Doacross:
            return False
        if inner_map_entry.map.schedule not in (dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential,
                                                dtypes.ScheduleType.CPU_Multicore):
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> Tuple[nodes.MapEntry, nodes.MapExit]:
        """
        TODO
        """
        new_params = self.inner_map_entry.map.params
        merged_entry, merged_exit = super().apply(graph, sdfg)
        subgraph = graph.scope_subgraph(merged_entry, True, True)
        for edge in subgraph.edges():
            memlet: Memlet = edge.data
            if memlet.schedule == dtypes.MemletScheduleType.Doacross_Source:
                # Nothing needs to be done here.
                pass
            elif memlet.schedule == dtypes.MemletScheduleType.Doacross_Source_Deferred:
                # Ensure the merged output is marked to resolve the deferred source.
                merged_entry.map.omp_doacross_multi_source = True
            elif memlet.schedule == dtypes.MemletScheduleType.Doacross_Sink:
                # Adjust the sink dependency offset to account for the new map ranges / dimensions.
                # Since the inner map must be a non-doacross map, we can append the inner map's parameters as unchanged
                # to the offsets of a sink.
                for par in new_params:
                    new_offset = [old for old in memlet.doacross_dependency_offset]
                    new_offset.append(symbolic.symbol(par))
                    memlet.doacross_dependency_offset = new_offset
        return
        """
        TODO
        """
        # Extract the parameters and ranges of the inner/outer maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)
        outer_map = outer_map_entry.map
        inner_map = inner_map_entry.map

        # Create merged map by inheriting attributes from outer map and using
        # the merge functions for parameters and ranges.
        merged_map = copy.deepcopy(outer_map)
        merged_map.label = outer_map.label
        merged_map.params = outer_map.params + inner_map.params
        merged_map.range = type(outer_map.range)(outer_map.range, inner_map.range)

        merged_entry = nodes.MapEntry(merged_map)
        merged_entry.in_connectors = outer_map_entry.in_connectors
        merged_entry.out_connectors = outer_map_entry.out_connectors

        merged_exit = nodes.MapExit(merged_map)
        merged_exit.in_connectors = outer_map_exit.in_connectors
        merged_exit.out_connectors = outer_map_exit.out_connectors

        graph.add_nodes_from([merged_entry, merged_exit])

        # Handle the case of dynamic map inputs in the inner map
        inner_dynamic_map_inputs = sdutil.dynamic_map_inputs(graph, inner_map_entry)
        for edge in inner_dynamic_map_inputs:
            remove_conn = (len(list(graph.out_edges_by_connector(edge.src, edge.src_conn))) == 1)
            conn_to_remove = edge.src_conn[4:]
            if remove_conn:
                merged_entry.remove_in_connector('IN_' + conn_to_remove)
                merged_entry.remove_out_connector('OUT_' + conn_to_remove)
            merged_entry.add_in_connector(edge.dst_conn, inner_map_entry.in_connectors[edge.dst_conn])
            outer_edge = next(graph.in_edges_by_connector(outer_map_entry, 'IN_' + conn_to_remove))
            graph.add_edge(outer_edge.src, outer_edge.src_conn, merged_entry, edge.dst_conn, outer_edge.data)
            if remove_conn:
                graph.remove_edge(outer_edge)

        # Redirect inner in edges.
        for edge in graph.out_edges(inner_map_entry):
            if edge.src_conn is None:  # Empty memlets
                graph.add_edge(merged_entry, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                continue

            # Get memlet path and edge
            path = graph.memlet_path(edge)
            ind = path.index(edge)
            # Add an edge directly from the previous source connector to the
            # destination
            graph.add_edge(merged_entry, path[ind - 1].src_conn, edge.dst, edge.dst_conn, edge.data)

        # Redirect inner out edges.
        for edge in graph.in_edges(inner_map_exit):
            if edge.dst_conn is None:  # Empty memlets
                graph.add_edge(edge.src, edge.src_conn, merged_exit, edge.dst_conn, edge.data)
                continue

            # Get memlet path and edge
            path = graph.memlet_path(edge)
            ind = path.index(edge)
            # Add an edge directly from the source to the next destination
            # connector
            graph.add_edge(edge.src, edge.src_conn, merged_exit, path[ind + 1].dst_conn, edge.data)

        # Redirect outer edges.
        sdutil.change_edge_dest(graph, outer_map_entry, merged_entry)
        sdutil.change_edge_src(graph, outer_map_exit, merged_exit)

        # Clean-up
        graph.remove_nodes_from([outer_map_entry, outer_map_exit, inner_map_entry, inner_map_exit])

        return merged_entry, merged_exit

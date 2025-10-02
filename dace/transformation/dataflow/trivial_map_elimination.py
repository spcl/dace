# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the trivial-map-elimination transformation. """

import dace
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties
from dace.memlet import Memlet


@make_properties
class TrivialMapElimination(transformation.SingleStateTransformation):
    """Implements the Trivial-Map Elimination pattern.

    Trivial-Map Elimination removes all dimensions containing only one
    element from a map. If this applies to all ranges the map is removed.
    Example: Map[i=0:I,j=7] -> Map[i=0:I]
    Example: Map[i=0  ,j=7] -> nothing

    There are some special cases:
    - GPU maps are ignored as they are syntactically needed.
    - If all map ranges are trivial and the map has dynamic map ranges,
        the map is not removed, and one map parameter is retained.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        if map_entry.map.schedule in (dace.dtypes.GPU_SCHEDULES + [dace.ScheduleType.GPU_Default]):
            return False
        if not any(r[0] == r[1] for r in map_entry.map.range):
            return False
        if (map_entry.map.get_param_num()) == 1 and (any(not e.dst_conn.startswith("IN_")
                                                         for e in graph.in_edges(map_entry) if not e.data.is_empty())):
            # There is only one map parameter and there are dynamic map ranges, this can not be resolved.
            return False
        return True

    def apply(self, graph, sdfg):
        map_entry = self.map_entry

        remaining_ranges = []
        remaining_params = []
        scope = graph.scope_subgraph(map_entry)
        for map_param, ranges in zip(map_entry.map.params, map_entry.map.range.ranges):
            map_from, map_to, _ = ranges
            if map_from == map_to:
                # Replace the map index variable with the value it obtained
                scope.replace(map_param, map_from)
            else:
                remaining_ranges.append(ranges)
                remaining_params.append(map_param)

        map_entry.map.range = remaining_ranges
        map_entry.map.params = remaining_params

        if len(remaining_params) != 0:
            # There are still some dimensions left, so no need to remove the map
            pass

        elif any(not e.dst_conn.startswith("IN_") for e in graph.in_edges(map_entry) if not e.data.is_empty()):
            # The map has dynamic map ranges, thus we can not remove the map.
            #  Instead we add one dimension back to keep the SDFG valid.
            map_entry.map.params = [map_param]
            map_entry.map.range = [ranges]

        else:
            # The map is empty and there are no dynamic map ranges.
            self.remove_empty_map(graph, sdfg)

    def remove_empty_map(self, graph, sdfg):
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)

        # Redirect map entry's out edges
        write_only_map = True
        for edge in graph.out_edges(map_entry):
            if edge.data.is_empty():
                continue
            # Add an edge directly from the previous source connector to the destination
            path = graph.memlet_path(edge)
            index = path.index(edge)
            graph.add_edge(path[index - 1].src, path[index - 1].src_conn, edge.dst, edge.dst_conn, edge.data)
            write_only_map = False

        # Redirect map exit's in edges.
        for edge in graph.in_edges(map_exit):
            path = graph.memlet_path(edge)
            index = path.index(edge)

            # Add an edge directly from the source to the next destination connector
            if len(path) > index + 1:
                graph.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)
                if write_only_map:
                    outer_exit = path[index + 1].dst
                    outer_entry = graph.entry_node(outer_exit)
                    if outer_entry is not None:
                        graph.add_edge(outer_entry, None, edge.src, None, Memlet())

        # Remove map
        graph.remove_nodes_from([map_entry, map_exit])

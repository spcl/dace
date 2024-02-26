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
            # Make sure the schedules are correct.
            if outer_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross:
                if inner_map_entry.map.schedule not in (dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential,
                                                        dtypes.ScheduleType.CPU_Multicore):
                    return False
            elif inner_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross:
                if outer_map_entry.map.schedule not in (dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential,
                                                        dtypes.ScheduleType.CPU_Multicore):
                    return False
            elif inner_map_entry.map.schedule != outer_map_entry.map.schedule:
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

        merged_entry, merged_exit = sdutil.merge_maps(graph, outer_map_entry, outer_map_exit, inner_map_entry,
                                                      inner_map_exit)

        if (outer_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross or
            inner_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross):
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
                    if outer_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross:
                        # Since the inner map must be a non-doacross map, we can append the inner map's parameters as
                        # unchanged to the offsets of a sink.
                        for par in inner_map_entry.map.params:
                            new_offset = [old for old in memlet.doacross_dependency_offset]
                            new_offset.append(symbolic.symbol(par))
                            memlet.doacross_dependency_offset = new_offset
                    elif inner_map_entry.map.schedule == dtypes.ScheduleType.CPU_Multicore_Doacross:
                        # The same, but in the other direction.
                        for par in outer_map_entry.map.params:
                            new_offset = [old for old in memlet.doacross_dependency_offset]
                            new_offset.append(symbolic.symbol(par))
                            memlet.doacross_dependency_offset = new_offset

        return merged_entry, merged_exit

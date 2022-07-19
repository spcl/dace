# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-collapse transformation. """

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

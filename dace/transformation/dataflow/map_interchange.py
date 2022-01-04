# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the map interchange transformation. """

from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.symbolic import symlist
from dace.transformation import transformation
from dace.sdfg.propagation import propagate_memlet
from dace.properties import make_properties


class MapInterchange(transformation.SingleStateTransformation):
    """ Implements the map-interchange transformation.
    
        Map-interchange takes two nested maps and interchanges their position.
    """

    outer_map_entry = transformation.PatternNode(nodes.MapEntry)
    inner_map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # TODO: Assuming that the subsets on the edges between the two map
        # entries/exits are the union of separate inner subsets, is it possible
        # that inverting these edges breaks the continuity of union? What about
        # the opposite?

        # Check the edges between the entries of the two maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry

        # Check that inner map range is independent of outer range
        map_deps = set()
        for s in inner_map_entry.map.range:
            map_deps |= set(map(str, symlist(s)))
        if any(dep in outer_map_entry.map.params for dep in map_deps):
            return False

        # Check that the destination of all the outgoing edges
        # from the outer map's entry is the inner map's entry.
        for e in graph.out_edges(outer_map_entry):
            if e.dst != inner_map_entry:
                return False
        # Check that the source of all the incoming edges
        # to the inner map's entry is the outer map's entry.
        for e in graph.in_edges(inner_map_entry):
            if e.src != outer_map_entry:
                return False
            # Check that dynamic input range memlets are independent of
            # first map range
            if e.dst_conn and not e.dst_conn.startswith('IN_'):
                memlet_deps = set()
                for s in e.data.subset:
                    memlet_deps |= set(map(str, symlist(s)))
                if any(dep in outer_map_entry.map.params for dep in memlet_deps):
                    return False

        # Check the edges between the exits of the two maps.
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Check that the destination of all the outgoing edges
        # from the inner map's exit is the outer map's exit.
        for e in graph.out_edges(inner_map_exit):
            if e.dst != outer_map_exit:
                return False
        # Check that the source of all the incoming edges
        # to the outer map's exit is the inner map's exit.
        for e in graph.in_edges(outer_map_exit):
            if e.src != inner_map_exit:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        # Extract the parameters and ranges of the inner/outer maps.
        outer_map_entry = self.outer_map_entry
        inner_map_entry = self.inner_map_entry
        inner_map_exit = graph.exit_node(inner_map_entry)
        outer_map_exit = graph.exit_node(outer_map_entry)

        # Switch connectors
        outer_map_entry.in_connectors, inner_map_entry.in_connectors = \
            inner_map_entry.in_connectors, outer_map_entry.in_connectors
        outer_map_entry.out_connectors, inner_map_entry.out_connectors = \
            inner_map_entry.out_connectors, outer_map_entry.out_connectors
        outer_map_exit.in_connectors, inner_map_exit.in_connectors = \
            inner_map_exit.in_connectors, outer_map_exit.in_connectors
        outer_map_exit.out_connectors, inner_map_exit.out_connectors = \
            inner_map_exit.out_connectors, outer_map_exit.out_connectors

        # Get edges between the map entries and exits.
        entry_edges = graph.edges_between(outer_map_entry, inner_map_entry)
        exit_edges = graph.edges_between(inner_map_exit, outer_map_exit)
        for e in entry_edges + exit_edges:
            graph.remove_edge(e)

        # Change source and destination of edges.
        sdutil.change_edge_dest(graph, outer_map_entry, inner_map_entry)
        sdutil.change_edge_src(graph, inner_map_entry, outer_map_entry)
        sdutil.change_edge_dest(graph, inner_map_exit, outer_map_exit)
        sdutil.change_edge_src(graph, outer_map_exit, inner_map_exit)

        # Add edges between the map entries and exits.
        new_entry_edges = []
        new_exit_edges = []
        for e in entry_edges:
            new_entry_edges.append(graph.add_edge(e.dst, e.src_conn, e.src, e.dst_conn, e.data))
        for e in exit_edges:
            new_exit_edges.append(graph.add_edge(e.dst, e.src_conn, e.src, e.dst_conn, e.data))

        # Repropagate memlets in modified region
        for e in new_entry_edges:
            path = graph.memlet_path(e)
            index = next(i for i, edge in enumerate(path) if e is edge)
            e.data.subset = propagate_memlet(graph, path[index + 1].data, outer_map_entry, True).subset
        for e in new_exit_edges:
            path = graph.memlet_path(e)
            index = next(i for i, edge in enumerate(path) if e is edge)
            e.data.subset = propagate_memlet(graph, path[index - 1].data, outer_map_exit, True).subset

    @staticmethod
    def annotates_memlets():
        return True

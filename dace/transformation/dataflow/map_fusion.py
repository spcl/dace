""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace import data, types, subsets, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import ShapeProperty
import sympy


def calc_set_union(set_a: subsets.Subset,
                   set_b: subsets.Subset) -> subsets.Range:
    """ Computes the union of two Subset objects. """

    if isinstance(set_a, subsets.Indices) or isinstance(
            set_b, subsets.Indices):
        raise NotImplementedError('Set union with indices is not implemented.')
    if not (isinstance(set_a, subsets.Range)
            and isinstance(set_b, subsets.Range)):
        raise TypeError('Can only compute the union of ranges.')
    if len(set_a) != len(set_b):
        raise ValueError('Range dimensions do not match')
    union = []
    for range_a, range_b in zip(set_a, set_b):
        union.append([
            sympy.Min(range_a[0], range_b[0]),
            sympy.Max(range_a[1], range_b[1]),
            sympy.Min(range_a[2], range_b[2]),
        ])
    return subsets.Range(union)


class MapFusion(pattern_matching.Transformation):
    """ Implements the map fusion pattern.

        Map Fusion takes two maps that are connected in series and have the 
        same range, and fuses them to one map. The tasklets in the new map are
        connected in the same manner as they were before the fusion.
    """

    _first_map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _second_map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapFusion._first_map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # The first map must have a non-conflicting map exit.
        # (cannot fuse with CR in the first map)
        first_map_entry = graph.nodes()[candidate[MapFusion._first_map_entry]]
        first_exits = graph.exit_nodes(first_map_entry)
        first_exit = first_exits[0]
        if any([e.data.wcr is not None for e in graph.in_edges(first_exit)]):
            return False

        # Check whether there is a pattern map -> data -> map.
        data_nodes = []
        for _, _, dst, _, _ in graph.out_edges(first_exit):
            if isinstance(dst, nodes.AccessNode):
                data_nodes.append(dst)
            else:
                return False
        second_map_entry = None
        for data_node in data_nodes:
            for _, _, dst, _, _ in graph.out_edges(data_node):
                if isinstance(dst, nodes.MapEntry):
                    if second_map_entry is None:
                        second_map_entry = dst
                    elif dst != second_map_entry:
                        return False
                else:
                    return False
        if second_map_entry is None:
            return False
        for src, _, _, _, _ in graph.in_edges(second_map_entry):
            if not src in data_nodes:
                return False

        # Check map spaces (this should be generalized to ignore order).
        first_range = first_map_entry.map.range
        second_range = second_map_entry.map.range
        if first_range != second_range:
            return False

        # Success
        candidate[MapFusion._second_map_entry] = graph.nodes().index(
            second_map_entry)

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_map_entry = graph.nodes()[candidate[MapFusion._first_map_entry]]
        second_map_entry = graph.nodes()[candidate[
            MapFusion._second_map_entry]]

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [first_map_entry, second_map_entry])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        first_map_entry = graph.nodes()[self.subgraph[
            MapFusion._first_map_entry]]
        first_map_exit = graph.exit_nodes(first_map_entry)[0]
        second_map_entry = graph.nodes()[self.subgraph[
            MapFusion._second_map_entry]]
        second_exits = graph.exit_nodes(second_map_entry)
        first_map_params = [
            symbolic.pystr_to_symbolic(p) for p in first_map_entry.map.params
        ]
        second_map_params = [
            symbolic.pystr_to_symbolic(p) for p in second_map_entry.map.params
        ]

        # Fix exits
        for exit_node in second_exits:
            if isinstance(exit_node, nodes.MapExit):
                exit_node.map = first_map_entry.map

        # Substitute symbols in second map.
        for _parent, _, _child, _, memlet in graph.bfs_edges(
                second_map_entry, reverse=False):
            for fp, sp in zip(first_map_params, second_map_params):
                for ind, r in enumerate(memlet.subset):
                    if isinstance(memlet.subset[ind], tuple):
                        begin = r[0].subs(sp, fp)
                        end = r[1].subs(sp, fp)
                        step = r[2].subs(sp, fp)
                        memlet.subset[ind] = (begin, end, step)
                    else:
                        memlet.subset[ind] = memlet.subset[ind].subs(sp, fp)

        transients = {}
        for _, _, dst, _, memlet in graph.out_edges(first_map_exit):
            if not memlet.data in transients:
                transients[memlet.data] = dst
        new_edges = []
        for src, src_conn, _, dst_conn, memlet in graph.in_edges(
                first_map_exit):
            new_memlet = dcpy(memlet)
            new_edges.append((src, src_conn, transients[memlet.data], dst_conn,
                              new_memlet))
        for _, src_conn, dst, dst_conn, memlet in graph.out_edges(
                second_map_entry):
            new_memlet = dcpy(memlet)
            new_edges.append((transients[memlet.data], src_conn, dst, dst_conn,
                              new_memlet))

        # Delete nodes/edges
        for edge in graph.in_edges(first_map_exit):
            graph.remove_edge(edge)
        for edge in graph.out_edges(second_map_entry):
            graph.remove_edge(edge)
        data_nodes = []
        for _, _, dst, _, _ in graph.out_edges(first_map_exit):
            data_nodes.append(dst)
        for data_node in data_nodes:
            for edge in graph.all_edges(data_node):
                graph.remove_edge(edge)
        graph.remove_node(first_map_exit)
        graph.remove_node(second_map_entry)

        # Add edges
        for edge in new_edges:
            graph.add_edge(*edge)

        # Reduce transient sizes
        for data_node in data_nodes:
            data_desc = data_node.desc(sdfg)
            if data_desc.transient:
                edges = graph.in_edges(data_node)
                subset = edges[0].data.subset
                for idx in range(1, len(edges)):
                    subset = calc_set_union(subset, edges[idx].data.subset)
                data_desc.shape = subset.bounding_box_size()
                data_desc.strides = list(subset.bounding_box_size())
                data_desc.offset = [0] * subset.dims()


pattern_matching.Transformation.register_pattern(MapFusion)

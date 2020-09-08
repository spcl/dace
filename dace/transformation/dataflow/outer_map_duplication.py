""" This module contains classes and functions that implement the outer map
    duplication transformation. """

from copy import deepcopy as dcpy
import dace
from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching


@registry.autoregister_params(singlestate=True)
@make_properties
class OuterMapDuplication(pattern_matching.Transformation):
    """ Implements the outer map duplication transformation.

        Outer map duplication is a restricted vertical map split. If a map scope
        contains only other independent scopes, then the outer map can be
        duplicated, once for each of the internal scopes.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(OuterMapDuplication._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[OuterMapDuplication._map_entry]]
        map_exit = graph.exit_nodes(map_entry)[0]
        entries = set()
        for _, _, dst, _, _ in graph.out_edges(map_entry):
            if not isinstance(dst, nodes.EntryNode):
                return False
            else:
                entries.add(dst)
        for src, _, _, _, _ in graph.in_edges(map_exit):
            if not isinstance(src, nodes.ExitNode):
                return False
        if len(entries) > 1:
            return True
        return False

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[OuterMapDuplication._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[self.subgraph[OuterMapDuplication._map_entry]]
        map_exit = graph.exit_nodes(map_entry)[0]

        # Find nested entry nodes.
        entries = set()
        for _, _, dst, _, _ in graph.out_edges(map_entry):
            entries.add(dst)
        
        # Duplicate outer map and move inner scopes.
        duplicate_map_entries = []
        for i, entry_node in enumerate(entries):

            if i == 0:
                continue

            exit_node = graph.exit_nodes(entry_node)[0]

            new_map_entry = dcpy(map_entry)
            new_map_exit = dcpy(map_exit)
            graph.add_nodes_from([new_map_entry, new_map_exit])
            duplicate_map_entries.append(new_map_entry)

            conn_dict = dict()
            edges = set()
            for e in graph.out_edges(map_entry):
                _, src_conn, dst, dst_conn, mem = e
                if src_conn in conn_dict.keys():
                    conn_dict[src_conn] += 1
                else:
                    conn_dict[src_conn] = 1
                if dst is entry_node:
                    graph.add_edge(new_map_entry, src_conn, dst, dst_conn,
                                   dcpy(mem))
                    edges.add(e)
            for e in edges:
                src_conn = e.src_conn
                graph.remove_edge(e)
                conn_dict[src_conn] -= 1
            for e in graph.in_edges(map_entry):
                src, src_conn, _, dst_conn, mem = e
                corr_src_conn = dst_conn
                if len(dst_conn) > 3 and dst_conn[:3] == 'IN_':
                    corr_src_conn = 'OUT_' + dst_conn[3:]
                if corr_src_conn in conn_dict.keys():
                    graph.add_edge(src, src_conn, new_map_entry, dst_conn,
                                   dcpy(mem))
                    if conn_dict[corr_src_conn] == 0:
                        graph.remove_edge(e)
            new_conn = set()
            for src, src_conn, _, dst_conn, mem in dace.sdfg.dynamic_map_inputs(
                graph, map_entry):
                new_conn.add(dst_conn)
                graph.add_edge(src, src_conn, new_map_entry, dst_conn,
                               dcpy(mem))
            new_map_entry.in_connectors = new_map_entry.in_connectors.union(new_conn)


            conn_dict = dict()
            edges = set()
            for e in graph.in_edges(map_exit):
                src, src_conn, _, dst_conn, mem = e
                if dst_conn in conn_dict.keys():
                    conn_dict[dst_conn] += 1
                else:
                    conn_dict[dst_conn] = 1
                if src is exit_node:
                    graph.add_edge(src, src_conn, new_map_exit, dst_conn,
                                   dcpy(mem))
                    edges.add(e)
            for e in edges:
                dst_conn = e.dst_conn
                graph.remove_edge(e)
                conn_dict[dst_conn] -= 1
            for e in graph.out_edges(map_exit):
                _, src_conn, dst, dst_conn, mem = e
                corr_dst_conn = src_conn
                if len(src_conn) > 4 and src_conn[:4] == 'OUT_':
                    corr_dst_conn = 'IN_' + src_conn[4:]
                if corr_dst_conn in conn_dict.keys():
                    graph.add_edge(new_map_exit, src_conn, dst, dst_conn,
                                   dcpy(mem))
                    if conn_dict[corr_dst_conn] == 0:
                        graph.remove_edge(e)
    
        return duplicate_map_entries
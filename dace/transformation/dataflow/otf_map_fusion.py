# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement the OTF map fusion transformation.
"""
from copy import deepcopy as dcpy
from collections import defaultdict
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import nodes as nds
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import data as dt


class OTFMapFusion(transformation.SingleStateTransformation):
    """ Performs fusion of two maps by replicating the contents of the first into the second map
        until all the input dependencies (memlets) of the second one are met.
    """
    first_map_exit = transformation.PatternNode(nds.ExitNode)
    array = transformation.PatternNode(nds.AccessNode)
    second_map_entry = transformation.PatternNode(nds.EntryNode)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # WCR: not supported on first map
        for _in_e in graph.in_edges(self.first_map_exit):
            if _in_e.data.wcr is not None:
                return False

        # Check intermediate nodes between both maps.
        for _, _, node, _, _ in graph.out_edges(self.first_map_exit):
            # Only map -> array -> map
            if not isinstance(node, nds.AccessNode):
                return False

            # Non-transient blocks removal of first map
            if not sdfg.arrays[node.data].transient:
                return False

            # Check that array is not co-produced by other parent map.
            producers = set(map(lambda edge: edge.src, graph.in_edges(node)))
            for prod in producers:
                if prod != self.first_map_exit:
                    return False

            # Check that array is not co-consumed by other child mao
            consumers = set(map(lambda edge: edge.dst, graph.out_edges(node)))
            for cons in consumers:
                if cons != self.second_map_entry:
                    return False

        # Success
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        first_map_entry = graph.entry_node(self.first_map_exit)

        intermediate_dnodes = set()
        for _, _, node, _, _ in graph.out_edges(self.first_map_exit):
            if not isinstance(node, nds.AccessNode):
                continue

            intermediate_dnodes.add(node)

        self._update_in_connectors(graph, intermediate_dnodes)
        self._replicate_first_map(sdfg, graph, first_map_entry, intermediate_dnodes)

        graph.remove_nodes_from(
            graph.all_nodes_between(first_map_entry, self.first_map_exit) | {first_map_entry, self.first_map_exit})

        for node in graph.nodes():
            if not isinstance(node, nds.AccessNode):
                continue

            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                graph.remove_node(node)

    def _update_in_connectors(self, graph, intermediate_dnodes):
        first_map_entry = graph.entry_node(self.first_map_exit)
        for dnode in intermediate_dnodes:
            for edge in graph.edges_between(dnode, self.second_map_entry):
                graph.remove_edge_and_connectors(edge)

        for edge in graph.in_edges(first_map_entry):
            if self.second_map_entry.add_in_connector(edge.dst_conn + "_"):
                graph.add_edge(edge.src, edge.src_conn, self.second_map_entry, edge.dst_conn + "_", edge.data)
            else:
                raise ValueError("Failed to connect")

    def _replicate_first_map(self, sdfg, graph, first_map_entry, intermediate_dnodes):
        for dnode in intermediate_dnodes:
            array_name = dnode.data
            array = sdfg.arrays[array_name]

            read_offsets = self._read_offsets(graph, array_name)

            # Replicate first map tasklets once for each read offset access and
            # connect them to other tasklets accordingly
            for offset, edges in read_offsets.items():
                new_nodes = self._copy_first_map_contents(sdfg, graph, first_map_entry)
                tmp_name = "__otf"
                tmp_name, _ = sdfg.add_scalar(tmp_name, array.dtype, transient=True, find_new_name=True)
                tmp_access = graph.add_access(tmp_name)

                for node in new_nodes:
                    for edge in graph.edges_between(node, self.first_map_exit):
                        graph.add_edge(edge.src, edge.src_conn, tmp_access, None, Memlet(tmp_name))
                        graph.remove_edge(edge)

                    for edge in graph.edges_between(first_map_entry, node):
                        memlet = dcpy(edge.data)
                        memlet.subset.offset(list(offset), negative=False)
                        self.second_map_entry.add_out_connector(edge.src_conn + "_")
                        graph.add_edge(self.second_map_entry, edge.src_conn + "_", node, edge.dst_conn, memlet)
                        graph.remove_edge(edge)

                for edge in edges:
                    graph.add_edge(tmp_access, None, edge.dst, edge.dst_conn, Memlet(tmp_name))

    def _read_offsets(self, state, array_name):
        """Compute offsets of read accesses in second map."""
        # Get output memlet of first tasklet
        output_edges = state.in_edges(self.first_map_exit)
        assert len(output_edges) == 1
        write_memlet = output_edges[0].data

        # Find read offsets by looping over second map entry connectors
        offsets = defaultdict(list)
        for edge in state.out_edges(self.second_map_entry):
            if edge.data.data == array_name:
                self.second_map_entry.remove_out_connector(edge.src_conn)
                state.remove_edge(edge)
                offset = OTFMapFusion._memlet_offsets(write_memlet, edge.data)
                offsets[offset].append(edge)

        return offsets

    def _copy_first_map_contents(self, sdfg, graph, first_map_entry):
        inter_nodes = list(graph.all_nodes_between(first_map_entry, self.first_map_exit) - {first_map_entry})
        new_inter_nodes = [dcpy(node) for node in inter_nodes]
        tmp_map = dict()
        for node in new_inter_nodes:
            if isinstance(node, nds.AccessNode):
                data = sdfg.arrays[node.data]
                if isinstance(data, dt.Scalar) and data.transient:
                    tmp_name = sdfg.temp_data_name()
                    sdfg.add_scalar(tmp_name, data.dtype, transient=True)
                    tmp_map[node.data] = tmp_name
                    node.data = tmp_name
            graph.add_node(node)
        id_map = {graph.node_id(old): graph.node_id(new) for old, new in zip(inter_nodes, new_inter_nodes)}

        def map_node(node):
            return graph.node(id_map[graph.node_id(node)])

        def map_memlet(memlet):
            memlet = dcpy(memlet)
            memlet.data = tmp_map.get(memlet.data, memlet.data)
            return memlet

        for edge in graph.edges():
            if edge.src in inter_nodes or edge.dst in inter_nodes:
                src = map_node(edge.src) if edge.src in inter_nodes else edge.src
                dst = map_node(edge.dst) if edge.dst in inter_nodes else edge.dst
                edge_data = map_memlet(edge.data)
                graph.add_edge(src, edge.src_conn, dst, edge.dst_conn, edge_data)

        return new_inter_nodes

    @staticmethod
    def _memlet_offsets(base_memlet, offset_memlet):
        """Compute subset offset of `offset_memlet` relative to `base_memlet`."""
        def offset(base_range, offset_range):
            b0, e0, s0 = base_range
            b1, e1, s1 = offset_range
            assert e1 - e0 == b1 - b0 and s0 == s1
            return int(e1 - e0)

        return tuple(offset(b, o) for b, o in zip(base_memlet.subset.ranges, offset_memlet.subset.ranges))

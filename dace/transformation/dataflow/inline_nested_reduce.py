# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy

from dace.sdfg import SDFG, SDFGState
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm


class InlineNestedReduce(pm.SingleStateTransformation):
    nested_sdfg = pm.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # NestedSDFG: Vanilla reduction
        # - Init state
        # - Reduce state
        # - Trivial init->reduce transition

        # Check: two states
        if len(self.nested_sdfg.sdfg.nodes()) != 2:
            return False

        # Check: Trivial transition
        interstate_edges = self.nested_sdfg.sdfg.edges()
        if len(interstate_edges) != 1:
            return False

        interstate_edge = interstate_edges[0]
        if not interstate_edge.data.is_unconditional():
            return False

        init_state = interstate_edge.src
        reduce_state = interstate_edge.dst

        # Check: Single reduction output
        out_connectors = set(self.nested_sdfg.out_connectors.keys())
        if len(out_connectors) > 1:
            return False

        # Init state
        # Check: Independent, i.e. not consuming any arrays from outside
        in_connectors = set(self.nested_sdfg.in_connectors.keys())
        for access_node in init_state.data_nodes():
            if access_node.data in in_connectors:
                return False

        # Reduce state
        # Check:
        # - inputs from in connectors
        # - outputs to out connectors
        for access_node in reduce_state.data_nodes():
            if reduce_state.in_degree(access_node) == 0:
                if not access_node.data in in_connectors:
                    return False
            elif reduce_state.out_degree(access_node) == 0:
                if not access_node.data in out_connectors:
                    return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        interstate_edge = self.nested_sdfg.sdfg.edges()[0]
        init_state = interstate_edge.src
        reduce_state = interstate_edge.dst

        out_connectors = set(self.nested_sdfg.out_connectors.keys())

        # Phase 1: Renaming
        # Find unique names for output arrays
        array_mapping = {}
        for out_conn in out_connectors:
            new_out_connector = sdfg._find_new_name(out_conn)
            assert not new_out_connector in sdfg.arrays

            # Rename array in nested SDFG
            # Create duplicate array in outer SDFG
            new_desc = copy.deepcopy(self.nested_sdfg.sdfg.arrays[out_conn])
            new_desc.transient = False
            sdfg.arrays[new_out_connector] = new_desc
            self.nested_sdfg.sdfg.arrays[new_out_connector] = new_desc

            # Add new out connector
            self.nested_sdfg.add_out_connector(new_out_connector)

            # Remember connector mapping
            array_mapping[out_conn] = new_out_connector

            del self.nested_sdfg.sdfg.arrays[out_conn]

        # Re-label init state with new array names
        for node in init_state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in array_mapping:
                node.data = array_mapping[node.data]

            for conn in list(node.in_connectors.keys()):
                if not conn.startswith("IN_"):
                    continue

                data = conn[3:]
                if data in array_mapping:
                    node.add_in_connector("IN_" + array_mapping[data])
                    node.remove_in_connector(conn)

            for conn in list(node.out_connectors.keys()):
                if not conn.startswith("OUT_"):
                    continue

                data = conn[4:]
                if data in array_mapping:
                    node.add_out_connector("OUT_" + array_mapping[data])
                    node.remove_out_connector(conn)

        for edge in init_state.edges():
            memlet = edge.data
            if memlet.data is None or not memlet.data in array_mapping:
                continue

            memlet.data = array_mapping[memlet.data]

            if not edge.src_conn is None and edge.src_conn.startswith("OUT_"):
                conn_data = edge.src_conn[4:]
                if conn_data in array_mapping:
                    edge.src_conn = "OUT_" + array_mapping[conn_data]

            if not edge.dst_conn is None and edge.dst_conn.startswith("IN_"):
                conn_data = edge.dst_conn[3:]
                if conn_data in array_mapping:
                    edge.dst_conn = "IN_" + array_mapping[conn_data]

        # Re-label reduce state with new array namees
        for node in reduce_state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in array_mapping:
                node.data = array_mapping[node.data]

            for conn in list(node.in_connectors.keys()):
                if not conn.startswith("IN_"):
                    continue

                data = conn[3:]
                if data in array_mapping:
                    node.add_in_connector("IN_" + array_mapping[data])
                    node.remove_in_connector(conn)

            for conn in list(node.out_connectors.keys()):
                if not conn.startswith("OUT_"):
                    continue

                data = conn[4:]
                if data in array_mapping:
                    node.add_out_connector("OUT_" + array_mapping[data])
                    node.remove_out_connector(conn)

        for edge in reduce_state.edges():
            memlet = edge.data
            if memlet.data is None or not memlet.data in array_mapping:
                continue

            memlet.data = array_mapping[memlet.data]

            if not edge.src_conn is None and edge.src_conn.startswith("OUT_"):
                conn_data = edge.src_conn[4:]
                if conn_data in array_mapping:
                    edge.src_conn = "OUT_" + array_mapping[conn_data]

            if not edge.dst_conn is None and edge.dst_conn.startswith("IN_"):
                conn_data = edge.dst_conn[3:]
                if conn_data in array_mapping:
                    edge.dst_conn = "IN_" + array_mapping[conn_data]

        # Swap out connectors for new out connectors
        for out_conn in out_connectors:
            for edge in graph.out_edges_by_connector(self.nested_sdfg, out_conn):
                new_out_conn = array_mapping[out_conn]

                memlet = copy.deepcopy(edge.data)
                graph.add_edge(edge.src, new_out_conn, edge.dst, edge.dst_conn, memlet)

                graph.remove_edge(edge)

            self.nested_sdfg.remove_out_connector(out_conn)

        # Update local variable
        out_connectors = set(self.nested_sdfg.out_connectors.keys())

        # Phase 2: Move states
        # Prepend new init state to this state
        new_init_state = sdfg.add_state_before(graph)

        # Add nodes to new init
        new_nodes = [copy.deepcopy(node) for node in init_state.nodes()]
        node_id_map = {}
        for old_node, new_node in zip(init_state.nodes(), new_nodes):
            new_init_state.add_node(new_node)

            node_id_map[init_state.node_id(old_node)] = new_init_state.node_id(new_node)

        # Add edges to new init
        for edge in init_state.edges():
            new_memlet = copy.deepcopy(edge.data)

            new_src = new_init_state.node(node_id_map[init_state.node_id(edge.src)])
            new_dst = new_init_state.node(node_id_map[init_state.node_id(edge.dst)])
            new_init_state.add_edge(new_src, edge.src_conn, new_dst, edge.dst_conn, new_memlet)

        self.nested_sdfg.sdfg.remove_node(init_state)

        # Add new write nodes
        for data in out_connectors:
            access_node = graph.add_access(data)

            memlet = Memlet.from_array(data, sdfg.arrays[data])
            new_edge = graph.add_edge(self.nested_sdfg, data, access_node, None, memlet)

            for edge in graph.edges_by_connector(self.nested_sdfg, data):
                if edge == new_edge:
                    continue

                graph.remove_edge(edge)

                memlet_ = copy.deepcopy(edge.data)
                graph.add_edge(access_node, None, edge.dst, edge.dst_conn, memlet_)

        from dace.transformation.interstate import InlineSDFG
        xform = InlineSDFG()
        xform.nested_sdfg = self.nested_sdfg
        xform.apply(graph, sdfg)

        for new_array in array_mapping.values():
            sdfg.arrays[new_array].transient = True

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation import transformation
from dace import memlet
from dace.sdfg.graph import OrderedDiGraph
from dace import memlet
from dace.sdfg import nodes, utils, graph as gr
from dace.sdfg import SDFGState
from dace.sdfg.propagation import propagate_memlet


class InMergeArrays(transformation.SingleStateTransformation, transformation.SimplifyPass):
    """ Merge duplicate arrays connected to the same scope entry. """

    array1 = transformation.PatternNode(nodes.AccessNode)
    array2 = transformation.PatternNode(nodes.AccessNode)
    map_entry = transformation.PatternNode(nodes.EntryNode)

    @classmethod
    def expressions(cls):
        # Matching
        #   o  o
        #   |  |
        # /======\

        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.array2)
        g.add_node(cls.map_entry)
        g.add_edge(cls.array1, cls.map_entry, None)
        g.add_edge(cls.array2, cls.map_entry, None)
        return [g]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg, permissive=False):
        # Ensure both arrays contain the same data
        arr1 = self.array1
        arr2 = self.array2
        if arr1.data != arr2.data:
            return False

        # Ensure only arr1's node ID contains incoming edges
        if graph.in_degree(arr2) > 0:
            return False

        # Ensure arr1 and arr2's node IDs are ordered (avoid duplicates)
        arr1_id = graph.node_id(self.array1)
        arr2_id = graph.node_id(self.array2)
        if (graph.in_degree(arr1) == 0 and graph.in_degree(arr2) == 0 and arr1_id >= arr2_id):
            return False

        map = self.map_entry

        # If array's connector leads directly to map, skip it
        if all(e.dst_conn and not e.dst_conn.startswith('IN_') for e in graph.edges_between(arr1, map)):
            return False
        if all(e.dst_conn and not e.dst_conn.startswith('IN_') for e in graph.edges_between(arr2, map)):
            return False

        if (any(e.dst != map for e in graph.out_edges(arr1)) or any(e.dst != map for e in graph.out_edges(arr2))):
            return False

        # Ensure arr1 and arr2 are the first two incoming nodes (avoid further
        # duplicates)
        all_source_nodes = set(
            graph.node_id(e.src) for e in graph.in_edges(map) if e.src != arr1 and e.src != arr2
            and e.src.data == arr1.data and e.dst_conn and e.dst_conn.startswith('IN_') and graph.in_degree(e.src) == 0)
        if any(nid < arr1_id or nid < arr2_id for nid in all_source_nodes):
            return False

        return True

    def match_to_str(self, graph):
        arr = self.array1
        map = self.map_entry
        nid1, nid2 = graph.node_id(self.array1), graph.node_id(self.array2)
        return '%s (%d, %d) -> %s' % (arr.data, nid1, nid2, map.label)

    def apply(self, graph, sdfg):
        array = self.array1
        map = self.map_entry
        map_edge = next(e for e in graph.out_edges(array) if e.dst == map)
        result_connector = map_edge.dst_conn[3:]

        # Find all other incoming access nodes without incoming edges
        source_edges = [
            e for e in graph.in_edges(map) if isinstance(e.src, nodes.AccessNode) and e.src.data == array.data
            and e.src != array and e.dst_conn and e.dst_conn.startswith('IN_') and graph.in_degree(e.src) == 0
        ]

        # Modify connectors to point to first array
        connectors_to_remove = set()
        for e in source_edges:
            connector = e.dst_conn[3:]
            connectors_to_remove.add(connector)
            for inner_edge in graph.out_edges(map):
                if inner_edge.src_conn[4:] == connector:
                    inner_edge._src_conn = 'OUT_' + result_connector

        # Remove other nodes from state
        graph.remove_nodes_from(set(e.src for e in source_edges))

        # Remove connectors from scope entry
        for c in connectors_to_remove:
            map.remove_in_connector('IN_' + c)
            map.remove_out_connector('OUT_' + c)

        # Re-propagate memlets
        edge_to_propagate = next(e for e in graph.out_edges(map) if e.src_conn[4:] == result_connector)
        map_edge._data = propagate_memlet(dfg_state=graph,
                                          memlet=edge_to_propagate.data,
                                          scope_node=map,
                                          union_inner_edges=True)


class OutMergeArrays(transformation.SingleStateTransformation, transformation.SimplifyPass):
    """ Merge duplicate arrays connected to the same scope entry. """

    array1 = transformation.PatternNode(nodes.AccessNode)
    array2 = transformation.PatternNode(nodes.AccessNode)
    map_exit = transformation.PatternNode(nodes.ExitNode)

    @classmethod
    def expressions(cls):
        # Matching
        # \======/
        #   |  |
        #   o  o

        g = OrderedDiGraph()
        g.add_node(cls.array1)
        g.add_node(cls.array2)
        g.add_node(cls.map_exit)
        g.add_edge(cls.map_exit, cls.array1, None)
        g.add_edge(cls.map_exit, cls.array2, None)
        return [g]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        arr1_id = self.subgraph[OutMergeArrays.array1]
        arr2_id = self.subgraph[OutMergeArrays.array2]

        # Ensure both arrays contain the same data
        arr1 = self.array1
        arr2 = self.array2
        if arr1.data != arr2.data:
            return False

        # Ensure only arr1's node ID contains outgoing edges
        if graph.out_degree(arr2) > 0:
            return False

        # Ensure arr1 and arr2's node IDs are ordered (avoid duplicates)
        if (graph.out_degree(arr1) == 0 and graph.out_degree(arr2) == 0 and arr1_id >= arr2_id):
            return False

        map = self.map_exit

        if (any(e.src != map for e in graph.in_edges(arr1)) or any(e.src != map for e in graph.in_edges(arr2))):
            return False

        # Ensure arr1 and arr2 are the first two sink nodes (avoid further
        # duplicates)
        all_sink_nodes = set(
            graph.node_id(e.dst) for e in graph.out_edges(map)
            if e.dst != arr1 and e.dst != arr2 and e.dst.data == arr1.data and e.src_conn
            and e.src_conn.startswith('OUT_') and graph.out_degree(e.dst) == 0)
        if any(nid < arr1_id or nid < arr2_id for nid in all_sink_nodes):
            return False

        return True

    def match_to_str(self, graph):
        arr = self.array1
        map = self.map_exit
        nid1, nid2 = graph.node_id(self.array1), graph.node_id(self.array2)
        return '%s (%d, %d) -> %s' % (arr.data, nid1, nid2, map.label)

    def apply(self, graph, sdfg):
        array = self.array1
        map = self.map_exit
        map_edge = next(e for e in graph.in_edges(array) if e.src == map)
        result_connector = map_edge.src_conn[4:]

        # Find all other outgoing access nodes without outgoing edges
        dst_edges = [
            e for e in graph.out_edges(map) if isinstance(e.dst, nodes.AccessNode) and e.dst.data == array.data
            and e.dst != array and e.src_conn and e.src_conn.startswith('OUT_') and graph.out_degree(e.dst) == 0
        ]

        # Modify connectors to point to first array
        connectors_to_remove = set()
        for e in dst_edges:
            connector = e.src_conn[4:]
            connectors_to_remove.add(connector)
            for inner_edge in graph.in_edges(map):
                if inner_edge.dst_conn[3:] == connector:
                    inner_edge.dst_conn = 'IN_' + result_connector

        # Remove other nodes from state
        graph.remove_nodes_from(set(e.dst for e in dst_edges))

        # Remove connectors from scope entry
        for c in connectors_to_remove:
            map.remove_in_connector('IN_' + c)
            map.remove_out_connector('OUT_' + c)

        # Re-propagate memlets
        edge_to_propagate = next(e for e in graph.in_edges(map) if e.dst_conn[3:] == result_connector)
        map_edge._data = propagate_memlet(dfg_state=graph,
                                          memlet=edge_to_propagate.data,
                                          scope_node=map,
                                          union_inner_edges=True)


class MergeSourceSinkArrays(transformation.SingleStateTransformation, transformation.SimplifyPass):
    """ Merge duplicate arrays that are source/sink nodes. """

    array1 = transformation.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        # Matching
        #   o  o
        return [utils.node_path_graph(cls.array1)]

        g = OrderedDiGraph()
        g.add_node(MergeSourceSinkArrays._array1)
        return [g]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        arr1_id = self.subgraph[MergeSourceSinkArrays.array1]
        arr1 = self.array1

        # Ensure array is either a source or sink node
        src_nodes = graph.source_nodes()
        sink_nodes = graph.sink_nodes()
        if arr1 in src_nodes:
            nodes_to_consider = src_nodes
        elif arr1 in sink_nodes:
            nodes_to_consider = sink_nodes
        else:
            return False

        # Ensure there are more nodes with the same data
        other_nodes = [
            graph.node_id(n) for n in nodes_to_consider
            if isinstance(n, nodes.AccessNode) and n.data == arr1.data and n != arr1
        ]
        if len(other_nodes) == 0:
            return False

        # Ensure arr1 is the first node to avoid further duplicates
        nid = min(other_nodes)
        if nid < arr1_id:
            return False

        return True

    def apply(self, graph, sdfg):
        array = self.array1
        if array in graph.source_nodes():
            src_node = True
            nodes_to_consider = graph.source_nodes()
            edges_to_consider = lambda n: graph.out_edges(n)
        else:
            src_node = False
            nodes_to_consider = graph.sink_nodes()
            edges_to_consider = lambda n: graph.in_edges(n)

        for node in nodes_to_consider:
            if node == array:
                continue
            if not isinstance(node, nodes.AccessNode):
                continue
            if node.data != array.data:
                continue
            for edge in list(edges_to_consider(node)):
                if src_node:
                    graph.add_edge(array, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                else:
                    graph.add_edge(edge.src, edge.src_conn, array, edge.dst_conn, edge.data)
                graph.remove_edge(edge)
            graph.remove_node(node)

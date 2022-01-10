# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation import transformation
from dace import memlet, registry
from dace.sdfg import nodes
from dace.sdfg.graph import OrderedDiGraph
from dace.sdfg.propagation import propagate_memlet


@registry.autoregister_params(singlestate=True, coarsening=True)
class InMergeArrays(transformation.Transformation):
    """ Merge duplicate arrays connected to the same scope entry. """

    _array1 = nodes.AccessNode("_")
    _array2 = nodes.AccessNode("_")
    _map_entry = nodes.EntryNode()

    @staticmethod
    def expressions():
        # Matching
        #   o  o
        #   |  |
        # /======\

        g = OrderedDiGraph()
        g.add_node(InMergeArrays._array1)
        g.add_node(InMergeArrays._array2)
        g.add_node(InMergeArrays._map_entry)
        g.add_edge(InMergeArrays._array1, InMergeArrays._map_entry, None)
        g.add_edge(InMergeArrays._array2, InMergeArrays._map_entry, None)
        return [g]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        arr1_id = candidate[InMergeArrays._array1]
        arr2_id = candidate[InMergeArrays._array2]

        # Ensure both arrays contain the same data
        arr1 = graph.node(arr1_id)
        arr2 = graph.node(arr2_id)
        if arr1.data != arr2.data:
            return False

        # Ensure only arr1's node ID contains incoming edges
        if graph.in_degree(arr2) > 0:
            return False

        # Ensure arr1 and arr2's node IDs are ordered (avoid duplicates)
        if (graph.in_degree(arr1) == 0 and graph.in_degree(arr2) == 0 and arr1_id >= arr2_id):
            return False

        map = graph.node(candidate[InMergeArrays._map_entry])

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

    @staticmethod
    def match_to_str(graph, candidate):
        arr = graph.node(candidate[InMergeArrays._array1])
        map = graph.node(candidate[InMergeArrays._map_entry])
        return '%s (%d, %d) -> %s' % (arr.data, candidate[InMergeArrays._array1], candidate[InMergeArrays._array2],
                                      map.label)

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)
        array = graph.node(self.subgraph[InMergeArrays._array1])
        map = graph.node(self.subgraph[InMergeArrays._map_entry])
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


@registry.autoregister_params(singlestate=True, coarsening=True)
class OutMergeArrays(transformation.Transformation):
    """ Merge duplicate arrays connected to the same scope entry. """

    _array1 = nodes.AccessNode("_")
    _array2 = nodes.AccessNode("_")
    _map_exit = nodes.ExitNode()

    @staticmethod
    def expressions():
        # Matching
        # \======/
        #   |  |
        #   o  o

        g = OrderedDiGraph()
        g.add_node(OutMergeArrays._array1)
        g.add_node(OutMergeArrays._array2)
        g.add_node(OutMergeArrays._map_exit)
        g.add_edge(OutMergeArrays._map_exit, OutMergeArrays._array1, None)
        g.add_edge(OutMergeArrays._map_exit, OutMergeArrays._array2, None)
        return [g]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        arr1_id = candidate[OutMergeArrays._array1]
        arr2_id = candidate[OutMergeArrays._array2]

        # Ensure both arrays contain the same data
        arr1 = graph.node(arr1_id)
        arr2 = graph.node(arr2_id)
        if arr1.data != arr2.data:
            return False

        # Ensure only arr1's node ID contains outgoing edges
        if graph.out_degree(arr2) > 0:
            return False

        # Ensure arr1 and arr2's node IDs are ordered (avoid duplicates)
        if (graph.out_degree(arr1) == 0 and graph.out_degree(arr2) == 0 and arr1_id >= arr2_id):
            return False

        map = graph.node(candidate[OutMergeArrays._map_exit])

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

    @staticmethod
    def match_to_str(graph, candidate):
        arr = graph.node(candidate[OutMergeArrays._array1])
        map = graph.node(candidate[OutMergeArrays._map_exit])
        return '%s (%d, %d) -> %s' % (arr.data, candidate[OutMergeArrays._array1], candidate[OutMergeArrays._array2],
                                      map.label)

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)
        array = graph.node(self.subgraph[OutMergeArrays._array1])
        map = graph.node(self.subgraph[OutMergeArrays._map_exit])
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


@registry.autoregister_params(singlestate=True, coarsening=True)
class MergeSourceSinkArrays(transformation.Transformation):
    """ Merge duplicate arrays that are source/sink nodes. """

    _array1 = nodes.AccessNode("_")

    @staticmethod
    def expressions():
        # Matching
        #   o  o

        g = OrderedDiGraph()
        g.add_node(MergeSourceSinkArrays._array1)
        return [g]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        arr1_id = candidate[MergeSourceSinkArrays._array1]
        arr1 = graph.node(arr1_id)

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

    @staticmethod
    def match_to_str(graph, candidate):
        arr = graph.node(candidate[MergeSourceSinkArrays._array1])
        if arr in graph.source_nodes():
            place = 'source'
        else:
            place = 'sink'
        return '%s array %s' % (place, arr.data)

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)
        array = graph.node(self.subgraph[MergeSourceSinkArrays._array1])
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

from dace.transformation import pattern_matching
from dace import memlet, registry
from dace.sdfg import nodes
from dace.sdfg import SDFGState
from dace.sdfg.propagation import propagate_memlet


@registry.autoregister_params(singlestate=True, strict=True)
class MergeArrays(pattern_matching.Transformation):
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

        g = SDFGState()
        g.add_node(MergeArrays._array1)
        g.add_node(MergeArrays._array2)
        g.add_node(MergeArrays._map_entry)
        g.add_edge(MergeArrays._array1, None, MergeArrays._map_entry, None,
                   memlet.Memlet())
        g.add_edge(MergeArrays._array2, None, MergeArrays._map_entry, None,
                   memlet.Memlet())
        return [g]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        arr1_id = candidate[MergeArrays._array1]
        arr2_id = candidate[MergeArrays._array2]

        # Ensure both arrays contain the same data
        arr1 = graph.node(arr1_id)
        arr2 = graph.node(arr2_id)
        if arr1.data != arr2.data:
            return False

        # Ensure only arr1's node ID contains incoming edges
        if graph.in_degree(arr2) > 0:
            return False

        # Ensure arr1 and arr2's node IDs are ordered (avoid duplicates)
        if (graph.in_degree(arr1) == 0 and graph.in_degree(arr2) == 0
                and arr1_id >= arr2_id):
            return False

        map = graph.node(candidate[MergeArrays._map_entry])

        # If arr1's connector leads directly to map, skip it
        if all(e.dst_conn and not e.dst_conn.startswith('IN_')
               for e in graph.edges_between(arr1, map)):
            return False

        if (any(e.dst != map for e in graph.out_edges(arr1))
                or any(e.dst != map for e in graph.out_edges(arr2))):
            return False

        # Ensure arr1 and arr2 are the first two incoming nodes (avoid further
        # duplicates)
        all_source_nodes = set(
            graph.node_id(e.src) for e in graph.in_edges(map) if e.src != arr1
            and e.src != arr2 and e.src.data == arr1.data and e.dst_conn
            and e.dst_conn.startswith('IN_') and graph.in_degree(e.src) == 0)
        if any(nid < arr1_id or nid < arr2_id for nid in all_source_nodes):
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        arr = graph.node(candidate[MergeArrays._array1])
        map = graph.node(candidate[MergeArrays._map_entry])
        return '%s (%d, %d) -> %s' % (arr.data, candidate[MergeArrays._array1],
                                      candidate[MergeArrays._array2], map.label)

    def apply(self, sdfg):
        graph = sdfg.node(self.state_id)
        array = graph.node(self.subgraph[MergeArrays._array1])
        map = graph.node(self.subgraph[MergeArrays._map_entry])
        map_edge = next(e for e in graph.out_edges(array) if e.dst == map)
        result_connector = map_edge.dst_conn[3:]

        # Find all other incoming access nodes without incoming edges
        source_edges = [
            e for e in graph.in_edges(map)
            if isinstance(e.src, nodes.AccessNode) and e.src.data == array.data
            and e.src != array and e.dst_conn and e.dst_conn.startswith('IN_')
            and graph.in_degree(e.src) == 0
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
        map_edge._data = propagate_memlet(dfg_state=graph,
                                          memlet=map_edge.data,
                                          scope_node=map,
                                          union_inner_edges=True)

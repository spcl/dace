# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes, registry
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.state import StateSubgraphView
from dace.symbolic import evaluate
from dace.transformation import transformation
from dace.properties import make_properties
import copy
import itertools


@registry.autoregister_params(singlestate=True)
@make_properties
class MapUnroll(transformation.Transformation):
    """
    Unrolls a map with constant ranges directly in the SDFG by replicating its
    subgraph for each iteration.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(MapUnroll._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        map_entry = graph.nodes()[candidate[MapUnroll._map_entry]]
        # Must be top-level map
        if graph.scope_dict()[map_entry] is not None:
            return False
        # All map ranges must be constant
        try:
            for begin, end, step in map_entry.map.range:
                evaluate(begin, sdfg.constants)
                evaluate(end, sdfg.constants)
                evaluate(step, sdfg.constants)
        except TypeError:
            return False
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[MapUnroll._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)

    def apply(self, sdfg):

        from dace.transformation.dataflow import TrivialMapElimination

        state = sdfg.nodes()[self.state_id]
        map_entry = state.nodes()[self.subgraph[MapUnroll._map_entry]]
        map_exit = state.exit_node(map_entry)

        # Collect all nodes in this weakly connected component
        seen = set()
        to_search = [map_entry]
        while to_search:
            node = to_search.pop()
            if node in seen:
                continue
            seen.add(node)
            for succ in state.successors(node):
                to_search.append(succ)
        to_search = [map_entry]
        seen.remove(map_entry)
        while to_search:
            node = to_search.pop()
            if node in seen:
                continue
            seen.add(node)
            for succ in state.predecessors(node):
                to_search.append(succ)
        subgraph = StateSubgraphView(state, seen)

        # Check for local memories that need to be replicated
        shared_transients = set(sdfg.shared_transients())
        # Extend this list with transients that are used outside this subgraph
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node not in seen:
                shared_transients.add(node.data)
        local_memories = []
        for name, desc in sdfg.arrays.items():
            if (desc.transient and name not in shared_transients
                    and desc.lifetime == dtypes.AllocationLifetime.Scope):
                local_memories.append(name)

        params = map_entry.map.params
        ranges = map_entry.map.range.ranges
        constant_ranges = []
        for r in ranges:
            begin = evaluate(r[0], sdfg.constants)
            end = evaluate(r[1], sdfg.constants)
            step = evaluate(r[2], sdfg.constants)
            end += step  # Make non-inclusive
            constant_ranges.append(range(begin, end, step))
        index_tuples = itertools.product(*constant_ranges)
        for t in index_tuples:
            suffix = "_" + "_".join(map(str, t))
            node_to_unrolled = {}
            # Copy all nodes
            for node in subgraph:
                if isinstance(node, nodes.NestedSDFG):
                    # Avoid deep-copying the nested SDFG
                    nsdfg = node.sdfg
                    node.sdfg = None
                unrolled_node = copy.deepcopy(node)
                node_to_unrolled[node] = unrolled_node  # Remember mapping
                if node == map_entry:
                    # Fix the map bounds to only this iteration
                    unrolled_node.map.range = [(i, i, 1) for i in t]
                if (isinstance(node, nodes.AccessNode)
                        and node.data in local_memories):
                    # If this is a local memory only used in this subgraph,
                    # we need to replicate it for each new subgraph
                    unrolled_name = node.data + suffix
                    if unrolled_name not in sdfg.arrays:
                        unrolled_desc = copy.deepcopy(sdfg.arrays[node.data])
                        sdfg.add_datadesc(unrolled_name, unrolled_desc)
                    unrolled_node.data = unrolled_name
                if isinstance(node, nodes.NestedSDFG):
                    # Reinstate the nested SDFG
                    node.sdfg = nsdfg
                    unrolled_node.sdfg = nsdfg
                    unrolled_node.sdfg.parent = None
                    unrolled_node.sdfg.parent_sdfg = None
                    unrolled_node.sdfg.parent_nsdfg_node = None
                state.add_node(unrolled_node)
            # Copy all edges
            for src, src_conn, dst, dst_conn, memlet in subgraph.edges():
                src = node_to_unrolled[src]
                dst = node_to_unrolled[dst]
                memlet = copy.deepcopy(memlet)
                if memlet.data in local_memories:
                    memlet.data = memlet.data + suffix
                state.add_edge(src, src_conn, dst, dst_conn, memlet)
            # Eliminate the now trivial map
            TrivialMapElimination.apply_to(
                sdfg,
                verify=False,
                annotate=False,
                save=False,
                _map_entry=node_to_unrolled[map_entry])

        # Now we can delete the original subgraph. This implicitly also remove
        # memlets between nodes
        state.remove_nodes_from(subgraph)

        # Remove local memories that were replicated
        for mem in local_memories:
            sdfg.remove_data(mem)

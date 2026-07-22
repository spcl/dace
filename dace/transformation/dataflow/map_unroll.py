# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data as dt, dtypes, symbolic, SDFG
from dace.sdfg import InterstateEdge, nodes, utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
import copy
import itertools


class MapUnroll(transformation.SingleStateTransformation):
    """Unrolls a constant-range top-level map by replicating its subgraph once per iteration,
    including any local-only data containers and nested SDFGs."""

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        # Must be top-level map
        if graph.scope_dict()[map_entry] is not None:
            return False
        # Flattening a device map strands its body on the host, still touching GPU_Global memory.
        # Only permissive callers (that delete the writes afterwards, e.g. MarkConstInit) may do it.
        if not permissive and map_entry.map.schedule in dtypes.GPU_SCHEDULES:
            return False
        # All map ranges must be constant
        try:
            for begin, end, step in map_entry.map.range:
                symbolic.evaluate(begin, sdfg.constants)
                symbolic.evaluate(end, sdfg.constants)
                symbolic.evaluate(step, sdfg.constants)
        except TypeError:
            return False
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        from dace.transformation.dataflow import TrivialMapElimination

        map_entry = self.map_entry

        subgraph = sdutil.weakly_connected_component(state, map_entry)

        # Check for local memories that need to be replicated
        local_memories = [
            name for name in sdutil.local_transients(sdfg, subgraph, entry_node=map_entry, include_nested=True)
            if not isinstance(sdfg.arrays[name], dt.Stream) and not isinstance(sdfg.arrays[name], dt.View)
        ]

        ranges = map_entry.map.range.ranges
        constant_ranges = []
        for r in ranges:
            begin = symbolic.evaluate(r[0], sdfg.constants)
            end = symbolic.evaluate(r[1], sdfg.constants)
            step = symbolic.evaluate(r[2], sdfg.constants)
            end += step  # Make non-inclusive
            constant_ranges.append(range(begin, end, step))
        index_tuples = itertools.product(*constant_ranges)
        for t in index_tuples:
            suffix = "_" + "_".join(map(str, t))
            node_to_unrolled = {}
            for node in subgraph:
                if isinstance(node, nodes.NestedSDFG):
                    # Copy node without its nested SDFG, then deepcopy the SDFG separately -- ~2x
                    # faster than a JSON round-trip, even amortized over every copy.
                    nsdfg = node.sdfg
                    node.sdfg = None
                    unrolled_node = copy.deepcopy(node)
                    node.sdfg = nsdfg
                    unrolled_nsdfg = copy.deepcopy(nsdfg)
                    unrolled_nsdfg.name = nsdfg.name + suffix
                    unrolled_nsdfg.parent = state
                    unrolled_nsdfg.parent_sdfg = sdfg
                    unrolled_nsdfg.update_cfg_list([])
                    unrolled_node.sdfg = unrolled_nsdfg
                    unrolled_nsdfg.parent_nsdfg_node = unrolled_node
                else:
                    unrolled_node = copy.deepcopy(node)
                    if node == map_entry:
                        # Fix the map bounds to only this iteration
                        unrolled_node.map.range = [(i, i, 1) for i in t]
                    if (isinstance(node, nodes.AccessNode) and node.data in local_memories):
                        unrolled_name = node.data + suffix
                        if unrolled_name not in sdfg.arrays:
                            unrolled_desc = copy.deepcopy(sdfg.arrays[node.data])
                            sdfg.add_datadesc(unrolled_name, unrolled_desc)
                        unrolled_node.data = unrolled_name
                state.add_node(unrolled_node)
                node_to_unrolled[node] = unrolled_node
            for src, src_conn, dst, dst_conn, memlet in subgraph.edges():
                src = node_to_unrolled[src]
                dst = node_to_unrolled[dst]
                memlet = copy.deepcopy(memlet)
                if memlet.data in local_memories:
                    memlet.data = memlet.data + suffix
                state.add_edge(src, src_conn, dst, dst_conn, memlet)

            for edge in subgraph.all_edges_recursive():
                if isinstance(edge, InterstateEdge):
                    for k, v in edge.data.assignments.items():
                        pass

            # Eliminate the now trivial map
            TrivialMapElimination.apply_to(sdfg,
                                           verify=False,
                                           annotate=False,
                                           save=False,
                                           map_entry=node_to_unrolled[map_entry])

        # Removing these nodes implicitly also removes their memlets.
        state.remove_nodes_from(subgraph)

        # Reset the cfg list if new nested SDFGs were added.
        if any(isinstance(node, nodes.NestedSDFG) for node in subgraph):
            sdfg.reset_cfg_list()

        for mem in local_memories:
            sdfg.remove_data(mem)

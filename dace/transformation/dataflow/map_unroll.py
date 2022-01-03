# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace import data as dt, dtypes, registry, symbolic, SDFG
from dace.sdfg import nodes, utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties
import copy
import itertools


@registry.autoregister_params(singlestate=True)
@make_properties
class MapUnroll(transformation.Transformation):
    """
    Unrolls a map with constant ranges in the top-level scope of an SDFG by
    replicating its subgraph for each iteration. If there are local data
    containers only used in this map, they will also be replicated, as will
    nested SDFGs found within.

    This transformation can be useful for forming weakly connected components
    that will be inferred as processing elements in an FPGA kernel.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(MapUnroll._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, permissive=False):
        map_entry = graph.nodes()[candidate[MapUnroll._map_entry]]
        # Must be top-level map
        if graph.scope_dict()[map_entry] is not None:
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
        subgraph = sdutil.weakly_connected_component(state, map_entry)

        # Save nested SDFGs to JSON, then deserialize them for every copy we
        # need to make
        nested_sdfgs = {}
        for node in subgraph:
            if isinstance(node, nodes.NestedSDFG):
                nested_sdfgs[node.sdfg] = node.sdfg.to_json()

        # Check for local memories that need to be replicated
        local_memories = [
            name for name in sdutil.local_transients(sdfg, subgraph, entry_node=map_entry, include_nested=True)
            if not isinstance(sdfg.arrays[name], dt.Stream) and not isinstance(sdfg.arrays[name], dt.View)
        ]

        params = map_entry.map.params
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
            # Copy all nodes
            for node in subgraph:
                if isinstance(node, nodes.NestedSDFG):
                    # Avoid deep-copying the nested SDFG
                    nsdfg = node.sdfg
                    # Don't copy the nested SDFG, as we will do this separately
                    node.sdfg = None
                    unrolled_node = copy.deepcopy(node)
                    node.sdfg = nsdfg
                    # Deserialize into a new SDFG specific to this copy
                    nsdfg_json = nested_sdfgs[nsdfg]
                    name = nsdfg_json["attributes"]["name"]
                    nsdfg_json["attributes"]["name"] += suffix
                    unrolled_nsdfg = SDFG.from_json(nsdfg_json)
                    nsdfg_json["attributes"]["name"] = name  # Reinstate
                    # Set all the references
                    unrolled_nsdfg.parent = state
                    unrolled_nsdfg.parent_sdfg = sdfg
                    unrolled_nsdfg.update_sdfg_list([])
                    unrolled_node.sdfg = unrolled_nsdfg
                    unrolled_nsdfg.parent_nsdfg_node = unrolled_node
                else:
                    unrolled_node = copy.deepcopy(node)
                    if node == map_entry:
                        # Fix the map bounds to only this iteration
                        unrolled_node.map.range = [(i, i, 1) for i in t]
                    if (isinstance(node, nodes.AccessNode) and node.data in local_memories):
                        # If this is a local memory only used in this subgraph,
                        # we need to replicate it for each new subgraph
                        unrolled_name = node.data + suffix
                        if unrolled_name not in sdfg.arrays:
                            unrolled_desc = copy.deepcopy(sdfg.arrays[node.data])
                            sdfg.add_datadesc(unrolled_name, unrolled_desc)
                        unrolled_node.data = unrolled_name
                state.add_node(unrolled_node)
                node_to_unrolled[node] = unrolled_node  # Remember mapping
            # Copy all edges
            for src, src_conn, dst, dst_conn, memlet in subgraph.edges():
                src = node_to_unrolled[src]
                dst = node_to_unrolled[dst]
                memlet = copy.deepcopy(memlet)
                if memlet.data in local_memories:
                    memlet.data = memlet.data + suffix
                state.add_edge(src, src_conn, dst, dst_conn, memlet)
            # Eliminate the now trivial map
            TrivialMapElimination.apply_to(sdfg,
                                           verify=False,
                                           annotate=False,
                                           save=False,
                                           _map_entry=node_to_unrolled[map_entry])

        # Now we can delete the original subgraph. This implicitly also remove
        # memlets between nodes
        state.remove_nodes_from(subgraph)

        # If we added a bunch of new nested SDFGs, reset the internal list
        if len(nested_sdfgs) > 0:
            sdfg.reset_sdfg_list()

        # Remove local memories that were replicated
        for mem in local_memories:
            sdfg.remove_data(mem)

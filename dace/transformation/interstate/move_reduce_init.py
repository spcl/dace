# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from typing import Set, Dict, Optional

import dace
from dace import symbolic, data as dt, InterstateEdge
from dace.sdfg import nodes, SDFG, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.passes.analysis import StateReachability
from dace.properties import make_properties


def _is_init_state(state: SDFGState) -> bool:
    """
    Check if a state is an initialization state for reduction.
    An init state has:
    - A single top-level map
    - A single tasklet inside the map with no data inputs
    - The tasklet writes to an output array
    """
    scope_dict = state.scope_dict()
    map_entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scope_dict[n] is None]

    if len(map_entries) != 1:
        return False

    map_entry = map_entries[0]

    scope_nodes = state.scope_subgraph(map_entry).nodes()
    tasklets = [n for n in scope_nodes if isinstance(n, nodes.Tasklet)]
    if len(tasklets) != 1:
        return False

    tasklet = tasklets[0]
    for e in state.in_edges(tasklet):
        if e.data.data is not None and not e.data.is_empty():
            return False

    has_output = False
    for e in state.out_edges(tasklet):
        if e.data.data is not None and not e.data.is_empty():
            has_output = True
            break

    return has_output


def _get_init_output_arrays(state: SDFGState) -> Set[str]:
    """Get the output arrays being initialized in the state."""
    outputs = set()
    for node in state.sink_nodes():
        if isinstance(node, nodes.AccessNode):
            outputs.add(node.data)
    return outputs


def _is_written_before(sdfg: SDFG,
                       state: SDFGState,
                       nsdfg_node: nodes.NestedSDFG,
                       array_name: str,
                       reachability: Optional[Dict[SDFGState, Set[SDFGState]]] = None) -> bool:
    """
    Check if an array is written before the nested SDFG in the parent SDFG.
    Uses StateReachability to find all states that can reach the current state,
    then checks their write sets.
    """
    if reachability is None:
        reachability_pass = StateReachability()
        all_reachability = reachability_pass.apply_pass(sdfg, {})
        reachability = all_reachability.get(sdfg.cfg_id, {})

    # Find all states that can reach the current state
    states_reaching_current = set()
    for s, reachable in reachability.items():
        if state in reachable:
            states_reaching_current.add(s)

    # Check if any reaching state writes to the array
    for src_state in states_reaching_current:
        _, write_set = src_state.read_and_write_sets()
        if array_name in write_set:
            return True

    # Check if another node in the same state writes to the array
    for node in state.nodes():
        if node == nsdfg_node:
            continue
        if isinstance(node, nodes.AccessNode) and node.data == array_name:
            if state.in_degree(node) > 0:
                for e in state.in_edges(node):
                    if e.src != nsdfg_node:
                        return True

    return False


def _substitute_symbols(nsdfg_node: nodes.NestedSDFG, rng: tuple) -> tuple:
    """Substitute nested SDFG symbols with outer values in a range tuple."""
    new_rng = list(rng)
    for inner_sym, outer_val in nsdfg_node.symbol_mapping.items():
        for i in range(3):
            if symbolic.issymbolic(new_rng[i]):
                new_rng[i] = new_rng[i].subs({inner_sym: outer_val})
    return tuple(new_rng)


@make_properties
@transformation.explicit_cf_compatible
class MoveReduceInitOutOfNestedSDFG(transformation.SingleStateTransformation):
    """
    Moves reduction initialization from a nested SDFG to a new state at the start
    of the SDFG. Having these initializations in NestedSDFGs blocks inlining.

    The transformation looks for nested SDFGs that have:
    1. An initialization state as the start state (single map, single tasklet with no inputs)
    2. The initialized array is not written before the nested SDFG

    After transformation:
    - A new state is added at the start of the SDFG with the initialization map
    - The nested SDFG's initialization state is removed
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    def can_be_applied(self, graph: SDFGState, expr_index, sdfg: SDFG, permissive=False):
        nsdfg_node = self.nested_sdfg
        nsdfg = nsdfg_node.sdfg

        start_state = nsdfg.start_state

        # Check if the start state matches the init pattern
        if not _is_init_state(start_state):
            return False

        init_outputs = _get_init_output_arrays(start_state)
        if not init_outputs:
            return False

        # Compute reachability once for all output checks
        reachability_pass = StateReachability()
        all_reachability = reachability_pass.apply_pass(sdfg, {})
        reachability = all_reachability.get(sdfg.cfg_id, {})

        # Verify each output array is valid for transformation
        for output in init_outputs:
            # Output must be an out_connector of the nested SDFG
            if output not in nsdfg_node.out_connectors:
                return False

            # Find the corresponding edge in the parent state
            outer_edge = None
            for e in graph.out_edges(nsdfg_node):
                if e.src_conn == output:
                    outer_edge = e
                    break

            if outer_edge is None:
                return False

            # The outer array must not be written before this nested SDFG
            outer_array = outer_edge.data.data

            if _is_written_before(sdfg, graph, nsdfg_node, outer_array, reachability):
                return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        nsdfg_node = self.nested_sdfg
        nsdfg = nsdfg_node.sdfg

        start_state = nsdfg.start_state
        successors = list(nsdfg.successors(start_state))
        next_state = successors[0] if successors else None

        # Create new init state at the start of the parent SDFG
        old_start = sdfg.start_state
        init_state = sdfg.add_state(label='reduce_init')
        sdfg.add_edge(init_state, old_start, InterstateEdge())
        sdfg.start_block = sdfg.node_id(init_state)

        # Build mapping from inner array names to outer (outside NestedSDFG) array names.
        # If the destination is a View, resolve to the underlying array.
        connector_map = {}
        for e in state.out_edges(nsdfg_node):
            if e.src_conn in _get_init_output_arrays(start_state):
                if isinstance(e.dst, nodes.AccessNode):
                    arr_name = e.dst.data
                    arr = sdfg.arrays.get(arr_name)
                    if isinstance(arr, dt.View):
                        view_edge = sdutil.get_view_edge(state, e.dst)
                        if view_edge is not None:
                            arr_name = view_edge.data.data
                    connector_map[e.src_conn] = arr_name
                else:
                    connector_map[e.src_conn] = e.data.data

        # Copy map nodes first

        node_map = {}
        for node in start_state.nodes():
            if isinstance(node, nodes.MapEntry):
                new_entry = copy.deepcopy(node)
                new_range = [_substitute_symbols(nsdfg_node, rng) for rng in new_entry.map.range]
                new_entry.map.range = dace.subsets.Range(new_range)
                node_map[node] = new_entry
                exit_node = start_state.exit_node(node)
                new_exit = copy.deepcopy(exit_node)

                # MapEntry and MapExit need to share the same Map object
                new_exit.map = new_entry.map
                node_map[exit_node] = new_exit

        # Copy remaining nodes (AccessNodes with renamed arrays, Tasklets, etc.)
        for node in start_state.nodes():
            if node in node_map:
                continue
            if isinstance(node, nodes.AccessNode):
                inner_name = node.data
                outer_name = connector_map.get(inner_name, inner_name)
                new_node = nodes.AccessNode(outer_name)
            else:
                new_node = copy.deepcopy(node)
            node_map[node] = new_node

        # Add all nodes to the new init state
        for node in start_state.nodes():
            init_state.add_node(node_map[node])

        # Copy edges with updated memlets (renamed arrays, substituted symbols)
        for edge in start_state.edges():
            src = node_map.get(edge.src)
            dst = node_map.get(edge.dst)
            if src is None or dst is None:
                continue

            new_memlet = copy.deepcopy(edge.data)
            if new_memlet.data is not None:
                inner_name = new_memlet.data
                outer_name = connector_map.get(inner_name, inner_name)
                new_memlet.data = outer_name

                if new_memlet.subset is not None:
                    new_subset = [_substitute_symbols(nsdfg_node, rng) for rng in new_memlet.subset]
                    new_memlet.subset = dace.subsets.Range(new_subset)

            init_state.add_edge(src, edge.src_conn, dst, edge.dst_conn, new_memlet)

        # Remove the init state from the nested SDFG and update its start block
        nsdfg.remove_node(start_state)

        if len(nsdfg.nodes()) > 0 and next_state is not None:
            nsdfg.start_block = nsdfg.node_id(next_state)
            nsdfg.reset_cfg_list()

        return init_state

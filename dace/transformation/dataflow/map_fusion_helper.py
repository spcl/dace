# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import dace
from dace import symbolic
from dace.sdfg import graph, nodes as nodes, propagation, validation
from dace.transformation import helpers


def find_parameter_remapping(
    first_map: nodes.Map,
    second_map: nodes.Map,
    simplify_ranges: bool = False,
) -> Optional[Dict[str, str]]:
    """Computes the parameter remapping for the parameters of the _second_ map.

    The returned `dict` maps the parameters of the second map (keys) to parameter
    names of the first map (values). Because of how the replace function works
    the `dict` describes how to replace the parameters of the second map
    with parameters of the first map.
    Parameters that already have the correct name and compatible range, are not
    included in the return value, thus the keys and values are always different.
    If no renaming at is _needed_, i.e. all parameter have the same name and range,
    then the function returns an empty `dict`.
    If no remapping exists, then the function will return `None`.

    :param first_map: The first map (these parameters will be replaced).
    :param second_map: The second map, these parameters acts as source.
    :param simplify_ranges: Perform simplification on the range expressions.

    :note: This function currently fails if the renaming is not unique. Consider the
        case were the first map has the structure `for i, j in map[0:20, 0:20]` and it
        writes `T[i, j]`, while the second map is equivalent to
        `for l, k in map[0:20, 0:20]` which reads `T[l, k]`. For this case we have
        the following valid remappings `{l: i, k: j}` and `{l: j, k: i}` but
        only the first one allows to fuse the map. This is because if the second
        one is used the second map will read `T[j, i]` which leads to a data
        dependency that can not be satisfied.
        To avoid this issue the renaming algorithm will process them in order, i.e.
        assuming that the order of the parameters in the map matches. But this is
        not perfect, the only way to really solve this is by trying possible
        remappings. At least the algorithm used here is deterministic.
    """

    # The parameter names
    first_params: List[str] = first_map.params
    second_params: List[str] = second_map.params

    if len(first_params) != len(second_params):
        return None

    if simplify_ranges:
        simp = lambda e: symbolic.simplify_ext(symbolic.simplify(e))  # noqa: E731 [lambda-assignment]
    else:
        simp = lambda e: e  # noqa: E731 [lambda-assignment]

    first_rngs: Dict[str, Tuple[Any, Any, Any]] = {
        param: tuple(simp(r) for r in rng)
        for param, rng in zip(first_params, first_map.range)
    }
    second_rngs: Dict[str, Tuple[Any, Any, Any]] = {
        param: tuple(simp(r) for r in rng)
        for param, rng in zip(second_params, second_map.range)
    }

    # Parameters of the second map that have not yet been matched to a parameter
    #  of the first map and the parameters of the first map that are still free.
    #  That we use a `list` instead of a `set` is intentional, because it counter
    #  acts the issue that is described in the doc string. Using a list ensures
    #  that they indexes are matched in order. This assume that in real world
    #  code the order of the loop is not arbitrary but kind of matches.
    unmapped_second_params: List[str] = list(second_params)
    unused_first_params: List[str] = list(first_params)

    # This is the result (`second_param -> first_param`), note that if no renaming
    #  is needed then the parameter is not present in the mapping.
    final_mapping: Dict[str, str] = {}

    # First we identify the parameters that already have the correct name.
    for param in set(first_params).intersection(second_params):
        first_rng = first_rngs[param]
        second_rng = second_rngs[param]

        if first_rng == second_rng:
            # They have the same name and the same range, this is already a match.
            #  Because the names are already the same, we do not have to enter them
            #  in the `final_mapping`
            unmapped_second_params.remove(param)
            unused_first_params.remove(param)

    # Check if no remapping is needed.
    if len(unmapped_second_params) == 0:
        return {}

    # Now we go through all the parameters that we have not mapped yet.
    #  All of them will result in a remapping.
    for unmapped_second_param in unmapped_second_params:
        second_rng = second_rngs[unmapped_second_param]
        assert unmapped_second_param not in final_mapping

        # Now look in all not yet used parameters of the first map which to use.
        for candidate_param in list(unused_first_params):
            candidate_rng = first_rngs[candidate_param]
            if candidate_rng == second_rng:
                final_mapping[unmapped_second_param] = candidate_param
                unused_first_params.remove(candidate_param)
                break
        else:
            # We did not find a candidate, so the remapping does not exist
            return None

    assert len(unused_first_params) == 0
    assert len(final_mapping) == len(unmapped_second_params)
    return final_mapping


def rename_map_parameters(
    first_map: nodes.Map,
    second_map: nodes.Map,
    second_map_entry: nodes.MapEntry,
    state: dace.SDFGState,
    simplify_ranges: bool = False,
) -> None:
    """Replaces the map parameters of the second map with names from the first.

    The replacement is done in a safe way, thus `{'i': 'j', 'j': 'i'}` is
    handled correct. The function assumes that a proper replacement exists.
    The replacement is computed by calling `find_parameter_remapping()`.

    :param first_map:  The first map (these are the final parameter).
    :param second_map: The second map, this map will be replaced.
    :param second_map_entry: The entry node of the second map.
    :param state: The SDFGState on which we operate.
    :param simplify_ranges: Perform simplification on the range expressions.
    """
    # Compute the replacement dict.
    repl_dict: Dict[str, str] = find_parameter_remapping(  # type: ignore[assignment]  # Guaranteed to be not `None`.
        first_map=first_map,
        second_map=second_map,
        simplify_ranges=simplify_ranges,
    )

    if repl_dict is None:
        raise RuntimeError("The replacement does not exist")
    if len(repl_dict) == 0:
        return

    second_map_scope = state.scope_subgraph(entry_node=second_map_entry)
    # Why is this thing in symbolic and not in replace?
    symbolic.safe_replace(
        mapping=repl_dict,
        replace_callback=second_map_scope.replace_dict,
    )

    # For some odd reason the replace function does not modify the range and
    #  parameter of the map, so we will do it the hard way.
    second_map.params = copy.deepcopy(first_map.params)
    second_map.range = copy.deepcopy(first_map.range)


def get_new_conn_name(
    edge_to_move: graph.MultiConnectorEdge[dace.Memlet],
    to_node: Union[nodes.MapExit, nodes.MapEntry],
    state: dace.SDFGState,
    scope_dict: Dict,
    never_consolidate_edges: bool = False,
    consolidate_edges_only_if_not_extending: bool = True,
) -> Tuple[str, bool]:
    """Determine the new connector name that should be used.

    The function returns a pair. The first element is the name of the connector
    name that should be used. The second element is a boolean that indicates if
    the connector name is already present on `to_node`, `True`, or if a new
    connector was created.

    The function honors the `self.never_consolidate_edges`, in which case
    a new connector is generated every time, leading to minimal subset but
    many connections. Furthermore, it will also consider
    `self.consolidate_edges_only_if_not_extending`. If it is set it will only
    create a new connection if this would lead to an increased subset.

    :note: In case `to_node` a MapExit or a nested map, the function will always
        generate a new connector.
    """
    assert edge_to_move.dst_conn.startswith("IN_")
    old_conn = edge_to_move.dst_conn[3:]

    # If we have a MapExit or have a nested Map we never consolidate or if
    #  especially requested.
    if (isinstance(to_node, nodes.MapExit) or scope_dict[to_node] is not None or never_consolidate_edges):
        return to_node.next_connector(old_conn), False

    # Now look for an edge that already referees to the data of the edge.
    edge_that_is_already_present = None
    for iedge in state.in_edges(to_node):
        if iedge.data.is_empty() or iedge.dst_conn is None:
            continue
        if not iedge.dst_conn.startswith("IN_"):
            continue
        if iedge.data.data == edge_to_move.data.data:
            # The same data is used so we reuse that connection.
            edge_that_is_already_present = iedge

    # No edge is there that is using the data, so create a new connector.
    #  TODO(phimuell): Probably should reuse the connector at `from_node`?
    if edge_that_is_already_present is None:
        return to_node.next_connector(old_conn), False

    # We also do not care if the consolidation leads to the extension of the
    #  subsets, thus we are done.
    if not consolidate_edges_only_if_not_extending:
        return edge_that_is_already_present.dst_conn[3:], True

    # We can only do the check for extension if both have a valid subset.
    edge_to_move_subset = edge_to_move.data.src_subset
    edge_that_is_already_present_subset = edge_that_is_already_present.data.src_subset
    if edge_to_move_subset is None or edge_that_is_already_present_subset is None:
        return to_node.next_connector(old_conn), False

    # The consolidation will not lead to an extension if either the edge that is
    #  there or the new edge covers each other.
    # NOTE: One could also say that we should only do that if `edge_that_is_already_there`
    #   covers the new one, but since the order, is kind of arbitrary, we test if
    #   either one covers.
    return ((edge_that_is_already_present.dst_conn[3:],
             True) if edge_that_is_already_present_subset.covers(edge_to_move_subset)
            or edge_to_move_subset.covers(edge_that_is_already_present_subset) else
            (to_node.next_connector(old_conn), False))


def relocate_nodes(
    from_node: Union[nodes.MapExit, nodes.MapEntry],
    to_node: Union[nodes.MapExit, nodes.MapEntry],
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    scope_dict: Dict,
    never_consolidate_edges: bool = False,
    consolidate_edges_only_if_not_extending: bool = True,
) -> None:
    """Move the connectors and edges from `from_node` to `to_nodes` node.

    This function will only rewire the edges, it does not remove the nodes
    themselves. Furthermore, this function should be called twice per Map,
    once for the entry and then for the exit.
    While it does not remove the node themselves if guarantees that the
    `from_node` has degree zero.
    The function assumes that the parameter renaming was already done.

    :param from_node: Node from which the edges should be removed.
    :param to_node: Node to which the edges should reconnect.
    :param state: The state in which the operation happens.
    :param sdfg: The SDFG that is modified.

    :note: After the relocation Memlet propagation should be run.
    """

    # Now we relocate empty Memlets, from the `from_node` to the `to_node`
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.out_edges(from_node))):
        helpers.redirect_edge(state, empty_edge, new_src=to_node)
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.in_edges(from_node))):
        helpers.redirect_edge(state, empty_edge, new_dst=to_node)

    # We now ensure that there is only one empty Memlet from the `to_node` to any other node.
    #  Although it is allowed, we try to prevent it.
    empty_targets: Set[nodes.Node] = set()
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
        if empty_edge.dst in empty_targets:
            state.remove_edge(empty_edge)
        empty_targets.add(empty_edge.dst)

    # Relocation of the edges that carry data.
    for edge_to_move in list(state.in_edges(from_node)):
        assert isinstance(edge_to_move.dst_conn, str)

        if not edge_to_move.dst_conn.startswith("IN_"):
            # Dynamic Map Range
            #  The connector name simply defines a variable name that is used,
            #  inside the Map scope to define a variable. We handle it directly.
            dmr_symbol = edge_to_move.dst_conn

            # TODO(phimuell): Check if the symbol is really unused in the target scope.
            if dmr_symbol in to_node.in_connectors:
                raise NotImplementedError(f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                                          f" to '{to_node}', but the symbol is already known there, but the"
                                          " renaming is not implemented.")
            if not to_node.add_in_connector(dmr_symbol, force=False):
                raise RuntimeError(  # Might fail because of out connectors.
                    f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'.")
            helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
            from_node.remove_in_connector(dmr_symbol)

        else:
            # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
            old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
            new_conn, conn_was_reused = get_new_conn_name(
                edge_to_move=edge_to_move,
                to_node=to_node,
                state=state,
                scope_dict=scope_dict,
                never_consolidate_edges=never_consolidate_edges,
                consolidate_edges_only_if_not_extending=consolidate_edges_only_if_not_extending,
            )

            # Now move the incoming edges of `to_node` to `from_node`. However,
            #  we only move `edge_to_move` if we have a new connector, if we
            #  reuse the connector we will simply remove it.
            dst_in_conn = "IN_" + new_conn
            for e in list(state.in_edges_by_connector(from_node, f"IN_{old_conn}")):
                if conn_was_reused and e is edge_to_move:
                    state.remove_edge(edge_to_move)
                    if state.degree(edge_to_move.src) == 0:
                        state.remove_node(edge_to_move.src)
                else:
                    helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn=dst_in_conn)

            # Now move the outgoing edges of `to_node` to `from_node`.
            dst_out_conn = "OUT_" + new_conn
            for e in list(state.out_edges_by_connector(from_node, f"OUT_{old_conn}")):
                helpers.redirect_edge(state, e, new_src=to_node, new_src_conn=dst_out_conn)

            # If we have used new connectors we must add the new connector names.
            if not conn_was_reused:
                to_node.add_in_connector(dst_in_conn)
                to_node.add_out_connector(dst_out_conn)

            # In any case remove the old connector name from the `from_node`.
            from_node.remove_in_connector("IN_" + old_conn)
            from_node.remove_out_connector("OUT_" + old_conn)

    # Check if we succeeded.
    if state.out_degree(from_node) != 0:
        raise validation.InvalidSDFGError(
            f"Failed to relocate the outgoing edges from `{from_node}`, there are still `{state.out_edges(from_node)}`",
            sdfg,
            sdfg.node_id(state),
        )
    if state.in_degree(from_node) != 0:
        raise validation.InvalidSDFGError(
            f"Failed to relocate the incoming edges from `{from_node}`, there are still `{state.in_edges(from_node)}`",
            sdfg,
            sdfg.node_id(state),
        )
    assert len(from_node.in_connectors) == 0
    assert len(from_node.out_connectors) == 0


def is_node_reachable_from(
    graph: dace.SDFGState,
    begin: nodes.Node,
    end: nodes.Node,
) -> bool:
    """Test if the node `end` can be reached from `begin`.

    Essentially the function starts a DFS at `begin`. If an edge is found that lead
    to `end` the function returns `True`. If the node is never found `False` is
    returned.

    :param graph: The graph to operate on.
    :param begin: The start of the DFS.
    :param end: The node that should be located.
    """

    def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
        return (edge.dst for edge in graph.out_edges(node))

    to_visit: List[nodes.Node] = [begin]
    seen: Set[nodes.Node] = set()

    while len(to_visit) > 0:
        node: nodes.Node = to_visit.pop()
        if node == end:
            return True
        elif node not in seen:
            to_visit.extend(next_nodes(node))
        seen.add(node)

    # We never found `end`
    return False


def is_parallel(
    graph: dace.SDFGState,
    node1: nodes.Node,
    node2: nodes.Node,
) -> bool:
    """Tests if `node1` and `node2` are parallel in the data flow graph.

    The function considers two nodes parallel in the data flow graph, if `node2`
    can not be reached from `node1` and vice versa. The function does not check
    the scope of the nodes.

    :param graph: The state on which we operate.
    :param node1: The first node to check.
    :param node2: The second node to check.
    """
    # The `all_nodes_between()` function traverse the graph and returns `None` if
    #  `end` was not found. We have to call it twice, because we do not know
    #  which node is upstream if they are not parallel.
    if is_node_reachable_from(graph=graph, begin=node1, end=node2):
        return False
    elif is_node_reachable_from(graph=graph, begin=node2, end=node1):
        return False
    return True


def can_topologically_be_fused(
    first_map_entry: nodes.MapEntry,
    second_map_entry: nodes.MapEntry,
    graph: Union[dace.SDFGState, dace.SDFG],
    sdfg: dace.SDFG,
    permissive: bool = False,
    only_inner_maps: bool = False,
    only_toplevel_maps: bool = False,
) -> Optional[Dict[str, str]]:
    """Performs basic checks if the maps can be fused.

    This function only checks constrains that are common between serial and
    parallel map fusion process, which includes:
    * The scope of the maps.
    * The scheduling of the maps.
    * The map parameters.

    :return: If the maps can not be topologically fused the function returns `None`.
        If they can be fused the function returns `dict` that describes parameter
        replacement, see `find_parameter_remapping()` for more.

    :param first_map_entry: The entry of the first (in serial case the top) map.
    :param second_map_exit: The entry of the second (in serial case the bottom) map.
    :param graph: The SDFGState in which the maps are located.
    :param sdfg: The SDFG itself.
    :param permissive: Currently unused.

    :note: It is invalid to call this function after nodes have been removed from the SDFG.
    """
    if only_inner_maps and only_toplevel_maps:
        raise ValueError(
            "Only one of `only_inner_maps` and `only_toplevel_maps` is allowed per MapFusionVertical instance.")

    # Ensure that both have the same schedule
    if first_map_entry.map.schedule != second_map_entry.map.schedule:
        return None

    # Fusing is only possible if the two entries are in the same scope.
    scope = graph.scope_dict()
    if scope[first_map_entry] != scope[second_map_entry]:
        return None
    elif only_inner_maps:
        if scope[first_map_entry] is None:
            return None
    elif only_toplevel_maps:
        if scope[first_map_entry] is not None:
            return None

    # We will now check if we can rename the Map parameter of the second Map such that they
    #  match the one of the first Map.
    param_repl = find_parameter_remapping(first_map=first_map_entry.map, second_map=second_map_entry.map)
    return param_repl

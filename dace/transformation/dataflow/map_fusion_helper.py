# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import itertools
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import sympy

import dace
from dace import subsets, symbolic
from dace.sdfg import graph, nodes as nodes, utils as sdutils, validation
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

    # A duplicated parameter collapses the `param -> range` dicts below, so the second pass
    #  would map the same name twice; a `params`/`range` length mismatch makes the `zip()`
    #  drop entries, so a name in the intersection is missing from the dict.
    if len(set(first_params)) != len(first_params) or len(set(second_params)) != len(second_params):
        return None
    if len(first_params) != first_map.range.dims() or len(second_params) != second_map.range.dims():
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

    # We now ensure that there is only one empty Memlet between `to_node` and any
    #  other node, i.e. we drop genuinely-duplicate empty Memlets while keeping every
    #  distinct one. The key is the `(src, dst)` pair, NOT `dst` alone: empty Memlets
    #  are ordering (WAW/WAR) dependencies, and two of them into `to_node` coming from
    #  *different* source nodes encode *different* dependencies (e.g. the write-ordering
    #  chains of two different arrays fused into one Map). Keying on `dst` alone would
    #  collapse every empty in-edge (they all share `dst == to_node`) down to one,
    #  dropping real dependencies and -- because the surviving one depends on edge
    #  iteration order -- producing an order-dependent miscompile.
    seen_empty_pairs: Set[Tuple[nodes.Node, nodes.Node]] = set()
    for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
        pair = (empty_edge.src, empty_edge.dst)
        if pair in seen_empty_pairs:
            state.remove_edge(empty_edge)
        seen_empty_pairs.add(pair)

    # Relocation of the edges that carry data.
    #  A passthrough connector may carry MORE THAN ONE edge -- a Map body that writes several
    #  disjoint slices of the same array through one tasklet output lands every one of them on
    #  the same `IN_x` (the CLOUDSC shape: five `zqxn2d[i, j, 0..4]` edges on a single MapExit
    #  connector). The passthrough branch below relocates the whole `IN_x` / `OUT_x` group in
    #  one go, so the group must be visited ONCE: the extra edges of a group are already
    #  relocated (and their edge objects stale) by the time the loop reaches them, and handling
    #  one again mints another connector pair on `to_node` that nothing is attached to --
    #  'Dangling in-connector IN_x' out of validate().
    relocated_in_conns: Set[str] = set()
    for edge_to_move in list(state.in_edges(from_node)):
        assert isinstance(edge_to_move.dst_conn, str)

        if not edge_to_move.dst_conn.startswith("IN_"):
            # Dynamic Map Range
            #  The connector name simply defines a variable name that is used,
            #  inside the Map scope to define a variable. We handle it directly.
            dmr_symbol = edge_to_move.dst_conn

            if dmr_symbol in to_node.in_connectors:
                # Same symbol, same value: the moved binding is redundant, so drop it instead.
                if not dynamic_map_range_binding_agrees(state, to_node, from_node, dmr_symbol):
                    raise NotImplementedError(f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                                              f" to '{to_node}', but the symbol is already known there, but the"
                                              " renaming is not implemented.")
                source = edge_to_move.src
                state.remove_edge(edge_to_move)
                from_node.remove_in_connector(dmr_symbol)
                if state.degree(source) == 0:
                    state.remove_node(source)
                continue
            if not to_node.add_in_connector(dmr_symbol, force=False):
                raise RuntimeError(  # Might fail because of out connectors.
                    f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'.")
            helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
            from_node.remove_in_connector(dmr_symbol)

        elif edge_to_move.dst_conn not in relocated_in_conns:
            # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
            relocated_in_conns.add(edge_to_move.dst_conn)
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


def safe_exit_node(
    state: dace.SDFGState,
    entry_node: nodes.EntryNode,
) -> Optional[nodes.ExitNode]:
    """The exit node of `entry_node`'s scope, or `None` if the scope is no longer intact.

    `SDFGState.exit_node()` raises a bare `StopIteration` when the exit node was removed and a
    `KeyError` when the entry node itself is gone -- both reachable from a stale match, and the
    former turns into a `RuntimeError` if it escapes into the matcher's generator (PEP 479).
    """
    try:
        return state.exit_node(entry_node)
    except (KeyError, StopIteration):
        return None


def scope_connectors_are_sound(
    state: dace.SDFGState,
    node: Union[nodes.MapEntry, nodes.MapExit],
) -> bool:
    """`True` if `node`'s connectors and edges are shaped the way the rewrite assumes.

    `relocate_nodes()` walks the in-edges of a scope node and moves the whole `IN_x`/`OUT_x`
    group at once. A connector without an edge is therefore never visited (and trips the final
    "everything was relocated" checks *after* the graph was rewritten), and a data edge without
    a connector has no counterpart to follow into the scope. Both shapes are already rejected by
    `SDFGState` validation, so refusing them here costs no valid SDFG.
    """
    bound_in_conns = {e.dst_conn for e in state.in_edges(node)}
    bound_out_conns = {e.src_conn for e in state.out_edges(node)}
    in_conns, out_conns = node.in_connectors, node.out_connectors

    for conn in in_conns:
        if conn not in bound_in_conns:
            return False
        if conn.startswith("IN_") and ("OUT_" + conn[3:]) not in out_conns:
            return False
    for conn in out_conns:
        if conn not in bound_out_conns:
            return False
        if conn.startswith("OUT_") and ("IN_" + conn[4:]) not in in_conns:
            return False

    # The other direction: an edge naming a connector that was never declared is relocated as a
    #  dynamic map range, which then collides on the target node.
    if any(conn is not None and conn not in in_conns for conn in bound_in_conns):
        return False
    if any(conn is not None and conn not in out_conns for conn in bound_out_conns):
        return False

    # An empty Memlet is an ordering edge and legitimately carries no connector.
    if None in bound_in_conns and any(e.dst_conn is None and not e.data.is_empty() for e in state.in_edges(node)):
        return False
    if None in bound_out_conns and any(e.src_conn is None and not e.data.is_empty() for e in state.out_edges(node)):
        return False
    return True


def map_scope_data_is_known(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    map_entry: nodes.MapEntry,
    map_exit: nodes.MapExit,
) -> bool:
    """`True` if every array name the Map scope mentions still has a descriptor in `sdfg`.

    A name that another transformation deleted surfaces as a `KeyError` deep inside the rewrite
    (`AccessNode.desc()`, `Memlet.try_initialize()`, `sdfg.arrays[...]`), so it is refused here.
    """
    scope = state.scope_subgraph(map_entry)
    boundary_edges = list(state.in_edges(map_entry)) + list(state.out_edges(map_exit))
    for edge in itertools.chain(scope.edges(), boundary_edges):
        if edge.data is not None and edge.data.data is not None and edge.data.data not in sdfg.arrays:
            return False
    boundary_nodes = (e.src for e in state.in_edges(map_entry))
    for node in itertools.chain(scope.nodes(), boundary_nodes, (e.dst for e in state.out_edges(map_exit))):
        if isinstance(node, nodes.AccessNode) and node.data not in sdfg.arrays:
            return False
    return True


def is_node_reachable_from(
    graph: dace.SDFGState,
    begin: nodes.Node,
    end: nodes.Node,
    ignore_empty_edges: bool = False,
) -> bool:
    """Test if the node `end` can be reached from `begin`.

    Essentially the function starts a DFS at `begin`. If an edge is found that lead
    to `end` the function returns `True`. If the node is never found `False` is
    returned.

    :param graph: The graph to operate on.
    :param begin: The start of the DFS.
    :param end: The node that should be located.
    :param ignore_empty_edges: Do not traverse empty Memlets, i.e. only follow real
        data flow. Used to tell an ordering-only connection apart from a data one.
    """

    def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
        return (edge.dst for edge in graph.out_edges(node)
                if not (ignore_empty_edges and edge.data is not None and edge.data.is_empty()))

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


def find_happens_before_connection(
    state: dace.SDFGState,
    first_map_entry: nodes.MapEntry,
    second_map_entry: nodes.MapEntry,
) -> Optional[List[graph.MultiConnectorEdge[dace.Memlet]]]:
    """The empty Memlets that are the *only* connection between the two Map scopes.

    ``StateFusionExtended`` collapses an interstate edge and re-imposes the ordering it
    used to enforce with empty (happens-before) Memlets. Two Maps that were in separate
    states and had a WAR/WAW hazard therefore end up ordered but *not* connected by data.
    Such Maps look non-parallel to `is_parallel()` even though fusing them may well be
    legal, so this function recognises that situation.

    The function returns the ordering edges if, and only if, `first_map_entry` is the
    upstream Map and every connection to `second_map_entry` is an empty Memlet that
    goes *directly* from the first Map (its MapExit, or an AccessNode that only the
    first Map writes) into `second_map_entry`. In every other case -- the Maps are
    parallel, the direction is reversed, real data flows between them, or a third node
    sits in between and would lose its own ordering -- `None` is returned.

    :param state: The state in which the two Maps are.
    :param first_map_entry: Entry of the Map that must be the upstream one.
    :param second_map_entry: Entry of the Map that must be the downstream one.
    """
    first_map_exit: Optional[nodes.MapExit] = safe_exit_node(state, first_map_entry)
    second_map_exit: Optional[nodes.MapExit] = safe_exit_node(state, second_map_entry)
    if first_map_exit is None or second_map_exit is None:
        return None

    # `first` has to be the upstream Map. If it is the downstream one we bail out; the
    #  matcher also offers the swapped pair, which is the one that is handled.
    if not is_node_reachable_from(graph=state, begin=first_map_exit, end=second_map_entry):
        return None
    if is_node_reachable_from(graph=state, begin=second_map_exit, end=first_map_entry):
        return None

    # If the Maps stay connected once the empty Memlets are ignored, then real data flows
    #  between them. That is `MapFusionVertical`'s business, not ours.
    if is_node_reachable_from(graph=state, begin=first_map_exit, end=second_map_entry, ignore_empty_edges=True):
        return None

    # Nodes that the first Map alone writes; an ordering edge may also start there
    #  (that is the shape a WAW hazard produces).
    own_outputs: Set[nodes.Node] = {
        e.dst
        for e in state.out_edges(first_map_exit)
        if isinstance(e.dst, nodes.AccessNode) and all(ie.src is first_map_exit for ie in state.in_edges(e.dst))
    }

    ordering_edges: List[graph.MultiConnectorEdge[dace.Memlet]] = []
    for edge in state.edges():
        if edge.data is None or not edge.data.is_empty():
            continue
        if edge.dst is not second_map_entry:
            continue
        if edge.src is not first_map_exit and edge.src not in own_outputs:
            # Something else orders the second Map as well. Dropping the edge would
            #  silently drop that dependency, so refuse.
            return None
        ordering_edges.append(edge)

    # Every path from the first to the second Map must be covered, otherwise the Maps
    #  would still not be parallel after the edges are removed.
    if not ordering_edges:
        return None
    reachable_without = _is_reachable_ignoring(state, first_map_exit, second_map_entry, set(ordering_edges))
    return None if reachable_without else ordering_edges


def _is_reachable_ignoring(
    state: dace.SDFGState,
    begin: nodes.Node,
    end: nodes.Node,
    ignored_edges: Set[graph.MultiConnectorEdge[dace.Memlet]],
) -> bool:
    """`is_node_reachable_from()` with a set of edges cut out of the graph."""
    to_visit: List[nodes.Node] = [begin]
    seen: Set[nodes.Node] = set()
    while to_visit:
        node = to_visit.pop()
        if node is end:
            return True
        if node in seen:
            continue
        seen.add(node)
        to_visit.extend(e.dst for e in state.out_edges(node) if e not in ignored_edges)
    return False


def _scope_boundary_accesses(
    state: dace.SDFGState,
    map_entry: nodes.MapEntry,
    param_repl: Optional[Dict[str, str]],
) -> Tuple[Dict[str, List[Tuple[nodes.Node, subsets.Subset]]], Dict[str, List[Tuple[nodes.Node, subsets.Subset]]]]:
    """Per-iteration reads and writes of a Map scope, keyed by data name.

    The subsets are taken from the edges immediately *inside* the scope nodes, so they
    describe what a single iteration touches (for a nested Map the propagated subset
    covers all of its iterations, which is the conservative and correct thing here).
    The node returned alongside a subset is the one an ordering edge has to attach to.

    :param state: The state in which the Map is.
    :param map_entry: The entry node of the Map.
    :param param_repl: Renaming applied to the subsets, see `find_parameter_remapping()`.
    """
    map_exit: Optional[nodes.MapExit] = safe_exit_node(state, map_entry)
    if map_exit is None:
        return None, None
    reads: Dict[str, List[Tuple[nodes.Node, subsets.Subset]]] = {}
    writes: Dict[str, List[Tuple[nodes.Node, subsets.Subset]]] = {}

    read_side = (state.out_edges(map_entry), lambda e: e.dst, reads)
    write_side = (state.in_edges(map_exit), lambda e: e.src, writes)
    for edges, node_of, target in (read_side, write_side):
        for edge in edges:
            if edge.data is None or edge.data.is_empty() or edge.data.data is None:
                continue
            data, subset = _boundary_access(state, edge)
            if data is None or subset is None:
                return None, None
            subset = copy.deepcopy(subset)
            if param_repl:
                symbolic.safe_replace(param_repl, subset.replace)
            target.setdefault(data, []).append((node_of(edge), subset))
    return reads, writes


def _boundary_access(
    state: dace.SDFGState,
    edge: graph.MultiConnectorEdge[dace.Memlet],
) -> Tuple[Optional[str], Optional[subsets.Subset]]:
    """What a boundary edge really touches, or `(None, None)` if that cannot be determined.

    A copy-Memlet names one END of the edge and carries the other in `other_subset`, so `data`
    alone can name an inner buffer. Views resolve to what they view.
    """
    outer = _outer_data_of_boundary_edge(state, edge)
    if outer is None:
        return None, None
    if edge.data.data == outer:
        return outer, edge.data.subset
    # The Memlet names the other end, so only `other_subset` describes the outer access.
    return outer, edge.data.other_subset


def _outer_data_of_boundary_edge(
    state: dace.SDFGState,
    edge: graph.MultiConnectorEdge[dace.Memlet],
) -> Optional[str]:
    """The de-aliased array reached by following `edge` out through its scope node."""
    connector = edge.dst_conn if isinstance(edge.dst, nodes.MapExit) else edge.src_conn
    if connector is None or not connector.startswith(("IN_", "OUT_")):
        return None
    scope_node = edge.dst if isinstance(edge.dst, nodes.MapExit) else edge.src
    if isinstance(scope_node, nodes.MapExit):
        outer_edges = list(state.out_edges_by_connector(scope_node, "OUT_" + connector[3:]))
        endpoints = [e.dst for e in outer_edges]
    else:
        outer_edges = list(state.in_edges_by_connector(scope_node, "IN_" + connector[4:]))
        endpoints = [e.src for e in outer_edges]
    if len(endpoints) != 1 or not isinstance(endpoints[0], nodes.AccessNode):
        return None
    return _dealias(state, endpoints[0])


def resolve_view_source(
    state: dace.SDFGState,
    view_node: nodes.AccessNode,
) -> Optional[nodes.AccessNode]:
    """The AccessNode that `view_node` ultimately views, or `None` if that is undecidable.

    `sdutils.get_view_edge()` reports every ambiguous View shape it has no rule for by
    returning `None` -- except one: a View with no incoming data edge and more than one
    outgoing edge falls through to an unguarded `in_edges[0]` and raises `IndexError`
    (`dace/sdfg/utils.py`). That escapes the `can_be_applied()` safety nets, which do not
    list `IndexError`, so it is refused here instead. The shape only occurs on a malformed
    state -- `SDFG.validate()` raises the very same `IndexError` on it -- so refusing it
    costs no valid SDFG.

    :param state: The state in which the View is.
    :param view_node: The View AccessNode to resolve.
    """
    if state.out_degree(view_node) > 1 and not any(not e.data.is_empty() for e in state.in_edges(view_node)):
        return None
    return sdutils.get_last_view_node(state, view_node)


def _dealias(state: dace.SDFGState, node: nodes.AccessNode) -> Optional[str]:
    """The name of the array `node` ultimately refers to, resolving Views."""
    desc = node.desc(state.sdfg)
    if not isinstance(desc, dace.data.View):
        return node.data
    viewed = resolve_view_source(state, node)
    return None if viewed is None else viewed.data


def _is_iteration_private(
    first_subset: subsets.Subset,
    second_subset: subsets.Subset,
    params: List[sympy.Symbol],
) -> bool:
    """Whether two accesses of one array can only collide within the *same* iteration.

    Fusing two Maps replaces a whole-Map ordering ("all of the first Map, then all of the
    second") with a per-iteration one, so a hazard is only still covered if no iteration
    can collide with a *different* iteration. That holds when some dimensions pin every
    Map parameter: a dimension pins `p` if both accesses reduce to the same single point
    there and that point is an injective function of `p` alone (`p`, `p + 1`, `2 * p`,
    ...). Two distinct parameter tuples then differ in some `p`, hence differ in the
    dimension that pins `p`, hence touch different elements.

    :param first_subset: Subset accessed by the first Map, in its own parameters.
    :param second_subset: Subset accessed by the second Map, already renamed.
    :param params: The (common) Map parameters.
    """
    if first_subset is None or second_subset is None:
        return False
    if isinstance(first_subset, subsets.Indices):
        first_subset = subsets.Range.from_indices(first_subset)
    if isinstance(second_subset, subsets.Indices):
        second_subset = subsets.Range.from_indices(second_subset)
    if not isinstance(first_subset, subsets.Range) or not isinstance(second_subset, subsets.Range):
        return False
    if first_subset.dims() != second_subset.dims():
        return False

    pinned: Set[sympy.Symbol] = set()
    for (first_begin, first_end, _), (second_begin, second_end, _) in zip(first_subset.ranges, second_subset.ranges):
        # A bound sympy can not handle leaves this dimension unproven, which only costs a refusal.
        try:
            first_begin = symbolic.pystr_to_symbolic(first_begin)
            first_end = symbolic.pystr_to_symbolic(first_end)
            second_begin = symbolic.pystr_to_symbolic(second_begin)
            second_end = symbolic.pystr_to_symbolic(second_end)
            # Only a single point in this dimension, and the same one on both sides, tells
            #  us anything about iterations that are not the same.
            if symbolic.simplify(first_begin - first_end) != 0 or symbolic.simplify(second_begin - second_end) != 0:
                continue
            if symbolic.simplify(first_begin - second_begin) != 0:
                continue
            for param in params:
                if param not in first_begin.free_symbols:
                    continue
                if any(other in first_begin.free_symbols for other in params if other is not param):
                    continue
                # Injective in `param` iff shifting it by one always moves the point.
                step = symbolic.simplify(first_begin.subs(param, param + 1) - first_begin)
                if step != 0 and not step.free_symbols:
                    pinned.add(param)
        except (TypeError, ValueError, AttributeError, sympy.SympifyError, sympy.PolynomialError):
            continue
    return all(param in pinned for param in params)


def analyze_happens_before_fusion(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    first_map_entry: nodes.MapEntry,
    second_map_entry: nodes.MapEntry,
    param_repl: Dict[str, str],
) -> Optional[Tuple[List[graph.MultiConnectorEdge[dace.Memlet]], List[Tuple[nodes.Node, nodes.Node]]]]:
    """Plan the fusion of two Maps that are ordered by happens-before edges only.

    Returns `None` if the Maps are not in that situation or if fusing them would not be
    safe. Otherwise it returns the ordering edges that have to be *removed* (they would
    become a self loop on the fused Map) together with the pairs of body nodes that have
    to be re-connected by an empty Memlet *inside* the fused scope: the whole-Map
    ordering the removed edges provided degenerates into a per-iteration one, and it is
    those inner edges that provide it.

    :param state: The state in which the two Maps are.
    :param sdfg: The SDFG on which we operate.
    :param first_map_entry: Entry of the upstream Map.
    :param second_map_entry: Entry of the downstream Map.
    :param param_repl: Renaming of the second Map's parameters, see `find_parameter_remapping()`.
    """
    ordering_edges = find_happens_before_connection(state, first_map_entry, second_map_entry)
    if ordering_edges is None:
        return None

    # The subset analysis below reads only the edges that cross the scope nodes, so it never
    #  books what an AccessNode *inside* a body touches, and a side effect is not expressed as
    #  a Memlet at all. Refuse what it can never be completed for, and remember the rest.
    inner_data: List[Dict[str, None]] = []  # `dict` as an ordered set, one entry per Map
    for map_entry in (first_map_entry, second_map_entry):
        scope_data: Dict[str, None] = {}
        for node in state.scope_subgraph(map_entry, False, False).nodes():
            if isinstance(node, nodes.AccessNode):
                # Non-transient data is reachable from outside this state entirely, so the
                #  "does the other Map name it too" test below would not bound the hazard.
                if not sdfg.arrays[node.data].transient:
                    return None
                data = _dealias(state, node)
                if data is None:
                    return None
                scope_data[data] = None
            # A side effect takes part in an ordering that no Memlet describes, so dropping the
            #  happens-before edge would silently turn a whole-Map ordering of the effects into a
            #  per-iteration interleaving of them, with nothing below able to notice.
            elif isinstance(node, nodes.CodeNode) and node.has_side_effects(sdfg):
                return None
        inner_data.append(scope_data)

    first_reads, first_writes = _scope_boundary_accesses(state, first_map_entry, None)
    second_reads, second_writes = _scope_boundary_accesses(state, second_map_entry, param_repl)
    # An access the scan could not read hides every hazard it takes part in.
    if first_reads is None or second_reads is None:
        return None

    # An inner access is invisible above, so it is only harmless while the OTHER Map does not
    #  name the same data -- otherwise the two share a buffer through a hazard that the subset
    #  test never gets to see, and the ordering edge would be deleted with nothing put in its
    #  place. Transience is NOT the discriminator: sibling Map scopes sharing even a
    #  `Scope`-lifetime transient get ONE allocation, folded up to their common parent scope by
    #  `framecode.py`, so it is exactly as shared as a global would be.
    first_inner, second_inner = inner_data[0].keys(), inner_data[1].keys()
    first_touched = first_inner | first_reads.keys() | first_writes.keys()
    second_touched = second_inner | second_reads.keys() | second_writes.keys()
    if not first_inner.isdisjoint(second_touched) or not second_inner.isdisjoint(first_touched):
        return None
    params = [symbolic.pystr_to_symbolic(param) for param in first_map_entry.map.params]

    # An ordering edge must end up in front of the second Map's access, and a nested scope
    #  is ordered as a whole, i.e. after its exit resp. before its entry.
    def order_source(node: nodes.Node) -> Optional[nodes.Node]:
        return safe_exit_node(state, node) if isinstance(node, nodes.EntryNode) else node

    def order_target(node: nodes.Node) -> Optional[nodes.Node]:
        return state.scope_dict().get(node) if isinstance(node, nodes.ExitNode) else node

    inner_pairs: List[Tuple[nodes.Node, nodes.Node]] = []
    # Read/read is not a hazard, the other three combinations are (WAW, RAW, WAR).
    hazards = [(first_writes, second_writes), (first_writes, second_reads), (first_reads, second_writes)]
    for first_side, second_side in hazards:
        for data, first_accesses in first_side.items():
            for first_node, first_subset in first_accesses:
                for second_node, second_subset in second_side.get(data, []):
                    if not _is_iteration_private(first_subset, second_subset, params):
                        return None
                    ordered_pair = (order_source(first_node), order_target(second_node))
                    # A broken nested scope leaves nothing to attach the ordering edge to.
                    if ordered_pair[0] is None or ordered_pair[1] is None:
                        return None
                    inner_pairs.append(ordered_pair)

    return ordering_edges, list(dict.fromkeys(inner_pairs))


def dynamic_map_range_edge(
    state: dace.SDFGState,
    map_entry: nodes.MapEntry,
    symbol: str,
) -> Optional[graph.MultiConnectorEdge]:
    """The single edge that binds dynamic-map-range `symbol` on `map_entry`, if there is one."""
    # A dangling connector yields no edge; a bare `StopIteration` escaping into the matcher's
    #  generator would be converted into a `RuntimeError` by PEP 479.
    return next(iter(state.in_edges_by_connector(map_entry, symbol)), None)


def dynamic_map_range_binding_agrees(
    state: dace.SDFGState,
    map_entry: nodes.MapEntry,
    other_map_entry: nodes.MapEntry,
    symbol: str,
) -> bool:
    """`True` if both Maps bind dynamic-map-range `symbol` to provably the same value."""
    edge = dynamic_map_range_edge(state, map_entry, symbol)
    other_edge = dynamic_map_range_edge(state, other_map_entry, symbol)
    if edge is None or other_edge is None:
        return False
    data = edge.data.data
    if (data != other_edge.data.data or edge.src_conn != other_edge.src_conn
            or edge.data.subset != other_edge.data.subset):
        return False
    if edge.src is other_edge.src:
        return True
    # Any other producer computes a value this scan cannot see, making the test below vacuous.
    if not isinstance(edge.src, nodes.AccessNode) or not isinstance(other_edge.src, nodes.AccessNode):
        return False
    # Distinct sources agree only if nothing writes that data in this state.
    return not any(state.in_degree(dn) for dn in state.data_nodes() if dn.data == data)


def dynamic_map_ranges_agree(
    first_map_entry: nodes.MapEntry,
    second_map_entry: nodes.MapEntry,
    state: dace.SDFGState,
) -> bool:
    """`True` if every dynamic-map-range symbol both Maps bind is bound to the same value."""
    shared = first_map_entry.dynamic_input_connectors & second_map_entry.dynamic_input_connectors
    if not all(dynamic_map_range_binding_agrees(state, first_map_entry, second_map_entry, symbol) for symbol in shared):
        return False
    # The second Map's bindings are moved onto the first Map by `relocate_nodes()`, and a name
    #  that is already an OUT connector there can neither be added nor renamed.
    return all(symbol not in first_map_entry.out_connectors
               for symbol in second_map_entry.dynamic_input_connectors - shared)


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
    :note: `only_inner_maps` and `only_toplevel_maps` are mutually exclusive; the transformations
        reject the combination in their constructor, i.e. it is a configuration error and never
        reported from here.
    """

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

    # A colliding dynamic map range cannot be renamed, only dropped when both bind the same value.
    if not dynamic_map_ranges_agree(first_map_entry, second_map_entry, graph):
        return None

    # We will now check if we can rename the Map parameter of the second Map such that they
    #  match the one of the first Map.
    param_repl = find_parameter_remapping(first_map=first_map_entry.map, second_map=second_map_entry.map)
    return param_repl

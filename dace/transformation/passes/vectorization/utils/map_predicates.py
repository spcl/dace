# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Boolean predicates and defensive assertions on maps / SDFGs.

The ``assert_X`` siblings are kept alongside their ``X`` counterparts
intentionally — per the locked policy
("defensive checks and assertions stay"), every loud-failure helper
remains available to callers. Removing or relaxing them would shift
silent corruption back into the pipeline.
"""
import sympy

import dace
from dace import SDFGState
from dace.sdfg.state import BreakBlock, ConditionalBlock


def has_maps(sdfg: dace.SDFG) -> bool:
    """
    Check if the given SDFG or any nested SDFG contains a MapEntry node.

    Args:
        sdfg (dace.SDFG): The SDFG to inspect.

    Returns:
        bool: True if any MapEntry exists in the SDFG hierarchy, False otherwise.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map is innermost (contains no nested maps or maps inside nested SDFGs).

    Args:
        map_entry (dace.nodes.MapEntry): The map entry node to test.
        state (SDFGState): The state containing the map entry.

    Returns:
        bool: True if the map has no inner maps or maps in nested SDFGs, False otherwise.
    """
    nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    if any(isinstance(node, dace.nodes.MapEntry) for node in nodes_between):
        return False
    return not any(isinstance(node, dace.nodes.NestedSDFG) and has_maps(node.sdfg) for node in nodes_between)


def map_consists_of_single_nsdfg_or_no_nsdfg(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map contains either a single NestedSDFG or none at all.

    Args:
        map_entry (dace.nodes.MapEntry): The map entry to check.
        graph: The graph containing the map.

    Returns:
        bool: True if the map contains a single NestedSDFG or no NestedSDFG, False otherwise.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    return (len(all_nodes) == 1 and isinstance(next(
        iter(all_nodes)), dace.nodes.NestedSDFG)) or not any(isinstance(_n, dace.nodes.NestedSDFG) for _n in all_nodes)


def get_single_nsdfg_inside_map(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dace.nodes.NestedSDFG:
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    if (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)):
        return next(iter(all_nodes))
    return None


def has_only_states(sdfg: dace.SDFG) -> bool:
    return all({isinstance(n, dace.SDFGState) for n in sdfg.nodes()})


def has_only_states_or_single_block_with_break_only(sdfg: dace.SDFG) -> bool:
    ifs = {n for n in sdfg.nodes() if isinstance(n, ConditionalBlock)}
    all_ifs_are_only_break = all({
        len(ifb.branches) == 1 and len(ifb.branches[0][1].nodes()) == 1
        and isinstance(ifb.branches[0][1].nodes()[0], BreakBlock)
        for ifb in ifs
    })
    non_ifs_non_states = {n for n in sdfg.nodes() if not isinstance(n, ConditionalBlock) and not isinstance(n, SDFGState)}
    return (all({isinstance(n, dace.SDFGState)
                 for n in sdfg.nodes()}) or (all_ifs_are_only_break and len(non_ifs_non_states) == 0))


def assert_maps_consist_of_single_nsdfg_or_no_nsdfg(sdfg: dace.SDFG) -> None:
    """
    Assert that each map contains either a single NestedSDFG or none at all.

    Args:
        sdfg (dace.SDFG): The SDFG to validate.

    Raises:
        AssertionError: If a map body contains more than one node or a mix of NestedSDFG and other nodes.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            if not map_consists_of_single_nsdfg_or_no_nsdfg(g, n):
                all_nodes = {
                    k
                    for k in g.all_nodes_between(n, g.exit_node(n))
                    if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
                }
                nsdfg_count = len(set([node for node in all_nodes if isinstance(node, dace.nodes.NestedSDFG)]))
                raise AssertionError(f"Got nodes {all_nodes} has {nsdfg_count} nSDFGs")


def no_other_subset(state, recursive: bool = True) -> bool:
    """
    Check if any edge in the state has an 'other_subset' attribute set.

    Args:
        state: The state to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has other_subset set, False otherwise.
    """
    for edge in state.edges():
        if edge.data.other_subset is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not no_other_subset_sdfg(node.sdfg, True):
                    return False
    return True


def no_other_subset_sdfg(sdfg: dace.SDFG, recursive: bool = True) -> bool:
    """
    Check if any edge in the SDFG has an 'other_subset' attribute set.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has other_subset set, False otherwise.
    """
    for state in sdfg.all_states():
        if not no_other_subset(state, recursive):
            return False
    return True


def assert_no_other_subset(sdfg: dace.SDFG, recursive: bool = True):
    """
    Assert that no edge in the SDFG has an 'other_subset' attribute set.

    This validation is needed because vectorization does not support other subsets.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs
    """
    assert no_other_subset_sdfg(sdfg, recursive), "Found edge with other_subset set"


def no_wcr(state, recursive: bool = True) -> bool:
    """
    Check if any edge in the state has a write-conflict resolution (WCR) operation.

    Args:
        state: The state to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has WCR set, False otherwise.
    """
    for edge in state.edges():
        if edge.data.wcr is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not no_wcr_sdfg(node.sdfg, True):
                    return False
    return True


def no_wcr_sdfg(sdfg: dace.SDFG, recursive: bool = True) -> bool:
    """
    Check if any edge in the SDFG has a write-conflict resolution (WCR) operation.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs

    Returns:
        bool: True if no edge has WCR set, False otherwise.
    """
    for state in sdfg.all_states():
        if not no_wcr(state, recursive):
            return False
    return True


def assert_no_wcr(sdfg: dace.SDFG, recursive: bool = True):
    """
    Assert that no edge in the SDFG has a write-conflict resolution (WCR) operation.

    WCR operations handle conflicting writes to the same memory location. This assertion
    ensures that the SDFG doesn't contain such operations, which are not supported by
    auto-vectorization.

    Args:
        sdfg: The SDFG to check
        recursive: If True, recursively check nested SDFGs
    """
    assert no_wcr_sdfg(sdfg, recursive), "Found edge with WCR set"


def last_dim_of_map_is_contiguous_accesses(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if the last dimension of a map performs contiguous accesses.

    Args:
        map_entry (dace.nodes.MapEntry): The map entry to check.
        state: The state containing the map.

    Returns:
        bool: True if all memlets contain the last map parameter, False otherwise.
    """
    nodes = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes)
    # Currently this enforced the map parameter to be involved in the memlet access
    # This will fail in a case such as:
    # _s2 = map_param + 1
    # A[_s2] as `map_param` will not be detected anymore.
    # This can be fixed by improving the analysis, as is an open TODO task.
    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data
        if memlet.subset is None:
            continue
        stride_one_idx = [i for i, s in enumerate(state.sdfg.arrays[edge.data.data].strides) if s == 1][0]
        b, e, s = memlet.subset[stride_one_idx]
        b_free_syms = b.free_symbols if hasattr(b, "free_symbols") else set()
        e_free_syms = e.free_symbols if hasattr(e, "free_symbols") else set()
        free_symbols = {str(s) for s in b_free_syms.union(e_free_syms)}
        last_param = str(list(map_entry.map.params)[-1])
        if last_param not in free_symbols and free_symbols != set():
            return False
    return True


def count_param_in_expr(expr, param_str: str):
    """
    Count occurrences of a parameter inside a SymPy expression, including
    inside function-call arguments.

    Compares by symbol *name*, not by SymPy ``==``, because DaCe's
    ``dace.symbolic.symbol`` is a SymPy ``Symbol`` subclass whose equality
    semantics carry extra attributes — two symbols with the same name but
    different DaCe metadata can compare unequal under ``==`` while being the
    same identifier from the pass's perspective.
    """
    if not isinstance(expr, sympy.Basic):
        return 0

    count = 0
    # 1) Count standalone symbol occurrences (match by name)
    for atom in expr.atoms(sympy.Symbol):
        if str(atom) == param_str:
            count += 1

    # 2) Count function-call argument occurrences (nested)
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.FunctionClass):
            # node is a function name, skip
            continue
        if isinstance(node, sympy.Function):
            for arg in node.args:
                count += count_param_in_expr(arg, param_str)

    return count


def map_param_appears_in_multiple_dimensions(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if the last map parameter appears across multiple dimensions or
    function-call argument usages in the state.

    Args:
        map_entry (dace.nodes.MapEntry): The map entry node.
        state (dace.SDFGState): The containing state.

    Returns:
        bool: True if the map parameter appears in >1 dimension or in function args.
    """

    last_param = str(map_entry.map.params[-1])
    print(f"Checking last map parameter: {last_param}")

    nodes_between = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes_between)

    total_appearances = 0

    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data

        # -------------------------
        # 1. APPEARANCES IN SUBSETS
        # -------------------------
        if memlet.subset is not None:
            subset_appearances = 0
            for (b, e, s) in memlet.subset:
                # Extract free symbols
                # Count occurrences in lower bound
                if hasattr(b, "free_symbols"):
                    subset_appearances += count_param_in_expr(b, last_param)

            print(f"[Subset] {last_param} appears {subset_appearances} times in memlet {memlet}")

            if subset_appearances >= 2:
                return True

    print(f"Total appearances of {last_param}: {total_appearances}")
    return False


def assert_last_dim_of_maps_are_contigous_accesses(sdfg: dace.SDFG):
    """
    Assert that the last dimension of all maps in an SDFG performs contiguous accesses.

    For each innermost map, this function checks that the last map parameter
    appears in every memlet within the map body, ensuring the last dimension
    corresponds to contiguous memory accesses. It also validates that all tasklets
    are properly enclosed in map scopes or nested SDFGs.

    Args:
        sdfg (dace.SDFG): The SDFG to check.

    Raises:
        Exception: If a tasklet is not enclosed in any map or valid nested SDFG scope.
        ValueError: If a node's parent scope is not a map, or the last map parameter
            does not appear in a memlet subset expression.
        AssertionError: If internal consistency assumptions about map nesting fail.
    """
    # ``find_state_of_nsdfg_node`` still lives in ``vectorization_utils.py`` and
    # migrates in slice S4 (``utils/nsdfg_reshape.py``). Imported lazily here to
    # avoid the obvious import cycle (``vectorization_utils`` re-exports from
    # this module, and this module would otherwise import from
    # ``vectorization_utils`` at module load time).
    from dace.transformation.passes.vectorization.vectorization_utils import find_state_of_nsdfg_node
    checked_map_entries = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            # Skip map entries/exits
            if isinstance(node, (dace.nodes.MapEntry, dace.nodes.MapExit)):
                continue

            map_entry = state.scope_dict()[node]
            if map_entry is None:
                if isinstance(node, dace.nodes.Tasklet):
                    parent_nsdfg = state.sdfg.parent_nsdfg_node
                    if parent_nsdfg is None:
                        continue
                    parent_state = find_state_of_nsdfg_node(sdfg, node)
                    parent_scope = parent_state.scope_dict()[parent_nsdfg]
                    if parent_scope is None or not isinstance(parent_scope, dace.nodes.MapEntry):
                        raise Exception(f"No NSDFGs that are not within Map scopes should be left, "
                                        f"check {parent_nsdfg} in state {parent_state}. Call inlineSDFG")
                else:
                    continue
            else:
                if not isinstance(map_entry, dace.nodes.MapEntry):
                    raise ValueError(f"Parent scope of node {node} is not a map, found {map_entry} in state {state}.")
                checked_map_entries.add(map_entry)

            if map_entry not in checked_map_entries:
                assert isinstance(
                    map_entry, dace.nodes.MapEntry), f"Parent scope of node {node} is not a map, returned {map_entry}."
                # Currently this enforced the map parameter to be involved in the memlet access
                # This will fail in a case such as:
                # _s2 = map_param + 1
                # A[_s2] as `map_param` will not be detected anymore.
                # This can be fixed by improving the analysis, as is an open TODO task.
                if not last_dim_of_map_is_contiguous_accesses(state, map_entry):
                    raise ValueError(f"Last map parameter must be in the memlet, "
                                     f"not in this case {map_entry}, {state}")


def map_has_branching_memlets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for out_conn in map_entry.out_connectors:
        out_egdges_of_out_conn = set(state.out_edges_by_connector(map_entry, out_conn))
        if len(out_egdges_of_out_conn) > 1:
            return True
    return False


def sdfg_has_nested_sdfgs(sdfg: dace.SDFG):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                return True
    return False


def map_has_nested_sdfgs(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            return True
    return False


def has_nsdfg_depth_more_than_one(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            if sdfg_has_nested_sdfgs(node.sdfg):
                return True
    return False

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
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbols


def has_maps(sdfg: dace.SDFG) -> bool:
    """
    Check if the SDFG or any nested SDFG contains a MapEntry node.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if any MapEntry exists in the hierarchy.
    """
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            return True
    return False


def is_innermost_map(state: SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map is innermost (no nested maps, including inside nested SDFGs).

    :param state: The state containing the map entry.
    :param map_entry: The map entry node to test.
    :returns: ``True`` if the map has no inner maps.
    """
    nodes_between = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    if any(isinstance(node, dace.nodes.MapEntry) for node in nodes_between):
        return False
    return not any(isinstance(node, dace.nodes.NestedSDFG) and has_maps(node.sdfg) for node in nodes_between)


def map_consists_of_single_nsdfg_or_no_nsdfg(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if a map contains either a single NestedSDFG or none at all.

    :param graph: The state containing the map.
    :param map_entry: The map entry to check.
    :returns: ``True`` if the map contains a single NestedSDFG or no NestedSDFG.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    return (len(all_nodes) == 1 and isinstance(next(
        iter(all_nodes)), dace.nodes.NestedSDFG)) or not any(isinstance(_n, dace.nodes.NestedSDFG) for _n in all_nodes)


def get_single_nsdfg_inside_map(graph: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> dace.nodes.NestedSDFG:
    """
    Return the sole NestedSDFG inside a map, or ``None`` if not exactly one.

    :param graph: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: The single NestedSDFG node, or ``None``.
    """
    all_nodes = {
        k
        for k in graph.all_nodes_between(map_entry, graph.exit_node(map_entry))
        if not isinstance(k, (dace.nodes.MapEntry, dace.nodes.MapExit))
    }
    if (len(all_nodes) == 1 and isinstance(next(iter(all_nodes)), dace.nodes.NestedSDFG)):
        return next(iter(all_nodes))
    return None


def has_only_states(sdfg: dace.SDFG) -> bool:
    """
    Check whether every top-level node of an SDFG is a plain SDFGState.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if no control-flow regions are present.
    """
    return all({isinstance(n, dace.SDFGState) for n in sdfg.nodes()})


def has_only_states_or_single_block_with_break_only(sdfg: dace.SDFG) -> bool:
    """
    Check that an SDFG has only states, or only conditional blocks whose sole branch is a break.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if the SDFG matches either shape.
    """
    ifs = {n for n in sdfg.nodes() if isinstance(n, ConditionalBlock)}
    all_ifs_are_only_break = all({
        len(ifb.branches) == 1 and len(ifb.branches[0][1].nodes()) == 1
        and isinstance(ifb.branches[0][1].nodes()[0], BreakBlock)
        for ifb in ifs
    })
    non_ifs_non_states = {
        n
        for n in sdfg.nodes() if not isinstance(n, ConditionalBlock) and not isinstance(n, SDFGState)
    }
    return (all({isinstance(n, dace.SDFGState)
                 for n in sdfg.nodes()}) or (all_ifs_are_only_break and len(non_ifs_non_states) == 0))


def _no_edge_attr_state(state, attr: str, recursive: bool) -> bool:
    """Return True iff no edge in ``state`` has the attribute set (``is not None``).

    With ``recursive=True``, recursively descends into NestedSDFG nodes.
    """
    for edge in state.edges():
        value = edge.data.wcr if attr == "wcr" else edge.data.other_subset
        if value is not None:
            return False
    if recursive:
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if not _no_edge_attr_sdfg(node.sdfg, attr, True):
                    return False
    return True


def _no_edge_attr_sdfg(sdfg: dace.SDFG, attr: str, recursive: bool) -> bool:
    """Return True iff no edge in any state of ``sdfg`` has the attribute set."""
    for state in sdfg.all_states():
        if not _no_edge_attr_state(state, attr, recursive):
            return False
    return True


def no_other_subset(state, recursive: bool = True) -> bool:
    """True iff no edge in ``state`` has ``other_subset`` set; recurses into NSDFGs by default."""
    return _no_edge_attr_state(state, "other_subset", recursive)


def no_wcr(state, recursive: bool = True) -> bool:
    """True iff no edge in ``state`` has WCR set; recurses into NSDFGs by default."""
    return _no_edge_attr_state(state, "wcr", recursive)


def last_dim_of_map_is_contiguous_accesses(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check if the last dimension of a map performs contiguous accesses.

    :param state: The state containing the map.
    :param map_entry: The map entry to check.
    :returns: ``True`` if every memlet's unit-stride dim involves the last map parameter.
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
        b_free_syms = free_symbols(b)
        e_free_syms = free_symbols(e)
        all_syms = {str(s) for s in b_free_syms.union(e_free_syms)}
        last_param = str(list(map_entry.map.params)[-1])
        if last_param not in all_syms and all_syms != set():
            return False
    return True


def count_param_in_expr(expr, param_str: str):
    """
    Count occurrences of a parameter in a SymPy expression, including function-call args.

    Matches by symbol name (not SymPy ``==``), since DaCe symbols with the
    same name but different metadata can compare unequal.

    :param expr: The SymPy expression to scan.
    :param param_str: The parameter name to count.
    :returns: Number of occurrences.
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
    Check if the last map parameter appears across multiple subset dimensions.

    :param state: The containing state.
    :param map_entry: The map entry node.
    :returns: ``True`` if the last parameter appears in more than one dimension.
    """

    last_param = str(map_entry.map.params[-1])

    nodes_between = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes_between)

    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data

        # Count occurrences of the last map parameter across all subset
        # dimensions of this memlet; flag if it appears more than once.
        if memlet.subset is not None:
            subset_appearances = 0
            for (b, e, s) in memlet.subset:
                if free_symbols(b):
                    subset_appearances += count_param_in_expr(b, last_param)

            if subset_appearances >= 2:
                return True

    return False


def is_linear_in_param(expr, param_str: str) -> bool:
    """
    Return whether ``expr`` is linear in ``param_str`` (form ``c*p + d``, ``c``/``d`` constant in ``p``).

    A bare integer / float literal counts as linear (coefficient 0).

    :param expr: The expression to classify.
    :param param_str: The parameter symbol name.
    :returns: ``True`` if ``expr`` is linear in the parameter.
    """
    if not isinstance(expr, sympy.Basic):
        return True  # plain int/float literal
    param_sym = sympy.Symbol(param_str)
    if param_sym not in expr.free_symbols:
        return True
    try:
        poly = sympy.Poly(expr, param_sym)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return False
    if poly.degree() > 1:
        return False
    # Coefficients must not themselves contain ``param_sym``.
    for c in poly.all_coeffs():
        if param_sym in free_symbols(c):
            return False
    return True


def map_param_dim_usage_is_linear_combo(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
    """
    Check that multi-dimension uses of the last map parameter are all linear in it.

    For every memlet whose last parameter appears in more than one
    dimension, each such dimension must be a point access whose begin
    expression is linear in the parameter. Memlets where the parameter is
    absent or used in only one dimension do not block the classification.

    :param state: The containing state.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if all multi-dim uses are linear (strided-lowerable).
    """
    last_param = str(map_entry.map.params[-1])
    nodes_between = list(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
    edges = state.all_edges(*nodes_between)
    for edge in edges:
        memlet: dace.memlet.Memlet = edge.data
        if memlet.subset is None:
            continue
        dims_with_param = []
        for d, (b, e, _) in enumerate(memlet.subset):
            if free_symbols(b) and count_param_in_expr(b, last_param) > 0:
                dims_with_param.append((d, b, e))
        if len(dims_with_param) < 2:
            continue
        for _, b, e in dims_with_param:
            if b != e:
                return False
            if not is_linear_in_param(b, last_param):
                return False
    return True


def map_has_branching_memlets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    """
    Check whether any map-entry out-connector feeds more than one edge.

    :param state: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if a single out-connector branches to multiple edges.
    """
    for out_conn in map_entry.out_connectors:
        out_egdges_of_out_conn = set(state.out_edges_by_connector(map_entry, out_conn))
        if len(out_egdges_of_out_conn) > 1:
            return True
    return False


def sdfg_has_nested_sdfgs(sdfg: dace.SDFG):
    """
    Check whether an SDFG contains any NestedSDFG node.

    :param sdfg: The SDFG to inspect.
    :returns: ``True`` if a NestedSDFG node is present.
    """
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                return True
    return False


def has_nsdfg_depth_more_than_one(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    """
    Check whether a map body contains a NestedSDFG that itself contains a NestedSDFG.

    :param state: The state containing the map.
    :param map_entry: The map entry to inspect.
    :returns: ``True`` if nested-SDFG depth exceeds one.
    """
    for node in state.all_nodes_between(map_entry, state.exit_node(map_entry)):
        if isinstance(node, dace.nodes.NestedSDFG):
            if sdfg_has_nested_sdfgs(node.sdfg):
                return True
    return False

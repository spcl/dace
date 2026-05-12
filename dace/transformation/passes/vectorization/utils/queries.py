# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Read-only query helpers used by the vectorization pipeline.

These helpers do not mutate the SDFG; they extract access subsets,
map-parameter relationships, or scalar conversions used by the
emission and prep passes. ``collect_vectorizable_arrays`` is not in
this module yet — it depends on ``_all_atoms`` and
``find_symbol_assignment`` from the lane-expansion module that
migrates in a later slice; it will move once those land here.
"""
import typing
from typing import Dict, Set, Tuple

import sympy

import dace


def to_ints(sym_epxr: dace.symbolic.SymExpr) -> typing.Union[int, None]:
    """
    Try to convert a symbolic expression to an integer.

    Args:
        sym_epxr (dace.symbolic.SymExpr): The symbolic expression to convert.

    Returns:
        int | None: The integer value if conversion succeeds, otherwise None.
    """
    try:
        return int(sym_epxr)
    except Exception:
        return None


def collect_non_unit_stride_accesses_in_map(sdfg: dace.SDFG, state: dace.SDFGState,
                                            map_entry: dace.nodes.MapEntry) -> Set[str]:
    """
    Determines which arrays can be vectorized based on their access patterns and symbol usage.
    The symbols used for accessing should not have any indirectness, meaning that they should
    not be accessing other Arrays on interstate assignemnts, this is expressed as a free function
    in sympy.

    The map parameter involve in vectorization should not appear in a multiplicaiton expression.
    E.g. loop (int i = 0; i < N; i ++) and access A[i] is ok but, A[i*2] means it is strided and it
    needs to be packed

    Args:
        sdfg: The SDFG to analyze.
        parent_nsdfg_node: NestedSDFG node.
        parent_state: State containing the NestedSDFG.
        invariant_scalars: Set of scalar names that are invariant across lanes (means these
            scalars to do not prevent vectorization)

    Returns:
        Dictionary mapping array names to a boolean indicating vectorizability.
    """
    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = map_entry
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]

    # Collect all subsets
    all_nodes = state.all_nodes_between(parent_map, state.exit_node(parent_map))
    all_accesses_to_arrays = {e.data.data: list() for e in state.all_edges(*all_nodes) if e.data.data is not None}
    for e in state.all_edges(*all_nodes):
        if e.data.data is not None:
            all_accesses_to_arrays[e.data.data].append(e.data.subset)

    for edge in state.all_edges(*all_nodes):
        if edge.data.other_subset is not None:
            raise NotImplementedError("other subset support not implemented")

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

    # Since no nestedSDFG, no indirectness may occur just check stridedness

    for arr_name, accesses in all_accesses_to_arrays.items():
        for access_subset in accesses:
            # Get the stride 1 dimension
            stride_one_dim = {i for i, stride in enumerate(sdfg.arrays[arr_name].strides) if stride == 1}.pop()
            b, e, s = access_subset[stride_one_dim]
            assert b == e
            assert s == 1

            # Evaluate the expression (b == e)
            access_expr = b  # use b since b==e
            if isinstance(access_expr, (dace.symbolic.SymExpr, sympy.Expr)):
                # Check for multipliers
                # If map_param appears multiplied in the expression, it is strided
                free_syms = {str(s) for s in access_expr.free_symbols}
                if len({
                        term
                        for term in access_expr.atoms(sympy.Mul)
                        if isinstance(term, sympy.Mul) and map_param in free_syms
                }) > 0:
                    array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_accesses_to_array_name(sdfg: dace.SDFG) -> Dict[Tuple[str, dace.subsets.Range], str]:
    """
    Collects all access subsets for each array in the SDFG.

    Args:
        sdfg: The SDFG to analyze.

    Returns:
        Dictionary mapping array names to a set of accessed subsets.
    """
    d = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")
            if edge.data.data is not None:
                if edge.data.data not in d:
                    d[edge.data.data] = set()
                d[edge.data.data].add(edge.data.subset)
    return d


def collect_all_memlets_to_dataname(sdfg: dace.SDFG) -> Dict[str, Set[dace.subsets.Range]]:
    """
    Collect all unique memlet subsets for each data array in the SDFG.

    This function traverses all states and edges in the SDFG and groups memlet subsets
    by the data array they access. Does not check interstate edges or conditionals.

    Args:
        sdfg: The SDFG to analyze

    Returns:
        A dictionary mapping data array names to sets of their accessed subsets
    """
    dataname_to_memlets = dict()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.data is not None:
                if edge.data.data not in dataname_to_memlets:
                    dataname_to_memlets[edge.data.data] = set()
                dataname_to_memlets[edge.data.data].add(edge.data.subset)

    return dataname_to_memlets


def parse_int_or_default(value, default=8):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

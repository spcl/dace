# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Read-only query helpers used by the vectorization pipeline.

These helpers do not mutate the SDFG; they extract access subsets,
map-parameter relationships, or scalar conversions used by the
emission and prep passes.
"""
import typing
from typing import Dict, Optional, Set, Tuple

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
            # Param appearing in more than one dimension (diagonal A[i,i] /
            # linear-combo A[2*i,i] / A[i,2*i]) is a non-unit-stride access
            # regardless of which dim happens to be stride-1; the per-lane
            # fan-out path linearises it through the array strides.
            dims_with_param = 0
            for dim_b, _, _ in access_subset:
                if hasattr(dim_b, "free_symbols") and any(str(s) == map_param for s in dim_b.free_symbols):
                    dims_with_param += 1
            if dims_with_param > 1:
                array_is_vectorizable[arr_name] = False
                continue

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


def collect_vectorizable_arrays(sdfg: dace.SDFG, parent_nsdfg_node: dace.nodes.NestedSDFG,
                                parent_state: dace.SDFGState, invariant_scalars: Set[str]) -> Dict[str, bool]:
    """
    Determines which arrays can be vectorized based on their access patterns and symbol usage.
    The symbols used for accessing should not have any indirectness, meaning that they should
    not be accessing other Arrays on interstate assignemnts, this is expressed as a free function
    in sympy.

    The map parameter involve in vectorization should not appear in a multiplicaiton expression.
    E.g. loop (int i = 0; i < N; i ++) and access A[i] is ok but, A[i*2] means it is strided and it
    needs to be packed

    Consider the case A[for_it_88, 0, jo] and interstate assignment has jo = B[for_it_88, 0]
    And the loop is over 0->for_it_88, this not vectorizable, so if any dimension involved uses the loop map
    param return false

    Args:
        sdfg: The SDFG to analyze.
        parent_nsdfg_node: NestedSDFG node.
        parent_state: State containing the NestedSDFG.
        invariant_scalars: Set of scalar names that are invariant across lanes (means these
            scalars to do not prevent vectorization)

    Returns:
        Dictionary mapping array names to a boolean indicating vectorizability.
    """
    # Lazy import to avoid an obvious cycle: ``utils.lane_expansion``
    # itself imports from ``utils.name_schemes`` but not from this
    # module — keep the import inside the function so callers don't
    # have to reason about load order between ``queries`` and
    # ``lane_expansion``.
    from dace.transformation.passes.vectorization.utils.lane_expansion import (
        _all_atoms,
        find_symbol_assignment,
    )

    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = parent_state.scope_dict()[parent_nsdfg_node]
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]
    parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)

    all_accesses_to_arrays = collect_accesses_to_array_name(sdfg)

    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")

    # Drop ``_iter_mask`` arrays entirely from the classification: they are
    # transient W-wide bool buffers filled per lane by GenerateIterationMask
    # (P3) and are NOT user data the vectorizer should touch — neither
    # vectorize (they are already W-wide) nor pack (they are not strided).
    # Their fill memlet is ``[0:W]`` which would fail the point-access
    # assert below; their packing path crashes on tasklet edges.
    for arr_name in list(all_accesses_to_arrays.keys()):
        if arr_name == "_iter_mask" or arr_name.startswith("_iter_mask_"):
            del all_accesses_to_arrays[arr_name]

    array_is_vectorizable = {k: True for k in all_accesses_to_arrays}

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
                # Strided iff the map_param is a *direct* factor of some Mul
                # term — e.g. ``2*i``.  Expressions like ``i + LEN_1D // 2``
                # contain a Mul atom ``LEN_1D / 2`` whose free symbols don't
                # include the map_param; subset-propagation can also leave
                # ``i - Min(i, i + LEN_1D // 2)``, whose Mul atom
                # ``-Min(i, i + LEN_1D // 2)`` carries ``i`` transitively via
                # ``Min`` but doesn't make the access strided.  Inspect the
                # Mul's direct args so neither case is misclassified.
                def _is_strided_mul(term: sympy.Mul) -> bool:
                    for f in term.args:
                        if isinstance(f, sympy.Symbol) and str(f) == map_param:
                            return True
                    return False

                if any(_is_strided_mul(term) for term in access_expr.atoms(sympy.Mul)):
                    array_is_vectorizable[arr_name] = False
                    raise Exception("TODO - I have not analyzed this case yet")

            if isinstance(b, (dace.symbolic.SymExpr, dace.symbolic.symbol, sympy.Expr)):
                if isinstance(b, (dace.symbolic.SymExpr, sympy.Expr)):
                    free_syms = {str(s) for s in b.free_symbols}
                else:
                    free_syms = {b}
                for free_sym in free_syms:
                    # Accessing map param is ok
                    if str(free_sym) == map_param:
                        continue
                    else:
                        # Other free symbols should not have indirect accesses
                        # Analysis tries find the first assignment in the CFG
                        assignment = find_symbol_assignment(sdfg, str(free_sym))
                        assert not (
                            assignment is None and str(free_sym) not in parent_syms_defined
                        ), f"Could not find an iedge assignment for {free_sym}, assignment {assignment}, parent symbols defined {parent_syms_defined}. {sdfg.label}, {sdfg.parent_nsdfg_node}: map param {map_param}"
                        # Loop invariant symbol passed from outside
                        if assignment is None:
                            continue

                        assignment_expr = dace.symbolic.SymExpr(assignment)
                        # Define functions to ignore (common arithmetic + piecewise + rounding)
                        ignored = {
                            sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs, sympy.floor,
                            sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan, sympy.sinh,
                            sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                        }

                        # Collect only user-defined or nonstandard functions - in intersate edge this means array accees
                        funcs = {f.name for f in assignment_expr.atoms(sympy.Function) if f.func not in ignored}
                        # Any array on the right-hand-side -> big problem
                        # Check for scalar / array accesses like this too
                        scalars = {str(s)
                                   for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars
                        # If scalar is invariant it should be ok?
                        #      {s
                        #       for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars)
                        if len(funcs) != 0 or len(scalars) != 0:
                            array_is_vectorizable[arr_name] = False

            # Go through non unit stride dimensions in case it those dimensions have unstructuredness
            for i, (b, e, s) in enumerate(access_subset):
                if i == stride_one_dim:
                    continue
                free_syms = set()
                if hasattr(b, "free_syms"):
                    free_syms = {str(s) for s in b.free_syms}
                if hasattr(b, "free_symbols"):
                    free_syms = {str(s) for s in b.free_symbols}

                if free_syms != set():
                    for free_sym in free_syms:
                        # Accessing map param is ok
                        if str(free_sym) == map_param:
                            continue
                        else:
                            # Other free symbols should not have indirect accesses
                            # Analysis tries find the first assignment in the CFG
                            assignment = find_symbol_assignment(sdfg, str(free_sym))

                            # If assignment is None, it is probably coming from parent map
                            parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)
                            if assignment is None:
                                assert str(
                                    free_sym
                                ) in parent_syms_defined, f"Could not find an iedge assignment for {free_sym} it is also not defined in symbols defined in nsdfg entry {parent_syms_defined}"
                                continue

                            assignment_expr = dace.symbolic.SymExpr(assignment)
                            # Define functions to ignore (common arithmetic + piecewise + rounding)
                            ignored = {
                                sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log, sympy.sqrt, sympy.Abs,
                                sympy.floor, sympy.ceiling, sympy.Min, sympy.Max, sympy.asin, sympy.acos, sympy.atan,
                                sympy.sinh, sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh
                            }
                            all_atoms = _all_atoms(assignment_expr, ignored)
                            all_atoms_str = {str(s) for s in all_atoms}

                            # Map parameter appears in inddirect access, array is not vectorizable
                            if map_param in all_atoms_str:
                                array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


def collect_element_write_subsets(state: dace.SDFGState) -> Optional[Dict[str, dace.subsets.Range]]:
    """Return ``{arr_name: subset}`` for every element-wise write in ``state``.

    A write is element-wise iff its memlet subset has
    ``num_elements_exact() == 1``. Returns ``None`` if any in-edge to an
    AccessNode in ``state`` violates that — both M3.1
    (``SameWriteSetIfElseToMergeCFG``) and M3.2 (``BranchNormalization``)
    skip the rewrite when this happens.

    Multiple writes to the same array within ``state`` keep only the
    last subset seen (matches the existing behaviour of both callers).
    """
    out: Dict[str, dace.subsets.Range] = {}
    for n in state.nodes():
        if not isinstance(n, dace.nodes.AccessNode):
            continue
        for e in state.in_edges(n):
            if e.data.data is None:
                continue
            try:
                if e.data.subset.num_elements_exact() != 1:
                    return None
            except Exception:
                return None
            out[n.data] = e.data.subset
    return out

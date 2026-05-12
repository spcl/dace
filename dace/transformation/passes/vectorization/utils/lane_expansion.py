# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Symbol lane fan-out helpers for the vectorization pipeline.

When a scalar symbol is computed by an interstate edge in an inner SDFG
that sits below a vectorized map, every lane needs its own copy of the
symbol (one per lane = ``vector_width`` copies). The helpers here build
those per-lane variants and stitch them back into the SDFG:

- ``expand_interstate_assignments_to_lanes``: the core fan-out. Rewrites
  every interstate-edge assignment ``sym = expr`` into a family
  ``sym_laneid_<i> = expr-with-vector-iter-shifted``, keyed via
  ``LaneIdScheme``. Already-lane-encoded LHS keys are carried through
  unchanged (idempotency).
- ``try_demoting_vectorizable_symbols``: demotes interstate-edge
  symbols whose value depends only on parent-scope state into scalar
  data nodes. This frees them from the lane-fan-out path entirely.
- ``resolve_missing_laneid_symbols``: bandage pass for any free
  ``sym_laneid_<i>`` symbol that survived the fan-out without a
  defining edge — synthesises the assignment from the parent-map state.
- ``assert_symbols_in_parent_map_symbols``: defensive validator that
  every ``*_laneid_*`` symbol in a missing-symbols set corresponds to a
  parent-map / parent-loop iterator. Loud failure preferred.
- ``find_symbol_assignment``: backwards traversal that returns the
  first interstate-edge assignment expression for a symbol.
- ``_all_atoms``: SymPy helper that collects every Symbol / Indexed /
  Function atom in an expression, used by both ``collect_vectorizable_arrays``
  (still in the legacy module pending S6d) and the fan-out itself.

This is the most fragile module in the vectorization pipeline; touch
with care. Latent issues (non-idempotency footguns, hand-rolled
name-parsing in ``resolve_missing_laneid_symbols``, etc.) are recorded
in the master plan rather than fixed during the mechanical S6c move.
"""
import re
from typing import Set

import sympy

import dace
import dace.sdfg.construction_utils as cutil
import dace.sdfg.utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.symbolic import DaceSympyPrinter

from dace.transformation.passes.vectorization.vectorization_utils import LaneIdScheme


def assert_symbols_in_parent_map_symbols(missing_symbols: Set[str], state: dace.SDFGState,
                                         nsdfg: dace.nodes.NestedSDFG):
    """
    Validates that given symbols correspond to loop variables in parent map scopes of a NestedSDFG.

    Args:
        missing_symbols: Symbols to validate (e.g., {"i_laneid_0", "j_laneid_1"}).
        state: The SDFG state.
        nsdfg: NestedSDFG node.

    Returns:
        Set of loop variable names found in the parent scopes.

    Raises:
        AssertionError if a symbol is not found in the loop scopes.
    """

    def validate_and_strip(strings):
        valid = []
        for s in strings:
            match = re.fullmatch(r'([A-Za-z_]\w*?)(\d+)$', s)
            assert match, f"No match in {strings} for a variable name"
            name, num = match.groups()
            valid.append((name, int(num)))
        return valid

    stripped_symbols = validate_and_strip(missing_symbols)
    loop_vars = {var for var, int_id in stripped_symbols}

    sdict = state.scope_dict()
    first_parent_map = sdict[nsdfg]
    parent_maps_and_loops = cutil.get_parent_map_and_loop_scopes(state.sdfg, first_parent_map, state)

    loop_symbols = set()
    for p in first_parent_map.map.params:
        loop_symbols.add(p)

    for map_or_loop in parent_maps_and_loops:
        if isinstance(map_or_loop, dace.nodes.MapEntry):
            for p in map_or_loop.map.params:
                loop_symbols.add(p)
        elif isinstance(map_or_loop, LoopRegion):
            loop_symbols.add(map_or_loop.loop_variable)

    for loop_var in loop_vars:
        loop_var = loop_var[:-len("_laneid_")] if loop_var.endswith("_laneid_") else loop_var
        assert loop_var in loop_symbols or loop_var in nsdfg.symbol_mapping, (
            f"{loop_var} not in parent-scope loop_symbols={loop_symbols} and not in "
            f"nsdfg.symbol_mapping={set(nsdfg.symbol_mapping.keys())}")

    return loop_vars


def find_symbol_assignment(sdfg: dace.SDFG, sym_name: str) -> str:
    """
    Finds the assignment expression of a given symbol by traversing the SDFG backwards.

    Args:
        sdfg: The SDFG to search.
        sym_name: Symbol to find.

    Returns:
        Assignment expression as a string, or None if not found.
    """

    # Pre-condition for vectorization
    assert all({isinstance(s, dace.SDFGState) for s in sdfg.nodes()})
    sink_state = {s for s in sdfg.nodes() if sdfg.out_degree(s) == 0}.pop()
    edges_to_check = sink_state.parent_graph.in_edges(sink_state)
    while edges_to_check:
        edge = edges_to_check.pop()

        for k, v in edge.data.assignments.items():
            if k == sym_name:
                return v

        edges_to_check += sink_state.parent_graph.in_edges(edge.src)

    return None
    #raise Exception("Symbol assignment not found")


def _all_atoms(expr, ignored=()):
    """
    Return a set of all atomic elements in a SymPy expression, including:
    - Symbols
    - Indexed symbols / arrays
    - Function calls
    - Numbers (optional)

    ignored: tuple of types to ignore, e.g., (sympy.Number,)
    """
    # Use expr.atoms to get all different types of atoms
    atoms = set()

    # Get all symbols
    atoms.update(expr.atoms(sympy.Symbol))

    # Get all Indexed (arrays)
    atoms.update(expr.atoms(sympy.Indexed))

    # Get all function symbols (but not the class, only instances)
    funcs = expr.atoms(sympy.Function)
    for f in funcs:
        if f.func not in ignored:
            atoms.add(f)
            # Also include arguments of the function
            atoms.update(f.args)

    return atoms


def expand_interstate_assignments_to_lanes(inner_sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG,
                                           state: dace.SDFGState, vector_width: int, invariant_data: Set[str],
                                           vector_map_param: str):
    # `sym = 0`
    # Would become
    # `sym_laneid_0 = 0, sym=sym_laneid_0, sym_laneid_1 = 0, sym_laneid_2 = 0, ....`
    # Assume:
    # `sym = A[_for_it] + 1`
    # Would become:
    # `sym_laneid_0 = A[_for_it + 0] + 1`, `sym = sym_laneid_0`, `sym_laneid_1 = A[_for_it + 1] + 1`, ...

    # Invariant data means that the data is constant across iterators
    # If all free symbols are from invariant data then duplication is not necessar

    # Pre-condition last dimension is the dimension we vectorize
    parent_map_entry = state.scope_dict()[nsdfg_node]
    assert parent_map_entry is not None and isinstance(parent_map_entry, dace.nodes.MapEntry)
    vectorized_param = vector_map_param
    #print(vector_map_param)

    for edge in inner_sdfg.all_interstate_edges():
        new_assignments = dict()
        assignments = edge.data.assignments

        # Idempotency: any LHS that already encodes a lane in its name is taken as
        # fixed (its lane is fully determined by the suffix). Re-expanding it would
        # produce <base>_laneid_<i>_laneid_<j> double-suffixed garbage. Carry the
        # already-expanded assignments through unchanged and drive the per-lane loop
        # only over the plain (un-encoded) keys.
        plain_assignments = {}
        for k, v in assignments.items():
            if LaneIdScheme.is_laneid(k):
                new_assignments[k] = v
            else:
                plain_assignments[k] = v

        for k, v in plain_assignments.items():
            original_v_expr = dace.symbolic.SymExpr(v)
            for i in range(vector_width):
                new_k = LaneIdScheme.make(k, i)
                v_expr = dace.symbolic.SymExpr(v)

                funcs = {str(f) for f in v_expr.atoms(sympy.Function)}
                non_func_free_syms = {str(s) for s in v_expr.free_symbols if str(s) not in funcs}
                array_accesses = {f for f in funcs if f in inner_sdfg.arrays}
                variant_array_accesses = (array_accesses.union(non_func_free_syms)) - invariant_data

                if len(variant_array_accesses) == 0:
                    # Whole expression is invariant — keep the original (un-expanded) symbol only.
                    new_assignments[k] = v
                    continue

                if new_k not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(new_k, inner_sdfg.symbols.get(k, dace.float64))

                # Replace the vector iterator with iter+lane
                v_expr = v_expr.subs(vectorized_param, f"({vectorized_param} + {i})")

                # Other free symbols are duplicated per-lane; symbols that already encode
                # a lane (parse non-trivially) are skipped so we never produce a doubly
                # lane-suffixed name.
                non_map_free_syms = {str(s)
                                     for s in original_v_expr.free_symbols} - ({vectorized_param}.union(
                                         inner_sdfg.free_symbols))
                assert vectorized_param not in non_map_free_syms

                for free_sym in non_map_free_syms:
                    free_sym_str = str(free_sym)
                    assert free_sym_str in inner_sdfg.arrays or free_sym_str in inner_sdfg.symbols

                    if LaneIdScheme.is_laneid(free_sym_str):
                        # Already lane-bound; its lane is fixed by the name. Don't re-encode.
                        continue

                    if free_sym_str in inner_sdfg.symbols:
                        if free_sym_str == vector_map_param:
                            raise AssertionError(
                                f"vector_map_param {vector_map_param!r} appeared in non_map_free_syms; "
                                f"upstream filtering is broken")
                        lane_sym = LaneIdScheme.make(free_sym_str, i)
                        v_expr = v_expr.subs(free_sym, lane_sym)
                        if lane_sym not in inner_sdfg.symbols:
                            inner_sdfg.add_symbol(lane_sym, inner_sdfg.symbols.get(free_sym_str, dace.float64))
                    else:
                        if isinstance(inner_sdfg.arrays[free_sym_str], dace.data.Scalar):
                            v_expr = v_expr.subs(free_sym, f"{free_sym}")
                        else:
                            assert inner_sdfg.arrays[free_sym_str].shape != (1, )
                            v_expr = v_expr.subs(free_sym, f"{free_sym}({i})")

                # ``DaceSympyPrinter`` prints array reads as ``arr[idx]``
                # (subscript form for names in the ``arrays`` set) and emits
                # ``(a and b)`` / ``(a or b)`` / ``(not a)`` directly for
                # ``sympy.Or``/``And``/``Not``, so the previous two-step
                # ``sympy.pycode`` + ``rewrite_boolean_functions_to_boolean_ops``
                # + ``convert_nonstandard_calls`` chain collapses to one print.
                printer = DaceSympyPrinter(set(inner_sdfg.arrays.keys()))
                new_v = printer.doprint(v_expr)
                new_assignments[new_k] = new_v

                if i == 0:
                    # Keep the original un-suffixed symbol bound to the lane-0 expansion so
                    # downstream consumers that haven't been retargeted yet still see it.
                    new_assignments[k] = new_v

        edge.data.assignments = new_assignments


def try_demoting_vectorizable_symbols(inner_sdfg: dace.SDFG) -> Set[str]:
    assigned_symbols = dict()
    for edge in inner_sdfg.all_interstate_edges():
        for k, v in edge.data.assignments.items():
            if k not in assigned_symbols:
                assigned_symbols[k] = set()
            assigned_symbols[k].add(v)

    demotable_symbols = set()
    for sym, sym_assignments in assigned_symbols.items():
        # Check that the access is to arrays and map param is involved
        all_function_args = set()
        #print(sym_assignments)
        for sym_assignment in sym_assignments:
            sym_assign_expr = dace.symbolic.SymExpr(sym_assignment)
            # Collect all array accesses (they are functions that are present in the sdfg)
            # Also try to support And and Or if this happens
            from sympy.logic.boolalg import And, Or
            atoms = (sym_assign_expr.atoms(sympy.Function) | sym_assign_expr.atoms(And) | sym_assign_expr.atoms(Or))
            funcs = {(getattr(a, "func", type(a)).__name__, a)
                     for a in atoms if hasattr(a, "func") and callable(a.func)}
            #print(funcs)
            for fname, f in funcs:
                #print(f"Check function: {fname} ({str(fname) in inner_sdfg.arrays})")
                if fname in inner_sdfg.arrays:
                    for arg in f.args:
                        all_function_args = all_function_args.union({str(s) for s in arg.free_symbols})

        # If all function args are s
        #print(f"{sym} <-(depends)- {all_function_args}")
        # if the depend set has no arrays or scalars we can do it
        data_in_dependence_set = {d for d in all_function_args if d in inner_sdfg.arrays}
        if len(data_in_dependence_set) == 0:
            demotable_symbols.add(sym)

    # Symbols used on memlets can't be demoted
    access_syms = set()
    for state in inner_sdfg.all_states():
        for edge in state.edges():
            if edge.data.subset is not None:
                dst = edge.dst
                available_syms = state.symbols_defined_at(dst)
                syms_used = {
                    str(s)
                    for s in edge.data.free_symbols if str(s) in inner_sdfg.symbols or str(s) in available_syms
                }
                access_syms = access_syms.union(syms_used)

    demotable_symbols = demotable_symbols - access_syms

    for demotable_symbol in demotable_symbols:
        stype = inner_sdfg.symbols[demotable_symbol]
        sdutil.demote_symbol_to_scalar(inner_sdfg, demotable_symbol, stype)

    return demotable_symbols


def resolve_missing_laneid_symbols(inner_sdfg, nsdfg, state, vector_map_param):
    """
    Resolve missing expanded loop symbols of the form ``loop_var_laneid_ID`` inside
    an SDFG nested in a vectorized map.

    During vectorized code generation, additional symbol variants such as
    ``i_laneid_0``, ``i_laneid_1`` may appear, but these are often not present in
    ``nsdfg.symbol_mapping``. This function reconstructs such missing symbols and
    inserts appropriate symbol assignments before the start block of the inner SDFG.

    Parameters
    ----------
    inner_sdfg : dace.SDFG
        The inner SDFG in which missing free symbols appear.

    nsdfg : dace.nodes.NestedSDFG
        The nested SDFG node that contains the symbol mapping to the outer SDFG.

    state : dace.SDFGState
        The state containing the NestedSDFG node. Used to look up parent map symbols.

    vector_map_param : str
        The name of the map iterator corresponding to the vector lane dimension.
        Symbols derived from this parameter will be rewritten as `vector_map_param + laneid`.

    Notes
    -----
    - Missing symbols must contain ``"_laneid_"``. Symbols not matching this pattern
      trigger an assertion.
    - Symbols belonging to the parent map (returned by
      ``assert_symbols_in_parent_map_symbols``) are *not* rewritten.
    - All rewritten symbols are assigned immediately before the start block of
      ``inner_sdfg`` via ``add_state_before``.

    Raises
    ------
    AssertionError
        If unexpected missing symbols remain after processing, or if symbols do not
        conform to the expected ``*_laneid_*`` pattern.

    """
    # Find missing symbols
    missing_symbols = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))
    #print(missing_symbols)

    # Determine which of the missing symbols correspond to parent map symbols
    map_symbols = assert_symbols_in_parent_map_symbols(missing_symbols, state, nsdfg)

    # Any symbol not in map_symbols must be auto-constructed
    unresolved = missing_symbols - map_symbols
    if len(unresolved) != 0:
        assignments = {}

        for missing_sym in unresolved:
            parsed = LaneIdScheme.parse(missing_sym)
            if parsed is None:
                raise NotImplementedError(f"Unexpected free symbol {missing_sym!r} without `_laneid_<i>` suffix; "
                                          f"cannot auto-construct")
            base, laneid = parsed

            if base == vector_map_param:
                # vector iterator -> add lane offset
                assignments[missing_sym] = f"{base} + {laneid}"
            else:
                # other iterators -> simply alias
                assignments[missing_sym] = base

        # Insert assignment state before the start block
        inner_sdfg.add_state_before(
            inner_sdfg.start_block,
            "pre_missing_assignment",
            is_start_state=True,
            assignments=assignments,
        )

    # Ensure no missing symbols remain
    remaining = set(inner_sdfg.free_symbols - set(nsdfg.symbol_mapping.keys()))
    assert len(remaining) == 0, \
        f"Remaining missing symbols after fix: {remaining}"

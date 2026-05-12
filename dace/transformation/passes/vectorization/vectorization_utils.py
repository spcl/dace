# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import sympy
import dace
from typing import Dict, Iterable, Optional, Set, Tuple, Union
from dace import SDFGState, typeclass
from dace import Any
from dace import List
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.graph import Edge
from dace.sdfg.state import BreakBlock, ConditionalBlock, LoopRegion
import dace.sdfg.tasklet_utils as tutil
import dace.sdfg.construction_utils as cutil
import dace.sdfg.utils as sdutil
from dace.symbolic import DaceSympyPrinter


class LaneIdScheme:
    """Centralised lane-id naming for the vectorization passes.

    The vectorization pipeline expands a single symbol used inside a vector tile into
    one symbol per lane, named ``<base>_laneid_<i>``. This class is the single owner
    of that scheme — every place in the codebase that constructs or inspects such a
    name must go through ``LaneIdScheme.make`` / ``LaneIdScheme.parse`` /
    ``LaneIdScheme.is_laneid`` instead of raw string concatenation or regex.

    Centralising the scheme is what makes the lane-expansion passes idempotent: a
    symbol that already encodes its lane in its name (parses non-trivially) is
    treated as fixed, never re-expanded into ``<base>_laneid_<i>_laneid_<j>``.
    """

    SUFFIX = "_laneid_"
    _PARSE_RE = re.compile(r"^(.*)_laneid_(\d+)$")

    @staticmethod
    def make(base: str, lane: int) -> str:
        """Build the lane-encoded name ``<base>_laneid_<lane>``."""
        return f"{base}{LaneIdScheme.SUFFIX}{lane}"

    @staticmethod
    def parse(name: str) -> Optional[Tuple[str, int]]:
        """Return ``(base, lane)`` if ``name`` ends with ``_laneid_<digits>``, else ``None``.

        For nested forms like ``foo_laneid_3_laneid_0`` the *trailing* lane is peeled
        once: the result is ``("foo_laneid_3", 0)``. Callers that want the original
        un-encoded base must call ``parse`` repeatedly until it returns ``None``.
        """
        m = LaneIdScheme._PARSE_RE.match(name)
        if m is None:
            return None
        return m.group(1), int(m.group(2))

    @staticmethod
    def is_laneid(name: str) -> bool:
        """True iff ``name`` matches the ``<base>_laneid_<digits>`` pattern."""
        return LaneIdScheme.parse(name) is not None


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    """ Convenience wrapper to make the .replace not in-place """
    new_subset = copy.deepcopy(subset)
    new_subset.replace(repl_dict)
    return new_subset


def repl_subset_to_use_laneid_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbol_offset: str,
                                     vector_map_param: str) -> dace.subsets.Range:
    """
    Apply a symbolic offset to all free symbols in a subset.

    This function replaces each free symbol in the subset with a new symbol
    that has the offset appended to its name (e.g., 'i' becomes 'i_{offset}' where offset is an integer).
    New symbols are automatically added to the SDFG if they don't exist.

    If symbol is vector map param always add + 1 instead of laneid

    Args:
        sdfg: The SDFG containing the subset
        subset: The subset whose symbols should be offset
        symbol_offset: String to append to each symbol name (should be an integer)
        add_missing_symbols: If True, adds symbol mappings and assignments for
                           free symbols in the parent SDFG

    Returns:
        A new subset with offset symbols applied

    Example:
        If subset contains symbol 'i' and symbol_offset is '_v':
        - 'i' becomes 'i_v'
        - Symbol 'i_v' is added to SDFG if not present
    """
    # Offset needs to be positive integer
    assert symbol_offset.isdigit()
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols

    repl_dict = {
        str(free_sym):
        str(free_sym) + "_laneid_" + str(symbol_offset) if str(free_sym) != vector_map_param else "(" + str(free_sym) +
        " + " + str(symbol_offset) + ")"
        for free_sym in free_syms
    }

    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        if str(free_sym) != vector_map_param:
            offset_symbol_name = str(free_sym) + "_laneid_" + str(symbol_offset)
            if offset_symbol_name not in sdfg.symbols:
                sdfg.add_symbol(offset_symbol_name, stype)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols - prev_sdfg_free_syms:
            raise Exception(
                "`repl_subset_to_use_laneid_offset` has introduced new free symbols (this will cause problems as the new symbols should not be free). This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue"
            )
    return new_subset


def repl_subset_to_use_with_int_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbols_to_offset: Set[str],
                                       int_offset: int) -> dace.subsets.Range:
    """
    Apply a int offset to all free symbols appearing on `symbols_to_offset` in a subset.

    that has the offset appended to its name (e.g., 'i' becomes 'i + {int_offset}' where offset is an integer).
    No new symbol is added
    """
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols
    new_range_list = []
    repl_dict = {str(free_sym): "(" + str(free_sym) + " + " + str(int_offset) + ")" for free_sym in symbols_to_offset}
    for (b, e, s) in subset:
        if hasattr(b, "subs"):
            nb = b.subs(repl_dict)
        else:
            nb = b
        if hasattr(e, "subs"):
            ne = e.subs(repl_dict)
        else:
            ne = e
        ns = 1
        new_range_list.append((nb, ne, ns))

    new_subset = dace.subsets.Range(new_range_list)

    for free_sym in free_syms:
        if str(free_sym) in sdfg.free_symbols - prev_sdfg_free_syms:
            raise Exception(
                "`repl_subset_to_use_with_int_offset` has introduced new free symbols (this will cause problems as the new symbols should not be free). This will result an invalid SDFG, either call with `add_missing_symbols=True` or fix this issue"
            )

    return new_subset


def replace_memlet_expression(state: SDFGState,
                              edges: Iterable[Edge[Memlet]],
                              old_subset_expr: dace.subsets.Range,
                              new_subset_expr: dace.subsets.Range,
                              repl_scalars_with_arrays: bool,
                              edges_to_skip: Set[Edge[Memlet]],
                              vector_numeric_type: typeclass,
                              dataname: Union[str, None] = None) -> Set[str]:
    """
    Replace memlet subsets matching a pattern with a new subset expression.

    Optionally converts scalar/size-1 arrays to arrays that match the new_subset_expr's sizes
    using the `vector_numeric_type` as dtype to accommodate the new subset dimensions.

    Args:
        state: The SDFG state containing the edges
        edges: Edges whose memlets should be checked and potentially replaced
        old_subset: The subset pattern to match
        new_subset: The replacement subset
        convert_scalars_to_arrays: If True, converts Scalar/size-1 Array nodes
                                  to proper Arrays with shape matching new_subset
        edges_to_skip: Set of edges that should not be modified (validation)
        vector_dtype: Data type to use when converting scalars to arrays
        dataname: if not None checks for memlet data too

    Raises:
        Exception: If an edge marked to skip is encountered during replacement
        because it indicates a bug in the auto-vectorization logic

    Side Effects:
        - Modifies memlet subsets on matching edges
        - May remove and re-add array data descriptors with new shapes
    """
    arr_dim = [((e + 1 - b) // s) for (b, e, s) in new_subset_expr]

    for edge in edges:
        src_node: dace.nodes.Node = edge.src
        dst_node: dace.nodes.Node = edge.dst

        if edge.data is not None and edge.data.subset == old_subset_expr:
            if edge in edges_to_skip:
                raise Exception("AA")
            if edge.data.data != dataname and dataname is not None:
                continue
            if repl_scalars_with_arrays:
                for data_node in [src_node, dst_node]:
                    if isinstance(data_node, dace.nodes.AccessNode):
                        arr = state.sdfg.arrays[data_node.data]
                        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array)
                                                                 and arr.shape == (1, )):
                            state.sdfg.remove_data(data_node.data, validate=False)
                            state.sdfg.add_array(name=data_node.data,
                                                 shape=tuple(arr_dim),
                                                 dtype=vector_numeric_type,
                                                 storage=arr.storage,
                                                 location=arr.location,
                                                 transient=arr.transient,
                                                 lifetime=arr.lifetime,
                                                 find_new_name=False)
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))


def expand_memlet_expression(state: SDFGState, edges: Iterable[Edge[Memlet]], edges_to_skip: Set[Edge[Memlet]],
                             vector_width: int) -> Set[Edge[Memlet]]:
    """
    Expand single-element memlet subsets along stride-1 dimensions to a given vector length.
    Pre-condition: all subset dimensions need to be 1

    For each memlet edge, this function modifies subsets that represent a single element
    and extend them to cover `vector_width` elements when the corresponding array stride is 1.
    Trying to modify an edge listed in `edges_to_skip` raises an error as it indicates a
    bug in the auto-vectorization logic.

    Args:
        state (SDFGState): The SDFG state containing the edges.
        edges (Iterable[Edge[Memlet]]): The memlet edges to inspect and possibly modify.
        edges_to_skip (Set[Edge[Memlet]]): Edges that should not be expanded.
        vector_width (int): The number of elements to expand contiguous subsets to.

    Returns:
        Set[Edge[Memlet]]: The set of edges whose memlets were modified.
    """
    modified_edges = set()
    for edge in edges:
        if edge.data is not None:
            if not all(((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset):
                print(
                    "Edge found where not all memlets subsets are length 1, if only one dimension matches to vector length then it is ok"
                )
                vlens = {((e + 1 - b) // s) == vector_width for b, e, s in edge.data.subset}
                if len(vlens) > 1:
                    raise Exception(
                        f"Memlet subsets for edge {edge}: {[(b, e, s) for b, e, s in edge.data.subset]},"
                        f"is not all length one or max 1 vector width subset: {[((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset]}"
                    )
                else:
                    # Do not do anything
                    continue
            new_subset_list = []
            for (b, e, s), stride in zip(edge.data.subset, state.sdfg.arrays[edge.data.data].strides):
                if stride == 1:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, b + vector_width - 1, s))
                else:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, e, s))
            new_subset_expr = dace.subsets.Range(new_subset_list)

            if new_subset_expr != edge.data.subset:
                edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))
                modified_edges.add(edge)
    return modified_edges


# Map / SDFG boolean predicates and their defensive ``assert_X`` siblings
# live in ``utils.map_predicates`` (split slice S3). Re-exported below so
# wildcard importers and named-import callers keep resolving them
# unchanged. Per the locked policy ("defensive checks and assertions stay"),
# the ``assert_X`` siblings are kept as-is alongside their boolean
# counterparts — they are not deleted, demoted, or rewritten.
from dace.transformation.passes.vectorization.utils.map_predicates import (  # noqa: E402, F401
    assert_last_dim_of_maps_are_contigous_accesses,
    assert_maps_consist_of_single_nsdfg_or_no_nsdfg,
    assert_no_other_subset,
    assert_no_wcr,
    count_param_in_expr,
    get_single_nsdfg_inside_map,
    has_maps,
    has_nsdfg_depth_more_than_one,
    has_only_states,
    has_only_states_or_single_block_with_break_only,
    is_innermost_map,
    last_dim_of_map_is_contiguous_accesses,
    map_consists_of_single_nsdfg_or_no_nsdfg,
    map_has_branching_memlets,
    map_has_nested_sdfgs,
    map_param_appears_in_multiple_dimensions,
    no_other_subset,
    no_other_subset_sdfg,
    no_wcr,
    no_wcr_sdfg,
    sdfg_has_nested_sdfgs,
)




# ``to_ints``, ``collect_non_unit_stride_accesses_in_map``,
# ``collect_accesses_to_array_name``, ``collect_all_memlets_to_dataname``,
# and ``parse_int_or_default`` live in ``utils.queries`` (split slice S1b).
# Re-exported below for backward compatibility — wildcard importers and
# named-import callers keep resolving the symbols from this module.
from dace.transformation.passes.vectorization.utils.queries import (  # noqa: E402, F401
    collect_accesses_to_array_name, collect_all_memlets_to_dataname, collect_non_unit_stride_accesses_in_map,
    parse_int_or_default, to_ints,
)


# ``get_vector_max_access_ranges``, ``find_state_of_nsdfg_node``,
# ``check_nsdfg_connector_array_shapes_match``,
# ``fix_nsdfg_connector_array_shapes_mismatch`` and ``reset_connectors``
# moved to ``utils.nsdfg_reshape`` (split slice S4a). Re-exported below
# for backward compatibility — wildcard importers and named-import callers
# keep resolving the names unchanged.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    check_nsdfg_connector_array_shapes_match,
    find_state_of_nsdfg_node,
    fix_nsdfg_connector_array_shapes_mismatch,
    get_vector_max_access_ranges,
    reset_connectors,
)


# ``prepare_vectorized_array``, ``compute_edge_subset``, ``process_in_edges``,
# ``process_out_edges`` moved to ``utils.nsdfg_reshape`` (split slice S4b).
# Re-exported below.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    compute_edge_subset,
    prepare_vectorized_array,
    process_in_edges,
    process_out_edges,
)

def offset_memlets(sdfg: dace.SDFG, dataname: str, offsets: List[dace.symbolic.SymExpr]):
    from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
    for _state, edge in walk_memlets_of(sdfg, dataname):
        subset = edge.data.subset.offset_new(dace.subsets.Range(offsets), negative=True)
        # If subset is not one dimensional we need to collapse 0 accesses
        collapsed_subset_list = [(b, e, s) for (b, e, s) in subset if (e + 1 - b) // s != 1]
        edge.data.subset = dace.subsets.Range(collapsed_subset_list)


# ``match_connector_to_data`` moved to ``utils.tasklets`` (S6b).

from dace.transformation.passes.vectorization.utils.tasklets import (  # noqa: E402, F401
    duplicate_access,
    insert_assignment_tasklet_from_src,
    insert_assignment_tasklet_to_dst,
    instantiate_tasklet_from_info,
    is_assignment_tasklet,
    is_vector_assign_tasklet,
    match_connector_to_data,
)


# ``assert_strides_are_packed_C_or_packed_Fortran`` lives in ``utils.layout``
# (split slice S1a). Re-exported below for backward compatibility — wildcard
# importers (``vectorize.py``, ``vectorize_break.py``, ``remove_vector_maps.py``)
# and named importers (tests) keep resolving the symbol from this module.
from dace.transformation.passes.vectorization.utils.layout import (  # noqa: E402, F401
    assert_strides_are_packed_C_or_packed_Fortran, )


# ``find_state_of_nsdfg_node`` moved to ``utils.nsdfg_reshape`` (S4a).


# ``check_nsdfg_connector_array_shapes_match`` moved to ``utils.nsdfg_reshape`` (S4a).


# ``fix_nsdfg_connector_array_shapes_mismatch`` moved to ``utils.nsdfg_reshape`` (S4a).


# ``extract_bracket_contents``, ``_DropDimsTransformer``, ``drop_dims_from_str``,
# ``drop_dims``, ``offset_symbol_in_expression`` and
# ``use_laneid_symbol_in_expression`` all live in ``utils.code_rewrite``
# (split slice S1c). Re-exported below for backward compatibility.
# ``STANDARD_FUNCS`` / ``FuncToSubscript`` / ``convert_nonstandard_calls``
# were deleted in S1c-bis — their sole caller now uses ``DaceSympyPrinter``.
from dace.transformation.passes.vectorization.utils.code_rewrite import (  # noqa: E402, F401
    drop_dims, drop_dims_from_str, extract_bracket_contents, offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)


# ``instantiate_tasklet_from_info`` moved to ``utils.tasklets`` (S6b).


# ``duplicate_access`` moved to ``utils.tasklets`` (S6b).


# ``replace_arrays_with_new_shape`` and ``copy_arrays_with_a_new_shape``
# moved to ``utils.arrays`` (S6a). Re-exported below alongside
# ``add_transient_arrays_from_list``.
from dace.transformation.passes.vectorization.utils.arrays import (  # noqa: E402, F401
    add_transient_arrays_from_list,
    copy_arrays_with_a_new_shape,
    replace_arrays_with_new_shape,
)


# Source/sink classification quad (get_{scalar,array}_{source,sink}_nodes) moved to ``utils.source_sink`` (S5).

from dace.transformation.passes.vectorization.utils.source_sink import (  # noqa: E402, F401
    check_writes_to_scalar_sinks_happen_through_assign_tasklets,
    expand_assignment_tasklets,
    get_array_sink_nodes,
    get_array_source_nodes,
    get_scalar_sink_nodes,
    get_scalar_source_nodes,
    input_is_zero_and_transient_accumulator,
    move_out_reduction,
    only_one_flop_after_source,
    reduce_before_use,
)


# ``add_transient_arrays_from_list`` moved to ``utils.arrays`` (S6a).


# ``is_assignment_tasklet`` moved to ``utils.tasklets`` (S6b).


# ``check_writes_to_scalar_sinks_happen_through_assign_tasklets`` moved to ``utils.source_sink`` (S5).


# ``only_one_flop_after_source`` moved to ``utils.source_sink`` (S5).


# ``input_is_zero_and_transient_accumulator`` moved to ``utils.source_sink`` (S5).


def replace_all_access_subsets(state: dace.SDFGState, name: str, new_subset_expr: str):
    """
    Replaces all memlet subsets for a given array in a state with a new subset expression.

    Args:
        state: The SDFG state to modify.
        name: Array name whose accesses are replaced.
        new_subset_expr: The new subset expression (e.g., "0:4").
    """
    for edge in state.edges():
        if edge.data is not None and edge.data.data == name:
            nm = dace.memlet.Memlet(expr=f"{name}[{new_subset_expr}]")
            edge.data = nm


# ``expand_assignment_tasklets`` moved to ``utils.source_sink`` (S5).


# ``reduce_before_use`` moved to ``utils.source_sink`` (S5).


# ``move_out_reduction`` moved to ``utils.source_sink`` (S5).


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
            if not match:
                state.sdfg.save("vectorize_failing.sdfg")
            assert match, f"No match in {strings} for a variable name"
            if match:
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
        if loop_var not in loop_symbols and loop_var not in nsdfg.symbol_mapping:
            state.sdfg.save("failing.sdfg")
        assert loop_var in loop_symbols or loop_var in nsdfg.symbol_mapping, f"{loop_var} not in {loop_symbols}"

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


def collect_vectorizable_arrays(sdfg: dace.SDFG, parent_nsdfg_node: dace.nodes.NestedSDFG, parent_state: SDFGState,
                                invariant_scalars: Set[str]) -> Set[str]:
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
    # Pre condition first parent maps is over the contiguous dimension and right most param if multi-dimensional
    parent_map = parent_state.scope_dict()[parent_nsdfg_node]
    assert isinstance(parent_map, dace.nodes.MapEntry)
    map_param = parent_map.map.params[-1]
    parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)

    all_accesses_to_arrays = collect_accesses_to_array_name(sdfg)
    #print(all_accesses_to_arrays)

    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.data.other_subset is not None:
                raise NotImplementedError("other subset support not implemented")

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
            #print(access_expr, type(access_expr))
            #print(isinstance(access_expr, (dace.symbolic.SymExpr, dace.symbolic.symbol, sympy.Expr)))
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
                        if assignment is None and str(free_sym) not in parent_syms_defined:
                            sdfg.save("failing_vectorization.sdfg")
                        assert not (
                            assignment is None and str(free_sym) not in parent_syms_defined
                        ), f"Could not find an iedge assignment for {free_sym}, assignemnt {assignment}, parent symbols defined {parent_syms_defined}. {sdfg.label}, {sdfg.parent_nsdfg_node}: map param {map_param}"
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
                        #print("Invariant", invariant_scalars)
                        #print("Non-invariant scalars",
                        #      {s
                        #       for s in assignment_expr.free_symbols if str(s) in sdfg.arrays} - invariant_scalars)
                        if len(funcs) != 0 or len(scalars) != 0:
                            #print(f"Indirect access detected: ({funcs}, {scalars}) for {arr_name}, is not vectorizable")
                            array_is_vectorizable[arr_name] = False

            # Go through non unit stride dimensions in case it those dimensions have unstructuredness
            for i, (b, e, s) in enumerate(access_subset):
                #print(i, ",", (b,e,s), "|", access_subset)
                if i == stride_one_dim:
                    continue
                #print(b, type(b),)
                free_syms = set()
                if hasattr(b, "free_syms"):
                    free_syms = {str(s) for s in b.free_syms}
                if hasattr(b, "free_symbols"):
                    free_syms = {str(s) for s in b.free_symbols}

                if free_syms != set():
                    #print(free_syms)
                    for free_sym in free_syms:
                        # Accessing map param is ok
                        #print("FS", free_syms)
                        if str(free_sym) == map_param:
                            continue
                        else:
                            # Other free symbols should not have indirect accesses
                            # Analysis tries find the first assignment in the CFG
                            assignment = find_symbol_assignment(sdfg, str(free_sym))

                            # If assignment is None, it is probably coming from parent map
                            parent_syms_defined = parent_state.symbols_defined_at(parent_nsdfg_node)
                            if assignment is None:
                                sdfg.save("failing_vectorization.sdfg")
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
                            #print(all_atoms_str)

                            # Map parameter appears in inddirect access, array is not vectorizable
                            if map_param in all_atoms_str:
                                array_is_vectorizable[arr_name] = False

    return array_is_vectorizable


# ``collect_non_unit_stride_accesses_in_map`` and ``collect_accesses_to_array_name``
# moved to ``utils.queries`` (split slice S1b). Re-exported at the top of this file.

# ``STANDARD_FUNCS`` / ``FuncToSubscript`` / ``convert_nonstandard_calls``
# were deleted in S1c-bis (replaced by ``DaceSympyPrinter`` at the
# ``expand_interstate_assignments_to_lanes`` callsite).


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


# ``collect_all_memlets_to_dataname`` moved to ``utils.queries`` (S1b).


# ``is_vector_assign_tasklet`` moved to ``utils.tasklets`` (S6b).


# ``insert_assignment_tasklet_from_src`` moved to ``utils.tasklets`` (S6b).


# ``insert_assignment_tasklet_to_dst`` moved to ``utils.tasklets`` (S6b).


# ``add_copies_before_and_after_nsdfg`` and ``find_copy_in_state`` moved
# to ``utils.nsdfg_reshape`` (split slice S4c). Re-exported below.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    add_copies_before_and_after_nsdfg,
    find_copy_in_state,
)


# ``map_has_branching_memlets`` moved to ``utils.map_predicates`` (S3).


# ``parse_int_or_default`` moved to ``utils.queries`` (S1b).


def sift_access_node_up(state: dace.SDFGState, node: dace.nodes.AccessNode, map_entry: dace.nodes.MapEntry):
    # We have MapEntry -> AccessNode -> DstNode
    # We move it up to be: AccessNode -> MapEntry -> DstNode
    # If access node's size is multiplied with the loop's dimensions

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)
    src_nodes = {ie.src for ie in in_edges}
    assert map_entry in src_nodes
    assert len(in_edges) == 1
    assert len(out_edges) == 1

    desc = state.sdfg.arrays[node.data]
    assert len(desc.shape) == len(map_entry.map.params)
    map_lengths = tuple([(e + 1 - b) // s for (b, e, s) in map_entry.map.range])
    # Vector map is one dimensional and has length 1 due to step size
    assert len(map_entry.map.params) == 1
    assert map_lengths[0] == 1

    ie = in_edges[0]
    oe = out_edges[0]
    # Rm access node's connection
    state.remove_edge(ie)
    state.remove_edge(oe)
    state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

    ies_from_connector = state.in_edges_by_connector(map_entry, ie.src_conn.replace("OUT_", "IN_"))
    for s_ie in ies_from_connector:
        state.remove_edge(s_ie)

        # Expand oe.data.subset
        new_subset_list = []
        p, (mb, me, ms) = map_entry.map.params[0], map_entry.map.range[0]
        for (b, e, s) in ie.data.subset:
            nb = b.subs(p, mb)
            ne = e.subs(p, mb)
            ns = s
            new_subset_list.append((nb, ne, ns))
        s_ie_subset = dace.subsets.Range(new_subset_list)

        state.add_edge(s_ie.src, s_ie.src_conn, node, None, dace.memlet.Memlet(data=s_ie.data.data, subset=s_ie_subset))
        state.add_edge(node, None, s_ie.dst, s_ie.dst_conn, copy.deepcopy(oe.data))


# ``sdfg_has_nested_sdfgs``, ``map_has_nested_sdfgs``, and
# ``has_nsdfg_depth_more_than_one`` moved to ``utils.map_predicates`` (S3).


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


def squeeze_memlets_of_packed_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                     array_accesses_to_be_packed: Set[str]):
    all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    all_edges = state.all_edges(*all_nodes)
    for edge in all_edges:
        if edge.data.data in array_accesses_to_be_packed:
            new_range_list = [(b, b, 1) for (b, e, s) in edge.data.subset]
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list))


def use_previous_subsets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, vector_width: int,
                         vectorizable_arrays: Set[str]):
    """
    Rewrite memlet subsets on edges leaving a single-parameter inner map so that
    structured vector accesses correctly refer to the surrounding parent map.

    The function performs:
        1. Extract the inner map parameter (e.g., `i`).
        2. Extract the parent's map lower bound symbol (e.g., `tile_i`).
        3. For each outgoing memlet:
             - Identify its corresponding incoming memlet (IN_x -> OUT_x).
             - Clone its subset.
             - Substitute outer -> inner symbols for begin/end expressions.
             - Adjust end bound when the subset length matches `vector_width` (=structured access).
             - Compute the memlet volume.
             - Assign a new Memlet with the updated subset and volume.

    Parameters
    ----------
    state : dace.SDFGState
        The current SDFG state containing the map entry.
    map_entry : dace.nodes.MapEntry
        The map entry whose outgoing memlet subsets will be rewritten.
        Must have exactly one map parameter.
    vector_width : int
        Width of the structured vector access. If a subset dimension has length
        equal to `vector_width`, we shrink its end bound by `vector_width - 1`.
        As in this case we have an exact subset, otherwise we pass a complete dimension or something in that fay that we cant change.

    Notes
    -----
    We cast symbolic expressions to string and re-sympify them to force SymPy
    to reattach the same symbol objects used by DaCe.
    """

    # Inner map has exactly one parameter, e.g., `i`.
    assert len(map_entry.map.params) == 1
    inner_param = map_entry.map.params[0]

    # Extract parent-map lower bound symbol as string, e.g. `[tile_i : ...]`.
    outer_param = str(map_entry.map.range[0][0])

    for out_edge in state.out_edges(map_entry):
        if out_edge.src_conn is None:
            continue

        # Find the corresponding incoming edge with IN_<idx> for OUT_<idx>.
        in_edges = set(state.in_edges_by_connector(map_entry, out_edge.src_conn.replace("OUT_", "IN_")))
        if not in_edges:
            continue

        # Safe: at most one incoming edge per OUT connector.
        assert len(in_edges) == 1
        in_edge = next(iter(in_edges))

        if in_edge.data.data not in vectorizable_arrays:
            continue

        # Copy original subset.
        orig_subset = copy.deepcopy(in_edge.data.subset)

        new_ranges = []
        volume = 1

        for (begin, end, stride) in orig_subset:
            # Rewrite begin expression
            if hasattr(begin, "subs"):
                begin_str = str(begin.subs({outer_param: inner_param}))
                new_begin = dace.symbolic.SymExpr(begin_str).simplify()
            else:
                new_begin = begin

            # Rewrite end expression
            if hasattr(end, "subs"):
                # Subset extent length
                extent = (end + 1 - begin) // stride
                # If exact vector access, shrink by vector_width - 1
                tail_adjust = vector_width - 1 if extent == vector_width else 0
                end_str = f"{end.subs({outer_param: inner_param})} - {tail_adjust}"
                new_end = dace.symbolic.SymExpr(end_str).simplify()
                volume *= extent
            else:
                new_end = end
            new_ranges.append((new_begin, new_end, stride))

        # Assign new memlet with updated subset and volume
        out_edge.data = dace.memlet.Memlet(
            data=out_edge.data.data,
            subset=dace.subsets.Range(new_ranges),
            volume=volume,
        )


# ``reset_connectors`` moved to ``utils.nsdfg_reshape`` (S4a).


def remove_map(map_entry: dace.nodes.MapEntry, state: dace.SDFGState):
    assert map_entry in state.nodes()
    map_exit = state.exit_node(map_entry)

    # Replace symbol dictionary
    repldict = {str(p): str(r[0]) for p, r in zip(map_entry.map.params, map_entry.map.range)}

    # Redirect map entry's out edges
    write_only_map = True
    for edge in state.out_edges(map_entry):
        if edge.data.is_empty() or edge.data.data is None:
            parent_map_entry = state.entry_node(map_entry)
            if parent_map_entry is not None:
                state.add_edge(parent_map_entry, None, edge.dst, edge.dst_conn, edge.data)
        else:
            # Add an edge directly from the previous source connector to the destination
            path = state.memlet_path(edge)
            index = path.index(edge)
            state.add_edge(path[index - 1].src, path[index - 1].src_conn, edge.dst, edge.dst_conn, edge.data)
            write_only_map = False

    # Redirect map exit's in edges.
    for edge in state.in_edges(map_exit):
        path = state.memlet_path(edge)
        index = path.index(edge)

        # Add an edge directly from the source to the next destination connector
        if len(path) > index + 1:
            state.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)

            if write_only_map:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)
                if outer_entry is not None:
                    if any({e.src == map_entry for e in state.in_edges(edge.src)}):
                        state.add_edge(outer_entry, None, edge.src, None, Memlet(None))
                    else:
                        for src in {e.src for e in state.in_edges(edge.src)}:
                            state.add_edge(outer_entry, None, src, None, Memlet(None))

            else:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)

    state.remove_node(map_entry)
    state.remove_node(map_exit)

    # Replace symbols
    all_nodes = state.all_nodes_between(outer_entry, outer_exit)
    all_edges = state.all_edges(*all_nodes)
    for n in all_nodes:
        if isinstance(n, dace.nodes.Tasklet):
            code_before = copy.deepcopy(n.code.as_string)
            tutil.tasklet_replace_code(n, repldict, py_only=False, use_sym_expr=False)
            #print("Repldict:", repldict, "\nCode Before:", code_before, "\nCode After:", n.code.as_string)
        if isinstance(n, dace.nodes.NestedSDFG):
            for k, v in repldict.items():
                if k in n.symbol_mapping:
                    sym_expr = dace.symbolic.SymExpr(n.symbol_mapping[k])
                    if k in {str(s) for s in sym_expr.free_symbols}:
                        printer = DaceSympyPrinter(arrays=state.sdfg.arrays)
                        n.symbol_mapping[v] = printer.doprint(sym_expr.subs(k, v))
                    else:
                        n.symbol_mapping[v] = n.symbol_mapping[k]
                    del n.symbol_mapping[k]
            n.sdfg.replace_dict(repldict)
            for k, v in repldict.items():
                assert k not in n.sdfg.symbols
                assert k not in n.sdfg.free_symbols
            # SDFG repldict does not change edge subsets
            for _is in n.sdfg.all_states():
                for _se in _is.edges():
                    if _se.data.data is not None:
                        _se.data.subset.replace(repldict)
    for e in all_edges:
        if e.data.data is None:
            continue
        e.data.subset.replace(repldict)


def try_clean_other_subset_going_out_from_map_entry(state: SDFGState, map_entry: dace.nodes.MapEntry):
    id = 0
    #state.sdfg.save("x.sdfg")
    for oe in state.out_edges(map_entry):
        #print(oe.data, oe.data.other_subset, oe.dst, type(oe.dst))
        if oe.data.other_subset is not None and isinstance(oe.dst, dace.nodes.AccessNode):
            assert oe.data.data is not None and oe.data.data != oe.dst.data
            # Add assignment tasklet
            t = state.add_tasklet(f"other_subset_assign_{id}", {"_in"}, {"_out"}, "_out = _in")
            state.remove_edge(oe)
            state.add_edge(oe.src, oe.src_conn, t, "_in", dace.memlet.Memlet(data=oe.data.data, subset=oe.data.subset))
            state.add_edge(t, "_out", oe.dst, oe.dst_conn,
                           dace.memlet.Memlet(data=oe.dst.data, subset=oe.data.other_subset))
            id += 1


def detect_halve_index(state: SDFGState, new_inner_map: dace.nodes.MapEntry, vector_length):
    all_nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
    map_param = new_inner_map.map.params[-1]
    all_edges = state.out_edges(new_inner_map)
    modified_nodes = set()
    modified_edges = set()
    for edge in all_edges:
        if edge.data.subset is not None:
            detected_param = None
            detected_divisor = None
            for b, e, s in edge.data.subset:
                param, divisor = detect_halve_index_impl(b)
                if param is not None and divisor is not None:
                    if detected_param is not None:
                        raise NotImplementedError(f"Multiple halve-indexed dimensions on memlet {edge.data}; "
                                                  f"only one supported (state {state.label}, edge {edge})")
                    detected_param = param
                    detected_divisor = divisor
            if detected_param is not None:
                # Multiply end expression with
                desc = state.sdfg.arrays[edge.data.data]
                arr_name, arr = state.sdfg.add_array(name=f"multiplexed_{edge.data.data}",
                                                     shape=(vector_length, ),
                                                     dtype=desc.dtype,
                                                     transient=True,
                                                     storage=dace.dtypes.StorageType.Register,
                                                     find_new_name=True)
                if vector_length % detected_divisor != 0:
                    raise NotImplementedError(f"vector_length={vector_length} not divisible by halve-index divisor "
                                              f"{detected_divisor} on memlet {edge.data}")
                t = state.add_tasklet(
                    "pack_tasklet", {"_in"}, {"_out"},
                    f"multiplex_elements(_in, _out, {vector_length // detected_divisor}, {detected_divisor});",
                    language=dace.dtypes.Language.CPP,
                    code_global=f'#include "dace/vector_intrinsics/multiplex.h"')
                modified_nodes.add(t)
                state.remove_edge(edge)
                new_range_list = list()
                # Detection means we should have b -> b+vector_length step size 1 on the param dim
                for (b, e, s) in edge.data.subset:
                    nb = b
                    if not hasattr(nb, "subs"):
                        raise NotImplementedError(f"detect_halve_index expected symbolic begin, got {type(nb)}: {nb}")
                    ne = nb.subs(detected_param, f"({detected_param}+{vector_length})")
                    ns = 1
                    new_range_list.append((nb, ne, ns))
                e1 = state.add_edge(edge.src, edge.src_conn, t, "_in",
                                    dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list)))
                access = state.add_access(arr_name)
                modified_nodes.add(access)
                modified_edges.add(e1)
                modified_edges.add(edge)
                e2 = state.add_edge(t, "_out", access, None,
                                    dace.memlet.Memlet.from_array(dataname=arr_name, datadesc=arr))
                e3 = state.add_edge(access, None, edge.dst, edge.dst_conn,
                                    dace.memlet.Memlet.from_array(dataname=arr_name, datadesc=arr))
                modified_edges.add(e2)
                modified_edges.add(e3)
    return modified_nodes, modified_edges


def detect_halve_index_impl(expr):
    """
    Detect patterns like int_floor(i, k) or floor_int(i, k)
    where k is ANY positive integer.

    Returns:
        (symbol, divisor) or (None, None)
    """
    # Only custom functions
    if isinstance(expr, sympy.Function) and expr.func.__name__ in ("int_floor", "floor_int"):
        if len(expr.args) != 2:
            return None, None

        i, den = expr.args

        # Divisor must be a positive integer
        if isinstance(i, sympy.Symbol) and isinstance(den, (int, sympy.Integer)) and den > 0:
            return i, int(den)

    return None, None

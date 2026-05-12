# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re
import sympy
import dace
from typing import Dict, Iterable, Optional, Set, Tuple, Union
from dace import SDFGState, typeclass
from dace import List
from dace.memlet import Memlet
from dace.sdfg.graph import Edge
import dace.sdfg.tasklet_utils as tutil
from dace.symbolic import DaceSympyPrinter


# ``LaneIdScheme`` moved to ``utils.name_schemes`` (S6d-a). Re-exported
# below so callers that did ``from …vectorization_utils import LaneIdScheme``
# keep working until S7 migrates every consumer to named imports.
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme  # noqa: E402, F401


# ``repl_subset``, ``repl_subset_to_use_laneid_offset``,
# ``repl_subset_to_use_with_int_offset``, ``replace_memlet_expression``,
# and ``expand_memlet_expression`` moved to ``utils.subsets`` (S6d-b).
# Re-exported alongside the rest of the subset/memlet rewrite family
# at the bottom of this file.


# Map / SDFG boolean predicates and their defensive ``assert_X`` siblings
# live in ``utils.map_predicates`` (split slice S3). Re-exported below so
# wildcard importers and named-import callers keep resolving them
# unchanged. Per the locked policy ("defensive checks and assertions stay"),
# the ``assert_X`` siblings are kept as-is alongside their boolean
# counterparts — they are not deleted, demoted, or rewritten.
from dace.transformation.passes.vectorization.utils.map_predicates import (  # noqa: E402, F401
    assert_last_dim_of_maps_are_contigous_accesses, assert_maps_consist_of_single_nsdfg_or_no_nsdfg,
    assert_no_other_subset, assert_no_wcr, count_param_in_expr, get_single_nsdfg_inside_map, has_maps,
    has_nsdfg_depth_more_than_one, has_only_states, has_only_states_or_single_block_with_break_only, is_innermost_map,
    last_dim_of_map_is_contiguous_accesses, map_consists_of_single_nsdfg_or_no_nsdfg, map_has_branching_memlets,
    map_has_nested_sdfgs, map_param_appears_in_multiple_dimensions, no_other_subset, no_other_subset_sdfg, no_wcr,
    no_wcr_sdfg, sdfg_has_nested_sdfgs,
)

# ``to_ints``, ``collect_non_unit_stride_accesses_in_map``,
# ``collect_accesses_to_array_name``, ``collect_all_memlets_to_dataname``,
# and ``parse_int_or_default`` live in ``utils.queries`` (split slice S1b).
# Re-exported below for backward compatibility — wildcard importers and
# named-import callers keep resolving the symbols from this module.
from dace.transformation.passes.vectorization.utils.queries import (  # noqa: E402, F401
    collect_accesses_to_array_name, collect_all_memlets_to_dataname, collect_non_unit_stride_accesses_in_map,
    collect_vectorizable_arrays, parse_int_or_default, to_ints,
)

# ``get_vector_max_access_ranges``, ``find_state_of_nsdfg_node``,
# ``check_nsdfg_connector_array_shapes_match``,
# ``fix_nsdfg_connector_array_shapes_mismatch`` and ``reset_connectors``
# moved to ``utils.nsdfg_reshape`` (split slice S4a). Re-exported below
# for backward compatibility — wildcard importers and named-import callers
# keep resolving the names unchanged.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    check_nsdfg_connector_array_shapes_match, find_state_of_nsdfg_node, fix_nsdfg_connector_array_shapes_mismatch,
    get_vector_max_access_ranges, reset_connectors,
)

# ``prepare_vectorized_array``, ``compute_edge_subset``, ``process_in_edges``,
# ``process_out_edges`` moved to ``utils.nsdfg_reshape`` (split slice S4b).
# Re-exported below.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    compute_edge_subset, prepare_vectorized_array, process_in_edges, process_out_edges,
)


# ``offset_memlets`` moved to ``utils.subsets`` (S6d-b).


# ``match_connector_to_data`` moved to ``utils.tasklets`` (S6b).

from dace.transformation.passes.vectorization.utils.tasklets import (  # noqa: E402, F401
    duplicate_access, insert_assignment_tasklet_from_src, insert_assignment_tasklet_to_dst,
    instantiate_tasklet_from_info, is_assignment_tasklet, is_vector_assign_tasklet, match_connector_to_data,
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
    add_transient_arrays_from_list, copy_arrays_with_a_new_shape, replace_arrays_with_new_shape,
)

# Source/sink classification quad (get_{scalar,array}_{source,sink}_nodes) moved to ``utils.source_sink`` (S5).

from dace.transformation.passes.vectorization.utils.source_sink import (  # noqa: E402, F401
    check_writes_to_scalar_sinks_happen_through_assign_tasklets, expand_assignment_tasklets, get_array_sink_nodes,
    get_array_source_nodes, get_scalar_sink_nodes, get_scalar_source_nodes, input_is_zero_and_transient_accumulator,
    move_out_reduction, only_one_flop_after_source, reduce_before_use,
)

# Lane-fan-out family moved to ``utils.lane_expansion`` (S6c). Re-exported
# below so callers that import from this module (and the rest of the
# legacy file, which still uses ``find_symbol_assignment`` and ``_all_atoms``
# inside ``collect_vectorizable_arrays``) keep resolving the symbols.
from dace.transformation.passes.vectorization.utils.lane_expansion import (  # noqa: E402, F401
    _all_atoms, assert_symbols_in_parent_map_symbols, expand_interstate_assignments_to_lanes, find_symbol_assignment,
    resolve_missing_laneid_symbols, try_demoting_vectorizable_symbols,
)


# Subset / memlet rewrite family moved to ``utils.subsets`` (S6d-b).
# Re-exported below so wildcard importers and named-import callers keep
# resolving the symbols unchanged.
from dace.transformation.passes.vectorization.utils.subsets import (  # noqa: E402, F401
    expand_memlet_expression,
    offset_memlets,
    repl_subset,
    repl_subset_to_use_laneid_offset,
    repl_subset_to_use_with_int_offset,
    replace_all_access_subsets,
    replace_memlet_expression,
    squeeze_memlets_of_packed_arrays,
    try_clean_other_subset_going_out_from_map_entry,
    use_previous_subsets,
)

# ``add_transient_arrays_from_list`` moved to ``utils.arrays`` (S6a).

# ``is_assignment_tasklet`` moved to ``utils.tasklets`` (S6b).

# ``check_writes_to_scalar_sinks_happen_through_assign_tasklets`` moved to ``utils.source_sink`` (S5).

# ``only_one_flop_after_source`` moved to ``utils.source_sink`` (S5).

# ``input_is_zero_and_transient_accumulator`` moved to ``utils.source_sink`` (S5).


# ``replace_all_access_subsets`` moved to ``utils.subsets`` (S6d-b).


# ``expand_assignment_tasklets`` moved to ``utils.source_sink`` (S5).

# ``reduce_before_use`` moved to ``utils.source_sink`` (S5).

# ``move_out_reduction`` moved to ``utils.source_sink`` (S5).

# ``assert_symbols_in_parent_map_symbols``, ``find_symbol_assignment``,
# and ``_all_atoms`` moved to ``utils.lane_expansion`` (S6c). Re-exported
# below alongside the rest of the lane-fan-out family.


# ``collect_vectorizable_arrays`` moved to ``utils.queries`` (S6d-c).
# Re-exported below alongside the rest of the queries family.


# ``collect_non_unit_stride_accesses_in_map`` and ``collect_accesses_to_array_name``
# moved to ``utils.queries`` (split slice S1b). Re-exported at the top of this file.

# ``STANDARD_FUNCS`` / ``FuncToSubscript`` / ``convert_nonstandard_calls``
# were deleted in S1c-bis (replaced by ``DaceSympyPrinter`` at the
# ``expand_interstate_assignments_to_lanes`` callsite).

# ``expand_interstate_assignments_to_lanes`` and ``try_demoting_vectorizable_symbols``
# moved to ``utils.lane_expansion`` (S6c). Re-exported below.

# ``collect_all_memlets_to_dataname`` moved to ``utils.queries`` (S1b).

# ``is_vector_assign_tasklet`` moved to ``utils.tasklets`` (S6b).

# ``insert_assignment_tasklet_from_src`` moved to ``utils.tasklets`` (S6b).

# ``insert_assignment_tasklet_to_dst`` moved to ``utils.tasklets`` (S6b).

# ``add_copies_before_and_after_nsdfg`` and ``find_copy_in_state`` moved
# to ``utils.nsdfg_reshape`` (split slice S4c). Re-exported below.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    add_copies_before_and_after_nsdfg, find_copy_in_state,
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

# ``resolve_missing_laneid_symbols`` moved to ``utils.lane_expansion`` (S6c).


# ``squeeze_memlets_of_packed_arrays`` moved to ``utils.subsets`` (S6d-b).


# ``use_previous_subsets`` moved to ``utils.subsets`` (S6d-b).


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


# ``try_clean_other_subset_going_out_from_map_entry`` moved to ``utils.subsets`` (S6d-b).


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

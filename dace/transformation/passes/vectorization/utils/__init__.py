# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vectorization helper utilities: layout/query inspectors, AST and memlet
rewriters, NSDFG reshaping, tasklet and array mutators, and naming schemes."""
from .name_schemes import LaneIdScheme, PackedNameScheme, VecNameScheme  # noqa: F401
from .layout import assert_strides_are_packed_C_or_packed_Fortran  # noqa: F401
from .queries import (  # noqa: F401
    collect_accesses_to_array_name, collect_all_memlets_to_dataname, collect_non_unit_stride_accesses_in_map,
    collect_vectorizable_arrays, parse_int_or_default, to_ints,
)
from .code_rewrite import (  # noqa: F401
    drop_dims, drop_dims_from_str, extract_bracket_contents, offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)
from .iteration import walk_memlets_of  # noqa: F401
from .nsdfg_reshape import (  # noqa: F401
    add_copies_before_and_after_nsdfg, check_nsdfg_connector_array_shapes_match, compute_edge_subset,
    find_copy_in_state, find_state_containing_node, fix_nsdfg_connector_array_shapes_mismatch,
    get_vector_max_access_ranges, prepare_vectorized_array, process_in_edges, process_out_edges, reset_connectors,
    sift_access_node_up,
)
from .tasklets import (  # noqa: F401
    duplicate_access, insert_assignment_tasklet_from_src, insert_assignment_tasklet_to_dst,
    instantiate_tasklet_from_info, is_assignment_tasklet, is_vector_assign_tasklet, match_connector_to_data,
)
from .arrays import (  # noqa: F401
    add_transient_arrays_from_list, copy_arrays_with_a_new_shape, replace_arrays_with_new_shape,
)
from .source_sink import (  # noqa: F401
    check_writes_to_scalar_sinks_happen_through_assign_tasklets, expand_assignment_tasklets, get_array_sink_nodes,
    get_array_source_nodes, get_scalar_sink_nodes, get_scalar_source_nodes, input_is_zero_and_transient_accumulator,
    move_out_reduction, only_one_flop_after_source, reduce_before_use,
)
from .map_ops import remove_map  # noqa: F401
from .multiplex import detect_halve_index, detect_halve_index_impl  # noqa: F401
from .subsets import (  # noqa: F401
    expand_memlet_expression, offset_memlets, repl_subset, repl_subset_to_use_laneid_offset,
    repl_subset_to_use_with_int_offset, replace_all_access_subsets, replace_memlet_expression,
    squeeze_memlets_of_packed_arrays, try_clean_other_subset_going_out_from_map_entry, use_previous_subsets,
)
from .lane_expansion import (  # noqa: F401
    _all_atoms, assert_symbols_in_parent_map_symbols, expand_interstate_assignments_to_lanes, find_symbol_assignment,
    resolve_missing_laneid_symbols, try_demoting_vectorizable_symbols,
)
from .map_predicates import (  # noqa: F401
    assert_last_dim_of_maps_are_contigous_accesses, assert_maps_consist_of_single_nsdfg_or_no_nsdfg,
    assert_no_other_subset, assert_no_wcr, count_param_in_expr, get_single_nsdfg_inside_map, has_maps,
    has_nsdfg_depth_more_than_one, has_only_states, has_only_states_or_single_block_with_break_only, is_innermost_map,
    last_dim_of_map_is_contiguous_accesses, map_consists_of_single_nsdfg_or_no_nsdfg, map_has_branching_memlets,
    map_has_nested_sdfgs, map_param_appears_in_multiple_dimensions, map_param_dim_usage_is_linear_combo,
    no_other_subset, no_other_subset_sdfg, no_wcr, no_wcr_sdfg, sdfg_has_nested_sdfgs,
)

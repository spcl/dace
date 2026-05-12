# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vectorization helper utilities.

This package is the destination of the planned split of the legacy
``vectorization_utils.py`` junk-drawer module. Modules are organised by
concern and added one slice at a time. The legacy
``vectorization_utils.py`` keeps re-exporting from here during the
migration; once every consumer is migrated the legacy file is deleted
(see plan slice S7).
"""
from .layout import assert_strides_are_packed_C_or_packed_Fortran  # noqa: F401
from .queries import (  # noqa: F401
    collect_accesses_to_array_name, collect_all_memlets_to_dataname, collect_non_unit_stride_accesses_in_map,
    parse_int_or_default, to_ints,
)
from .code_rewrite import (  # noqa: F401
    drop_dims, drop_dims_from_str, extract_bracket_contents, offset_symbol_in_expression,
    use_laneid_symbol_in_expression,
)
from .iteration import walk_memlets_of  # noqa: F401
from .nsdfg_reshape import (  # noqa: F401
    add_copies_before_and_after_nsdfg,
    check_nsdfg_connector_array_shapes_match,
    compute_edge_subset,
    find_copy_in_state,
    find_state_of_nsdfg_node,
    fix_nsdfg_connector_array_shapes_mismatch,
    get_vector_max_access_ranges,
    prepare_vectorized_array,
    process_in_edges,
    process_out_edges,
    reset_connectors,
)
from .arrays import (  # noqa: F401
    add_transient_arrays_from_list,
    copy_arrays_with_a_new_shape,
    replace_arrays_with_new_shape,
)
from .source_sink import (  # noqa: F401
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
from .map_predicates import (  # noqa: F401
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

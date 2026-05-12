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
# to ``utils.nsdfg_reshape`` (split slice S4c). ``sift_access_node_up``
# moved to the same module in S6d-d. Re-exported below.
from dace.transformation.passes.vectorization.utils.nsdfg_reshape import (  # noqa: E402, F401
    add_copies_before_and_after_nsdfg, find_copy_in_state, sift_access_node_up,
)

# ``map_has_branching_memlets`` moved to ``utils.map_predicates`` (S3).

# ``parse_int_or_default`` moved to ``utils.queries`` (S1b).


# ``sift_access_node_up`` moved to ``utils.nsdfg_reshape`` (S6d-d).


# ``sdfg_has_nested_sdfgs``, ``map_has_nested_sdfgs``, and
# ``has_nsdfg_depth_more_than_one`` moved to ``utils.map_predicates`` (S3).

# ``resolve_missing_laneid_symbols`` moved to ``utils.lane_expansion`` (S6c).


# ``squeeze_memlets_of_packed_arrays`` moved to ``utils.subsets`` (S6d-b).


# ``use_previous_subsets`` moved to ``utils.subsets`` (S6d-b).


# ``reset_connectors`` moved to ``utils.nsdfg_reshape`` (S4a).


# ``remove_map`` moved to ``utils.map_ops`` (S6d-e). Re-exported at the
# bottom of this file. The plan originally proposed promoting it to
# ``dace.sdfg.utils`` because the body has no vectorization-specific
# logic, but the actual call set is narrow (``RemoveVectorMaps`` only)
# and keeping the helper inside the vectorization package matches the
# user's reuse-threshold directive.


# ``try_clean_other_subset_going_out_from_map_entry`` moved to ``utils.subsets`` (S6d-b).


# ``detect_halve_index`` and ``detect_halve_index_impl`` moved to
# ``utils.multiplex`` (S6d-f). Re-exported at the bottom of this file.


from dace.transformation.passes.vectorization.utils.map_ops import remove_map  # noqa: E402, F401

from dace.transformation.passes.vectorization.utils.multiplex import (  # noqa: E402, F401
    detect_halve_index,
    detect_halve_index_impl,
)

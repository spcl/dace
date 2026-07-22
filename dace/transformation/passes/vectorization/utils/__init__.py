# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vectorization helper utilities: query inspectors, AST and memlet rewriters,
array mutators, naming schemes, and map predicates."""
from .name_schemes import LaneIdScheme  # noqa: F401
from .code_rewrite import (  # noqa: F401
    offset_symbol_in_expression, use_laneid_symbol_in_expression,
)
from .iteration import walk_memlets_of  # noqa: F401
from .arrays import (  # noqa: F401
    add_transient_arrays_from_list, replace_arrays_with_new_shape,
)
from .subsets import (  # noqa: F401
    repl_subset, repl_subset_to_use_laneid_offset, replace_all_access_subsets,
)
from .map_predicates import (  # noqa: F401
    count_param_in_expr, get_single_nsdfg_inside_map, has_maps, has_nsdfg_depth_more_than_one, has_only_states,
    has_only_states_or_single_block_with_break_only, is_innermost_map, last_dim_of_map_is_contiguous_accesses,
    map_consists_of_single_nsdfg_or_no_nsdfg, map_has_branching_memlets, map_param_appears_in_multiple_dimensions,
    map_param_dim_usage_is_linear_combo, no_other_subset, no_wcr, sdfg_has_nested_sdfgs,
)

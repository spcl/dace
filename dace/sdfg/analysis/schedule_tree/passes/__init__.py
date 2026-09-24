# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Passes for schedule trees.

Each module holds one family of passes; all passes are importable from this package.
"""
from dace.sdfg.analysis.schedule_tree.passes.folding import fold_guards
from dace.sdfg.analysis.schedule_tree.passes.canonicalization import forward_substitute_conditions
from dace.sdfg.analysis.schedule_tree.passes.canonicalization import remove_dead_assignments
from dace.sdfg.analysis.schedule_tree.passes.canonicalization import pair_complementary_guards
from dace.sdfg.analysis.schedule_tree.passes.splitting import split_iteration_spaces
from dace.sdfg.analysis.schedule_tree.passes.splitting import merge_contiguous_loops
from dace.sdfg.analysis.schedule_tree.passes.unswitching import unswitch_invariant_guards
from dace.sdfg.analysis.schedule_tree.passes.if_conversion import convert_diamonds_to_selects
from dace.sdfg.analysis.schedule_tree.passes.if_conversion import hoist_select_arms
from dace.sdfg.analysis.schedule_tree.passes.rerolling import reroll_statements
from dace.sdfg.analysis.schedule_tree.passes.rerolling import fuse_rolled_loops
from dace.sdfg.analysis.schedule_tree.passes.dead_code import remove_unused_and_duplicate_labels
from dace.sdfg.analysis.schedule_tree.passes.dead_code import remove_empty_scopes
from dace.sdfg.analysis.schedule_tree.passes.dead_code import remove_dead_stores
from dace.sdfg.analysis.schedule_tree.passes.flattening import flatten_contiguous_nests
from dace.sdfg.analysis.schedule_tree.passes.loop_range_reduction import reduce_loop_ranges

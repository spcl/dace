# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop range reduction: the sequence of passes that shrinks loops by their conditions."""

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.canonicalization import (forward_substitute_conditions,
                                                                      pair_complementary_guards,
                                                                      remove_dead_assignments)
from dace.sdfg.analysis.schedule_tree.passes.common import (AccessIndex, prune_empty)
from dace.sdfg.analysis.schedule_tree.passes.dead_code import (remove_dead_stores)
from dace.sdfg.analysis.schedule_tree.passes.folding import (fold_guards)
from dace.sdfg.analysis.schedule_tree.passes.splitting import (merge_contiguous_loops, split_iteration_spaces)
from dace.sdfg.analysis.schedule_tree.passes.unswitching import (unswitch_invariant_guards)


def reduce_loop_ranges(stree: tn.ScheduleTreeScope,
                       max_ranges: int = 32,
                       max_enumeration: int = 1 << 20,
                       min_trip_count: int = 1) -> int:
    """
    Split and shrink loops and maps according to the conditions in their bodies, and move conditions that do not
    change within a loop out of it.

    ``for i in range(N): if 1 <= i < M: A`` becomes ``for i in range(1, min(N, M)): A``; with ``cst = [0, 0, 0, 1, 1,
    0, 0, 2]`` a compile-time constant, ``for k in range(8): S; if cst[k] > 0: A`` becomes ``for k in range(3): S``,
    ``for k in range(3, 5): S; A``, ``for k in range(5, 7): S``, ``for k in range(7, 8): S; A``; and ``for i: if
    cst[k] == 0: A else: B`` becomes ``if cst[k] == 0: for i: A else: for i: B``.

    This runs, in order, the passes that each do one part of it and can be checked on their own:
    :func:`pair_complementary_guards`, :func:`forward_substitute_conditions` and :func:`remove_dead_assignments`
    (canonicalization), :func:`fold_guards`, :func:`unswitch_invariant_guards` (before splitting, so an invariant
    condition leaves a loop once rather than once per part), :func:`split_iteration_spaces`,
    :func:`remove_dead_stores` and :func:`merge_contiguous_loops`, and removes the scopes these leave empty. Converting the remaining data-dependent branches to
    selects (:func:`convert_diamonds_to_selects`) and rolling unrolled code (:func:`reroll_statements`,
    :func:`fuse_rolled_loops`) are separate steps.

    :param stree: The schedule tree (or subtree) to transform in place.
    :param max_ranges: Do not split a scope into more than this many copies.
    :param max_enumeration: Upper bound on the iterates evaluated for a guard over compile-time constant data.
    :param min_trip_count: Do not split within loops of fewer iterations than this (see
                           :func:`split_iteration_spaces`).
    :return: The number of conditions folded, conditions moved out of loops, and loops and maps split.
    """
    pair_complementary_guards(stree)
    forward_substitute_conditions(stree)
    remove_dead_assignments(stree)
    changed = fold_guards(stree, max_enumeration)
    changed += unswitch_invariant_guards(stree)
    changed += split_iteration_spaces(stree, max_ranges, max_enumeration, min_trip_count)
    remove_dead_stores(stree)
    merge_contiguous_loops(stree)
    prune_empty(stree, AccessIndex(stree.get_root()))
    return changed

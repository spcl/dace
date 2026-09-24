# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Removing code without effect: unused labels, empty scopes and dead stores."""
import ast
from typing import List, Set, Tuple

from dace import data, dtypes, symbolic
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes.common import (condition_of, is_pure, memlets_of, names_read,
                                                            names_written)


def remove_unused_and_duplicate_labels(stree: tn.ScheduleTreeScope):
    """
    Removes unused and duplicate labels from the schedule tree.

    :param stree: The schedule tree to remove labels from.
    """

    class FindGotos(tn.ScheduleNodeVisitor):

        def __init__(self):
            self.gotos: Set[str] = set()

        def visit_GotoNode(self, node: tn.GotoNode):
            if node.target is not None:
                self.gotos.add(node.target)

    class RemoveLabels(tn.ScheduleNodeTransformer):

        def __init__(self, labels_to_keep: Set[str]) -> None:
            self.labels_to_keep = labels_to_keep
            self.labels_seen = set()

        def visit_StateLabel(self, node: tn.StateLabel):
            if node.state.name not in self.labels_to_keep:
                return None
            if node.state.name in self.labels_seen:
                return None
            self.labels_seen.add(node.state.name)
            return node

    fg = FindGotos()
    fg.visit(stree)
    return RemoveLabels(fg.gotos).visit(stree)


def remove_empty_scopes(stree: tn.ScheduleTreeScope):
    """
    Removes empty scopes from the schedule tree.

    :warning: This pass is not safe to use for for-loops, as it will remove indices that may be used after the loop.
    """

    class RemoveEmptyScopes(tn.ScheduleNodeTransformer):

        def visit_scope(self, node: tn.ScheduleTreeScope):
            if len(node.children) == 0:
                return None

            return self.generic_visit(node)

    return RemoveEmptyScopes().visit(stree)


def remove_dead_stores(stree: tn.ScheduleTreeScope) -> int:
    """
    Remove statements whose results are never read: tasklets, copies and symbol assignments that only write transient
    scalars (or symbols) that are dead where they are written, i.e. overwritten or not read on every path to a read.

    Liveness is computed backwards over the tree: a branch keeps what any of its alternatives reads, and a loop or map
    keeps what its next iteration reads (a fixed point over the body) as well as what follows it, since it may not
    run at all. This removes, e.g., a temporary that each part of a split loop computes but only some parts use, which
    :func:`remove_dead_assignments` (which only removes values read nowhere) keeps. Only pure statements are removed;
    around other kinds of scopes (unstructured control flow) every transient scalar is considered live.

    :param stree: The schedule tree (or subtree) to transform in place.
    :return: The number of statements removed.
    """
    root = stree.get_root()
    containers = root.containers
    tracked = {name for name, desc in containers.items() if desc.transient and isinstance(desc, data.Scalar)}
    in_descriptors = set().union(*(map(str, desc.free_symbols) for desc in containers.values()))
    tracked |= {name for name in root.symbols if name not in containers and name not in in_descriptors}
    everything = frozenset(tracked)
    dead: List[tn.ScheduleTreeNode] = []

    def statement_sets(node: tn.ScheduleTreeNode) -> Tuple[Set[str], Set[str]]:
        """Names a statement reads, and names it certainly overwrites (not through dynamic or accumulating writes)."""
        reads, writes = names_read(node) & tracked, names_written(node) & tracked
        for memlet in memlets_of(node, 'out_memlets'):
            if memlet.dynamic or memlet.wcr is not None:
                writes.discard(memlet.data)
                if memlet.wcr is not None and memlet.data in tracked:
                    reads.add(memlet.data)
        return reads, writes

    def removable(node: tn.ScheduleTreeNode, live: Set[str]) -> bool:
        if not isinstance(node, (tn.TaskletNode, tn.CopyNode, tn.AssignNode)):
            return False
        written = names_written(node)
        if not written or not written <= tracked or written & live:
            return False
        if isinstance(node, tn.TaskletNode):
            if node.node.language != dtypes.Language.Python or getattr(node.node, 'side_effects', False):
                return False
            if any(m.dynamic or m.wcr is not None for m in memlets_of(node, 'out_memlets')):
                return False
            return is_pure(ast.Module(body=list(node.node.code.code), type_ignores=[]))
        if isinstance(node, tn.AssignNode):
            return is_pure(getattr(node.value.code[0], 'value', node.value.code[0]))
        return True

    def body(children: List[tn.ScheduleTreeNode], live: Set[str], record: bool) -> Set[str]:
        """Live names before ``children`` given those live after them; records dead statements if ``record``."""
        live = set(live)
        k = len(children) - 1
        while k >= 0:
            node = children[k]
            if isinstance(node, (tn.ElifScope, tn.ElseScope)):
                # Find the chain start and process the whole chain at once
                start = k
                while start > 0 and isinstance(children[start], (tn.ElifScope, tn.ElseScope)):
                    start -= 1
                chain = children[start:k + 1]
                before = set()
                for branch in chain:
                    before |= body(branch.children, live, record)
                    condition = condition_of(branch)
                    if condition is not None:
                        before |= set(symbolic.symbols_in_ast(condition)) & tracked
                if not isinstance(chain[-1], tn.ElseScope):
                    before |= live
                live = before
                k = start - 1
                continue
            if isinstance(node, tn.IfScope) and not isinstance(node, tn.StateIfScope):
                before = body(node.children, live, record) | live
                condition = condition_of(node)
                live = before | (
                    (set(symbolic.symbols_in_ast(condition)) & tracked) if condition is not None else set(everything))
            elif isinstance(node, (tn.ForScope, tn.MapScope)):
                header = names_read(node) & tracked
                carried = set(live)
                while True:  # The next iteration reads what the body needs at its start
                    start = body(node.children, carried | header, False)
                    grown = carried | start
                    if grown == carried:
                        break
                    carried = grown
                if record:
                    body(node.children, carried | header, True)
                live = carried | header
            elif isinstance(node, tn.ScheduleTreeScope):
                live = set(everything)  # Unstructured or unknown: assume everything is read
            else:
                if record and removable(node, live):
                    dead.append(node)
                else:
                    reads, writes = statement_sets(node)
                    live = (live - writes) | reads
            k -= 1
        return live

    body(stree.children, set(everything) if stree is not root else set(), True)
    dead_ids = {id(n) for n in dead}
    for scope in [n for n in stree.preorder_traversal() if isinstance(n, tn.ScheduleTreeScope)]:
        kept = [c for c in scope.children if id(c) not in dead_ids]
        if len(kept) < len(scope.children):
            scope.children = []
            scope.add_children(kept)
    return len(dead)

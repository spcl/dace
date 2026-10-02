# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Assortment of passes for schedule trees.
"""

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from typing import List, Optional, Set, Tuple


def remove_unused_and_duplicate_labels(stree: tn.ScheduleTreeScope):
    """
    Removes unused and duplicate labels from the schedule tree. Labels in general blocks are only removed if duplicate.

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
            # Labels in general blocks separate their blocks, even if no goto jumps to them
            if node.name not in self.labels_to_keep and not isinstance(node.parent, tn.GBlock):
                return None
            if node.name in self.labels_seen:
                return None
            self.labels_seen.add(node.name)
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


def _contains_goto(node: tn.ScheduleTreeNode, label: Optional[str]) -> bool:
    """
    Returns True if the given node is or contains a goto to the given label (``None`` for exit gotos).
    """
    return any(isinstance(n, tn.GotoNode) and n.target == label for n in node.preorder_traversal())


def _conditional_chain_end(nodes: List[tn.ScheduleTreeNode], start: int) -> int:
    """
    Returns the index after the last branch of the if/elif/else chain that starts at the given index.
    """
    end = start + 1
    while end < len(nodes) and isinstance(nodes[end], (tn.ElifScope, tn.ElseScope)):
        end += 1
    return end


def _may_fall_through(nodes: List[tn.ScheduleTreeNode], label: Optional[str]) -> bool:
    """
    Returns False if every path through the given nodes ends with a goto to the given label.
    """
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if isinstance(node, tn.GotoNode) and node.target == label:
            return False
        if isinstance(node, tn.IfScope):
            end = _conditional_chain_end(nodes, index)
            chain = nodes[index:end]
            if isinstance(chain[-1],
                          tn.ElseScope) and not any(_may_fall_through(branch.children, label) for branch in chain):
                return False
            index = end
            continue
        index += 1
    return True


def _eliminate_gotos(
        nodes: List[tn.ScheduleTreeNode], label: Optional[str],
        updates: List[Tuple[tn.ScheduleTreeScope, List[tn.ScheduleTreeNode]]]) -> Optional[List[tn.ScheduleTreeNode]]:
    """
    Returns a version of the given nodes, which are followed by the given label, without gotos to that label.

    Does not modify the tree. Instead, the new children of affected scopes are appended to ``updates``.

    :param nodes: The nodes to rewrite.
    :param label: The label that follows the nodes (``None`` for the exit of the program).
    :param updates: A list of scopes and their new children, which is extended by this function.
    :return: The rewritten nodes, or None if the gotos cannot be eliminated.
    """
    result: List[tn.ScheduleTreeNode] = []
    index = 0
    while index < len(nodes):
        node = nodes[index]

        # Everything after an unconditional goto to the label is unreachable
        if isinstance(node, tn.GotoNode) and node.target == label:
            return result

        if not _contains_goto(node, label):
            result.append(node)
            index += 1
            continue

        # Gotos out of loops, dataflow scopes, or general blocks cannot be eliminated structurally
        if not isinstance(node, tn.IfScope):
            return None

        # Statements after the conditional chain only run on branches that do not jump to the label, so they are
        # moved into those branches (including an implicit else branch)
        end = _conditional_chain_end(nodes, index)
        chain = nodes[index:end]
        rest = nodes[end:]
        has_else = isinstance(chain[-1], tn.ElseScope)
        falls_through = {id(branch) for branch in chain if _may_fall_through(branch.children, label)}
        if rest and len(falls_through) + (0 if has_else else 1) > 1:
            return None  # Would duplicate the statements after the chain

        for branch in chain:
            body = _eliminate_gotos(branch.children + (rest if id(branch) in falls_through else []), label, updates)
            if body is None:
                return None
            updates.append((branch, body))
        result.extend(chain)

        if rest and not has_else:
            else_body = _eliminate_gotos(rest, label, updates)
            if else_body is None:
                return None
            else_scope = tn.ElseScope(children=[])
            updates.append((else_scope, else_body))
            result.append(else_scope)
        return result

    return result


def eliminate_forward_gotos(scope: tn.ScheduleTreeScope, end: int, label: Optional[str]) -> bool:
    """
    Removes gotos to a label that directly follows a range of children in a scope by restructuring conditionals.

    For example, ``if c: goto L; A; label L:`` becomes ``if c: pass; else: A; label L:``. Statements following an
    unconditional goto to the label are unreachable and removed. The label itself is kept.

    :param scope: The scope whose children contain the gotos.
    :param end: The index of the label in the children of ``scope``, or the number of children for ``label=None``.
    :param label: The name of the label, or ``None`` for exit gotos at the end of the program.
    :return: True if all gotos to the label within the range were removed. If False, the tree is left unmodified.
    """
    updates: List[Tuple[tn.ScheduleTreeScope, List[tn.ScheduleTreeNode]]] = []
    new_children = _eliminate_gotos(scope.children[:end], label, updates)
    if new_children is None:
        return False

    updates.append((scope, new_children + scope.children[end:]))
    for updated_scope, children in updates:
        for child in children:
            child.parent = updated_scope
        updated_scope.children = children
    return True

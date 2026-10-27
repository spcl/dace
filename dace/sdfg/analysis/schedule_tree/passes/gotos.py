# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Eliminating forward gotos by restructuring conditionals."""

from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _contains_goto(node: tn.ScheduleTreeNode, label: str | None) -> bool:
    """
    Returns True if the given node is or contains a goto to the given label (``None`` for exit gotos).
    """
    return any(isinstance(n, tn.GotoNode) and n.target == label for n in node.preorder_traversal())


def _conditional_chain_end(nodes: list[tn.ScheduleTreeNode], start: int) -> int:
    """
    Returns the index after the last branch of the if/elif/else chain that starts at the given index.
    """
    end = start + 1
    while end < len(nodes) and isinstance(nodes[end], (tn.ElifScope, tn.ElseScope)):
        end += 1
    return end


def _may_fall_through(nodes: list[tn.ScheduleTreeNode], label: str | None) -> bool:
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
            if isinstance(chain[-1], tn.ElseScope) and not any(
                _may_fall_through(branch.children, label) for branch in chain
            ):
                return False
            index = end
            continue
        index += 1
    return True


def _eliminate_gotos(
    nodes: list[tn.ScheduleTreeNode],
    label: str | None,
    updates: list[tuple[tn.ScheduleTreeScope, list[tn.ScheduleTreeNode]]],
) -> list[tn.ScheduleTreeNode] | None:
    """
    Returns a version of the given nodes, which are followed by the given label, without gotos to that label.

    Does not modify the tree. Instead, the new children of affected scopes are appended to ``updates``.

    :param nodes: The nodes to rewrite.
    :param label: The label that follows the nodes (``None`` for the exit of the program).
    :param updates: A list of scopes and their new children, which is extended by this function.
    :return: The rewritten nodes, or None if the gotos cannot be eliminated.
    """
    result: list[tn.ScheduleTreeNode] = []
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


def eliminate_forward_gotos(scope: tn.ScheduleTreeScope, end: int, label: str | None) -> bool:
    """
    Removes gotos to a label that directly follows a range of children in a scope by restructuring conditionals.

    For example, ``if c: goto L; A; label L:`` becomes ``if c: pass; else: A; label L:``. Statements following an
    unconditional goto to the label are unreachable and removed. The label itself is kept.

    :param scope: The scope whose children contain the gotos.
    :param end: The index of the label in the children of ``scope``, or the number of children for ``label=None``.
    :param label: The name of the label, or ``None`` for exit gotos at the end of the program.
    :return: True if all gotos to the label within the range were removed. If False, the tree is left unmodified.
    """
    updates: list[tuple[tn.ScheduleTreeScope, list[tn.ScheduleTreeNode]]] = []
    new_children = _eliminate_gotos(scope.children[:end], label, updates)
    if new_children is None:
        return False

    updates.append((scope, new_children + scope.children[end:]))
    for updated_scope, children in updates:
        for child in children:
            child.parent = updated_scope
        updated_scope.children = children
    return True

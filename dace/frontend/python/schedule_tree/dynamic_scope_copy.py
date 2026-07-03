"""Normalize frontend dynamic scope inputs into dedicated schedule-tree copy nodes."""

from __future__ import annotations

import ast
import copy
from typing import Optional, Set

from dace.sdfg.analysis.schedule_tree import treenodes as tn


def promote_dynamic_scope_copies(root: tn.ScheduleTreeRoot) -> None:
    """Rewrite scalar-copy tasklets that feed dynamic frontend scopes.

    The direct Python schedule-tree frontend outlines unresolved subscript
    expressions into scalar temporaries such as ``__stree_idx = A[i]`` before a
    dynamic ``FrontendMap``. Those nodes are semantically dynamic scope inputs,
    so normalize them into ``DynScopeCopyNode`` to match the schedule-tree IR
    contract used by SDFG-derived trees.
    """

    _DynamicScopeCopyPromoter().visit(root)


class _DynamicScopeCopyPromoter(tn.ScheduleNodeTransformer):

    def visit_scope(self, node: tn.ScheduleTreeScope):
        self.generic_visit(node)

        for index, child in enumerate(node.children):
            dynamic_inputs = _frontend_dynamic_input_names(child)
            if not dynamic_inputs:
                continue

            cursor = index - 1
            while cursor >= 0:
                sibling = node.children[cursor]
                if isinstance(sibling, tn.DynScopeCopyNode):
                    cursor -= 1
                    continue
                if not isinstance(sibling, tn.TaskletNode):
                    break

                replacement = _dynscope_replacement(sibling, dynamic_inputs)
                if replacement is None:
                    break

                replacement.parent = node
                node.children[cursor] = replacement
                dynamic_inputs.discard(replacement.target)
                cursor -= 1

        return node


def _frontend_dynamic_input_names(node: tn.ScheduleTreeNode) -> Set[str]:
    if not isinstance(node, tn.MapScope) or not isinstance(node.node, tn.FrontendMap):
        return set()

    result: Set[str] = set()
    for start, stop, step in node.node.ranges:
        for expr in (start, stop, step):
            name = _simple_name(expr)
            if name is not None:
                result.add(name)
    return result


def _simple_name(expr: str) -> Optional[str]:
    try:
        parsed = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    return parsed.body.id if isinstance(parsed.body, ast.Name) else None


def _dynscope_replacement(node: tn.TaskletNode, dynamic_inputs: Set[str]) -> Optional[tn.DynScopeCopyNode]:
    if len(node.in_memlets) != 1 or len(node.out_memlets) != 1:
        return None

    target = next(iter(node.out_memlets.values())).data
    if target not in dynamic_inputs or not _is_direct_assignment_to(node, target):
        return None

    memlet = copy.deepcopy(next(iter(node.in_memlets.values())))
    return tn.DynScopeCopyNode(target=target, memlet=memlet)


def _is_direct_assignment_to(node: tn.TaskletNode, target: str) -> bool:
    code = getattr(node.node, 'code', None)
    text = getattr(code, 'as_string', None)
    if not text:
        return False

    try:
        parsed = ast.parse(text)
    except SyntaxError:
        return False

    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assign) or len(parsed.body[0].targets) != 1:
        return False

    assign = parsed.body[0]
    if not isinstance(assign.targets[0], ast.Name) or assign.targets[0].id != target:
        return False

    return isinstance(assign.value, (ast.Name, ast.Attribute, ast.Subscript))

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrites every ``merge(c, t, e)`` call in a tasklet body to the
floating-point factor form ``c * t + (1 - c) * e``; only the code string
changes (connectors and edges are untouched)."""
import ast
import copy
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.transformation import pass_pipeline as ppl


class _MergeToFpFactor(ast.NodeTransformer):
    """Replaces every ``merge(c, t, e)`` call with ``(c)*(t) + (1 - c)*(e)``."""

    def __init__(self):
        """Initialize with no rewrite recorded yet."""
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Lower a ``merge(c, t, e)`` call to FP-factor arithmetic, recursing first.

        :param node: the call node being visited.
        :returns: the rewritten node, or the original if it is not a 3-arg merge.
        """
        # Recurse first so nested ``merge(merge(...), ...)`` lowers inside-out.
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == "merge" and len(node.args) == 3):
            return node
        c, t, e = node.args
        c_copy = copy.deepcopy(c)
        ct = ast.BinOp(left=c, op=ast.Mult(), right=t)
        one_minus_c = ast.BinOp(left=ast.Constant(value=1), op=ast.Sub(), right=c_copy)
        one_minus_c_times_e = ast.BinOp(left=one_minus_c, op=ast.Mult(), right=e)
        replacement = ast.BinOp(left=ct, op=ast.Add(), right=one_minus_c_times_e)
        self.changed = True
        return ast.copy_location(replacement, node)


@properties.make_properties
class LowerMergeToFpFactor(ppl.Pass):
    """Lower every ``merge(c, t, e)`` call in tasklet bodies to ``c*t + (1-c)*e``."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        """This pass modifies tasklets."""
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """This pass never needs reapplication."""
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite all merge calls in Python tasklet bodies of ``sdfg``.

        :param sdfg: the SDFG whose tasklets are rewritten in place.
        :param _: unused pipeline results.
        :returns: number of tasklets rewritten, or None if none changed.
        """
        rewritten = 0
        for state in sdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.Tasklet):
                    continue
                code = node.code.as_string if isinstance(node.code, CodeBlock) else str(node.code)
                if "merge(" not in code:
                    continue
                try:
                    tree = ast.parse(code, mode="exec")
                except SyntaxError:
                    # Non-Python tasklet body (e.g. raw CPP). The K1=fp_factor
                    # path only sees Python tasklets emitted by M3.1b/M3.2;
                    # leave anything else alone.
                    continue
                transformer = _MergeToFpFactor()
                new_tree = transformer.visit(tree)
                if not transformer.changed:
                    continue
                ast.fix_missing_locations(new_tree)
                node.code = CodeBlock(ast.unparse(new_tree))
                rewritten += 1
        return rewritten or None

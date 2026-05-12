# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``LowerMergeToFpFactor`` — rewrites every ``merge(c, t, e)`` call inside a
``Tasklet`` body as the floating-point factor form ``c * t + (1 - c) * e``.

This is the downstream rewriter for the ``K1 = fp_factor`` pipeline branch.
The canonical merge IR emitted by ``SameWriteSetIfElseToMergeCFG`` (M3.1b)
and ``BranchNormalization`` (M3.2) is what every prep pass produces; this
pass turns those merge calls into FP-factor arithmetic so that downstream
emission falls back on the existing ``vector_mul`` / ``vector_add``
templates instead of the ``vector_select`` / mask-blend path used by the
``K1 = merge`` branch.

Tasklet connectors and edges are unchanged: only the ``code`` string is
rewritten. ``c`` may be either an in-connector (the typical M3.1b/M3.2
shape, e.g. ``_c``) or a free symbol that the merge tasklet carried in
its body (the cond-as-symbol shape M3.1b/M3.2 emit when the cond is not
resolvable to an array). Both cases collapse to the same FP-factor body.

After this pass runs, no ``Tasklet`` body in the SDFG contains a
``merge(...)`` call.
"""
import ast
import copy
from typing import Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.transformation import pass_pipeline as ppl


class _MergeToFpFactor(ast.NodeTransformer):
    """Replaces every ``merge(c, t, e)`` call with ``(c)*(t) + (1 - c)*(e)``.

    Tracks whether any rewrite happened so the caller can skip writing the
    code back when nothing changed.
    """

    def __init__(self):
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
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
    """Lower every ``merge(c, t, e)`` call in tasklet bodies to ``c*t + (1-c)*e``.

    See module docstring for the contract.
    """

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
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

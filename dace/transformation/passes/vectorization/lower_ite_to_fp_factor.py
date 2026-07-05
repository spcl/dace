# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrites every ``ITE(c, t, e)`` call in a tasklet body to the
floating-point factor form ``c * t + (1 - c) * e``; only the code string
changes (connectors and edges are untouched)."""
import ast
from typing import Optional

import dace
from dace import properties
from dace.frontend.python.astutils import unparse
from dace.properties import CodeBlock
from dace.transformation import pass_pipeline as ppl


class _ITEToFpFactor(ast.NodeTransformer):
    """Replaces every ``ITE(c, t, e)`` call with ``cf*t + (1 - cf)*e`` where
    ``cf = <cast_dtype>(c)`` promotes the (typically ``bool``) condition to the
    arm dtype, so the ``cf * t`` tile binop is uniform-dtype (the K-dim tile path
    refuses a mixed ``bool * double`` binop). ``SplitTasklets`` later splits ``cf``
    into a standalone cast tasklet that ``ConvertTaskletsToTileOps`` lowers to a
    ``TileUnop`` cast. When ``cast_dtype`` is ``None`` (or a ``bool`` output, where
    no promotion is needed) the condition is used as-is.
    """

    def __init__(self, cast_dtype: Optional[str] = None):
        self.changed = False
        self._cast_dtype = cast_dtype

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Lower an ``ITE(c, t, e)`` call to FP-factor arithmetic, recursing first.

        :param node: the call node being visited.
        :returns: the rewritten node, or the original if it is not a 3-arg ITE.
        """
        # Recurse first so nested ``ITE(ITE(...), ...)`` lowers inside-out; the arms
        # are then rendered back to source with the shared ``astutils`` unparser.
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == "ITE" and len(node.args) == 3):
            return node
        c, t, e = (unparse(arg) for arg in node.args)
        cf = f"dace.{self._cast_dtype}({c})" if self._cast_dtype is not None else f"({c})"
        replacement = ast.parse(f"{cf} * ({t}) + (1 - {cf}) * ({e})", mode="eval").body
        self.changed = True
        return ast.copy_location(replacement, node)


@properties.make_properties
class LowerITEToFpFactor(ppl.Pass):
    """Lower every ``ITE(c, t, e)`` call in tasklet bodies to ``c*t + (1-c)*e``."""

    CATEGORY: str = "Vectorization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _ite_output_dtype(self, sdfg: dace.SDFG, state, tasklet) -> Optional[str]:
        """The ``dace`` dtype name (e.g. ``"float64"``) to promote the condition to
        -- the tasklet's single output-array dtype (== the ITE arms' dtype). Returns
        ``None`` when it cannot be resolved or the output is already ``bool`` (no
        promotion needed), so the condition is left unchanged."""
        out_edges = [e for e in state.out_edges(tasklet) if e.data is not None and e.data.data is not None]
        if not out_edges:
            return None
        dtype = sdfg.arrays[out_edges[0].data.data].dtype
        name = dtype.to_string()
        if name == "bool" or name not in vars(dace):
            return None
        return name

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite all ITE calls in Python tasklet bodies of ``sdfg``.

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
                if "ITE(" not in code:
                    continue
                try:
                    tree = ast.parse(code, mode="exec")
                except SyntaxError:
                    # Non-Python tasklet body (e.g. raw CPP). The K1=fp_factor
                    # path only sees Python tasklets emitted by M3.1b/M3.2;
                    # leave anything else alone.
                    continue
                transformer = _ITEToFpFactor(self._ite_output_dtype(sdfg, state, node))
                new_tree = transformer.visit(tree)
                if not transformer.changed:
                    continue
                ast.fix_missing_locations(new_tree)
                node.code = CodeBlock(unparse(new_tree))
                rewritten += 1
        return rewritten or None

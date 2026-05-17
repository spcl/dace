# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Retype integer-valued floating-point ``**`` exponents to ``int``.

A Python tasklet expression ``base ** 2.0`` carries a *float* exponent
constant.  DaCe's C++ unparser routes that through the general
``dace::math::pow(base, 2.0)`` (libm ``pow``) path, whereas an *integer*
exponent ``base ** 2`` is emitted as ``dace::math::ipow(base, 2)`` --
plain left-to-right repeated multiplication (``base*base``), which is
exactly what a Fortran/C reference compiler produces for a small integer
power.  The two are not bit-identical: libm ``pow`` rounds through a
``exp(y*log(x))`` style evaluation and can differ in the trailing bit,
which then cascades through long real(8) reduction chains.

This pass walks every Python tasklet and rewrites the exponent of each
``**`` whose right operand is an integer-valued float literal (``2.0``,
``-3.0``, ``(2.0)``) to the corresponding ``int`` (``2``, ``-3``, ``2``)
so codegen takes the deterministic ``ipow`` path.  Only the exponent
*literal* is retyped -- the base sub-expression is untouched (no
duplication, no connector renumbering), so the rewrite is safe at any
point after tasklet emission.  Genuinely fractional exponents
(``0.5``, ``0.333``) are integer-valued-checked and left alone.
"""

import ast
from typing import Optional

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


class _ExponentIntegerizer(ast.NodeTransformer):
    """Rewrite integer-valued float ``**`` exponents to ``int`` literals."""

    def __init__(self):
        self.rewrites = 0

    @staticmethod
    def _as_int_constant(node: ast.AST) -> Optional[ast.AST]:
        """Return an ``int``-valued replacement for an integer-valued
        float exponent node, or ``None`` if it is not one.

        Handles a bare ``Constant`` (``2.0``) and a unary-minus over a
        ``Constant`` (``-2.0``); the sign is folded into the integer so
        the result is a single literal codegen recognises.

        :param node: the ``**`` right-operand AST node.
        :returns: a new ``ast.Constant`` holding an ``int``, or ``None``.
        """
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, float) and node.operand.value.is_integer()):
            return ast.copy_location(ast.Constant(value=-int(node.operand.value)), node)
        if (isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value.is_integer()):
            return ast.copy_location(ast.Constant(value=int(node.value)), node)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            repl = self._as_int_constant(node.right)
            if repl is not None:
                node.right = repl
                self.rewrites += 1
        return node


@explicit_cf_compatible
class IntegerizePowerExponents(ppl.Pass):
    """Retype integer-valued float ``**`` exponents in tasklets to ``int``.

    Runs after tasklet splitting (the bridge's post-generation stage):
    by then every tasklet body is final, so retyping an exponent literal
    only flips the codegen branch from libm ``pow`` to repeated-multiply
    ``ipow`` -- bit-matching the Fortran/C reference -- without touching
    the base expression or any connector.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite every Python tasklet's integer-valued float ``**``
        exponents to ``int``.

        :param sdfg: the SDFG to transform (recursively, including
            nested SDFGs).
        :returns: the number of exponents rewritten, or ``None`` if the
            pass made no change.
        """
        total = 0
        for node, _parent in sdfg.all_nodes_recursive():
            if not isinstance(node, dace.nodes.Tasklet):
                continue
            if node.code.language != dace.dtypes.Language.Python:
                continue
            body = node.code.code
            if not isinstance(body, list):
                continue
            tr = _ExponentIntegerizer()
            for stmt in body:
                tr.visit(stmt)
            if tr.rewrites:
                for stmt in body:
                    ast.fix_missing_locations(stmt)
                total += tr.rewrites
        return total or None

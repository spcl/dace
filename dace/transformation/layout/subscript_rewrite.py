# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
r"""Reindexing array accesses inside interstate-edge assignments.

An interstate assignment holds a whole expression, not a bare access, so the layout passes cannot
find an access by matching text: a greedy ``name\[(.*)\]`` over ``A[i, j, k] * B[i, j, k]`` runs to
B's closing bracket and reports five indices for a rank-3 access, and a substitution built from that
match swallows every other term in the expression.

``pystr_to_symbolic`` already lowers ``A[i, j, k]`` to :class:`~dace.symbolic.Subscript`, and
``symstr`` prints it back, so the access is located structurally instead.

Sympy canonicalizes as it parses, so a rewritten assignment comes back re-associated: ``0.5 * (A +
B)`` prints as ``0.5*A + 0.5*B``, and ``A / 3.0`` as ``0.3333333333333333*A``. That is ACCEPTED here
(``PermuteDimensions`` has round-tripped its assignments this way all along); do not reintroduce text
matching to avoid it.
"""

from typing import Callable, Sequence, Tuple

import sympy

from dace import symbolic


def rewrite_subscript_indices(
        expr_str: str, name: str, new_indices: Callable[[Tuple[symbolic.SymbolicType, ...]],
                                                        Sequence[symbolic.SymbolicType]]) -> str:
    """``expr_str`` with the indices of every ``name[...]`` access replaced by ``new_indices``.

    :param expr_str: The assignment expression to rewrite.
    :param name: The accessed container to reindex; other containers are left alone.
    :param new_indices: Maps an access' old index tuple to its new one.
    :return: The rewritten expression, or ``expr_str`` unchanged if it holds no such access.
    """
    if name not in expr_str:  # cheap reject before parsing
        return expr_str
    try:
        expr = symbolic.pystr_to_symbolic(expr_str)
    except (SyntaxError, TypeError, AttributeError):
        return expr_str
    if not isinstance(expr, sympy.Basic):
        return expr_str

    rewritten = rewrite_expression(expr, name, new_indices)
    if rewritten is expr:
        return expr_str
    return symbolic.symstr(rewritten, arrayexprs=frozenset(symbolic.arrays(rewritten)))


def rewrite_expression(
    expr: symbolic.SymbolicType, name: str,
    new_indices: Callable[[Tuple[symbolic.SymbolicType, ...]],
                          Sequence[symbolic.SymbolicType]]) -> symbolic.SymbolicType:
    """:func:`rewrite_subscript_indices` on a parsed expression. Bottom-up, so an access nested in
    another access' index is reindexed too."""
    if not expr.args:
        return expr
    args = tuple(rewrite_expression(a, name, new_indices) for a in expr.args)
    # the container is ``args[0]`` of a Subscript, never the function -- ``expr.func`` is ``Subscript``
    if isinstance(expr, symbolic.Subscript) and str(args[0]) == name:
        args = (args[0], ) + tuple(new_indices(args[1:]))
    if args == expr.args:
        return expr
    return expr.func(*args)

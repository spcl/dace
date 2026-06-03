# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Polymorphism helpers for symbolic / non-symbolic operands.

DaCe subset bounds, memlet expressions, and constants flow through the
pipeline as a mix of:

* :class:`sympy.Expr` — fully-symbolic expressions carrying ``free_symbols``,
  ``is_Integer``, ``subs``, ``atoms``.
* Python ``int`` / ``float`` — plain primitives that do **not** expose those
  attributes.

Every callsite that wanted to inspect such an operand previously wrote a
defensive ``getattr(expr, "free_symbols", set())`` (or matching ``hasattr``
guard). The helpers below centralise that polymorphism so the call sites
stay clean and a future tightening of the inputs (e.g. requiring sympy
everywhere) only needs to touch one place.

All helpers are read-only and return well-typed Python objects.
"""
from typing import Any, Mapping, Set


def free_symbols(expr: Any) -> Set[Any]:
    """Return ``expr.free_symbols`` when the operand exposes it, else ``set()``.

    Plain Python ``int`` / ``float`` literals have no free symbols, so the
    empty set is the right identity for set unions and membership tests.

    :param expr: Operand to inspect (typically a sympy expression or a
        Python primitive).
    :returns: The set of free symbols, or an empty set when ``expr`` is a
        primitive without a ``free_symbols`` attribute.
    """
    fs = getattr(expr, "free_symbols", None)
    return fs if fs is not None else set()


def free_symbol_names(expr: Any) -> Set[str]:
    """Return the **names** of ``free_symbols(expr)`` as strings.

    Equivalent to ``{str(s) for s in free_symbols(expr)}``; the dominant
    callsite shape across the pipeline. Centralises the ``str()`` cast.

    :param expr: Operand to inspect.
    :returns: Free-symbol names, or an empty set for primitives.
    """
    return {str(s) for s in free_symbols(expr)}


def is_integer(expr: Any) -> bool:
    """Return whether ``expr`` is an integer (sympy ``Integer`` or Python ``int``).

    The sympy form exposes ``is_Integer``; Python primitives don't but
    ``isinstance(expr, int)`` captures them. ``bool`` is a subclass of
    ``int``; it counts as integer here (matches sympy's behaviour).

    :param expr: Operand to inspect.
    :returns: ``True`` when ``expr`` is an integer value.
    """
    if isinstance(expr, int):
        return True
    return bool(getattr(expr, "is_Integer", False))


def subs(expr: Any, mapping: Mapping[Any, Any]) -> Any:
    """Apply ``expr.subs(mapping)`` when supported; return ``expr`` otherwise.

    A Python primitive has nothing to substitute, so returning it
    unchanged is the natural identity. The callsite no longer has to
    branch on ``hasattr(expr, "subs")``.

    :param expr: Operand to rewrite.
    :param mapping: Mapping of replacements (sympy semantics).
    :returns: The rewritten expression, or ``expr`` when it has no
        ``subs`` method.
    """
    fn = getattr(expr, "subs", None)
    return fn(mapping) if fn is not None else expr


def atoms(expr: Any) -> Set[Any]:
    """Return ``expr.atoms()`` when supported, else the empty set.

    :param expr: Operand to inspect.
    :returns: The set of atomic terms, or an empty set for primitives.
    """
    fn = getattr(expr, "atoms", None)
    return fn() if fn is not None else set()


def atoms_of(expr: Any, *types: type) -> Set[Any]:
    """Return ``expr.atoms(*types)`` when supported, else the empty set.

    Sympy's ``atoms`` accepts one or more atom-type filters; Python
    primitives don't expose ``atoms`` at all, so callers writing
    ``expr.atoms(Subscript)`` need a guard. This helper provides that
    guard centrally.

    :param expr: Operand to inspect.
    :param types: Atom-type classes to filter (e.g. ``Subscript``).
    :returns: The set of matching atoms, or an empty set for primitives.
    """
    fn = getattr(expr, "atoms", None)
    return fn(*types) if fn is not None else set()

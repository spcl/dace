# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Polymorphism helpers for symbolic / non-symbolic operands.

DaCe subset bounds and memlet expressions flow through the pipeline as a mix of
:class:`sympy.Basic` expressions and plain Python ``int`` / ``float``. Only sympy carries
``free_symbols``, so every call site would otherwise have to test for it. These helpers
centralise that one test, keyed on the type rather than on attribute presence.
"""
from typing import Any, Set

import sympy


def free_symbols(expr: Any) -> Set[Any]:
    """Free symbols of ``expr``, or an empty set for a non-symbolic operand.

    A Python numeric literal has no free symbols, so the empty set is the right identity
    for the set unions and membership tests at the call sites.

    :param expr: Operand to inspect.
    :returns: The set of free symbols.
    """
    return expr.free_symbols if isinstance(expr, sympy.Basic) else set()


def free_symbol_names(expr: Any) -> Set[str]:
    """Names of :func:`free_symbols` as strings -- the dominant call-site shape.

    :param expr: Operand to inspect.
    :returns: Free-symbol names.
    """
    return {str(s) for s in free_symbols(expr)}

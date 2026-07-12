# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""AST-based code-rewrite helpers for the vectorization pipeline.

Manipulate Python expression / statement strings (CodeBlock bodies, interstate-edge assignment
RHSs, loop and conditional-block conditions), round-tripped through ``ast.unparse``.
"""
from typing import Optional, Set

import dace
from dace.symbolic import DaceSympyPrinter
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def offset_symbol_in_expression(expr_str: str,
                                symbol_to_offset: str,
                                offset: int,
                                arrays: Optional[Set[str]] = None) -> str:
    """Return a new expression string with a symbol incremented by an offset.

    :param expr_str: The original expression as a string.
    :param symbol_to_offset: The symbol within the expression to offset.
    :param offset: The integer value to add to the symbol.
    :param arrays: Array names in scope, passed to ``DaceSympyPrinter`` so array reads round-trip
        as ``arr[idx]`` not ``arr(idx)``. Callers with an SDFG in scope pass
        ``set(sdfg.arrays.keys())``. Defaults to empty set.
    :returns: The offset expression string, or the original unchanged if the symbol is absent.
    :raises Exception: If ``expr_str`` contains an ``Eq(`` equality term.
    """
    if "Eq(" in expr_str:
        raise Exception(expr_str)
    expr = dace.symbolic.SymExpr(expr_str)
    sym_to_change = None
    for free_sym in expr.free_symbols:
        if str(free_sym) == symbol_to_offset:
            sym_to_change = free_sym
            break
    if sym_to_change is None:
        return expr_str
    offsetted_expr = f"({sym_to_change} + {offset})"
    offset_expr = expr.subs(sym_to_change, offsetted_expr)
    return DaceSympyPrinter(arrays if arrays is not None else set()).doprint(offset_expr)


def use_laneid_symbol_in_expression(expr_str: str,
                                    symbol_to_offset: str,
                                    offset: int,
                                    vector_map_param: str = None,
                                    arrays: Optional[Set[str]] = None) -> str:
    """Return a new expression string with a symbol replaced by its lane-id variant.

    ``sym1`` -> ``sym1_laneid_<offset>``, except ``vector_map_param`` -> ``(sym + offset)``.

    :param expr_str: The original expression as a string.
    :param symbol_to_offset: The symbol within the expression to offset.
    :param offset: The lane index / integer offset to apply.
    :param vector_map_param: The vector map parameter name; matching symbols are offset rather
        than lane-suffixed.
    :param arrays: Array names in scope, passed to ``DaceSympyPrinter`` so array reads print as
        ``arr[idx]``.
    :returns: The rewritten expression string, or the original unchanged if the symbol is absent.
    :raises Exception: If ``expr_str`` contains an ``Eq(`` equality term.
    """
    if "Eq(" in expr_str:
        raise Exception(expr_str)
    expr = dace.symbolic.SymExpr(expr_str)
    sym_to_change = None
    for free_sym in expr.free_symbols:
        if str(free_sym) == symbol_to_offset:
            sym_to_change = free_sym
            break
    if sym_to_change is None:
        return expr_str
    if vector_map_param is not None and str(sym_to_change) == vector_map_param:
        offsetted_expr = f"({sym_to_change} + {offset})"
    else:
        offsetted_expr = f"({LaneIdScheme.make_dim(str(sym_to_change), 0, int(offset))})"
    offset_expr = expr.subs(sym_to_change, offsetted_expr)
    # ``arrays`` rationale: see ``offset_symbol_in_expression``.
    return DaceSympyPrinter(arrays if arrays is not None else set()).doprint(offset_expr)

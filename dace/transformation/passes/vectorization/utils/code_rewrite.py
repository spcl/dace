# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""AST-based code-rewrite helpers for the vectorization pipeline.

Manipulate Python expression / statement strings (CodeBlock bodies, interstate-edge assignment
RHSs, loop and conditional-block conditions), round-tripped through ``ast.unparse``.
"""
import ast
import re
from typing import Optional, Set, Tuple

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.symbolic import DaceSympyPrinter
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def extract_bracket_contents(expr: str, name: str):
    """Extract contents inside the brackets following ``name`` in an expression.

    Handles nested brackets + quoted strings so commas inside them are not split on.

    :param expr: The input expression to search within.
    :param name: The variable / function name to match before the brackets.
    :returns: ``(full_match, parts)``: ``full_match`` = matched ``<name>[...]`` substring (empty
        if no match), ``parts`` = list of top-level comma-separated elements inside the brackets.
    """
    # Find <name>[...]
    pattern = rf'\b{name}\s*\[(.*)\]'
    match = re.search(pattern, expr)
    if not match:
        return "", []

    full_match = match.group(0)
    inside = match.group(1)
    parts = []
    current = []
    depth = 0
    in_string = None

    for ch in inside:
        if in_string:
            current.append(ch)
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            current.append(ch)
        elif ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)

    # Add last part
    if current:
        parts.append(''.join(current).strip())

    return full_match, parts


class _DropDimsTransformer(ast.NodeTransformer):
    """Drop dimensions from every Subscript access of a named array.

    Rewrites ``Name(dataname)[indices]`` keeping only indices where ``dim_mask[i] == 1``.
    """

    def __init__(self, dataname: str, dim_mask: Tuple[int, ...]):
        self.dataname = dataname
        self.dim_mask = dim_mask

    def _filter_indices(self, slice_node: ast.AST) -> ast.AST:
        # Pre-3.9, slice was wrapped in ast.Index. Unwrap if present.
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):  # pragma: no cover - py<3.9
            inner = slice_node.value
            new_inner = self._filter_indices(inner)
            return ast.Index(value=new_inner)
        if isinstance(slice_node, ast.Tuple):
            elts = slice_node.elts
            if len(elts) != len(self.dim_mask):
                raise ValueError(f"drop_dims_from_str: array {self.dataname!r} accessed with {len(elts)} dimensions, "
                                 f"dim_mask has {len(self.dim_mask)}")
            kept = [e for e, m in zip(elts, self.dim_mask) if m]
            if len(kept) == 1:
                return kept[0]
            return ast.Tuple(elts=kept, ctx=ast.Load())
        # Single-element subscript (1-D access)
        if len(self.dim_mask) != 1:
            raise ValueError(f"drop_dims_from_str: array {self.dataname!r} accessed with 1 dimension, "
                             f"dim_mask has {len(self.dim_mask)}")
        if self.dim_mask[0]:
            return slice_node
        # Dropping the only dim collapses the subscript; return the bare value (caller swaps Subscript->value).
        return None

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.dataname:
            new_slice = self._filter_indices(node.slice)
            if new_slice is None:
                return node.value
            node.slice = new_slice
        return node


def drop_dims_from_str(src_str: str, dim_mask: Tuple[int], dataname: str) -> str:
    """Remove dimensions from every access of ``dataname`` in a Python expression string.

    ``dim_mask`` 0 -> drop dim, 1 -> keep. AST-based -> all occurrences updated. Returns input
    unchanged if no matching access found or input does not parse as Python.

    :param src_str: Source expression / statement(s).
    :param dim_mask: Tuple of 0/1 indicating which dimensions to keep.
    :param dataname: Name of the array whose accesses to rewrite.
    :returns: Modified string with dimensions dropped from each occurrence.
    :raises ValueError: If any access has a dimension count different from ``len(dim_mask)``.
    """
    try:
        tree = ast.parse(src_str)
    except SyntaxError:
        return src_str
    transformer = _DropDimsTransformer(dataname, tuple(dim_mask))
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def drop_dims(sdfg: dace.SDFG, dim_mask: Tuple[int], dataname: str) -> None:
    """Remove dimensions from all accesses to an array throughout an SDFG.

    Updates memlet subsets on edges, loop conditions / init / update statements, conditional
    branch conditions, and interstate-edge assignments.

    :param sdfg: The SDFG to modify.
    :param dim_mask: Tuple of 0/1, which dimensions to keep (1 = keep, 0 = drop); length must
        match array dimensionality.
    :param dataname: Name of the array whose accesses to update.
    """

    for cfg_region in sdfg.all_control_flow_regions():

        # Handle loop regions: update loop condition, init, and update statements
        if isinstance(cfg_region, LoopRegion):
            # Update loop condition (e.g., "i < N and A[i] > 0")
            cfg_region.loop_condition = CodeBlock(
                drop_dims_from_str(cfg_region.loop_condition.as_string, dim_mask, dataname),
                cfg_region.loop_condition.language)

            # Update loop update statement (e.g., "i = i + 1")
            cfg_region.update_statement = CodeBlock(
                drop_dims_from_str(cfg_region.update_statement.as_string, dim_mask, dataname),
                cfg_region.update_statement.language)

            # Update loop initialization (e.g., "i = A[0]")
            cfg_region.init_statement = CodeBlock(
                drop_dims_from_str(cfg_region.init_statement.as_string, dim_mask, dataname),
                cfg_region.init_statement.language)

        # Handle conditional blocks: update branch conditions
        elif isinstance(cfg_region, ConditionalBlock):
            # Each branch is a (condition, block) tuple
            for i, (condition, block) in enumerate(cfg_region.branches):
                # Update condition expression
                new_condition = CodeBlock(drop_dims_from_str(condition.as_string, dim_mask, dataname),
                                          condition.language)
                # Replace the branch with updated condition
                cfg_region.branches[i] = (new_condition, block)

    from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
    for _state, edge in walk_memlets_of(sdfg, dataname):
        # An earlier rewrite may have already collapsed this memlet to lower dimensionality
        # (common after a previous prepare / vectorize pass touched the same array); skip such
        # memlets rather than re-collapsing.
        if len(edge.data.subset) != len(dim_mask):
            continue

        # Build new subset with filtered dimensions
        new_subset = []
        for (begin, end, step), mask_bit in zip(edge.data.subset, dim_mask):
            if mask_bit:  # Keep this dimension
                new_subset.append((begin, end, step))

        # Create new memlet with reduced dimensionality
        edge.data = dace.memlet.Memlet(data=dataname, subset=dace.subsets.Range(new_subset))

    for interstate_edge in sdfg.all_interstate_edges():
        # Skip edges without assignments
        if interstate_edge.data.assignments == {}:
            continue

        # Update each assignment expression
        new_assignments = {}
        for var_name, expr in interstate_edge.data.assignments.items():
            # Apply dimension dropping to the expression
            new_assignments[var_name] = drop_dims_from_str(expr, dim_mask, dataname)

        # Replace assignments
        interstate_edge.data.assignments = new_assignments


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


# ``STANDARD_FUNCS`` / ``FuncToSubscript`` / ``convert_nonstandard_calls`` deleted in S1c-bis.
# Their sole caller (``expand_interstate_assignments_to_lanes``) now emits the lane-suffixed
# assignment via ``DaceSympyPrinter(arrays).doprint``, which prints array reads as ``arr[idx]``
# natively and emits ``and``/``or``/``not`` for ``sympy.And``/``Or``/``Not`` -> AST round-trip no
# longer needed.

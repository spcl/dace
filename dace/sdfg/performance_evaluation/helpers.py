# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Helper functions shared by the SDFG performance analyses: element UUIDs, fixed-point symbol
substitution, and static-symbol detection. """

import re
from typing import Dict

import sympy as sp

from dace import SDFG, SDFGState, dtypes, nodes
from dace.sdfg.state import BreakBlock, ContinueBlock, ReturnBlock, UnstructuredControlFlow
from dace.symbolic import pystr_to_symbolic, symbol

UUID_SEPARATOR = '/'


def ids_to_string(cfg_id, state_id=-1, node_id=-1, edge_id=-1):
    return (str(cfg_id) + UUID_SEPARATOR + str(state_id) + UUID_SEPARATOR + str(node_id) + UUID_SEPARATOR +
            str(edge_id))


def get_uuid(element, state=None):
    if isinstance(element, SDFG):
        return ids_to_string(element.cfg_id)
    elif isinstance(element, SDFGState):
        return ids_to_string(element.parent_graph.cfg_id, element.block_id)
    elif isinstance(element, nodes.Node):
        return ids_to_string(state.parent_graph.cfg_id, state.block_id, state.node_id(element))
    else:
        return ids_to_string(-1)


def has_unstructured_control_flow(sdfg: SDFG) -> bool:
    """
    Check whether the SDFG contains control flow the performance analyses do not model.

    They assume structured control flow -- loops as ``LoopRegion`` and branches as
    ``ConditionalBlock`` -- and model neither non-local exits nor legacy state machines. The
    following are therefore reported as unsupported: a legacy loop (a cycle not encapsulated in a
    ``LoopRegion``), unstructured branching (a block with more than one outgoing edge), and
    ``break`` / ``continue`` / ``return`` (``BreakBlock`` / ``ContinueBlock`` / ``ReturnBlock``).

    :param sdfg: The SDFG to inspect.
    :return: True if any (possibly nested) control-flow region is unstructured.
    """
    for region in sdfg.all_control_flow_regions(recursive=True):
        if isinstance(region, UnstructuredControlFlow) or region.has_cycles():
            return True
        for block in region.nodes():
            if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)) or len(region.out_edges(block)) > 1:
                return True
    return False


def subs_till_fixed_point(expr: sp.Expr, symbol_map: Dict[sp.Expr, sp.Expr]) -> sp.Expr:
    """
    Apply a symbol mapping to a symbolic expression repeatedly until a fixed point is reached.

    Requires that the symbol mapping has no cyclic dependencies, otherwise it would not converge.

    :param expr: The expression to substitute into (non-symbolic values are returned unchanged).
    :param symbol_map: Mapping from symbols to their replacement expressions.
    :return: The expression after substituting to a fixed point.
    """
    if not isinstance(expr, sp.Expr):
        return expr
    prev = None
    curr = expr
    while prev != curr:
        prev = curr
        curr = curr.subs(symbol_map)
    return curr


def get_static_symbols(sdfg: SDFG) -> Dict[str, sp.Expr]:
    """
    Find the symbols that are assigned at exactly one point in the SDFG (i.e., statically known).

    A symbol is static if it is written by a single length-1 access (from a tasklet performing one
    assignment, or by a single copy from another access node). Symbols written in more than one
    place are excluded.

    :param sdfg: The SDFG for which to find static symbols and their assignments.
    :return: Mapping from each static symbol name to its defining expression (resolved to a fixed
             point). String keys let callers both substitute and index the result by symbol name.
    """
    # Strip type-cast prefixes (e.g. ``dace.float64``, ``int``) from a tasklet RHS so the cast does
    # not interfere with the symbolic parse below. The DaCe type names are derived from
    # ``dace.dtypes`` (rather than hard-coded), and matched longest-first so ``dace.float64`` wins
    # over ``float``.
    cast_names = {'int', 'float', 'complex', 'bool'}
    cast_names |= {f'dace.{name}' for name in dir(dtypes) if isinstance(getattr(dtypes, name), dtypes.typeclass)}
    type_regex = re.compile("|".join(re.escape(name) for name in sorted(cast_names, key=len, reverse=True)))

    static_symbol_mapping: Dict[sp.Symbol, sp.Expr] = {symbol(a): symbol(a) for a in sdfg.arg_names}
    non_static_symbols = set()
    for node, containing_state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.AccessNode):
            continue
        if containing_state.in_degree(node) != 1:
            continue
        edge = containing_state.in_edges(node)[0]
        source = edge.src
        if edge.data.volume != 1:
            continue

        if isinstance(source, nodes.Tasklet):
            tasklet = source
            in_map = {}
            out_map = {}
            # Incoming edges: symbols feeding the tasklet.
            for e in containing_state.in_edges(tasklet):
                if not isinstance(e.src, nodes.AccessNode):
                    continue
                in_map[e.dst_conn] = str(e.src.data)
            # Outgoing edges: symbols written by the tasklet (expected to be a single edge).
            for e in containing_state.out_edges(tasklet):
                if not isinstance(e.dst, nodes.AccessNode):
                    continue
                out_map[e.src_conn] = str(e.dst.data)

            in_map = {symbol(k): symbol(v) for k, v in in_map.items()}
            out_map = {symbol(k): symbol(v) for k, v in out_map.items()}
            code = tasklet.code.as_string.strip()
            # Expect a single assignment.
            lines = [l.strip() for l in code.splitlines() if l.strip()]
            if len(lines) > 1:
                non_static_symbols.add(node.data)
                continue
            lhs, rhs = lines[0].split('=', 1)
            lhs = lhs.strip()
            rhs = type_regex.sub("", rhs.strip())
            lhs_sympy = pystr_to_symbolic(lhs).subs(out_map)

            if lhs_sympy not in static_symbol_mapping.keys():
                try:
                    static_symbol_mapping[lhs_sympy] = pystr_to_symbolic(rhs).subs(in_map)
                except Exception:
                    non_static_symbols.add(lhs_sympy)
            else:
                non_static_symbols.add(lhs_sympy)

        elif isinstance(source, nodes.AccessNode):
            data_sym = symbol(source.data)
            if data_sym not in static_symbol_mapping.keys():
                static_symbol_mapping[data_sym] = symbol(node.data)
            else:
                non_static_symbols.add(data_sym)

    static_symbol_mapping = {k: v for (k, v) in static_symbol_mapping.items() if k not in non_static_symbols}
    return {str(k): subs_till_fixed_point(v, static_symbol_mapping) for k, v in static_symbol_mapping.items()}

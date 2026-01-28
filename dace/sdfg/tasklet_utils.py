# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tasklet Classification Utilities

This module provides utilities for analyzing and classifying DaCe tasklets based on their
computational patterns. It parses tasklet code to determine the types of operations, operands,
and constants involved. It also provides utilities furhter manipulate and analyze tasklets.
"""

import re
import sympy
import dace
from typing import Dict, Tuple, Set
from dace.properties import CodeBlock
from enum import Enum
import ast
import typing


class TaskletType(Enum):
    """
    Enumeration of supported tasklet computational patterns.

    Each pattern represents a specific combination of input types (arrays, scalars, symbols)
    and operation types (assignment, binary operation, unary operation).

    Note: inside a tasklet you always have scalars, it is about he connector types
    Assignment Operations:
        ARRAY_ARRAY_ASSIGNMENT: Direct array-to-array copy (e.g., a = b)
        ARRAY_SYMBOL_ASSIGNMENT: Symbol/constant assignment to array (e.g., a = sym)
        ARRAY_SCALAR_ASSIGNMENT: Scalar variable assignment to array (e.g., a = scl)
        SCALAR_ARRAY_ASSIGNMENT: Array assignment to scalar variable (e.g., scl = a)
        SCALAR_SCALAR_ASSIGNMENT: Scalar assignment to scalar variable (e.g., scl = scl)

    Binary Operations with Arrays:
        ARRAY_SYMBOL: Array with symbol/constant (e.g., out = arr + 5, out = arr * N)
        ARRAY_SCALAR: Array with scalar variable (e.g., out = arr + scl)
        ARRAY_ARRAY: Two arrays (e.g., out = arr1 + arr2)

    Binary Operations with Scalars/Symbols:
        SCALAR_SYMBOL: Scalar with symbol/constant (e.g., out = scl + 5)
        SCALAR_SCALAR: Two scalars (e.g., out = scl1 + scl2)
        SYMBOL_SYMBOL: Two symbols (e.g., out = sym1 + sym2)

    Unary Operations:
        UNARY_ARRAY: Single array operand (e.g., out = abs(arr), out = arr * arr)
        UNARY_SCALAR: Single scalar operand (e.g., out = abs(scl), out = scl * scl)
        UNARY_SYMBOL: Single symbol operand (e.g., out = abs(sym), out = sym * sym)
    """
    ARRAY_ARRAY_ASSIGNMENT = "array_array_assignment"
    ARRAY_SYMBOL_ASSIGNMENT = "array_symbol_assignment"
    ARRAY_SCALAR_ASSIGNMENT = "array_scalar_assignment"
    SCALAR_ARRAY_ASSIGNMENT = "scalar_array_assignment"
    SCALAR_SCALAR_ASSIGNMENT = "scalar_scalar_assignment"
    SCALAR_SYMBOL_ASSIGNMENT = "scalar_symbol_assignment"
    SCALAR_SYMBOL = "scalar_symbol"
    ARRAY_SYMBOL = "array_symbol"
    ARRAY_SCALAR = "array_scalar"
    SCALAR_ARRAY = "scalar_array"
    ARRAY_ARRAY = "array_array"
    UNARY_ARRAY = "unary_array"
    UNARY_SYMBOL = "unary_symbol"
    UNARY_SCALAR = "unary_scalar"
    SCALAR_SCALAR = "scalar_scalar"
    SYMBOL_SYMBOL = "symbol_symbol"


def _token_split(string_to_check: str) -> Set[str]:
    """
    Splits a string into a set of tokens, keeping delimiters, and returns all tokens.
    The input string is split on empty space and brackets (` `, `(`, `)`, `[`, `]`).

    Parameters
    ----------
    string_to_check : str
        The string to split into tokens.
    pattern_str : str
        (Unused in this function, kept for consistency with token_match)

    Returns
    -------
    Set[str]
        The set of tokens extracted from the string.
    """
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens}

    return tokens


def replace_code(code_str: str,
                 code_lang: dace.dtypes.Language,
                 repldict: Dict[str, str],
                 py_only: bool = False,
                 use_sym_expr: bool = True) -> str:
    """
    Replaces variables in a code string according to a replacement dictionary.
    Supports Python symbolic substitution and fallback string-based replacement.

    Parameters
    ----------
    code_str : str
        The code string to modify.
    code_lang : dace.dtypes.Language
        The programming language of the code.
    repldict : Dict[str, str]
        Mapping from variable names to their replacements.
    py_only: bool
        Replace only if python tasklet
    Returns
    -------
    str
        The modified code string with replacements applied.
    """

    def _str_replace(lhs: str, rhs: str) -> str:
        code_str = token_replace_dict(rhs, repldict)
        return f"{lhs.strip()} = {code_str.strip()}"

    def _str_replace_nsplit(code_str: str) -> str:
        new_code_str = token_replace_dict(code_str, repldict)
        return new_code_str

    if code_lang == dace.dtypes.Language.Python and use_sym_expr:
        try:
            lhs, rhs = code_str.split(" = ")
            lhs = lhs.strip()
            rhs = rhs.strip()
        except Exception as e:
            try:
                new_rhs_sym_expr = dace.symbolic.SymExpr(code_str).subs(repldict)
                cleaned_expr = sympy.pycode(new_rhs_sym_expr, allow_unknown_functions=True).strip()
                return f"{cleaned_expr}"
            except Exception as e:
                return code_str
        try:
            new_rhs_sym_expr = dace.symbolic.SymExpr(rhs).subs(repldict)
            cleaned_expr = sympy.pycode(new_rhs_sym_expr, allow_unknown_functions=True).strip()
            return f"{lhs.strip()} = {cleaned_expr}"
        except Exception as e:
            return _str_replace(lhs, rhs)
    else:
        if not py_only:
            try:
                lhs, rhs = code_str.split(" = ")
                lhs = lhs.strip()
                rhs = rhs.strip()
                return _str_replace(lhs, rhs)
            except Exception as e:
                try:
                    return _str_replace_nsplit(code_str)
                except Exception as e:
                    return code_str
        else:
            assert False


def tasklet_replace_code(tasklet: dace.nodes.Tasklet,
                         repldict: Dict[str, str],
                         py_only: bool = True,
                         use_sym_expr: bool = True):
    """
    Replaces symbols in a tasklet's code according to a replacement dictionary.
    Updates the tasklet's code in place.

    Parameters
    ----------
    tasklet : dace.nodes.Tasklet
        The tasklet whose code to modify.
    repldict : Dict[str, str]
        Mapping from variable names to their replacements.

    Returns
    -------
    None
    """
    new_code = replace_code(tasklet.code.as_string,
                            tasklet.code.language,
                            repldict,
                            py_only=py_only,
                            use_sym_expr=use_sym_expr)
    tasklet.code = CodeBlock(code=new_code, language=tasklet.code.language)


def extract_bracket_tokens(s: str) -> list[tuple[str, list[str]]]:
    """
    Extracts all contents inside [...] along with the token before the '[' as the name.

    Args:
        s (str): Input string.

    Returns:
        List of tuples: [(name_token, string inside brackes)]
    """
    results = []

    # Pattern to match <name>[content_inside]
    pattern = re.compile(r'(\b\w+)\[([^\]]*?)\]')

    for match in pattern.finditer(s):
        name = match.group(1)  # token before '['
        content = match.group(2).split()  # split content inside brackets into tokens

        results.append((name, " ".join(content)))

    return {k: v for (k, v) in results}


def remove_bracket_tokens(s: str) -> str:
    """
    Removes all [...] patterns from the string.

    Args:
        s (str): Input string.

    Returns:
        str: String with all [...] removed.
    """
    return re.sub(r'\[.*?\]', '', s)


def _extract_constant_from_ast_str(src: str) -> str:
    """
    Extract a numeric constant from a Python code string using AST parsing.

    Supports both direct constants (e.g., 42, 3.14) and unary operations on constants
    (e.g., -5, +3.14). The function walks the AST tree to find constant nodes.

    Args:
        src: Python code string containing a constant (e.g., "x + 3.14" or "y - (-5)")

    Returns:
        String representation of the constant value

    Raises:
        ValueError: If no constant is found in the source string

    Examples:
        >>> _extract_constant_from_ast_str("x + 3.14")
        '3.14'
        >>> _extract_constant_from_ast_str("y + (-5)")
        '-5'
    """
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            if isinstance(node.op, ast.USub):
                return f"-{node.operand.value}"
            elif isinstance(node.op, ast.UAdd):
                return str(node.operand.value)

    raise ValueError("No constant found")


def _split_code_on_assignment(code_str: str) -> Tuple[str, str]:
    """
    Returns the LHS and RHS of the first assignment in a Python tasklet.

    Args:
        node: A Python tasklet node.

    Returns:
        A tuple (lhs_str, rhs_str) where both are strings representing
        the left-hand side and right-hand side of the first assignment.
    """
    # Parse the tasklet code into an AST
    code_ast = ast.parse(code_str)

    # Find the first assignment statement
    assign_node = next((n for n in code_ast.body if isinstance(n, ast.Assign)), None)
    if assign_node is None:
        raise ValueError("No assignment found in tasklet code.")

    # Convert LHS to string
    lhs_node = assign_node.targets[0]  # handle simple assignments only
    lhs_str = ast.unparse(lhs_node).strip()

    # Convert RHS to string
    rhs_node = assign_node.value
    rhs_str = ast.unparse(rhs_node).strip()

    assert isinstance(lhs_str, str)
    assert isinstance(rhs_str, str)

    return lhs_str, rhs_str


def _extract_non_connector_syms_from_tasklet(node: dace.nodes.Tasklet, state) -> typing.Set[str]:
    """
    Identify free symbols in tasklet code that are not input/output connectors.

    This function extracts all symbolic variables from the right-hand side of the tasklet's
    code expression and filters out those that correspond to input/output connectors,
    leaving only the actual free symbols (e.g., SDFG symbols or constants).

    Args:
        node: The tasklet node to analyze (must be a Python tasklet)

    Returns:
        Set of symbol names that appear in the code but are not connectors

    Examples:
        For a tasklet "out = in_a + N" with connectors {in_a, out}, this returns {"N"}
        For a tasklet "out = in_x * alpha + beta" with connectors {in_x, out}, this returns {"alpha", "beta"}

    Note:
        Requires the tasklet to use Python language and have valid symbolic expressions.
    """
    assert isinstance(node, dace.nodes.Tasklet)
    assert node.code.language == dace.dtypes.Language.Python
    connectors = {str(s) for s in set(node.in_connectors.keys()).union(set(node.out_connectors.keys()))}
    code_lhs, code_rhs = _split_code_on_assignment(node.code.as_string)
    code_rhs = code_rhs.replace("math.", "")
    all_syms = {
        str(s)
        for s in dace.symbolic.symbols_in_code(code_rhs,
                                               potential_symbols={str(s)
                                                                  for s in state.symbols_defined_at(node)})
    }
    real_free_syms = all_syms - connectors
    free_non_connector_syms = {str(s) for s in real_free_syms}
    return free_non_connector_syms


def _extract_non_connector_bound_syms_from_tasklet(code_str: str) -> typing.Set[str]:
    """
    Recursively extract all literal constants (numbers, strings, booleans, None)
    from a Python AST node.

    Args:
        node (ast.AST): The AST node or subtree to traverse.

    Returns:
        List of constants (int, float, str, bool, None, etc.)
    """
    constants = []
    node = ast.parse(code_str, mode="exec")

    class ConstantExtractor(ast.NodeVisitor):

        def visit_Constant(self, n):
            constants.append(n.value)

        # For compatibility with Python <3.8
        def visit_Num(self, n):  # type: ignore
            constants.append(n.n)

        def visit_Str(self, n):  # type: ignore
            constants.append(n.s)

        def visit_NameConstant(self, n):  # type: ignore
            constants.append(n.value)

    ConstantExtractor().visit(node)
    return {str(c) for c in constants}


_BINOP_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.And: "and",
    ast.Or: "or",
}
"""Mapping from AST binary operation nodes to their string representations."""

_UNARY_SYMBOLS = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
}
"""Mapping from AST unary operation nodes to their string representations."""

_CMP_SYMBOLS = {
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
"""Mapping from AST comparison operation nodes to their string representations."""

_SUPPORTED_OPS = {
    '*',
    '+',
    '-',
    '/',
    '>',
    '<',
    '>=',
    '<=',
    '==',
    '!=',
    'and',
    'or',
    'not',
}
"""Set of supported binary and comparison operators."""

_SUPPORTED = {
    '*',
    '+',
    '-',
    '/',
    '>',
    '<',
    '>=',
    '<=',
    '==',
    '!=',
    'abs',
    'exp',
    'sqrt',
    'log',
    'ln',
    'exp',
    'pow',
    'min',
    'max',
    'and',
    'or',
    'not',
}
"""Set of all supported operations including functions."""


def _extract_single_op(src: str, default_to_assignment: bool = False) -> str:
    """
    Extract the single supported operation from Python code.

    Parses the code string and identifies exactly one supported operation. The operation
    can be a binary operator (+, -, *, /), comparison operator (>, <, etc.), or a
    function call (abs, exp, sqrt, etc.).

    Args:
        src: Python code string (should be parseable into an AST) (e.g., "out = a + b" or "out = sqrt(x)")
        default_to_assignment: If True, return "=" when no operation is found;
                               if False, raise ValueError

    Returns:
        The operation symbol (e.g., "+", "*") or function name (e.g., "sqrt", "abs")

    Note:
        This function assumes tasklet contains a single operation.
        You can run the pass `SplitTasklets` to get such tasklets.
    """

    tree = ast.parse(src)
    found = None

    for node in ast.walk(tree):
        op = None

        if isinstance(node, ast.BinOp):
            op = _BINOP_SYMBOLS.get(type(node.op), None)
        elif isinstance(node, ast.UnaryOp):
            op = _UNARY_SYMBOLS.get(type(node.op), None)
        elif isinstance(node, ast.BoolOp):
            op = _BINOP_SYMBOLS.get(type(node.op), None)
        elif isinstance(node, ast.Compare):
            assert len(node.ops) == 1
            op = _CMP_SYMBOLS.get(type(node.ops[0]), None)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                op = node.func.id
            elif isinstance(node.func, ast.Attribute):
                op = node.func.attr

        if op is None:
            continue

        if op not in _SUPPORTED:
            print(f"Found unsupported op {op} in {src}")

        if found is not None:
            raise ValueError(
                f"More than one supported operation found, first found: {found}, second: {op} for tasklets {node} ({ast.unparse(node)})"
            )

        found = op

    code_rhs = src.split(" = ")[-1].strip()
    try:
        tree = ast.parse(code_rhs, mode="eval")
        call_node = tree.body
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
            found = func_name
    except SyntaxError as e:
        pass

    if found is None:
        if default_to_assignment is True:
            found = "="
        else:
            raise ValueError(f"No supported operation found for code_str: {src}")

    return found


def _match_connector_to_data(state: dace.SDFGState, tasklet: dace.nodes.Tasklet) -> Dict:
    """
    Map input connector names to their corresponding data descriptors.

    Creates a dictionary that maps each input connector of the tasklet to its
    associated data descriptor (array or scalar) by examining the incoming edges.

    Args:
        state: The SDFG state containing the tasklet
        tasklet: The tasklet node whose connectors to map

    Returns:
        Dictionary mapping connector names (str) to data descriptors (dace.data.Data)

    """
    tdict = dict()
    for ie in state.in_edges(tasklet):
        if ie.data is not None:
            tdict[ie.dst_conn] = state.sdfg.arrays[ie.data.data]

    return tdict


def _get_scalar_and_array_arguments(state: dace.SDFGState, tasklet: dace.nodes.Tasklet) -> Tuple[Set[str], Set[str]]:
    """
    Separate tasklet input connectors into scalars and arrays.

    Returns:
        Tuple of (scalar_connectors, array_connectors) where each is a set of connector names
    """
    tdict = _match_connector_to_data(state, tasklet)
    scalars = {k for k, v in tdict.items() if isinstance(v, dace.data.Scalar)}
    arrays = {k for k, v in tdict.items() if isinstance(v, dace.data.Array)}
    return scalars, arrays


def _reorder_rhs(code_str: str, op: str, rhs1: str, rhs2: str) -> Tuple[str, str]:
    """
    Determine the correct left-right ordering of operands based on their appearance in code.

    For binary operations, this function analyzes the code to determine which operand
    appears on the left side of the operator and which appears on the right. This is
    important for non-commutative operations like subtraction and division.

    Args:
        code_str: Full tasklet code string (e.g., "out = a - b")
        op: Operation symbol (e.g., "-", "*", "min")
        rhs1: First operand name
        rhs2: Second operand name

    Returns:
        Tuple of (left_operand, right_operand) in the order they appear in the code

    Note:
        For function calls, uses AST parsing to extract arguments in order.
        For operators, splits the code by the operator symbol.
    """
    code_rhs = code_str.split(" = ")[-1].strip()
    if op not in _SUPPORTED_OPS:
        try:
            tree = ast.parse(code_rhs, mode="eval")
            call_node = tree.body
            if not isinstance(call_node, ast.Call):
                raise ValueError(f"Expected a function call in expression: {code_rhs}")

            args = [ast.get_source_segment(code_rhs, arg).strip() for arg in call_node.args]
            left_string, right_string = args[0:2]
            assert len(args) == 2
        except SyntaxError as e:
            raise ValueError(f"Failed to parse function expression: {code_rhs}") from e
    else:
        left_string, right_string = [_token_split(cstr.strip()) for cstr in code_rhs.split(op)]

    if rhs1 in left_string and rhs2 in left_string:
        if rhs1 != rhs2:
            raise Exception(
                "SSA tasklet, rhs1 and rhs2 both can't appear on left side of the operand (unless they are the same and repeated)"
            )

    if rhs1 in right_string and rhs2 in right_string:
        if rhs1 != rhs2:
            raise Exception(
                "SSA tasklet, rhs1 and rhs2 both can't appear on right side of the operand (unless they are the same and repeated)"
            )

    if rhs1 in left_string and rhs2 in right_string:
        return rhs1, rhs2

    if rhs1 in right_string and rhs2 in left_string:
        return rhs2, rhs1

    if rhs1 not in left_string and rhs2 not in right_string:
        raise Exception(
            f"SSA tasklet, rhs1 appears in none of the substrings rhs1: {rhs1} string: {left_string} -op- {right_string}"
        )

    if rhs2 not in left_string and rhs2 not in right_string:
        raise Exception(
            f"SSA tasklet, rhs2 appears in none of the substrings, rhs2: {rhs1} string: {left_string} -op- {right_string}"
        )


def count_name_occurrences(expr: str, name: str) -> int:
    """
    Count how many times a given variable name appears in an expression.

    Uses AST parsing to accurately count variable name occurrences, distinguishing
    between actual variable references and other uses of the same string.

    Args:
        expr: Expression to parse (e.g., "a + b * a")
        name: Variable name to count (e.g., "a")

    Returns:
        Number of times the variable appears in the expression

    Examples:
        >>> count_name_occurrences("a + b * a", "a")
        2
        >>> count_name_occurrences("x * x * x", "x")
        3

    Note:
        This is used to distinguish between unary operations (single occurrence)
        and binary operations where the same operand appears twice (e.g., x * x).
    """
    tree = ast.parse(expr, mode="eval")
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            count += 1
    return count


def classify_tasklet(state: dace.SDFGState, node: dace.nodes.Tasklet) -> Dict:
    """
    Analyze a tasklet and return its classification with metadata.

    This is the main entry point for tasklet classification. It inspects the tasklet's
    code, input/output connectors, and data descriptors to determine the tasklet type
    and extract relevant metadata for code generation.

    Args:
        state: The SDFG state containing the tasklet
        node: The tasklet node to classify

    Returns:
        Dictionary with the following keys:
            - type (TaskletType): The classified tasklet type
            - lhs (str): Output connector name (left-hand side variable)
            - rhs1 (str or None):  Input connector/operand name left of the operator/first function argument
            - rhs2 (str or None): Input connector/operand name right of the operator/secpnd function argument
            - constant1 (str or None): First constant/symbol value left of the operator/first function argument
            - constant2 (str or None): Second constant/symbol value right of the operator/secpnd function argument
            - op (str): Operation symbol or function name

    Notes:
        - Left of the operator is c1 or rhs1 and right of the operator is c2 or rhs2, regardless of the number of constants or expressions

    Raises:
        AssertionError: If tasklet has more than 1 output connector
        NotImplementedError: If tasklet pattern is not supported
        ValueError: If code cannot be parsed or contains unsupported operations

    Classification Logic:
        (Output can be scalar / array)
        Single Input (n_in == 1):
            - Direct assignment: a = b
            - Array/scalar with constant: a = b + 5
            - Array/scalar with symbol: a = b * N
            - Unary operation: a = abs(b) or a = b * b

        Two Inputs (n_in == 2):
            - Two arrays: a = b + c
            - Array and scalar: a = b * scl
            - Two scalars: a = scl1 + scl2

        Zero Inputs (n_in == 0):
            - Symbol assignment: a = N
            - Two symbols: a = N + M
            - Unary symbol: a = abs(N)

    Examples:
        >>> # For tasklet "out = in_a + 5"
        >>> result = classify_tasklet(state, tasklet_node)
        >>> result
        {
            'type': TaskletType.ARRAY_SYMBOL,
            'lhs': 'out',
            'rhs1': 'in_a',
            'rhs2': None,
            'constant1': None,
            'constant2': '5',
            'op': '+'
        }
        # For more see the unit tests

    Constraints:
        - Tasklet must have exactly 1 output connector
        - Tasklet must use Python language
        - Code must contain at most one operation (See SplitTasklets pass to enforce this easily)
    """
    in_conns = list(node.in_connectors.keys())
    out_conns = list(node.out_connectors.keys())
    n_in = len(in_conns)
    n_out = len(out_conns)

    assert n_out <= 1, "Only support tasklets with at most 1 output in this pass"
    lhs = next(iter(node.out_connectors.keys())) if n_out == 1 else None

    assert isinstance(node, dace.nodes.Tasklet)
    code: CodeBlock = node.code
    assert code.language == dace.dtypes.Language.Python
    code_str: str = code.as_string

    info_dict = {"type": None, "lhs": lhs, "rhs1": None, "rhs2": None, "constant1": None, "constant2": None, "op": None}

    assert n_out == 1

    if n_in == 1:
        rhs = in_conns[0]
        in_edges = {ie for ie in state.in_edges_by_connector(node, rhs)}
        assert len(in_edges) == 1, f"expected 1 in-edge for connector {rhs}, found {len(in_edges)}"
        rhs_data_name = in_edges.pop().data.data
        rhs_data = state.sdfg.arrays[rhs_data_name]
        out_edges = {oe for oe in state.out_edges_by_connector(node, lhs)}
        assert len(out_edges) == 1, f"expected 1 out-edge for connector {lhs}, found {len(out_edges)}"
        lhs_data_name = out_edges.pop().data.data
        lhs_data = state.sdfg.arrays[lhs_data_name]

        # Assignment operators it will return op <- `=` and always populate `rhs1`
        if code_str == f"{lhs} = {rhs}" or code_str == f"{lhs} = {rhs};":
            lhs_datadesc = lhs_data
            rhs_datadesc = rhs_data
            ttype = None
            if isinstance(lhs_datadesc, dace.data.Array) and isinstance(rhs_datadesc, dace.data.Array):
                ttype = TaskletType.ARRAY_ARRAY_ASSIGNMENT
            elif isinstance(lhs_datadesc, dace.data.Array) and isinstance(rhs_datadesc, dace.data.Scalar):
                ttype = TaskletType.ARRAY_SCALAR_ASSIGNMENT
            elif isinstance(lhs_datadesc, dace.data.Scalar) and isinstance(rhs_datadesc, dace.data.Array):
                ttype = TaskletType.SCALAR_ARRAY_ASSIGNMENT
            elif isinstance(lhs_datadesc, dace.data.Scalar) and isinstance(rhs_datadesc, dace.data.Scalar):
                ttype = TaskletType.SCALAR_SCALAR_ASSIGNMENT
            else:
                raise ValueError(f"Unsupported Assignment Type {lhs_datadesc} <- {rhs_datadesc}")
            info_dict.update({"type": ttype, "op": "=", "rhs1": rhs})
            return info_dict

        has_constant = False
        constant = None
        try:
            constant = _extract_constant_from_ast_str(code_str)
            has_constant = True
        except Exception as e:
            has_constant = False

        free_non_connector_syms = _extract_non_connector_syms_from_tasklet(node, state)
        if len(free_non_connector_syms) == 1:
            has_constant = True
            constant = free_non_connector_syms.pop()

        if not has_constant:
            # If the rhs arrays appears repeatedly it means we have an operator like `a = b * b`
            # In case the occurence equaling two, repeat the `rhs` argument
            rhs_occurence_count = count_name_occurrences(code_str.split(" = ")[1].strip(), rhs)
            if isinstance(rhs_data, dace.data.Array):
                rhs2 = None if rhs_occurence_count == 1 else rhs
                ttype = TaskletType.UNARY_ARRAY if rhs_occurence_count == 1 else TaskletType.ARRAY_ARRAY
                info_dict.update({"type": ttype, "rhs1": rhs, "rhs2": rhs2, "op": _extract_single_op(code_str)})
                return info_dict
            elif isinstance(rhs_data, dace.data.Scalar):
                rhs2 = None if rhs_occurence_count == 1 else rhs
                ttype = TaskletType.UNARY_SCALAR if rhs_occurence_count == 1 else TaskletType.SCALAR_SCALAR
                info_dict.update({"type": ttype, "rhs1": rhs, "rhs2": rhs2, "op": _extract_single_op(code_str)})
                return info_dict
            else:
                raise Exception(f"Unhandled case in tasklet type (1) {rhs_data}, {type(rhs_data)}")
        else:
            # Handle the correct order, left-of the operand is `1` and right is `2`
            op = _extract_single_op(code_str)
            reordered = _reorder_rhs(code_str, op, rhs, constant)
            rhs1 = rhs if reordered[0] == rhs else None
            rhs2 = rhs if reordered[1] == rhs else None
            constant1 = constant if reordered[0] == constant else None
            constant2 = constant if reordered[1] == constant else None
            if isinstance(rhs_data, dace.data.Array):
                info_dict.update({
                    "type": TaskletType.ARRAY_SYMBOL,
                    "rhs1": rhs1,
                    "rhs2": rhs2,
                    "constant1": constant1,
                    "constant2": constant2,
                    "op": _extract_single_op(code_str)
                })
                return info_dict
            elif isinstance(rhs_data, dace.data.Scalar):
                info_dict.update({
                    "type": TaskletType.SCALAR_SYMBOL,
                    "rhs1": rhs1,
                    "rhs2": rhs2,
                    "constant1": constant1,
                    "constant2": constant2,
                    "op": _extract_single_op(code_str)
                })
                return info_dict
            else:
                raise Exception("Unhandled case in tasklet type (2) {rhs_data}, {type(rhs_data)}")

    elif n_in == 2:
        op = _extract_single_op(code_str)
        rhs1, rhs2 = in_conns[0], in_conns[1]
        rhs1, rhs2 = _reorder_rhs(code_str, op, rhs1, rhs2)

        lhs = next(iter(node.out_connectors.keys()))
        scalars, arrays = _get_scalar_and_array_arguments(state, node)
        assert len(scalars) + len(arrays) == 2

        if len(arrays) == 2 and len(scalars) == 0:
            info_dict.update({"type": TaskletType.ARRAY_ARRAY, "rhs1": rhs1, "rhs2": rhs2, "op": op})
            return info_dict
        elif len(scalars) == 1 and len(arrays) == 1:
            array_arg = next(iter(arrays))
            scalar_arg = next(iter(scalars))
            ttype = TaskletType.ARRAY_SCALAR if rhs1 == array_arg else TaskletType.SCALAR_ARRAY
            if ttype == TaskletType.ARRAY_SCALAR:
                assert rhs2 == scalar_arg
            else:
                assert rhs1 == scalar_arg
            assert rhs1 is not None
            assert rhs2 is not None
            info_dict.update({"type": ttype, "rhs1": rhs1, "rhs2": rhs2, "op": op})
            return info_dict
        elif len(scalars) == 2:
            info_dict.update({"type": TaskletType.SCALAR_SCALAR, "rhs1": rhs1, "rhs2": rhs2, "op": op})
            return info_dict
    elif n_in == 0:
        free_syms = _extract_non_connector_syms_from_tasklet(node, state)
        bound_syms = _extract_non_connector_bound_syms_from_tasklet(node.code.as_string)
        op = _extract_single_op(code_str, default_to_assignment=True)
        if len(free_syms) == 2:
            assert len(bound_syms) == 0
            free_sym1 = free_syms.pop()
            free_sym2 = free_syms.pop()
            free_sym1, free_sym2 = _reorder_rhs(code_str, op, free_sym1, free_sym2)
            info_dict.update({
                "type": TaskletType.SYMBOL_SYMBOL,
                "constant1": free_sym1,
                "constant2": free_sym2,
                "op": _extract_single_op(code_str)
            })
            return info_dict
        elif len(free_syms) == 1:
            if op == "=":
                free_sym1 = free_syms.pop()
                info_dict.update({"type": TaskletType.ARRAY_SYMBOL_ASSIGNMENT, "constant1": free_sym1, "op": "="})
                return info_dict
            else:
                free_sym1 = free_syms.pop()
                rhs_occurence_count = count_name_occurrences(code_str.split(" = ")[1].strip(), free_sym1)
                if rhs_occurence_count == 2:
                    assert len(bound_syms) == 0
                    c1, c2 = free_sym1, None
                    ttype = TaskletType.UNARY_SYMBOL
                else:
                    # It might be sym1 op 2.0 (constant literal, doesn't have to be 2.0)
                    # But also a function
                    assert len(bound_syms) <= 1
                    if len(bound_syms) == 1:
                        bound_sym1 = bound_syms.pop()
                        # Make sure order is correct
                        c1, c2 = _reorder_rhs(code_str, op, free_sym1, bound_sym1)
                        ttype = TaskletType.SYMBOL_SYMBOL
                    else:
                        c1, c2 = free_sym1, None
                        ttype = TaskletType.UNARY_SYMBOL

                info_dict.update({"type": ttype, "constant1": c1, "constant2": c2, "op": op})
                return info_dict
        else:
            if len(bound_syms) == 2:
                c1 = bound_syms.pop()
                c2 = bound_syms.pop()
                c1, c2 = _reorder_rhs(code_str, op, c1, c2)
                if c1 == c2:
                    ttype = TaskletType.UNARY_SYMBOL
                else:
                    ttype = TaskletType.SYMBOL_SYMBOL
                info_dict.update({"type": ttype, "constant1": c1, "constant2": c2, "op": op})
                return info_dict
            else:
                assert len(bound_syms) == 1
                # Could be a function call on a constant like `f(2.0)`, but also `= 0.0`
                if op == "=":
                    c1 = bound_syms.pop()
                    # Assignment operators it will return op <- `=` and always populate `rhs1`
                    if code_str == f"{lhs} = {c1}" or code_str == f"{lhs} = {c1};":
                        out_edges = {oe for oe in state.out_edges_by_connector(node, lhs)}
                        assert len(out_edges) == 1, f"expected 1 out-edge for connector {lhs}, found {len(out_edges)}"
                        lhs_data_name = out_edges.pop().data.data
                        lhs_data = state.sdfg.arrays[lhs_data_name]
                        lhs_datadesc = lhs_data
                        ttype = None
                        if isinstance(lhs_datadesc, dace.data.Array):
                            ttype = TaskletType.ARRAY_SYMBOL_ASSIGNMENT
                        elif isinstance(lhs_datadesc, dace.data.Scalar):
                            ttype = TaskletType.SCALAR_SYMBOL_ASSIGNMENT
                        else:
                            raise ValueError(f"Unsupported Assignment Type {lhs_datadesc} <- {c1}")
                        info_dict.update({"type": ttype, "op": "=", "constant1": c1})
                        return info_dict
                else:
                    c1 = bound_syms.pop()
                    ttype = TaskletType.UNARY_SYMBOL
                    info_dict.update({"type": ttype, "constant1": c1, "constant2": None, "op": op})
                    return info_dict

    raise NotImplementedError("Unhandled case in detect tasklet type")


import ast


class FuncToOp(ast.NodeTransformer):
    """
    Rewrite:
        or(a, b)  →  a or b
        and(a, b) →  a and b
        not(a)    →  not a
    """

    def visit_Call(self, node):
        self.generic_visit(node)

        # Only handle simple function names (not attributes)
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            # NOT
            if fname.lower() == "not" and len(node.args) == 1:
                return ast.UnaryOp(op=ast.Not(), operand=node.args[0])
            # OR
            if fname.lower() == "or" and len(node.args) == 2:
                return ast.BoolOp(op=ast.Or(), values=node.args)
            # AND
            if fname.lower() == "and" and len(node.args) == 2:
                return ast.BoolOp(op=ast.And(), values=node.args)

        return node


def rewrite_boolean_functions_to_boolean_ops(src: str) -> str:
    """Parse -> rewrite -> unparse"""
    tree = ast.parse(src)
    tree = FuncToOp().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

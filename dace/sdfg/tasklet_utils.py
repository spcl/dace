# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tasklet code utilities: whether a tasklet reads or assigns a name, and rewriting its code."""

import ast
import re

import sympy

import dace
from dace.properties import CodeBlock
from dace.sdfg.narrowing import as_basic

#: A name being assigned in a non-Python body: ``x =``, ``x +=``, ``x <<=``, but not ``x ==`` /
#: ``x !=`` / ``x <=`` / ``x >=``. Used where the body cannot be parsed, so it is deliberately
#: over-eager: reporting an assignment that is not one only trips a sanity check.
_NON_PYTHON_ASSIGNMENT = r"(?<![\w.]){name}\s*(?:[+\-*/%&|^]|<<|>>)?=(?!=)"


def token_replace_dict(code: str, repldict: dict[str, str]) -> str:
    """
    Replaces exact token matches in a code string using a replacement dictionary.
    Tokens are split using whitespace and common delimiters (` `, `(`, `)`, `[`, `]`).

    Parameters
    ----------
    code : str
        The code string in which to replace tokens.
    repldict : Dict[str, str]
        Mapping from token names to their replacement strings.

    Returns
    -------
    str
        The code string with replacements applied.
    """

    # Split while keeping delimiters
    tokens = re.split(r"(\s+|[()\[\]])", code)

    # Replace tokens that exactly match src
    tokens = [repldict[token.strip()] if token.strip() in repldict else token for token in tokens]

    # Recombine everything
    return "".join(tokens).strip()


def token_match(string_to_check: str, pattern_str: str) -> str:
    """
    Checks if a given pattern string exists as a token in the input string.
    The input string is split on empty space and brackets (` `, `(`, `)`, `[`, `]`).

    Parameters
    ----------
    string_to_check : str
        The string to search for the token.
    pattern_str : str
        The token to search for.

    Returns
    -------
    bool
        True if the token exists, False otherwise.
    """

    # Split while keeping delimiters
    tokens = re.split(r"(\s+|[()\[\]])", string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens}

    return pattern_str in tokens


def tasklet_assigns_name(tasklet: dace.nodes.Tasklet, name: str) -> bool:
    """Report whether ``tasklet``'s body assigns to ``name``.

    A tasklet body is not necessarily a single assignment -- it can hold several statements, or
    none at all (a guard tasklet is ``if s > 0: abort()``) -- so this reads the
    statements rather than splitting the source on ``" = "``.

    :param tasklet: The tasklet whose body to inspect.
    :param name: The name to look for on the left-hand side of an assignment.
    :returns: ``True`` if the body assigns ``name``.
    """
    code = tasklet.code.as_string
    if name not in code:
        return False

    if tasklet.code.language != dace.dtypes.Language.Python:
        return re.search(_NON_PYTHON_ASSIGNMENT.format(name=re.escape(name)), code) is not None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Not parseable as Python despite the language tag: fall back to the textual rule rather
        # than claiming the body assigns nothing.
        return re.search(_NON_PYTHON_ASSIGNMENT.format(name=re.escape(name)), code) is not None

    for node in ast.walk(tree):
        targets = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.NamedExpr)):
            targets = (node.target,)
        for target in targets:
            if any(isinstance(sub, ast.Name) and sub.id == name for sub in ast.walk(target)):
                return True
    return False


def tasklet_has_symbol(tasklet: dace.nodes.Tasklet, symbol_str: str) -> bool:
    """
    Checks if a symbol is present in a tasklet's code. Uses symbolic analysis of sympy.
    Checks functions and function arguments too.

    Parameters
    ----------
    tasklet : dace.nodes.Tasklet
        The tasklet whose code to inspect.
    symbol_str : str
        The symbol name to search for.

    Returns
    -------
    bool
        True if the symbol exists in the tasklet's code, False otherwise.
    """
    code = tasklet.code.as_string

    # Python tasklet case
    if tasklet.code.language == dace.dtypes.Language.Python:
        try:
            # Split into LHS and RHS (single assignment inside Python tasklet)
            lhs, rhs = tasklet.code.as_string.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Direct match: symbol is LHS variable
            if symbol_str == lhs:
                return True

            # Parse symbolic expression
            sym_expr = dace.symbolic.SymExpr(rhs)

            # Collect ALL symbols we find
            found = set()

            # 1. Free symbols (simple variables like a, b, i, N, etc.)
            found |= {str(s) for s in as_basic(sym_expr).free_symbols}

            # 2. Function names + symbols inside arguments
            for func in as_basic(sym_expr).atoms(sympy.Function):
                # Add function name
                found.add(str(func.func))

                # Add all symbols appearing in function arguments
                for arg in func.args:
                    # All symbols within this argument expression
                    found |= {str(s) for s in arg.free_symbols}

            return symbol_str in found

        except Exception:
            # Fallback to token-based match if symbolic parsing fails
            return token_match(code, symbol_str)

    # Non-Python tasklet (e.g., C++)
    else:
        return token_match(code, symbol_str)


def replace_code(
    code_str: str,
    code_lang: dace.dtypes.Language,
    repldict: dict[str, str],
    py_only: bool = False,
    use_sym_expr: bool = True,
) -> str:
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


def tasklet_replace_code(
    tasklet: dace.nodes.Tasklet, repldict: dict[str, str], py_only: bool = True, use_sym_expr: bool = True
):
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
    new_code = replace_code(
        tasklet.code.as_string, tasklet.code.language, repldict, py_only=py_only, use_sym_expr=use_sym_expr
    )
    tasklet.code = CodeBlock(code=new_code, language=tasklet.code.language)

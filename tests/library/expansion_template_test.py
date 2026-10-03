# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""No library expansion ships a ``{...}`` placeholder it never interpolated.

A library node builds its C++ as a Python string. Drop the ``f`` prefix, or hand a
``str.format_map`` template a field its mapping does not carry, and the brace text reaches the
emitted source verbatim; the node still expands and still generates code, and only the vendor
compiler ever complains. One mechanical cuBLAS-to-dialect refactor left six of these at once
(``gemv``, ``batched_matmul``, ``symv``, ``tensordot``, ``csrmv``, ``csrmm``), each fatal on both
backends and none visible to any test that stops at expansion.

This reads the sources rather than any one node's output, so a node with no GPU test -- which is
most of them -- is covered too.
"""
import ast
import pathlib
import re

import dace

#: A placeholder naming an expansion-local object. Literal braces in emitted C++ are block braces,
#: which carry no such name, so this does not fire on them.
PLACEHOLDER = re.compile(r'\{(cls|node|dialect|self)\b[^}]*\}')

LIBRARIES = pathlib.Path(dace.__file__).parent / 'libraries'


def formatted_literals(tree: ast.AST) -> set:
    """Literals that ARE interpolated, by being the receiver of ``.format``/``.format_map``."""
    receivers = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('format', 'format_map') and isinstance(node.func.value, ast.Constant)):
            receivers.add(id(node.func.value))
    return receivers


def uninterpolated(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text())
    formatted = formatted_literals(tree)
    return [(node.lineno, m.group(0)) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in formatted
            for m in PLACEHOLDER.finditer(node.value)]


def test_no_expansion_leaks_an_uninterpolated_placeholder() -> None:
    offenders = {
        str(p.relative_to(LIBRARIES)): hits
        for p in sorted(LIBRARIES.rglob('*.py')) if (hits := uninterpolated(p))
    }
    assert not offenders, ('these string literals carry a placeholder nothing interpolates, so the braces reach the '
                           f'emitted C++ verbatim: {offenders}. Add the missing f prefix, or put the field in the '
                           'mapping the template is formatted with.')

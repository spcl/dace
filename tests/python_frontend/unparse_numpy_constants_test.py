# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unparsing of NumPy scalar constants (NumPy 2 reprs such as ``np.True_`` are not valid in generated code)."""

import ast

import numpy as np

from dace.frontend.python import astutils


def _unparse_constant(value) -> str:
    tree = ast.parse("x = 0")
    tree.body[0].value = astutils.create_constant(value, tree.body[0].value)
    return astutils.unparse(tree).strip()


def test_unparse_numpy_bool():
    assert _unparse_constant(np.True_) == "x = True"
    assert _unparse_constant(np.False_) == "x = False"


def test_unparse_numpy_number():
    assert _unparse_constant(np.int32(3)) == "x = 3"
    assert _unparse_constant(np.float64(2.5)) == "x = 2.5"


if __name__ == "__main__":
    test_unparse_numpy_bool()
    test_unparse_numpy_number()

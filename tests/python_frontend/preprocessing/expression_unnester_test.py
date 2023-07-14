# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the preprocessing functionality of unnesting expressions. """

import ast
import inspect
import itertools
import numpy as np


from collections.abc import Callable
from dace.frontend.python.astutils import _remove_outer_indentation
from dace.frontend.python import preprocessing as pr
from numpy import typing as npt
from typing import Any


def _unnest(func: Callable[..., Any], expected_var_num: int) -> Callable[..., Any]:

    function_ast = ast.parse(_remove_outer_indentation(inspect.getsource(func)))

    name_getter = pr.NameGetter()
    name_getter.visit(function_ast)
    program_names = name_getter.names

    function_ast = pr.ParentSetter().visit(function_ast)
    unnester = pr.ExpressionUnnester(names=program_names)
    function_ast = unnester.visit(function_ast)
    for parent, attr, idx, node in reversed(unnester.ast_nodes_to_add):
        getattr(parent, attr).insert(idx, node)
    
    ast.fix_missing_locations(function_ast)
    print(ast.unparse(function_ast))

    _validate_unnesting(function_ast, expected_var_num)

    code = compile(function_ast, filename='<ast>', mode='exec')
    namespace = {**globals()}
    exec(code, namespace)
    unnested_function = namespace[func.__name__]

    return unnested_function


def _validate_unnesting(unnested_ast: ast.AST, expected_var_num: int) -> None:

    name_getter = pr.NameGetter()
    name_getter.visit(unnested_ast)
    program_names = name_getter.names

    for i in range(expected_var_num):
        name = f'__var_{i}'
        assert name in program_names
    assert f'__var_{expected_var_num}' not in program_names


def test_BoolOp():

    def original_function(a: bool, b: bool, c: bool, d: bool, e: bool) -> bool:
        e = (a and (b or False)) or (c and (d or e))
        return e
    
    new_function = _unnest(original_function, 4)

    for (a, b, c, d, e) in itertools.permutations([True, False], 5):
        assert original_function(a, b, c, d, e) == new_function(a, b, c, d, e)


def test_NamedExpr():

    def original_function(a: int) -> int:
        y = (x := a + 7) + x
        return y
    
    new_function = _unnest(original_function, 1)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10,))
    for a in randints:
        assert original_function(a) == new_function(a)


def test_BinOp():

    def original_function(a: int, b: int) -> int:
        c = ((a + b) * (a - b)) ** 2
        return c
    
    new_function = _unnest(original_function, 3)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_UnaryOp():

    def original_function(a: int, b: int) -> int:
        c = - (a + b)
        return b
    
    new_function = _unnest(original_function, 1)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Lambda():

    def original_function(a: int, b: int) -> int:
        f = lambda x: x + a * 2
        return f(b)
    
    new_function = _unnest(original_function, 0)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_IfExp():

    def original_function(a: int, b: int) -> int:
        c = a - b if a > b else b - a
        return c
    
    new_function = _unnest(original_function, 3)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Dict():

    def original_function(a: int, b: int) -> int:
        c = {a + b: a - b, a - b: a + b}
        return c
    
    new_function = _unnest(original_function, 4)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Set():
    
    def original_function(a: int, b: int) -> int:
        c = {a + b, a - b}
        return c
    
    new_function = _unnest(original_function, 2)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


if __name__ == '__main__':
    test_BoolOp()
    test_NamedExpr()
    test_BinOp()
    test_UnaryOp()
    test_Lambda()
    test_IfExp()
    test_Dict()
    test_Set()

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
from typing import Any, Dict, List, Tuple


def _unnest(func: Callable[..., Any], expected_var_num: int, context: Dict[str, Any] = None) -> Callable[..., Any]:

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
    context = context or {}
    namespace = {**globals(), **context}
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


def test_Await():

    import asyncio

    async def original_function(a: int) -> int:
        await asyncio.sleep(0.001 * a * 2)
        return a
    
    new_function = _unnest(original_function, 3, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(1, 10, size=(10,))
    for a in randints:
        assert asyncio.run(original_function(a)) == asyncio.run(new_function(a))


def test_Yield():

    def original_function(a: int) -> int:
        yield a + 3
        yield a + 2
        yield a + 1
    
    new_function = _unnest(original_function, 3)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10,))
    for a in randints:
        assert list(original_function(a)) == list(new_function(a))


def test_YieldFrom():

    def x(n: int) -> int:
        for i in range(n):
            yield i
    
    def y(n: int) -> int:
        for i in reversed(range(n)):
            yield i

    def original_function(n: int) -> int:
        yield from itertools.chain(x(n),  y(n))
    
    new_function = _unnest(original_function, 4, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(1, 10, size=(10,))
    for n in randints:
        assert list(original_function(n)) == list(new_function(n))


def test_Compare():

    def original_function(a: int, b: int) -> int:
        c = a + b > a > a - b > b
        return c
    
    new_function = _unnest(original_function, 2)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Call():

    def x(n: int) -> int:
        return 2 * n

    def original_function(a: int, b: int) -> int:
        c = x(a + b) * x(a - b)
        return c
    
    new_function = _unnest(original_function, 4, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_JoinedStr():
    # NOTE: Also tests FormattedValue

    def original_function(a: int) -> str:
        string = f'"sin({a}) is {np.sin(a):.3}"'
        return string
    
    print(original_function(5))

    new_function = _unnest(original_function, 2, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10,))
    for a in randints:
        assert original_function(a) == new_function(a)


def test_Attribute():

    def original_function(a: npt.NDArray[np.int32], b: int) -> int:
        c = (a + b).T.size
        return c
    
    new_function = _unnest(original_function, 2, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for s, b in randints:
        a = np.arange(s, dtype=np.int32)
        assert original_function(a, b) == new_function(a, b)

def test_Subscript():

    def original_function(a: npt.NDArray[np.int32], b: int) -> int:
        c = (a + b)[a[b - 1]]
        return c
    
    new_function = _unnest(original_function, 3, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(1, 100, size=(10,))
    for b in randints:
        a = np.arange(b, dtype=np.int32)
        assert original_function(a, b) == new_function(a, b)


def test_List():

    def original_function(a: int, b: int) -> List[int]:
        c = [a + b, a - b]
        return c
    
    new_function = _unnest(original_function, 2)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Tuple():

    def original_function(a: int, b: int) -> Tuple[int, int]:
        c = (a + b, a - b)
        return c
    
    new_function = _unnest(original_function, 2)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Slice():

    def original_function(a: npt.NDArray[np.int32], b: npt.NDArray[np.int32], c: npt.NDArray[np.int32]) -> int:
        c = a[b[c[0]:c[1]]]
        return c
    
    new_function = _unnest(original_function, 4, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(1, 100, size=(10,))
    for s in randints:
        a = np.arange(s, dtype=np.int32)
        b = np.arange(s, dtype=np.int32)
        n = rng.integers(0, s)
        c = np.array([n, min(s-1, n + 2)], dtype=np.int32)
        assert np.array_equal(original_function(a, b, c), new_function(a, b, c))


if __name__ == '__main__':
    test_BoolOp()
    test_NamedExpr()
    test_BinOp()
    test_UnaryOp()
    test_Lambda()
    test_IfExp()
    test_Dict()
    test_Set()
    test_Await()
    test_Yield()
    test_YieldFrom()
    test_Compare()
    test_Call()
    test_JoinedStr()
    test_Attribute()
    test_Subscript()
    test_List()
    test_Tuple()
    test_Slice()

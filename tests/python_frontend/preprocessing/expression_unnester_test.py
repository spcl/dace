# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the preprocessing functionality of unnesting expressions. """

import ast
import inspect
import itertools
import numpy as np
import warnings

from collections.abc import Callable
from dace.frontend.python.astutils import _remove_outer_indentation
from dace.frontend.python import preprocessing as pr
from dataclasses import dataclass, field
from numpy import typing as npt
from typing import Any, Dict, List, Tuple


##### Helper functions #####


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


##### Tests for Statements #####


def test_Return():

    def original_function(a: int, b: int) -> int:
        return a + b
    
    new_function = _unnest(original_function, 1)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_Delete():

    def original_function(a: Dict[int, Any], b: int) -> int:
        for i in range(b):
            del a[b - i]
            
    
    new_function = _unnest(original_function, 1)

    rng = np.random.default_rng(42)
    randints = rng.integers(10, 100, size=(10,))
    for s in randints:
        ref = {i: i for i in range(s)}
        val = {i: i for i in range(s)}
        original_function(ref, s - 1)
        new_function(val, s - 1)
        assert ref == val


def test_Assign():

    def original_function(a: int, b: int) -> int:
        c, d = a + b, a - b
        return c, d
    
    new_function = _unnest(original_function, 2)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_AugAssign():

    def original_function(a: int, b: int) -> int:
        a += b
        return a
    
    new_function = _unnest(original_function, 0)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_AnnAssign():

    def original_function(a: int, b: int) -> int:
        c: int = a + b
        return c
    
    new_function = _unnest(original_function, 0)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_For():

    def original_function(a: int, b: int) -> int:
        for i in range(b):
            a += i
        return a
    
    new_function = _unnest(original_function, 0)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_While():

    def original_function(a: int, b: int) -> int:
        while min(a, b) < b:
            a += 1
        return a
    
    new_function = _unnest(original_function, 0)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


def test_If():

    def original_function(a: int, b: int) -> int:
        if a < b:
            a += 1
        return a
    
    new_function = _unnest(original_function, 1)

    rng = np.random.default_rng(42)
    randints = rng.integers(-99, 100, size=(10, 2))
    for (a, b) in randints:
        assert original_function(a, b) == new_function(a, b)


##### Tests for Expressions #####


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
    
    new_function = _unnest(original_function, 1)

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


def test_Attribute_1():

    @dataclass
    class MyClass:
        a: npt.NDArray[np.int32]
        b: npt.NDArray[np.int32] = field(default=None, init=False)

    def original_function(arr: MyClass, b: int) -> int:
        arr.a += b
        arr.b = MyClass(arr.a)
        arr.b.b = MyClass(arr.a)
    
    new_function = _unnest(original_function, 3, locals())

    a_ref = MyClass(np.arange(10, dtype=np.int32))
    a_val = MyClass(np.arange(10, dtype=np.int32))

    original_function(a_ref, 1)
    new_function(a_val, 1)

    assert np.array_equal(a_ref.a, a_val.a)
    assert np.array_equal(a_ref.b.a, a_val.b.a)
    assert np.array_equal(a_ref.b.b.a, a_val.b.b.a)
    assert a_val.b.b.b is None


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


def test_Slice_1():

    def original_function(a: npt.NDArray[np.int32], b: npt.NDArray[np.int32], c: npt.NDArray[np.int32]):
        a[b[c[0]:c[1]]] = 1000
    
    new_function = _unnest(original_function, 4, locals())

    rng = np.random.default_rng(42)
    randints = rng.integers(1, 100, size=(10,))
    for s in randints:
        a_ref = np.arange(s, dtype=np.int32)
        a_val = a_ref.copy()
        b = np.arange(s, dtype=np.int32)
        n = rng.integers(0, s)
        c = np.array([n, min(s-1, n + 2)], dtype=np.int32)
        original_function(a_ref, b, c)
        new_function(a_val, b, c)
        assert np.array_equal(a_ref, a_val)


##### Mixed tests #####


def test_mixed():

    try:
        from scipy import sparse
    except ImportError:
        warnings.warn('Skipping mixed test, scipy not installed')
        return
    
    def original_function(A: sparse.csr_matrix, B: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        for i, j in itertools.product(range(A.shape[0]), range(B.shape[1])):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                C[i, j] += A.data[k] * B[A.indices[k], j]
        return C
    
    new_function = _unnest(original_function, 26, locals())

    rng = np.random.default_rng(42)
    for _ in range(10):
        A = sparse.random(20, 10, density=0.1, format='csr', dtype=np.float32, random_state=rng)
        B = rng.random((10, 5), dtype=np.float32)
        assert np.allclose(original_function(A, B), new_function(A, B))


def test_mixed_1():

    try:
        from scipy import sparse
    except ImportError:
        warnings.warn('Skipping mixed test, scipy not installed')
        return
    
    def original_function(A: List[sparse.csr_matrix], B: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        C = np.zeros((len(A), A[0].shape[0], B.shape[-1]), dtype=np.float32)
        for l, i, j in itertools.product(range(len(A)), range(A[0].shape[0]), range(B.shape[-1])):
            for k in range(A[l].indptr[i], A[l].indptr[i + 1]):
                C[l, i, j] += A[l].data[k] * B[l, A[l].indices[k], j]
        return C
    
    new_function = _unnest(original_function, 37, locals())

    rng = np.random.default_rng(42)
    for _ in range(10):
        A = [sparse.random(20, 10, density=0.1, format='csr', dtype=np.float32, random_state=rng) for _ in range(5)]
        B = rng.random((5, 10, 5), dtype=np.float32)
        assert np.allclose(original_function(A, B), new_function(A, B))


def test_mixed_2():

    try:
        from scipy import sparse
    except ImportError:
        warnings.warn('Skipping mixed test, scipy not installed')
        return
    
    def original_function(A: sparse.csr_matrix,
                          B: npt.NDArray[np.float32],
                          C: npt.NDArray[np.float32]) -> sparse.csr_matrix:
        D = sparse.csr_matrix((np.zeros(A.nnz, dtype=A.dtype), A.indices, A.indptr), shape=A.shape)
        for i in range(A.shape[0]):
            for j in range(A.indptr[i], A.indptr[i + 1]):
                for k in range(B.shape[1]):
                    D.data[j] += A.data[j] * B[i, k] * C[k, A.indices[j]]
        return D
    
    new_function = _unnest(original_function, 28, locals())

    rng = np.random.default_rng(42)
    for _ in range(10):
        A = sparse.random(20, 10, density=0.1, format='csr', dtype=np.float32, random_state=rng)
        B = rng.random((20, 5), dtype=np.float32)
        C = rng.random((5, 10), dtype=np.float32)
        assert np.allclose(original_function(A, B, C).todense(), new_function(A, B, C).todense())


def test_mixed_3():

    def match(b1: int, b2: int) -> int:
        if b1 + b2 == 3:
            return 1
        else:
            return 0

    def original_function(N: int, seq: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:

        table = np.zeros((N, N), np.int32)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N):
                if j - 1 >= 0:
                    table[i, j] = max(table[i, j], table[i, j - 1])
                if i + 1 < N:
                    table[i, j] = max(table[i, j], table[i + 1, j])
                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        table[i,
                            j] = max(table[i, j],
                                    table[i + 1, j - 1] + match(seq[i], seq[j]))
                    else:
                        table[i, j] = max(table[i, j], table[i + 1, j - 1])
                for k in range(i + 1, j):
                    table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

        return table

    new_function = _unnest(original_function, 58, locals())

    rng = np.random.default_rng(42)
    for _ in range(10):
        N = rng.integers(10, 20)
        seq = rng.integers(0, 4, N)
        assert np.allclose(original_function(N, seq), new_function(N, seq))


def test_mixed_4():

    # -----------------------------------------------------------------------------
    # From Numpy to Python
    # Copyright (2017) Nicolas P. Rougier - BSD license
    # More information at https://github.com/rougier/numpy-book
    # -----------------------------------------------------------------------------

    def original_function(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
        # Adapted from https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
        Xi, Yi = np.mgrid[0:xn, 0:yn]
        X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
        Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
        C = X + Y * 1j
        N_ = np.zeros(C.shape, dtype=np.int64)
        Z_ = np.zeros(C.shape, dtype=np.complex128)
        Xi.shape = Yi.shape = C.shape = xn * yn

        Z = np.zeros(C.shape, np.complex128)
        for i in range(itermax):
            if not len(Z):
                break

            # Compute for relevant points only
            np.multiply(Z, Z, Z)
            np.add(Z, C, Z)

            # Failed convergence
            I = abs(Z) > horizon
            N_[Xi[I], Yi[I]] = i + 1
            Z_[Xi[I], Yi[I]] = Z[I]

            # Keep going with those who have not diverged yet
            np.logical_not(I, I)  # np.negative(I, I) not working any longer
            Z = Z[I]
            Xi, Yi = Xi[I], Yi[I]
            C = C[I]
        return Z_.T, N_.T
    
    new_function = _unnest(original_function, 36, locals())

    rng = np.random.default_rng(42)
    for _ in range(10):
        xmin = rng.random()
        xmax = xmin + rng.random()
        ymin = rng.random()
        ymax = ymin + rng.random()
        xn = rng.integers(10, 20)
        yn = rng.integers(10, 20)
        itermax = rng.integers(10, 20)
        assert np.allclose(original_function(xmin, xmax, ymin, ymax, xn, yn, itermax)[0],
                           new_function(xmin, xmax, ymin, ymax, xn, yn, itermax)[0])


if __name__ == '__main__':
    test_Return()
    test_Delete()
    test_Assign()
    test_AugAssign()
    test_AnnAssign()
    test_For()
    test_While()
    test_If()
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
    test_Attribute_1()
    test_Subscript()
    test_List()
    test_Tuple()
    test_Slice()
    test_Slice_1()
    test_mixed()
    test_mixed_1()
    test_mixed_2()
    test_mixed_3()
    test_mixed_4()

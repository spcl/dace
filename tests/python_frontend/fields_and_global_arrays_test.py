# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests class fields and external arrays. """
import dace
from dace.frontend.python.common import DaceSyntaxError
import numpy as np
from dataclasses import dataclass
import pytest
import time


def test_bad_closure():
    """ 
    Testing functions that should not be in the closure (must be implemented as
    callbacks).
    """
    with pytest.raises(DaceSyntaxError):

        @dace.program
        def badprog(A: dace.float64[20]):
            # Library function that does not return the same value every time
            A[:] = time.time()

        A = np.random.rand(20)
        badprog(A)


def test_dynamic_closure():
    """ 
    Testing for function closure that was not defined before the program.
    """
    @dace.program
    def exttest_readonly():
        return A + 1

    A = np.random.rand(20)
    assert np.allclose(exttest_readonly(), A + 1)


def test_external_ndarray_readonly():
    A = np.random.rand(20)

    @dace.program
    def exttest_readonly():
        return A + 1

    assert np.allclose(exttest_readonly(), A + 1)


def test_external_ndarray_modify():
    A = np.random.rand(20)

    @dace.program
    def exttest_modify():
        A[:] = 1

    exttest_modify()
    assert np.allclose(A, 1)


def test_external_dataclass():
    @dataclass
    class MyObject:
        my_a: dace.float64[20]

    dc = MyObject(np.random.rand(20))

    @dace.program
    def exttest():
        dc.my_a[:] = 5

    exttest()
    assert np.allclose(dc.my_a, 5)


def test_dataclass_method():
    @dataclass
    class MyObject:
        my_a: dace.float64[20]

        def __init__(self) -> None:
            self.my_a = np.random.rand(20)

        @dace.method
        def something(self, B: dace.float64[20]):
            self.my_a += B

    dc = MyObject()
    acopy = np.copy(dc.my_a)
    b = np.random.rand(20)
    dc.something(b)
    assert np.allclose(dc.my_a, acopy + b)


def test_dataclass_method_cache():
    @dataclass
    class MyObject:
        my_a: dace.float64[20]

        def __init__(self) -> None:
            self.my_a = np.random.rand(20)

        @dace.method
        def something(self, B: dace.float64[20]):
            self.my_a += B

    dc = MyObject()
    acopy = np.copy(dc.my_a)
    b = np.random.rand(20)
    dc.something(b)
    dc.something(b)
    assert np.allclose(dc.my_a, acopy + b + b)


def test_dataclass_method_aot():
    """ AOT compilation of dataclass methods. """
    @dataclass
    class MyObject:
        my_a: dace.float64[20]

        def __init__(self) -> None:
            self.my_a = np.random.rand(20)

        @dace.method
        def something(self, B: dace.float64[20]):
            self.my_a += B

    dc = MyObject()
    csdfg = dc.something.compile()
    acopy = np.copy(dc.my_a)
    b = np.random.rand(20)
    csdfg(b, **dc.something.__sdfg_closure__())
    assert np.allclose(dc.my_a, acopy + b)


def test_object_method():
    """ JIT-based inference of fields at call time. """
    class MyObject:
        def __init__(self) -> None:
            self.my_a = np.random.rand(20)

        @dace.method
        def something(self, B: dace.float64[20]):
            self.my_a += B

    obj = MyObject()
    acopy = np.copy(obj.my_a)
    b = np.random.rand(20)
    obj.something(b)
    assert np.allclose(obj.my_a, acopy + b)


def test_object_newfield():
    # This syntax (adding new fields at dace.method runtime) is disallowed
    with pytest.raises(DaceSyntaxError):

        class MyObject:
            @dace.method
            def something(self, B: dace.float64[20]):
                self.my_newfield = B

        obj = MyObject()
        b = np.random.rand(20)
        obj.something(b)
        assert np.allclose(obj.my_newfield, b)


def test_object_constant():
    class MyObject:
        q: dace.constant

        def __init__(self) -> None:
            self.q = 5

        @dace.method
        def something(self, B: dace.float64[20]):
            return B + self.q

    obj = MyObject()
    A = np.random.rand(20)
    B = obj.something(A)
    assert np.allclose(B, A + 5)

    # Ensure constant was folded
    assert 'q' not in obj.something.to_sdfg().generate_code()[0].clean_code


def test_external_cache():
    """ 
    If data descriptor changes from compile time to call time, warn and 
    recompile.
    """
    A = np.random.rand(20)

    @dace.program
    def plusglobal(B):
        return A + B

    B = np.random.rand(20)
    assert np.allclose(plusglobal(B), A + B)

    # Now modify the global
    A = np.random.rand(30)
    B = np.random.rand(30)
    assert np.allclose(plusglobal(B), A + B)


def test_nested_objects():
    """ Multiple objects with multiple "self" values and same field names. """
    class ObjA:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)

        @dace.method
        def nested(self, A):
            return A + self.q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja.nested(A)

    A = np.random.rand(20)
    obj = ObjB(5)
    expected = A + obj.q + A + (obj.q * 2)

    result = obj.outer(A)
    assert np.allclose(expected, result)


def test_nested_constants():
    class ObjA:
        def __init__(self, q) -> None:
            self.q = q

        @dace.method
        def nested(self, A):
            return A + self.q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = q
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja.nested(A)

    A = np.random.rand(20)
    obj = ObjB(5)
    expected = A + obj.q + A + (obj.q * 2)

    result = obj.outer(A)
    assert np.allclose(expected, result)


def test_nested_object_access():
    class ObjA:
        def __init__(self, q) -> None:
            self.q = q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = q
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja.q

    A = np.random.rand(20)
    obj = ObjB(5)
    expected = A + obj.q + (obj.q * 2)

    result = obj.outer(A)
    assert np.allclose(expected, result)


def test_same_field_different_classes():
    """ 
    Testing for correctness in the existence of the same object in multiple
    contexts.
    """
    class A:
        def __init__(self, arr) -> None:
            self.arr = arr

    class B(A):
        def __init__(self, arr) -> None:
            super().__init__(arr)
            self.arr2 = arr

        @dace.method
        def mymethod(self, A):
            self.arr[:] = 1
            self.arr2[:] = A

    field = np.random.rand(20)
    param = np.random.rand(20)
    obj = B(field)
    obj.mymethod(param)

    # Ensure only one array was created
    assert len(next(iter(obj.mymethod._cache.cache.keys())).closure_types) == 1

    assert np.allclose(obj.arr, param)


def test_object_methods_ref_across_methods():
    """ 
    JIT-based inference of fields at call time, same attribute used on 
    different levels of nesting.
    """
    class MyObject:
        def __init__(self) -> None:
            self.my_a = np.random.rand(20)
            self.my_b = np.random.rand(20)

        @dace.method
        def something_else(self):
            self.my_b[...] = self.my_a

        @dace.method
        def something(self, B: dace.float64[20]):
            self.my_a += B
            self.something_else()

    obj = MyObject()
    acopy = np.copy(obj.my_a)
    b = np.random.rand(20)
    obj.something(b)
    assert np.allclose(obj.my_a, acopy + b)
    assert np.allclose(obj.my_a, obj.my_b)


def test_nested_objects_call():
    class ObjA:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)

        @dace.method
        def __call__(self, A):
            return A + self.q

    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.full([20], q)
            self.obja = ObjA(q * 2)

        @dace.method
        def outer(self, A):
            return A + self.q + self.obja(A)

    A = np.random.rand(20)
    obj = ObjB(5)
    expected = A + obj.q + A + (obj.q * 2)
    result = obj.outer(A)
    assert np.allclose(expected, result)


class MyObjA:
    @dace.method
    def method_a(self, A):
        A[...] = 1.0 + A

    @dace.method
    def method_b(self, B):
        B[...] = 2.0

    @dace.method
    def switch(self, A, B, C):
        if C is None:
            self.method_b(A)
        else:
            self.method_a(B)


class MyObjB:
    def __init__(self) -> None:
        self.obja = MyObjA()

    @dace.method
    def arg_none_explicit(self, A, B):
        self.obja.switch(A, B, None)

    @dace.method
    def arg_field(self, A, B, C):
        self.obja.switch(A, B, C)


def test_arg_none_explicit():
    A = np.random.rand(20)
    B = np.random.rand(20)
    obj = MyObjB()
    expected = np.empty_like(A)
    expected[...] = 2.0
    obj.arg_none_explicit(A, B)
    assert np.allclose(expected, A)


def test_arg_field():
    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.random.rand(20)
    obj = MyObjB()
    expected = np.empty_like(A)
    expected[...] = 1.0 + B
    obj.arg_field(A, B, C)
    assert np.allclose(expected, B)


def test_nested_methods_different_inner_objects():
    class ObjA:
        def __init__(self, key):
            self.key = key

        @dace.method
        def method(self, A):
            if self.key == "1":
                A[...] = A + 1.0
            if self.key == "2":
                A[...] = A + 2.0

    class ObjB:
        def __init__(self) -> None:
            self.obja1 = ObjA("1")
            self.obja2 = ObjA("2")

        @dace.method
        def call_them_both(self, A):
            self.obja1.method(A)
            self.obja2.method(A)
            return A

    A = np.zeros(20)
    obj = ObjB()
    expected = A + 1.0 + 2.0
    result = obj.call_them_both(A)
    assert np.allclose(expected, result)


def test_constant_closure_cache():
    class Obj:
        def __init__(self, q) -> None:
            self.q = q

        @dace.method
        def __call__(self, A):
            return A + self.q

    obj = Obj(2)
    A = np.random.rand(20)
    expected = A + obj.q
    assert np.allclose(obj(A), expected)

    assert len(obj.__call__._cache.cache) == 1

    # Should not replace cache
    expected = A + obj.q
    assert np.allclose(obj(A), expected)
    assert len(obj.__call__._cache.cache) == 1

    obj.q = 5
    expected = A + obj.q
    assert np.allclose(obj(A), expected)
    if obj.__call__._cache.size > 1:
        assert len(obj.__call__._cache.cache) == 2


def test_constant_closure_cache_nested():
    class ObjB:
        def __init__(self, q) -> None:
            self.q = q

    class ObjA:
        def __init__(self, obj) -> None:
            self.obj = obj

        @dace.method
        def __call__(self, A):
            return A + self.obj.q

    obj = ObjA(ObjB(2))
    A = np.random.rand(20)
    expected = A + obj.obj.q
    assert np.allclose(obj(A), expected)

    obj.obj.q = 5
    expected = A + obj.obj.q
    assert np.allclose(obj(A), expected)
    if obj.__call__._cache.size > 1:
        assert len(obj.__call__._cache.cache) == 2


def test_array_closure_cache():
    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.random.rand(q)

    class ObjA:
        def __init__(self, obj) -> None:
            self.obj = obj

        @dace.method
        def __call__(self, A):
            return A + self.obj.q[0]

    obj = ObjA(ObjB(2))
    A = np.random.rand(20)
    expected = A + obj.obj.q[0]
    assert np.allclose(obj(A), expected)

    obj.obj.q = np.random.rand(5)
    expected = A + obj.obj.q[0]
    assert np.allclose(obj(A), expected)
    if obj.__call__._cache.size > 1:
        assert len(obj.__call__._cache.cache) == 2


def test_array_closure_cache_nested():
    class ObjB:
        def __init__(self, q) -> None:
            self.q = np.random.rand(20)

        @dace.method
        def nested(self, A):
            return A + self.q

    class ObjA:
        def __init__(self, obj) -> None:
            self.obj = obj

        @dace.method
        def __call__(self, A):
            return self.obj.nested(A)

    obj = ObjA(ObjB(2))
    A = np.random.rand(20)
    expected = A + obj.obj.q
    assert np.allclose(obj(A), expected)


if __name__ == '__main__':
    test_bad_closure()
    test_dynamic_closure()
    test_external_ndarray_readonly()
    test_external_ndarray_modify()
    test_external_dataclass()
    test_dataclass_method()
    test_dataclass_method_cache()
    test_dataclass_method_aot()
    test_object_method()
    test_object_newfield()
    test_object_constant()
    test_external_cache()
    test_nested_objects()
    test_nested_constants()
    test_nested_object_access()
    test_same_field_different_classes()
    test_object_methods_ref_across_methods()
    test_nested_objects_call()
    test_arg_none_explicit()
    test_arg_field()
    test_nested_methods_different_inner_objects()
    test_constant_closure_cache()
    test_constant_closure_cache_nested()
    test_array_closure_cache()
    test_array_closure_cache_nested()

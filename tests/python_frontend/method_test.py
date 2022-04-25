# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests dace.program as class methods """
import dace
import numpy as np
import sys


class MyTestClass:
    """ Test class with various values, lifetimes, and call types. """
    classvalue = 2

    def __init__(self, n=5) -> None:
        self.n = n

    @dace.method
    def method_jit(self, A):
        return A + self.n

    @dace.method
    def method(self, A: dace.float64[20]):
        return A + self.n

    @dace.method
    def __call__(self, A: dace.float64[20]):
        return A * self.n

    @dace.method
    def other_method_caller(self, A: dace.float64[20]):
        return self.method(A) + 2 + self(A)

    @staticmethod
    @dace.program
    def static(A: dace.float64[20]):
        return A + A

    @staticmethod
    @dace.program
    def static_withclass(A: dace.float64[20]):
        return A + MyTestClass.classvalue

    @classmethod
    @dace.method
    def clsmethod(cls, A):
        return A + cls.classvalue


class MyTestCallAttributesClass:
    class SDFGMethodTestClass:
        def __sdfg__(self, *args, **kwargs):
            @dace.program
            def call(A):
                A[:] = 7.0

            return call.__sdfg__(*args)

        def __sdfg_signature__(self):
            return ['A'], []

    def __init__(self, n=5) -> None:
        self.n = n
        self.call_me = MyTestCallAttributesClass.SDFGMethodTestClass()

    @dace.method
    def method_jit(self, A):
        self.call_me(A)
        return A + self.n

    @dace.method
    def __call__(self, A):
        self.call_me(A)
        return A * self.n

    @dace.method
    def method(self, A: dace.float64[20]):
        self.call_me(A)
        return A + self.n

    @dace.method
    def method_jit_with_scalar_arg(self, A, b):
        self.call_me(A)
        return A + b


def test_method_jit():
    A = np.random.rand(20)
    cls = MyTestClass(10)
    assert np.allclose(cls.method_jit(A), A + 10)


def test_method():
    A = np.random.rand(20)
    cls = MyTestClass(10)
    assert np.allclose(cls.method(A), A + 10)


def test_method_cache():
    A = np.random.rand(20)
    cls1 = MyTestClass(10)
    cls2 = MyTestClass(11)
    assert np.allclose(cls1.method(A), A + 10)
    assert np.allclose(cls1.method(A), A + 10)
    assert np.allclose(cls2.method(A), A + 11)


def test_callable():
    A = np.random.rand(20)
    cls = MyTestClass(12)
    assert np.allclose(cls(A), A * 12)


def test_static():
    A = np.random.rand(20)
    assert np.allclose(MyTestClass.static(A), A + A)


def test_static_withclass():
    A = np.random.rand(20)
    # TODO(later): Make cache strict w.r.t. globals and locals used in program
    # assert np.allclose(MyTestClass.static_withclass(A), A + 2)
    # Modify value
    MyTestClass.classvalue = 3
    assert np.allclose(MyTestClass.static_withclass(A), A + 3)


def test_classmethod():
    # Only available in Python 3.9+
    if sys.version_info >= (3, 9):
        A = np.random.rand(20)
        # Modify value first
        MyTestClass.classvalue = 4
        assert np.allclose(MyTestClass.clsmethod(A), A + 4)


def test_nested_methods():
    A = np.random.rand(20)
    cls = MyTestClass()
    assert np.allclose(cls.other_method_caller(A), (A * 5) + (A + 5) + 2)


def mydec(a):
    def mutator(func):
        dp = dace.program(func)

        @dace.program
        def mmm(A: dace.float64[20]):
            res = dp(A, a)
            return res

        sdfg = mmm.to_sdfg()
        return sdfg

    return mutator


def someprog(A: dace.float64[20], a: dace.float64):
    res = A + a
    return res


def someprog_indirection(a):
    return mydec(a)(someprog)


def test_decorator():
    @dace.program(constant_functions=True)
    def otherprog(A: dace.float64[20]):
        res = np.empty_like(A)
        someprog_indirection(3)(A=A, __return=res)
        return res

    sdfg = otherprog.to_sdfg()
    A = np.random.rand(20)
    assert np.allclose(sdfg(A), A + 3)


def test_sdfgattr_method_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(10)
    assert np.allclose(cls.method_jit(A), 17)


def test_sdfgattr_callable_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(12)
    assert np.allclose(cls(A), 84)


def test_sdfgattr_method_annotated_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(14)
    assert np.allclose(cls.method(A), 21)


def test_sdfgattr_method_jit_with_scalar():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(10)
    assert np.allclose(cls.method_jit_with_scalar_arg(A, 2.0), 9.0)


def test_nested_field_in_map():
    class B:
        def __init__(self) -> None:
            self.field = np.random.rand(10, 10)

        @dace.method
        def callee(self):
            return self.field[1, 1]

    class A:
        def __init__(self, nested: B):
            self.nested = nested

        @dace.method
        def tester(self):
            val = np.ndarray([2], np.float64)
            for i in dace.map[0:2]:
               val[i] = self.nested.callee()
            return val

    obj = A(B())
    result = obj.tester()

    assert np.allclose(result, np.array([obj.nested.field[1, 1], obj.nested.field[1, 1]]))


if __name__ == '__main__':
    test_method_jit()
    test_method()
    test_method_cache()
    test_callable()
    test_static()
    test_static_withclass()
    test_classmethod()
    test_nested_methods()
    test_decorator()
    test_sdfgattr_method_jit()
    test_sdfgattr_callable_jit()
    test_sdfgattr_method_annotated_jit()
    test_sdfgattr_method_jit_with_scalar()
    test_nested_field_in_map()

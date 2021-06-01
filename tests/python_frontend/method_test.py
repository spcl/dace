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


if __name__ == '__main__':
    test_method_jit()
    test_method()
    test_method_cache()
    test_callable()
    test_static()
    test_static_withclass()
    test_classmethod()

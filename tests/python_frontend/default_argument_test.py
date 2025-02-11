# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for using default arguments (mutable or not) in different contexts. """
import dace
import numpy as np


def test_default_arg():
    @dace.program
    def tester(arr: dace.float64[20], qmin: float = 0.0):
        arr[:] = qmin

    myarr = np.random.rand(20)
    tester(myarr)
    assert np.allclose(myarr, 0.0)
    tester(myarr, 2.0)
    assert np.allclose(myarr, 2.0)


def test_single_nested_default_arg_jit():
    class MyClass:
        def __call__(self, arr, qmin=0.0):
            arr[:] = qmin

    a = MyClass()

    @dace.program
    def tester(arr, qmin2):
        a(arr)

    myarr = np.random.rand(20)
    tester(myarr, 2.0)
    assert np.allclose(myarr, 0.0)


def test_nested_default_arg_jit():
    class MyClass:
        def __call__(self, arr, qmin=0.0):
            self.nested(arr, qmin)

        def nested(self, arr, qmin):
            arr[:] = qmin

    a = MyClass()

    @dace.program
    def tester(arr, qmin2):
        a(arr)

    myarr = np.random.rand(20)
    tester(myarr, 2.0)
    assert np.allclose(myarr, 0.0)


def test_nested_default_arg():
    class MyClass:
        def __call__(self, arr: dace.float64[20], qmin: float = 0.0):
            self.nested(arr, qmin)

        def nested(self, arr: dace.float64[20], qmin: float):
            arr[:] = qmin

    a = MyClass()

    @dace.program
    def tester(arr: dace.float64[20], qmin2: float):
        a(arr)

    myarr = np.random.rand(20)
    tester(myarr, 2.0)
    assert np.allclose(myarr, 0.0)


def test_nested_default_arg_reuse():
    class MyClass:
        def __call__(self, arr: dace.float64[20], qmin: float = 0.0):
            self.nested(arr, qmin)

        def nested(self, arr: dace.float64[20], qmin: float):
            arr[:] = qmin

    a = MyClass()

    @dace.program
    def tester(arr: dace.float64[20], qmin: float):
        a(arr)

    myarr = np.random.rand(20)
    tester(myarr, 2.0)
    assert np.allclose(myarr, 0.0)


def test_nested_default_arg_reuse_2():
    class MyClass:
        def __call__(self, arr: dace.float64[20], qmin: float = 0.0):
            self.nested(arr, qmin)

        def nested(self, arr: dace.float64[20], qmin: float):
            arr[:] = qmin

    a = MyClass()

    @dace.program
    def tester(arr: dace.float64[20], arr2: dace.float64[20], qmin: float):
        a(arr, qmin=1.0)
        a(arr2)

    myarr = np.random.rand(20)
    myarr2 = np.random.rand(20)
    tester(myarr, myarr2, 2.0)
    assert np.allclose(myarr, 1.0)
    assert np.allclose(myarr2, 0.0)


def test_default_arg_object():
    @dace.program
    def tester(arr: dace.float64[20], defarg: dace.float64[20] = np.ones(20)):
        defarg += 1
        arr[:] = defarg

    myarr = np.random.rand(20)
    b = np.full(20, 5.0)
    tester(myarr)
    assert np.allclose(myarr, 2.0)
    tester(myarr, b)
    assert np.allclose(myarr, 6.0)
    tester(myarr)
    assert np.allclose(myarr, 3.0)


def test_nested_default_arg_object():
    class MyClass:
        def __call__(self, arr: dace.float64[20], defarg: dace.float64[20] = np.ones(20)):
            defarg += 1
            arr[:] = defarg

    a = MyClass()

    @dace.program
    def tester(arr: dace.float64[20]):
        a(arr)

    myarr = np.random.rand(20)
    tester(myarr)
    assert np.allclose(myarr, 2.0)
    tester(myarr)
    assert np.allclose(myarr, 3.0)


if __name__ == '__main__':
    test_default_arg()
    test_single_nested_default_arg_jit()
    test_nested_default_arg_jit()
    test_nested_default_arg()
    test_nested_default_arg_reuse()
    test_nested_default_arg_reuse_2()
    test_default_arg_object()
    test_nested_default_arg_object()

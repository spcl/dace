# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests automatic detection and parsing of nested functions and methods that are
not annotated with @dace decorators.
"""
import dace
from dace.frontend.python.common import DaceSyntaxError, SDFGConvertible
from dataclasses import dataclass
import numpy as np
import pytest
from typing import List, Tuple


@dataclass
class SomeClass:
    q: float

    def method(self, a):
        return a * self.q

    def __call__(self, a):
        return self.method(a)


def freefunction(A):
    return A + 1


def test_autodetect_function():
    """
    Tests auto-detection of parsable free functions in the Python frontend.
    """

    @dace
    def adf(A):
        return freefunction(A)

    A = np.random.rand(20)
    B = adf(A)
    assert np.allclose(B, A + 1)


def test_autodetect_method():
    obj = SomeClass(0.5)

    @dace
    def adm(A):
        return obj.method(A)

    A = np.random.rand(20)
    B = adm(A)
    assert np.allclose(B, A / 2)


def test_autodetect_callable_object():
    obj = SomeClass(0.5)

    @dace
    def adco(A):
        return obj(A)

    A = np.random.rand(20)
    B = adco(A)
    assert np.allclose(B, A / 2)


def test_nested_function_method():

    @dataclass
    class TestClass:
        some_field: int

        def some_method(self, q):
            return q * self.some_field

    obj = TestClass(5)

    def nested(a):
        return a + 1 + obj.some_method(a)

    @dace
    def nfm(a: dace.float64[20]):
        return nested(a)

    A = np.random.rand(20)
    ref = nfm.f(A)
    daceres = nfm(A)
    assert np.allclose(ref, daceres)


def test_function_that_needs_replacement():

    @dace
    def notworking(a: dace.float64[20]):
        return np.allclose(a, a)

    A = np.random.rand(20)
    with dace.config.set_temporary('frontend', 'typed_callbacks_only', value=True):
        with pytest.raises(DaceSyntaxError):
            with pytest.warns(match="Automatically creating callback"):
                notworking(A)


@pytest.mark.parametrize('typed_callbacks', (False, True))
def test_nested_autoparse(typed_callbacks):

    def notworking_nested(a):
        return np.allclose(a, a)

    @dace
    def notworking2(a: dace.float64[20]):
        return notworking_nested(a)

    A = np.random.rand(20)

    with dace.config.set_temporary('frontend', 'typed_callbacks_only', value=typed_callbacks):
        if typed_callbacks:
            with pytest.raises(DaceSyntaxError, match='numpy.allclose'):
                with pytest.warns(match="Automatically creating callback"):
                    notworking2(A)
        else:
            with pytest.warns(match="Automatically creating callback"):
                with pytest.warns(match="Cannot infer return type"):
                    result = notworking2(A)
            assert result is True


def test_nested_recursion_fail():

    def nested_a(a):
        if a[0] < -2.0:
            return a[0]
        a[0] -= 1.0
        return nested_a(a)

    @dace
    def recursive_autoparse(a: dace.float64[20]):
        return nested_a(a)

    A = np.random.rand(20)
    with dace.config.set_temporary('frontend', 'typed_callbacks_only', value=True):
        with pytest.raises(DaceSyntaxError, match='nested_a'):
            with pytest.warns(match="due to recursion"):
                recursive_autoparse(A)


def test_nested_recursion2_fail():

    def nested_a(a):
        a[0] -= 1.0
        return nested_b(a)

    def nested_b(a):
        if a[0] < -2.0:
            return a[0]
        return nested_a(a)

    @dace
    def recursive_autoparse(a: dace.float64[20]):
        return nested_a(a)

    A = np.random.rand(20)
    with dace.config.set_temporary('frontend', 'typed_callbacks_only', value=True):
        with pytest.raises(DaceSyntaxError, match='nested_'):
            with pytest.warns(match="due to recursion"):
                recursive_autoparse(A)


def test_nested_autoparse_dec_fail():

    def some_decorator(f):
        return f

    @some_decorator
    def notworking_nested(a):
        return a

    @dace
    def notworking3(a: dace.float64[20]):
        return notworking_nested(a)

    A = np.random.rand(20)
    with dace.config.set_temporary('frontend', 'typed_callbacks_only', value=True):
        with pytest.raises(DaceSyntaxError, match='notworking_nested'):
            with pytest.warns(match="Automatically creating callback"):
                notworking3(A)


def freefunction2(A):
    A += 2


def test_autodetect_function_in_for():
    """
    Tests auto-detection of parsable free functions in a for loop.
    """

    @dace
    def adff(A):
        for _ in range(5):
            freefunction2(A)

    A = np.random.rand(20)
    ref = np.copy(A)
    adff(A)
    assert np.allclose(A, ref + 2 * 5)


def test_error_handling():

    class NotConvertible(SDFGConvertible):

        def __call__(self, a):
            import numpy as np
            print('A very pythonic method', a)

        def __sdfg__(self, *args, **kwargs):
            # Raise a special type of error that does not naturally occur in dace
            raise NotADirectoryError('I am not really convertible')

        def __sdfg_signature__(self):
            return ([], [])

    A = np.random.rand(20)

    with pytest.raises(NotADirectoryError):

        @dace.program
        def testprogram(A, nc: dace.compiletime):
            nc(A)

        testprogram(A, NotConvertible())


def test_nested_class_error_handling():

    def not_convertible(f):

        class NotConvertibleMethod(SDFGConvertible):

            def __sdfg__(self, *args, **kwargs):
                # Raise a special type of error that does not naturally occur in dace
                raise NotADirectoryError('I am not really convertible')

            def __sdfg_signature__(self):
                return ([], [])

        return NotConvertibleMethod()

    class MaybeConvertible:

        @not_convertible
        def __call__(self, a):
            import numpy as np
            print('A very pythonic method', a)

    A = np.random.rand(20)

    with pytest.raises(NotADirectoryError):

        @dace.program
        def testprogram(A, nc: dace.compiletime):
            nc(A)

        testprogram(A, MaybeConvertible())


def test_loop_unrolling():

    @dace.program
    def called(A):
        A += 1

    n = 5

    @dace.program
    def program(A):
        for i in dace.unroll(range(n)):
            called(A)

    A = np.random.rand(20)
    expected = A + n
    program(A)
    assert np.allclose(A, expected)


def test_type_hints_in_nested_call():
    """
    Tests that type hints are correctly propagated to nested functions, ignoring
    existing type hints if the nested function is not decorated.
    """

    def nested(a: int, b: List[float], c) -> Tuple[float, float]:
        return np.sum(b) + a, c

    @dace
    def outer(a: dace.float64[20], result: dace.float64[2]):
        ret1, ret2 = nested(5, a, 3.0)
        result[0] = ret1
        result[1] = ret2

    A = np.random.rand(20)
    res = np.zeros(2)
    ref = np.copy(res)
    ref[0] = np.sum(A) + 5
    ref[1] = 3.0
    outer(A, res)
    assert np.allclose(res, ref)


@pytest.mark.parametrize('decorated', (False, True))
def test_explicit_type_hints_in_nested_call(decorated):
    """
    Tests that type hints are not ignored if the nested function is decorated.
    """

    if decorated:

        @dace
        def nested(a: dace.float64[20], b: dace.float64[16]):
            b += a
    else:
        # This function is not decorated, so the type hints should be ignored
        def nested(a: dace.float64[20], b: dace.float64[16]):
            b += a

    @dace
    def outer(a: dace.float64[20]):
        nested(a, a)

    A = np.random.rand(20)
    a_ref = A * 2

    if decorated:
        with pytest.raises(IndexError):
            outer(A)
    else:
        outer(A)
        assert np.allclose(A, a_ref)


if __name__ == '__main__':
    test_autodetect_function()
    test_autodetect_method()
    test_autodetect_callable_object()
    test_nested_function_method()
    test_function_that_needs_replacement()
    test_nested_autoparse(False)
    test_nested_autoparse(True)
    test_nested_recursion_fail()
    test_nested_recursion2_fail()
    test_nested_autoparse_dec_fail()
    test_autodetect_function_in_for()
    test_error_handling()
    test_nested_class_error_handling()
    test_loop_unrolling()
    test_type_hints_in_nested_call()
    test_explicit_type_hints_in_nested_call(False)
    test_explicit_type_hints_in_nested_call(True)

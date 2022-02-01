# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests loop unrolling functionality. """
import dace
from dace.frontend.python import astutils
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.preprocessing import LoopUnroller, DaceSyntaxError
import numpy as np
import pytest


def test_native_unroll():
    """ Tests that unrolling functionality works. """
    a = 0
    for i in dace.unroll(range(2, 4)):
        a += i * i

    assert a == 13


def test_dace_unroll():
    """ Tests that unrolling functionality works within DaCe programs. """
    @dace.program
    def tounroll(A: dace.float64[1]):
        for i in dace.unroll(range(1, 4)):
            A[0] += i * i

    src_ast, fname, _, _ = astutils.function_to_ast(tounroll.f)
    lu = LoopUnroller(tounroll.global_vars, fname)
    unrolled = lu.visit(src_ast)
    assert len(unrolled.body[0].body) == 3

    a = np.zeros([1])
    tounroll(a)
    assert a[0] == 14


def test_dace_unroll_multistatement():
    """ Tests unrolling functionality with multiple statements. """
    @dace.program
    def tounroll(A: dace.float64[1]):
        for i in dace.unroll(range(1, 4)):
            A[0] += i * i
            if i in (3, ):
                A[0] += 2

    src_ast, fname, _, _ = astutils.function_to_ast(tounroll.f)
    lu = LoopUnroller(tounroll.global_vars, fname)
    unrolled = lu.visit(src_ast)
    assert len(unrolled.body[0].body) == 6

    a = np.zeros([1])
    tounroll(a)
    assert a[0] == 16


def test_dace_unroll_break():
    """ Tests unrolling functionality with control flow statements. """
    @dace.program
    def tounroll(A: dace.float64[1]):
        for i in dace.unroll(range(1, 4)):
            A[0] += i * i
            if i in (2, 3):
                break

    src_ast, fname, _, _ = astutils.function_to_ast(tounroll.f)
    lu = LoopUnroller(tounroll.global_vars, fname)
    with pytest.raises(DaceSyntaxError):
        unrolled = lu.visit(src_ast)


def test_dace_unroll_generator():
    """
    Tests that dace does not unroll arbitrary generators by default, but does
    so if explicitly defined with dace.unroll.
    """
    def mygenerator():
        for i in range(5):
            yield i * i

    a = np.zeros([1])

    with pytest.raises(DaceSyntaxError):

        @dace.program
        def tounroll_fail(A: dace.float64[1]):
            for val in mygenerator():
                A += val

        tounroll_fail(a)

    @dace.program
    def tounroll(A: dace.float64[1]):
        for val in dace.unroll(mygenerator()):
            A += val

    tounroll(a)
    assert a[0] == 30


def test_auto_unroll_tuple():
    """ Tests that unrolling functionality works automatically on tuples. """
    @dace.program
    def tounroll(A: dace.float64[1], B: dace.float64[2], C: dace.float64[1]):
        for arr in (A, B[1], C, B[0]):
            arr += 5

    a = np.zeros([1])
    b = np.zeros([2])
    c = np.zeros([1])
    tounroll(a, b, c)
    assert a[0] == 5
    assert b[0] == 5
    assert b[1] == 5
    assert c[0] == 5


def test_auto_unroll_dictionary():
    """
    Tests that unrolling functionality works automatically on dictionaries.
    """
    @dace.program
    def tounroll(A: dace.float64[1], d: dace.constant):
        for val in d:
            A += val

    a = np.zeros([1])
    d = {1: 2, 3: 4}
    tounroll(a, d)
    assert a[0] == 4


def test_auto_unroll_dictionary_method():
    """
    Tests that unrolling functionality works automatically on dict methods.
    """
    @dace.program
    def tounroll(A: dace.float64[1], d: dace.constant):
        for val in d.values():
            A += val

    a = np.zeros([1])
    d = {1: 2, 3: 4}
    tounroll(a, d)
    assert a[0] == 6


# Raise error if ndarray is the generator and dace.unroll was not specified
def test_ndarray_generator():
    @dace.program
    def tounroll(A: dace.float64[1], values: dace.float64[5]):
        for val in values:
            A += val

    a = np.zeros([1])
    v = np.random.rand(5)
    with pytest.raises(DaceSyntaxError):
        tounroll(a, v)
        assert a[0] == np.sum(v)


def test_tuple_elements_enumerate():
    @dace.program
    def tounroll(A: dace.float64[3]):
        for i, val in enumerate([1, 2, 3]):
            A[i] += val

    a = np.zeros([3])
    tounroll(a)
    assert np.allclose(a, np.array([1, 2, 3]))


def test_tuple_elements_zip():
    a1 = [2, 3, 4]
    a2 = (4, 5, 6)

    @dace.program
    def tounroll(A: dace.float64[1]):
        for a, b in zip(a1, a2):
            A += 2 * a + b

    a = np.zeros([1])
    tounroll(a)
    assert np.allclose(a, (2 + 3 + 4) * 2 + (4 + 5 + 6))


@pytest.mark.parametrize('thres', [-1, 0, 5])
def test_unroll_threshold(thres):
    with dace.config.set_temporary('frontend', 'unroll_threshold', value=thres):

        @dace.program
        def tounroll(A: dace.float64[10]):
            for i in range(6, 10):
                A[i] = i
            for j in range(6):
                A[j] = j + 1

        sdfg = tounroll.to_sdfg()
        if thres < 0:
            assert 'i' in sdfg.symbols and 'j' in sdfg.symbols
        elif thres == 0:
            assert 'i' not in sdfg.symbols and 'j' not in sdfg.symbols
        elif thres == 5:
            assert 'i' not in sdfg.symbols and 'j' in sdfg.symbols

        A = np.random.rand(10)
        ref = np.copy(A)
        ref[0:6] = np.arange(1, 7)
        ref[6:10] = np.arange(6, 10)

        sdfg(A)

        assert np.allclose(A, ref)


def test_deepcopy():
    class Nocopy(SDFGConvertible):
        def __sdfg__(self, *args, **kwargs):
            @dace
            def bla(a: dace.float64[20]):
                return a

            return bla.to_sdfg()

        def __sdfg_closure__(self, reevaluate=None):
            return {}

        def __sdfg_signature__(self):
            return [['a'], []]

        def __deepcopy__(self, memo):
            raise ValueError('DO NOT COPY ME PLEASE')

    nocopy = Nocopy()

    @dace.program
    def someprogram(a):
        for i in dace.unroll(range(3)):
            a += i * nocopy(a)

    b = np.random.rand(20)
    expected = 6 * b
    someprogram(b)
    assert np.allclose(b, expected)


def test_arrays_keys():
    class Wrapper:
        def __init__(self) -> None:
            self._an_array = np.ones((12), np.float64)

        def __str__(self) -> str:
            return f"I am an array {self._an_array}"

        def __repr__(self) -> str:
            return self.__str__()

        @property
        def arr(self):
            return self._an_array
    d = {'0a0': Wrapper(), '1b1': Wrapper()}
    expected = {'0a0': d['0a0'].arr + 1, '1b1': d['1b1'].arr + 1}

    @dace.program
    def prog():
        for arr in d.keys():
            d[arr].arr += 1

    prog()
    assert np.allclose(d['0a0'].arr, expected['0a0'])
    assert np.allclose(d['1b1'].arr, expected['1b1'])


if __name__ == '__main__':
    test_native_unroll()
    test_dace_unroll()
    test_dace_unroll_multistatement()
    test_dace_unroll_break()
    test_dace_unroll_generator()
    test_auto_unroll_tuple()
    test_auto_unroll_dictionary()
    test_auto_unroll_dictionary_method()
    test_ndarray_generator()
    test_tuple_elements_enumerate()
    test_tuple_elements_zip()
    test_unroll_threshold(-1)
    test_unroll_threshold(0)
    test_unroll_threshold(5)
    test_deepcopy()
    test_arrays_keys()

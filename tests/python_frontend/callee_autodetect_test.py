# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests automatic detection and parsing of nested functions and methods that are
not annotated with @dace decorators.
"""
import dace
from dace.frontend.python.common import DaceSyntaxError
from dataclasses import dataclass
import numpy as np
import pytest


@dataclass
class TestClass:
    some_field: int

    def some_method(self, q):
        return q * self.some_field


def test_nested_function_method():
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
    with pytest.raises(DaceSyntaxError):
        notworking(A)


if __name__ == '__main__':
    test_nested_function_method()
    test_function_that_needs_replacement()

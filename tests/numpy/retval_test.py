# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

# The nanobind interface forbids a caller-provided `__return` buffer by default
# (it is opt-in via compiler.nanobind_allow_return_override); this test asserts
# the always-allow ctypes behavior, so it is ctypes-only. The nanobind behavior
# (both config states) is covered in tests/codegen/nanobind_interface_test.py.
skip_return_override_on_nanobind = pytest.mark.skipif(
    dace.Config.get('compiler', 'interface') == 'nanobind',
    reason='nanobind forbids a caller-provided __return buffer by default (opt-in)')


@dace.program
def oneret(A: dace.float64[20]):
    return A * 2


def test_oneret():
    A = np.random.rand(20)
    result = oneret(A)
    assert np.allclose(result, A * 2)


@dace.program
def multiret(A: dace.float64[20]):
    return A * 3, A * 4, A


def test_multiret():
    A = np.random.rand(20)
    result = multiret(A)
    assert np.allclose(result[0], A * 3)
    assert np.allclose(result[1], A * 4)
    assert np.allclose(result[2], A)


@dace.program
def nested_ret(A: dace.float64[20]):
    return oneret(A) + 1


def test_nested_ret():
    A = np.random.rand(20)
    result = nested_ret(A)
    assert np.allclose(result, A * 2 + 1)


@skip_return_override_on_nanobind
def test_return_override():
    A = np.random.rand(20)
    result = np.random.rand(20)
    result2 = oneret(A, __return=result)
    assert np.allclose(result, A * 2)
    assert np.allclose(result2, A * 2)


if __name__ == '__main__':
    test_oneret()
    test_multiret()
    test_nested_ret()
    test_return_override()

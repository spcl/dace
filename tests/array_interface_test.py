# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace


@pytest.fixture
def ctypes_interface(monkeypatch):
    """Pins ``compiler.interface`` to ctypes for tests that assert ctypes-specific behavior.

    Under the default ``auto`` these SDFGs would select the nanobind interface,
    where the asserted behavior differs: nanobind ndarray arguments accept numpy/DLPack only (no __array_interface__-style coercion).
    The ``DACE_compiler_interface`` env var overrides ``set_temporary``, so it
    is dropped first.
    """
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    with dace.config.set_temporary('compiler', 'interface', value='ctypes'):
        yield


class ArrayWrapper:

    def __init__(self, array, **kwargs):
        self.array = array

    @property
    def __array_interface__(self):
        return self.array.__array_interface__


@pytest.mark.usefixtures('ctypes_interface')
def test_array_interface_input():

    @dace.program
    def simple_program(A: dace.float64[3, 3, 3]):
        A += 1

    simple_program.compile()

    A = np.ones((3, 3, 3))
    Awrap = ArrayWrapper(A)

    simple_program(A=Awrap)

    np.testing.assert_equal(A, 2)


if __name__ == "__main__":
    test_array_interface_input()

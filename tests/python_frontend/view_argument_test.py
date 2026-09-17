# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


@pytest.fixture
def ctypes_interface(monkeypatch):
    """Pins ``compiler.interface`` to ctypes for tests that assert ctypes-specific behavior.

    Under the default ``auto`` these SDFGs would select the nanobind interface,
    where the asserted behavior differs: nanobind passes views zero-copy via DLPack by design; compiler.allow_view_arguments is a
    ctypes-marshalling concept.
    The ``DACE_compiler_interface`` env var overrides ``set_temporary``, so it
    is dropped first.
    """
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    with dace.config.set_temporary('compiler', 'interface', value='ctypes'):
        yield


@dace.program
def viewtest(A: dace.float64[20, 20]):
    return A + 1


@pytest.mark.usefixtures('ctypes_interface')
def test_view_argument():
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=False):
        with pytest.raises(TypeError):
            A = np.random.rand(20, 20)
            viewtest(A.T)


def test_view_argument_override():
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        A = np.random.rand(40, 20)
        result = viewtest(A[20:, :])
        assert np.allclose(result, A[20:, :] + 1)


if __name__ == '__main__':
    test_view_argument()
    test_view_argument_override()

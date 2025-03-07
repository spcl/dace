# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

##################
# Lists
from dace.frontend.python.common import DaceSyntaxError

global_axes = [0, 2, 1]


@dace
def global_list_program(A: dace.int32[3, 2, 4]):
    return np.transpose(A, axes=global_axes)


def test_global_func_access_global_list():
    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = global_list_program(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=global_axes))


def test_local_func_access_global_list():

    @dace
    def local_list_program(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=global_axes)

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list_program(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=global_axes))


def test_local_list():
    local_axes = [1, 2, 0]

    @dace
    def local_list(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=local_axes)

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=local_axes))


@pytest.mark.skip('Syntax is not yet supported')
def test_local_list_with_slice():
    local_axes = [1, 2, 0, 100]

    @dace
    def local_list(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=local_axes[0:-1])

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=local_axes[0:-1]))


def test_local_list_with_symbols():
    N = dace.symbol('N')
    local_shape = [N, 4]

    @dace
    def local_list(A: dace.int32[N, 2, 4]):
        result = dace.define_local(local_shape, dace.int32)
        result[:] = np.sum(A, axis=1)
        return result

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list(A=inp.copy())
    assert np.allclose(result, np.sum(inp.copy(), axis=1))


def test_local_list_nested_lists():
    N = dace.symbol('N')
    local_shape = [[N], 4]

    @dace
    def local_list(A: dace.int32[N, 2, 4]):
        result = dace.define_local(local_shape, dace.int32)
        result[:] = np.sum(A, axis=1)
        return result

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)

    with pytest.raises(TypeError):
        result = local_list(A=inp.copy())


if __name__ == "__main__":
    test_global_func_access_global_list()
    test_local_func_access_global_list()
    test_local_list()
    # test_local_list_with_slice()
    test_local_list_with_symbols()
    test_local_list_nested_lists()

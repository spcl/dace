# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

##################
# Lists
global_axes = [0, 2, 1]


@dace
def global_list_program(A: dace.int32[3, 2, 4]):
    return np.transpose(A, axes=global_axes)


def global_func_access_global_list_test():
    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = global_list_program(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=global_axes))


def local_func_access_global_list_test():
    @dace
    def local_list_program(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=global_axes)

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list_program(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=global_axes))


def local_list_test():
    local_axes = [1, 2, 0]

    @dace
    def local_list(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=local_axes)

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=local_axes))


def local_list_test_with_slice():
    local_axes = [1, 2, 0, 100]

    @dace
    def local_list(A: dace.int32[3, 2, 4]):
        return np.transpose(A, axes=local_axes[0:-2])

    inp = np.random.randint(0, 10, (3, 2, 4)).astype(np.int32)
    result = local_list(A=inp.copy())
    assert np.allclose(result, np.transpose(inp.copy(), axes=local_axes))


if __name__ == "__main__":
    local_func_access_global_list_test()
    global_func_access_global_list_test()
    local_list_test()

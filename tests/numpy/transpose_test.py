# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from common import compare_numpy_output
from dace.library import change_default
import dace.libraries.blas as blas

M, N = 24, 24


@dace.program
def transpose_test(A: dace.float32[M, N], B: dace.float32[M, N]):
    B[:] = np.transpose(A)


@compare_numpy_output()
def test_transpose_axes0(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 2, 0])


@compare_numpy_output()
def test_transpose_axes1(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 1, 0, 2])


@compare_numpy_output()
def test_transpose_axes2(A: dace.float32[10, 5, 3, 2]):
    return np.transpose(A, axes=[3, 0, 2])


def test_transpose():
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros([M, N], dtype=np.float32)
    transpose_test(A, B)

    realB = np.transpose(A)
    rel_error = np.linalg.norm(B - realB) / np.linalg.norm(realB)
    print('Relative_error:', rel_error)
    assert rel_error <= 1e-5


@pytest.mark.parametrize('implementation', ['pure', 'cuTENSOR'])
def test_transpose_libnode(implementation):
    axes = [1, 0, 2]
    axis_sizes = [4, 2, 3]

    @dace.program
    def fn(A, B):
        B[:] = np.transpose(A, axes=axes)

    with change_default(blas, implementation):
        permuted_sizes = [axis_sizes[i] for i in axes]
        x = np.arange(np.prod(axis_sizes)).reshape(axis_sizes).astype(np.float32)
        y = np.zeros(permuted_sizes).astype(np.float32)

        sdfg = fn.to_sdfg(x, y)
        if implementation == 'cuTENSOR':
            sdfg.apply_gpu_transformations()
        sdfg.simplify()
        sdfg.expand_library_nodes()

        sdfg = sdfg.compile()
        sdfg(A=x, B=y)

        ref = np.transpose(x, axes=axes)
        print(ref)
        print(y)
        assert np.allclose(ref, y), "Result doesn't match reference!"


if __name__ == '__main__':
    test_transpose_axes0()
    test_transpose_axes1()
    test_transpose_axes2()
    test_transpose()
    test_transpose_libnode('pure')
    test_transpose_libnode('cuTENSOR')

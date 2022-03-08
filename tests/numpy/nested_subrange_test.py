# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def nested_subrange(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    for i, j in dace.map[0:3, 0:4]:
        tmp = A[:, i, j]
        B[i, j] = tmp[0] + tmp[1]


@dace.program
def subrange1(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    B[:] = A[0, :, :] + A[1, :, :]


@dace.program
def subrange2(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    tmp0 = A[0, :, :]
    tmp1 = A[1, :, :]
    for i, j in dace.map[0:3, 0:4]:
        B[i, j] = tmp0[i, j] + tmp1[i, j]


@dace.program
def subrange3(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    B[:] = A[0] + A[1, :, :]


@dace.program
def subrange_of_subrange(A: dace.float32[2, 3, 4, 5], B: dace.float32[4]):
    B[:] = A[:, 0, :, 0][0, :]


@dace.program
def subrange_of_subrange_nested(A: dace.float32[2, 3, 4, 5], B: dace.float32[4]):
    for i, j, k in dace.map[0:3, 0:5, 0:2]:
        B[:] = A[:, i, :, j][k, :]


def onetest(program):
    A = np.random.rand(2, 3, 4).astype(np.float32)
    expected = A[0, :, :] + A[1, :, :]
    B = np.random.rand(3, 4).astype(np.float32)

    sdfg = program.to_sdfg()
    sdfg(A=A, B=B)

    diff = np.linalg.norm(expected - B)
    print('Difference:', diff)
    assert diff < 1e-5


def onetest_subrange_of_subrange(program):
    A = np.random.rand(2, 3, 4, 5).astype(np.float32)
    expected = A[0, 0, :, 0]
    for i in range(2):
        for j in range(3):
            for k in range(5):
                A[i, j, :, k] = expected

    B = np.random.rand(4).astype(np.float32)

    sdfg = program.to_sdfg()
    sdfg(A=A, B=B)

    diff = np.linalg.norm(expected - B)
    print('Difference:', diff)
    assert diff < 1e-5


def test_nested_subrange():
    onetest(nested_subrange)


def test_subrange1():
    onetest(subrange1)


def test_subrange2():
    onetest(subrange2)


def test_subrange3():
    onetest(subrange3)


def test_subrange_of_subrange():
    onetest_subrange_of_subrange(subrange_of_subrange)


def test_subrange_of_subrange_nested():
    onetest_subrange_of_subrange(subrange_of_subrange_nested)


if __name__ == '__main__':
    test_nested_subrange()
    test_subrange1()
    test_subrange2()
    test_subrange3()
    test_subrange_of_subrange()
    test_subrange_of_subrange_nested()

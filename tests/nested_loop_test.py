# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def nested_loop_test(A: dace.int32[1]):
    for i in range(11):
        for j in range(5):
            with dace.tasklet:
                in_a << A[0]
                out_a >> A[0]
                out_a = in_a + 1


def test_loop_loop_indirect():
    @dace.program
    def loop_with_value(A: dace.float64[20, 20], ind: dace.int64[20]):
        for i in range(20):
            for j in range(ind[i]):
                A[i, j] = j

    A = np.random.rand(20, 20)
    ind = np.random.randint(low=0, high=19, size=(20, ), dtype=np.int64)
    expected = A.copy()

    loop_with_value(A, ind)

    loop_with_value.f(expected, ind)
    assert np.allclose(A, expected)


def test_map_loop_indirect():
    @dace.program
    def loop_with_value(A: dace.float64[20, 20], ind: dace.int64[20]):
        for i in dace.map[0:20]:
            for j in range(ind[i]):
                A[i, j] = j

    A = np.random.rand(20, 20)
    ind = np.random.randint(low=0, high=19, size=(20, ), dtype=np.int64)
    expected = A.copy()

    loop_with_value(A, ind)

    loop_with_value.f(expected, ind)
    assert np.allclose(A, expected)


def test_map_loop_indirect_2():
    @dace.program
    def loop_with_value(A: dace.float64[20, 20], ind: dace.int64[20]):
        for i in dace.map[0:20]:
            for j in range(ind[i], ind[i] + 1):
                A[i, j] = j

    A = np.random.rand(20, 20)
    ind = np.random.randint(low=0, high=19, size=(20, ), dtype=np.int64)
    expected = A.copy()

    loop_with_value(A, ind)

    loop_with_value.f(expected, ind)
    assert np.allclose(A, expected)


def test_map_map_indirect():
    @dace.program
    def loop_with_value(A: dace.float64[20, 20], ind: dace.int64[20]):
        for i in dace.map[0:20]:
            for j in dace.map[0:ind[i]]:
                A[i, j] = j

    A = np.random.rand(20, 20)
    ind = np.random.randint(low=0, high=19, size=(20, ), dtype=np.int64)
    expected = A.copy()

    loop_with_value(A, ind)

    loop_with_value.f(expected, ind)
    assert np.allclose(A, expected)


def test_map_map_indirect_2():
    @dace.program
    def loop_with_value(A: dace.float64[20, 20], ind: dace.int64[20]):
        for i in dace.map[0:20]:
            for j in dace.map[ind[i]:ind[i] + 1]:
                A[i, j] = j

    A = np.random.rand(20, 20)
    ind = np.random.randint(low=0, high=19, size=(20, ), dtype=np.int64)
    expected = A.copy()

    loop_with_value(A, ind)

    loop_with_value.f(expected, ind)
    assert np.allclose(A, expected)


def test():
    A = np.zeros(1).astype(np.int32)
    nested_loop_test(A)

    assert A[0] == 11 * 5


if __name__ == "__main__":
    test()
    test_loop_loop_indirect()
    test_map_loop_indirect()
    test_map_loop_indirect_2()
    test_map_map_indirect()
    test_map_map_indirect_2()

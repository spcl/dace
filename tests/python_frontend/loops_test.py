# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace.frontend.python.common import DaceSyntaxError


@dace.program
def for_loop():
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = 0
    for i in range(0, 10, 2):
        A[i] = i
    return A


def test_for_loop():
    A = for_loop()
    A_ref = np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int32)
    assert (np.array_equal(A, A_ref))


@dace.program
def for_loop_with_break_continue():
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = 0
    for i in range(20):
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        A[i] = i
    return A


def test_for_loop_with_break_continue():
    A = for_loop_with_break_continue()
    A_ref = np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int32)
    assert (np.array_equal(A, A_ref))


@dace.program
def nested_for_loop():
    A = dace.ndarray([10, 10], dtype=dace.int32)
    A[:] = 0
    for i in range(20):
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        for j in range(20):
            if j >= 10:
                break
            if j % 2 == 1:
                continue
            A[i, j] = j
    return A


def test_nested_for_loop():
    A = nested_for_loop()
    A_ref = np.zeros([10, 10], dtype=np.int32)
    for i in range(0, 10, 2):
        A_ref[i] = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0]
    assert (np.array_equal(A, A_ref))


@dace.program
def while_loop():
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = 0
    i = 0
    while (i < 10):
        A[i] = i
        i += 2
    return A


def test_while_loop():
    A = while_loop()
    A_ref = np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int32)
    assert (np.array_equal(A, A_ref))


@dace.program
def while_loop_with_break_continue():
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = 0
    i = -1
    while i < 20:
        i += 1
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        A[i] = i
    return A


def test_while_loop_with_break_continue():
    A = while_loop_with_break_continue()
    A_ref = np.array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0], dtype=np.int32)
    assert (np.array_equal(A, A_ref))


@dace.program
def nested_while_loop():
    A = dace.ndarray([10, 10], dtype=dace.int32)
    A[:] = 0
    i = -1
    while i < 20:
        i += 1
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        j = -1
        while j < 20:
            j += 1
            if j >= 10:
                break
            if j % 2 == 1:
                continue
            A[i, j] = j
    return A


def test_nested_while_loop():
    A = nested_while_loop()
    A_ref = np.zeros([10, 10], dtype=np.int32)
    for i in range(0, 10, 2):
        A_ref[i] = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0]
    assert (np.array_equal(A, A_ref))


@dace.program
def nested_for_while_loop():
    A = dace.ndarray([10, 10], dtype=dace.int32)
    A[:] = 0
    for i in range(20):
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        j = -1
        while j < 20:
            j += 1
            if j >= 10:
                break
            if j % 2 == 1:
                continue
            A[i, j] = j
    return A


def test_nested_for_while_loop():
    A = nested_for_while_loop()
    A_ref = np.zeros([10, 10], dtype=np.int32)
    for i in range(0, 10, 2):
        A_ref[i] = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0]
    assert (np.array_equal(A, A_ref))


@dace.program
def nested_while_for_loop():
    A = dace.ndarray([10, 10], dtype=dace.int32)
    A[:] = 0
    i = -1
    while i < 20:
        i += 1
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        for j in range(20):
            if j >= 10:
                break
            if j % 2 == 1:
                continue
            A[i, j] = j
    return A


def test_nested_while_for_loop():
    A = nested_while_for_loop()
    A_ref = np.zeros([10, 10], dtype=np.int32)
    for i in range(0, 10, 2):
        A_ref[i] = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0]
    assert (np.array_equal(A, A_ref))


@dace.program
def map_with_break_continue():
    A = dace.ndarray([10], dtype=dace.int32)
    A[:] = 0
    for i in dace.map[0:20]:
        if i >= 10:
            break
        if i % 2 == 1:
            continue
        A[i] = i
    return A


def test_map_with_break_continue():
    try:
        map_with_break_continue()
    except Exception as e:
        if isinstance(e, DaceSyntaxError):
            return 0
    assert (False)


@dace.program
def nested_map_for_loop():
    A = np.ndarray([10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):
            A[i, j] = i * 10 + j
    return A


def test_nested_map_for_loop():
    ref = np.zeros([10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            ref[i, j] = i * 10 + j
    val = nested_map_for_loop()
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_for_for_loop():
    A = np.ndarray([10, 10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):
            for k in range(10):
                A[i, j, k] = i * 100 + j * 10 + k
    return A


def test_nested_map_for_for_loop():
    ref = np.zeros([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                ref[i, j, k] = i * 100 + j * 10 + k
    val = nested_map_for_for_loop()
    assert (np.array_equal(val, ref))


@dace.program
def nested_for_map_for_loop():
    A = np.ndarray([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in dace.map[0:10]:
            for k in range(10):
                A[i, j, k] = i * 100 + j * 10 + k
    return A


def test_nested_for_map_for_loop():
    ref = np.zeros([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                ref[i, j, k] = i * 100 + j * 10 + k
    val = nested_for_map_for_loop()
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_for_loop_with_tasklet():
    A = np.ndarray([10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):

            @dace.tasklet
            def comp():
                out >> A[i, j]
                out = i * 10 + j

    return A


def test_nested_map_for_loop_with_tasklet():
    ref = np.zeros([10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            ref[i, j] = i * 10 + j
    val = nested_map_for_loop_with_tasklet()
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_for_for_loop_with_tasklet():
    A = np.ndarray([10, 10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):
            for k in range(10):

                @dace.tasklet
                def comp():
                    out >> A[i, j, k]
                    out = i * 100 + j * 10 + k

    return A


def test_nested_map_for_for_loop_with_tasklet():
    ref = np.zeros([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                ref[i, j, k] = i * 100 + j * 10 + k
    val = nested_map_for_for_loop_with_tasklet()
    assert (np.array_equal(val, ref))


@dace.program
def nested_for_map_for_loop_with_tasklet():
    A = np.ndarray([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in dace.map[0:10]:
            for k in range(10):

                @dace.tasklet
                def comp():
                    out >> A[i, j, k]
                    out = i * 100 + j * 10 + k

    return A


def test_nested_for_map_for_loop_with_tasklet():
    ref = np.zeros([10, 10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                ref[i, j, k] = i * 100 + j * 10 + k
    val = nested_for_map_for_loop_with_tasklet()
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_for_loop_2(B: dace.int64[10, 10]):
    A = np.ndarray([10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):
            A[i, j] = 2 * B[i, j] + i * 10 + j
    return A


def test_nested_map_for_loop_2():
    B = np.ones([10, 10], dtype=np.int64)
    ref = np.zeros([10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            ref[i, j] = 2 + i * 10 + j
    val = nested_map_for_loop_2(B)
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_for_loop_with_tasklet_2(B: dace.int64[10, 10]):
    A = np.ndarray([10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in range(10):

            @dace.tasklet
            def comp():
                inp << B[i, j]
                out >> A[i, j]
                out = 2 * inp + i * 10 + j

    return A


def test_nested_map_for_loop_with_tasklet_2():
    B = np.ones([10, 10], dtype=np.int64)
    ref = np.zeros([10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(10):
            ref[i, j] = 2 + i * 10 + j
    val = nested_map_for_loop_with_tasklet_2(B)
    assert (np.array_equal(val, ref))


@dace.program
def nested_map_with_symbol():
    A = np.zeros([10, 10], dtype=np.int64)
    for i in dace.map[0:10]:
        for j in dace.map[i:10]:
            A[i, j] = i * 10 + j
    return A


def test_nested_map_with_symbol():
    ref = np.zeros([10, 10], dtype=np.int64)
    for i in range(10):
        for j in range(i, 10):
            ref[i, j] = i * 10 + j
    val = nested_map_with_symbol()
    assert (np.array_equal(val, ref))


def test_for_else():
    @dace.program
    def for_else(A: dace.float64[20]):
        for i in range(1, 20):
            if A[i] >= 10:
                A[0] = i
                break
            if i % 2 == 1:
                continue
            A[i] = i
        else:
            A[0] = -1.0

    A = np.random.rand(20)
    A_2 = np.copy(A)
    expected_1 = np.copy(A)
    expected_2 = np.copy(A)

    expected_2[6] = 20.0
    for_else.f(expected_1)
    for_else.f(expected_2)

    for_else(A)
    assert np.allclose(A, expected_1)

    A_2[6] = 20.0
    for_else(A_2)
    assert np.allclose(A_2, expected_2)


def test_while_else():
    @dace.program
    def while_else(A: dace.float64[2]):
        while A[0] < 5.0:
            if A[1] < 0.0:
                A[0] = -1.0
                break
            A[0] += 1.0
        else:
            A[1] = 1.0
            A[1] = 1.0

    A = np.array([0.0, 0.0])
    expected = np.array([5.0, 1.0])
    while_else(A)
    assert np.allclose(A, expected)

    A = np.array([0.0, -1.0])
    expected = np.array([-1.0, -1.0])
    while_else(A)
    assert np.allclose(A, expected)


if __name__ == "__main__":
    test_for_loop()
    test_for_loop_with_break_continue()
    test_nested_for_loop()
    test_while_loop()
    test_while_loop_with_break_continue()
    test_nested_while_loop()
    test_nested_for_while_loop()
    test_nested_while_for_loop()
    test_map_with_break_continue()
    test_nested_map_for_loop()
    test_nested_map_for_for_loop()
    test_nested_for_map_for_loop()
    test_nested_map_for_loop_with_tasklet()
    test_nested_map_for_for_loop_with_tasklet()
    test_nested_for_map_for_loop_with_tasklet()
    test_nested_map_for_loop_2()
    test_nested_map_for_loop_with_tasklet_2()
    test_nested_map_with_symbol()
    test_for_else()
    test_while_else()

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import random


def test_python_min1():
    @dace.program
    def python_min1(a: dace.int64):
        return min(a)

    for _ in range(100):
        a = random.randint(-10, 10)
        assert (python_min1(a)[0] == a)


def test_python_max1():
    @dace.program
    def python_max1(a: dace.int64):
        return max(a)

    for _ in range(100):
        a = random.randint(-10, 10)
        assert (python_max1(a)[0] == a)


def test_python_min2():
    @dace.program
    def python_min2(a: dace.int64, b: dace.int64):
        return min(a, b)

    for _ in range(100):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        assert (python_min2(a, b)[0] == min(a, b))


def test_python_max2():
    @dace.program
    def python_max2(a: dace.int64, b: dace.int64):
        return max(a, b)

    for _ in range(100):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        assert (python_max2(a, b)[0] == max(a, b))


def test_python_min3():
    @dace.program
    def python_min3(a: dace.int64, b: dace.int64, c: dace.int64):
        return min(a, b, c)

    for _ in range(100):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        assert (python_min3(a, b, c)[0] == min(a, b, c))


def test_python_max3():
    @dace.program
    def python_max3(a: dace.int64, b: dace.int64, c: dace.int64):
        return max(a, b, c)

    for _ in range(100):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        assert (python_max3(a, b, c)[0] == max(a, b, c))


def test_python_abs():
    @dace.program
    def python_abs(a: dace.int64):
        b = abs(a)
        return b + 1

    assert python_abs(-1) == (abs(-1) + 1)


if __name__ == "__main__":
    test_python_min1()
    test_python_max1()
    test_python_min2()
    test_python_max2()
    test_python_min3()
    test_python_max3()
    test_python_abs()

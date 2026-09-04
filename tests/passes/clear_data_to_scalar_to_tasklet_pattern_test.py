# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace

N = dace.symbol('N')


@dace.program
def t1(A: dace.float64[N]):
    sc = 0
    for i in dace.map[0:N:1]:
        sc = A[i]
        A[i] = 2 * sc


@dace.program
def t2(A: dace.float64[N], B: dace.float64[N]):
    sc = 0
    for i in dace.map[0:N:1]:
        sc = A[i]
        A[i] = 2 * sc
    for i in dace.map[0:N:1]:
        sc = B[i]
        A[i] = 1.1 * sc


def test_t1():
    sdfg = t1.to_sdfg()
    sdfg.validate()


def test_t2():
    sdfg = t2.to_sdfg()
    sdfg.validate()


if __name__ == "__main__":
    test_t1()
    test_t2()

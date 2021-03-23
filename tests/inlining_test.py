# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def transpose(input, output):
    @dace.map(_[0:H, 0:W])
    def compute(i, j):
        a << input[j, i]
        b >> output[i, j]
        b = a


@dace.program
def bla(A, B, alpha):
    @dace.tasklet
    def something():
        al << alpha
        a << A[0, 0]
        b >> B[0, 0]
        b = al * a


@dace.program
def myprogram(A, B, cst):
    transpose(A, B)
    bla(A, B, cst)


def test():
    myprogram.compile(dace.float32[W, H], dace.float32[H, W], dace.int32)


def test_regression_reshape_unsqueeze():
    nsdfg = dace.SDFG("nested_reshape_node")
    nstate = nsdfg.add_state()
    nsdfg.add_array("input", [3, 3], dace.float64)
    nsdfg.add_view("view", [3, 3], dace.float64)
    nsdfg.add_array("output", [9], dace.float64)

    R = nstate.add_read("input")
    A = nstate.add_access("view")
    W = nstate.add_write("output")

    mm1 = dace.Memlet("input[0:3, 0:3] -> 0:3, 0:3")
    mm2 = dace.Memlet("view[0:3, 0:2] -> 3:9")

    nstate.add_edge(R, None, A, None, mm1)
    nstate.add_edge(A, None, W, None, mm2)

    @dace.program
    def test_reshape_unsqueeze(A: dace.float64[3, 3], B: dace.float64[9]):
        nsdfg(input=A, output=B)

    sdfg = test_reshape_unsqueeze.to_sdfg(strict=False)
    sdfg.apply_strict_transformations()
    sdfg.validate()

    a = np.random.rand(3, 3)
    b = np.random.rand(9)
    regb = np.copy(b)
    regb[3:9] = a[0:3, 0:2].reshape([6])
    sdfg(A=a, B=b)

    assert np.allclose(b, regb)


if __name__ == "__main__":
    test()
    test_regression_reshape_unsqueeze()

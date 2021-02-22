# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp

W = dp.symbol('W')
H = dp.symbol('H')


@dp.program
def transpose(input, output):
    @dp.map(_[0:H, 0:W])
    def compute(i, j):
        a << input[j, i]
        b >> output[i, j]
        b = a


@dp.program
def bla(A, B, alpha):
    @dp.tasklet
    def something():
        al << alpha
        a << A[0, 0]
        b >> B[0, 0]
        b = al * a


@dp.program
def myprogram(A, B, cst):
    transpose(A, B)
    bla(A, B, cst)


def test():
    myprogram.compile(dp.float32[W, H], dp.float32[H, W], dp.int32)


if __name__ == "__main__":
    test()
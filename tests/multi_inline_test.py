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
def bla(input, output):
    transpose(input, output)


@dp.program
def multi_inline(A, B):
    bla(A, B)


def test():
    multi_inline.compile(dp.float32[W, H], dp.float32[H, W])


if __name__ == "__main__":
    test()

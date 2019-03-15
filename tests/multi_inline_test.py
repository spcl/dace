#!/usr/bin/env python
import dace as dp

W = dp.symbol('W')
H = dp.symbol('H')


@dp.external_function
def transpose(input, output):
    @dp.map(_[0:H, 0:W])
    def compute(i, j):
        a << input[j, i]
        b >> output[i, j]
        b = a


@dp.external_function
def bla(input, output):
    dp.call(transpose, input, output)


@dp.program
def myprogram(A, B):
    dp.call(bla, A, B)


if __name__ == '__main__':
    dp.compile(myprogram, dp.float32[W, H], dp.float32[H, W])

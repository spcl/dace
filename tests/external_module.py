# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def transpose(input: dace.float32[W, H], output: dace.float32[H, W]):

    @dace.map(_[0:H, 0:W])
    def compute(i, j):
        a << input[j, i]
        b >> output[i, j]
        b = a

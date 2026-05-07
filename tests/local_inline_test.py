# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')


@dace.program
def local_inline_inner(AA, BB):
    tmp = dace.define_local([W], AA.dtype)

    @dace.map(_[0:W])
    def compute(i):
        a << AA[i]
        b >> tmp[i]
        b = -a

    @dace.map(_[0:W])
    def compute2(i):
        a << tmp[i]
        b >> BB[i]
        b = a + 1


@dace.program
def local_inline(A: dace.float64[W], B: dace.float64[W], C: dace.float64[W]):
    local_inline_inner(A, B)
    local_inline_inner(B, C)


def test():
    W = 3

    A = dace.ndarray([W])
    B = dace.ndarray([W])
    C = dace.ndarray([W])

    A[:] = np.mgrid[0:W]
    B[:] = 0.0
    C[:] = 0.0

    local_inline(A, B, C)

    diff = np.linalg.norm((-(-A + 1) + 1) - C) / W
    print("Difference:", diff)
    print("==== Program end ====")
    assert diff <= 1e-5


if __name__ == "__main__":
    test()

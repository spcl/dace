# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import dace
import numpy as np

N = 12


@dace.program
def program(input, output):
    @dace.map(_[0:N])
    def tasklet(i):
        a << input[i]
        b >> output(1, lambda a, b: a + b)
        b = a


def test_cr_complex():
    print('CR non-atomic (complex value) test')

    A = np.random.rand(N).astype(np.complex128)
    A += np.random.rand(N).astype(np.complex128) * 1j
    B = np.ndarray([1], dtype=A.dtype)
    B[0] = 0

    program(A, B)

    diff = abs(np.sum(A) - B[0])
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_cr_complex()

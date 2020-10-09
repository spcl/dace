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


if __name__ == "__main__":
    print('CR non-atomic (complex value) test')

    A = np.random.rand(N).astype(np.complex128)
    A += np.random.rand(N).astype(np.complex128) * 1j
    B = np.ndarray([1], dtype=A.dtype)
    B[0] = 0

    program(A, B)

    diff = abs(np.sum(A) - B[0])
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)

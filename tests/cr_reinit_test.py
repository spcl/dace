#!/usr/bin/env python
from __future__ import print_function

import dace
import numpy as np

N = 12


@dace.program
def program(input, output):
    for t in range(3):

        @dace.map(_[0:N])
        def tasklet(i):
            a << input[i]
            b >> output(1, lambda a, b: a + b, 0)
            b = a


if __name__ == "__main__":
    print('CR re-initialization test')

    A = np.random.rand(N)
    B = np.ndarray([1], dtype=A.dtype)
    B[0] = 100

    program(A, B)

    diff = abs(3 * np.sum(A) - B[0])
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)

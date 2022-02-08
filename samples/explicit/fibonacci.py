# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np


@dp.program
def fibonacci(iv: dp.int32[1], res: dp.float32[1]):
    S = dp.define_stream(dp.int32, 0)

    # Initialize stream with input value
    @dp.tasklet
    def init():
        i << iv
        s >> S
        s = i

    @dp.consume(S, 4)
    def cons(elem, p):
        sout >> S(-1)
        val >> res(-1, lambda a, b: a + b)[0]

        if elem == 1:
            val = 1
        elif elem > 1:  # Recurse by pushing smaller values
            sout = elem - 1
            sout = elem - 2


def fibonacci_py(v):
    """ Computes the Fibonacci sequence at point v. """
    if v == 0:
        return 0
    if v == 1:
        return 1
    return fibonacci_py(v - 1) + fibonacci_py(v - 2)


if __name__ == '__main__':
    print('Fibonacci recursion using consume - Python frontend')
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = 10
    output[0] = 0
    regression = fibonacci_py(input[0])

    fibonacci(input, output)

    diff = (regression - output[0])**2
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)

# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Sample showing the Consume scope by unrolling a Fibonacci sequence recursive computation. """
import argparse
import dace
import numpy as np


@dace.program
def fibonacci(iv: dace.int32[1], res: dace.float32[1]):
    # Define an unbounded stream
    S = dace.define_stream(dace.int32, 0)

    # Initialize stream with input value
    with dace.tasklet:
        i << iv
        s >> S
        s = i

    # Consume elements from the stream, with 4 processing elements in parallel.
    # The consume scope can push new values onto the stream S as it is working
    @dace.consume(S, 4)
    def scope(elem, p):
        # Set dynamic outgoing memlet to `S` (with -1 as the volume)
        sout >> S(-1)
        # The end result `res` has a sum write-conflict resolution with dynamic volume
        val >> res(-1, lambda a, b: a + b)

        # Setting `sout` to a value pushes it onto the stream
        if elem == 1:
            # End of recursion, set `val` to 1 to add it to `res`
            val = 1
        elif elem > 1:
            # Otherwise, recurse by pushing smaller values
            sout = elem - 1
            sout = elem - 2
        # A value of zero does not incur any action


def fibonacci_py(v):
    """ Computes the Fibonacci sequence at point v. """
    if v == 0:
        return 0
    if v == 1:
        return 1
    return fibonacci_py(v - 1) + fibonacci_py(v - 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int, nargs="?", default=10)
    args = parser.parse_args()

    # Set initial array to number, and output to zero
    input = np.ndarray([1], np.int32)
    output = np.ndarray([1], np.float32)
    input[0] = args.number
    output[0] = 0
    regression = fibonacci_py(input[0])

    fibonacci(input, output)

    diff = (regression - output[0])**2
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)

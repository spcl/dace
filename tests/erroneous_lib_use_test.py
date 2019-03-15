#!/usr/bin/env python
from __future__ import print_function

import dace
from dace.codegen.compiler import DuplicateDLLError, CompilationError
import numpy as np


# Dynamically creates DaCe programs with the same name
def program_generator(size, factor):
    @dace.program(
        dace.float64[size], dace.float64[size], size=size, factor=factor)
    def program(input, output):
        @dace.map(_[0:size])
        def tasklet(i):
            a << input[i]
            b >> output[i]
            b = a * factor

    return program


if __name__ == "__main__":
    print('Reloadable DaCe program test')

    array_one = np.random.rand(10).astype(np.float64)
    array_two = np.random.rand(20).astype(np.float64)
    output_one = np.zeros(10, dtype=np.float64)
    output_two = np.zeros(20, dtype=np.float64)

    prog_one = program_generator(10, 2.0)
    prog_two = program_generator(20, 4.0)

    # This should NOT work (the two SDFGs will compile over the same file)
    try:
        func1 = dace.compile(prog_one)
        func2 = dace.compile(prog_two)

        func1(array_one, output_one)
        func2(array_two, output_two)

        diff1 = np.linalg.norm(2.0 * array_one - output_one) / 10.0
        diff2 = np.linalg.norm(4.0 * array_two - output_two) / 20.0
        print("Differences:", diff1, diff2)
        print("This should definitely not work!")
        exit(1)
    except (DuplicateDLLError, CompilationError):
        print("Exception successfully caught, test passed")
        exit(0)

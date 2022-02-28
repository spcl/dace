# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple program showing the `dace.map` syntax and profiling. """

import argparse
import dace
import numpy as np

# Define a symbol so that the vectors could have arbitrary sizes and compile the code once
# (this step is not necessary for arrays with known sizes)
N = dace.symbol('N')


# Define the program
def laplace(A: dace.float64[N], T: dace.int64):
    # Define transient (temporary) array
    tmp = np.zeros_like(A)

    # This loop will remain a loop
    for _ in range(T):
        # This loop will become a parallel map
        for i in dace.map[1:N - 1]:
            tmp[i] = A[i - 1] - 2 * A[i] + A[i + 1]
            # Alternatively, a "NumPy way" to write the kernel also works
            # tmp[i] = np.sum(A[i - 1:i + 2] * np.array([1, -2, 1]))

        for i in dace.map[1:N - 1]:
            A[i] = tmp[i - 1] - 2 * tmp[i] + tmp[i + 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1024)
    parser.add_argument("iterations", type=int, nargs="?", default=100)
    args = parser.parse_args()

    # Create a data-centric version of the program
    dace_laplace = dace.program(laplace)

    # Set initial values
    A = np.random.rand(args.N)

    # Time the result by enabling profiling
    with dace.config.set_temporary('profiling', value=True):
        dace_laplace(A, args.iterations)

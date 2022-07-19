# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Simple program showing the DaCe Python interface via scalar multiplication and vector addition. """

import argparse
import dace
import numpy as np

# Define a symbol so that the vectors could have arbitrary sizes and compile the code once
# (this step is not necessary for arrays with known sizes)
N = dace.symbol('N')


# Define the data-centric program with type hints
# (without this step, Just-In-Time compilation is triggered every call)
@dace.program
def axpy(a: dace.float64, x: dace.float64[N], y: dace.float64[N]):
    return a * x + y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1024)
    args = parser.parse_args()

    # Initialize arrays
    a = np.random.rand()
    x = np.random.rand(args.N)
    y = np.random.rand(args.N)

    # Call the program (the value of N is inferred by dace automatically)
    z = axpy(a, x, y)

    # Check result
    expected = a * x + y
    print("Difference:", np.linalg.norm(z - expected))

# Vector addition with explicit dataflow. Computes Z += X + Y
# Can be used for simple vectorization test

import dace
import numpy as np
import argparse

N = dace.symbol("N")


@dace.program
def vec_sum(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_x << x[i]
        in_y << y[i]
        in_z << z[i]
        out >> z[i]

        out = in_x + in_y + in_z


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    parser.add_argument("--compile-only",
                        default=False,
                        action="store_true",
                        dest="compile-only")
    args = vars(parser.parse_args())
    dace.config.Config.set("compiler", "intel_fpga", "mode", value="emulator")

    N.set(args["N"])

    print('Vectors addition %d' % (N.get()))

    # Initialize arrays: X, Y and Z
    X = np.random.rand(N.get()).astype(dace.float32.type)
    Y = np.random.rand(N.get()).astype(dace.float32.type)
    Z = np.random.rand(N.get()).astype(dace.float32.type)

    Z_exp = X + Y + Z

    #compute expected result
    vec_sum(X, Y, Z)
    diff = np.linalg.norm(Z_exp - Z) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)

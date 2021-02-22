# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Type inference test with annotated types. """

import dace
import numpy as np
import argparse
N = dace.symbol("N")


@dace.program
def type_inference(x: dace.float32[N], y: dace.float32[N]):
    @dace.map
    def comp(i: _[0:N]):
        in_x << x[i]
        in_y << y[i]
        out >> y[i]

        # computes y[i]=(int)x[i] + ((int)y[i])*2.1
        var1 = int(in_x)
        var2: int = in_y
        var3 = 2.1 if (i>1 and i<10) else 2.1 # Just for the sake of testing
        res = var1 + var3 * var2
        out = res


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    parser.add_argument(
        "--compile-only",
        default=False,
        action="store_true",
        dest="compile-only")
    args = vars(parser.parse_args())
    dace.config.Config.set("compiler", "intel_fpga", "mode", value="emulator")

    N.set(args["N"])

    print('Vectors addition %d' % (N.get()))

    # Initialize vector: X
    X = np.random.uniform(-10, 0, N.get()).astype(dace.float32.type)
    Y = np.random.uniform(-10, 0, N.get()).astype(dace.float32.type)
    # compute expected result
    Z = np.zeros(N.get())
    for i in range(0, N.get()):
        Z[i] = int(X[i]) + int(Y[i]) * 2.1

    type_inference(X, Y)

    diff = np.linalg.norm(Z - Y) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)

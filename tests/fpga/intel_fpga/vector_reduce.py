# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Sums all the element of the vector with a reduce. """

import dace
import numpy as np
import argparse

N = dace.symbol('N')


@dace.program
def vector_reduce(x: dace.float32[N], s: dace.scalar(dace.float32)):
    #transient
    tmp = dace.define_local([N], dtype=x.dtype)

    @dace.map
    def sum(i: _[0:N]):
        in_x << x[i]
        out >> tmp[i]

        out = in_x

    dace.reduce(lambda a, b: a + b, tmp, s, axis=(0), identity=0)


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
    s = dace.scalar(dace.float32)

    vector_reduce(X, s)
    #compute expected result
    s_exp = 0.0
    for x in X:
        s_exp += x
    print(s)
    print(s_exp)
    diff = np.linalg.norm(s_exp - s) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" 
Vector addition with explicit dataflow. Computes Z += X + Y
Can be used for simple vectorization test
"""

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
    parser.add_argument("vectorize_first", choices=["true", "false"])
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = parser.parse_args()
    dace.config.Config.set("compiler", "intel_fpga", "mode", value="emulator")

    N.set(args.N)

    print('Vectors addition %d' % (N.get()))

    # Initialize arrays: X, Y and Z
    X = np.random.rand(N.get()).astype(dace.float32.type)
    Y = np.random.rand(N.get()).astype(dace.float32.type)
    Z = np.random.rand(N.get()).astype(dace.float32.type)

    Z_exp = X + Y + Z

    sdfg = vec_sum.to_sdfg()

    if args.vectorize_first == "true":
        transformations = [
            dace.transformation.dataflow.vectorization.Vectorization,
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG
        ]
        transformation_options = [{}, {
            "propagate_parent": True,
            "postamble": False
        }]
    elif args.vectorize_first == "false":
        transformations = [
            dace.transformation.interstate.fpga_transform_sdfg.
            FPGATransformSDFG,
            dace.transformation.dataflow.vectorization.Vectorization
        ]
        transformation_options = [{}, {
            "propagate_parent": True,
            "postamble": False
        }]

    sdfg.apply_transformations(transformations, transformation_options)

    sdfg(x=X, y=Y, z=Z, N=N)

    diff = np.linalg.norm(Z_exp - Z) / N.get()
    if diff > 1e-5:
        raise ValueError("Difference: {}".format(diff))

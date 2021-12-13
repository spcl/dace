# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vector addition with explicit dataflow. Computes Z += X + Y
Can be used for simple vectorization test
"""

import dace
from dace.fpga_testing import fpga_test
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


def run_vec_sum(vectorize_first: bool):

    N.set(24)

    # Initialize arrays: X, Y and Z
    X = np.random.rand(N.get()).astype(dace.float32.type)
    Y = np.random.rand(N.get()).astype(dace.float32.type)
    Z = np.random.rand(N.get()).astype(dace.float32.type)

    Z_exp = X + Y + Z

    sdfg = vec_sum.to_sdfg()

    if vectorize_first:
        transformations = [
            dace.transformation.dataflow.vectorization.Vectorization,
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG
        ]
        transformation_options = [{
            "target": dace.ScheduleType.FPGA_Device,
        }, {}]
    else:
        transformations = [
            dace.transformation.interstate.fpga_transform_sdfg.
            FPGATransformSDFG,
            dace.transformation.dataflow.vectorization.Vectorization
        ]
        transformation_options = [{}, {
            "target": dace.ScheduleType.FPGA_Device,
        }]

    assert sdfg.apply_transformations(transformations,
                                      transformation_options) == 2

    sdfg(x=X, y=Y, z=Z, N=N)

    diff = np.linalg.norm(Z_exp - Z) / N.get()
    if diff > 1e-5:
        raise ValueError("Difference: {}".format(diff))

    return sdfg


@fpga_test()
def test_vec_sum_vectorize_first():
    return run_vec_sum(True)


@fpga_test()
def test_vec_sum_fpga_transform_first():
    return run_vec_sum(False)


if __name__ == "__main__":
    test_vec_sum_vectorize_first(None)
    test_vec_sum_fpga_transform_first(None)

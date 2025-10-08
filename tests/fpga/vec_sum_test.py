# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vector addition with explicit dataflow. Computes Z += X + Y
Can be used for simple vectorization test
"""

import dace
from dace.fpga_testing import fpga_test, xilinx_test
import numpy as np
from dace.config import set_temporary
import pytest


def run_vec_sum(vectorize_first: bool):
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

    n = 24

    # Initialize arrays: X, Y and Z
    rng = np.random.default_rng(42)
    X = rng.random(n, dtype=np.float32)
    Y = rng.random(n, dtype=np.float32)
    Z = rng.random(n, dtype=np.float32)
    ref = X + Y + Z

    sdfg = vec_sum.to_sdfg()

    if vectorize_first:
        transformations = [
            dace.transformation.dataflow.vectorization.Vectorization,
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG
        ]
        transformation_options = [{"propagate_parent": True, "postamble": False}, {}]
    else:
        transformations = [
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG,
            dace.transformation.dataflow.vectorization.Vectorization
        ]
        transformation_options = [{}, {"propagate_parent": True, "postamble": False}]

    assert sdfg.apply_transformations(transformations, transformation_options) == 2

    sdfg(x=X, y=Y, z=Z, N=n)

    print(f"ref ({ref.shape}): {ref}")
    print(f"Z ({Z.shape}): {Z}")

    diff = np.linalg.norm(ref - Z) / n
    if diff > 1e-5:
        raise ValueError("Difference: {}".format(diff))

    return sdfg


@fpga_test(assert_ii_1=False)
def test_vec_sum_vectorize_first():
    return run_vec_sum(True)


# TODO: Investigate and re-enable if possible.
@pytest.mark.skip(reason="Unexplained CI Regression")
@fpga_test(assert_ii_1=False, intel=False)
def test_vec_sum_fpga_transform_first():
    return run_vec_sum(False)


@xilinx_test(assert_ii_1=True)
def test_vec_sum_vectorize_first_decoupled_interfaces():
    # For this test, decoupled read/write interfaces are needed to achieve II=1
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        return run_vec_sum(True)


@xilinx_test(assert_ii_1=True)
def test_vec_sum_fpga_transform_first_decoupled_interfaces():
    # For this test, decoupled read/write interfaces are needed to achieve II=1
    with set_temporary("compiler", "xilinx", "decouple_array_interfaces", value=True):
        with set_temporary('testing', 'serialization', value=False):
            return run_vec_sum(True)


if __name__ == "__main__":
    test_vec_sum_vectorize_first(None)
    test_vec_sum_fpga_transform_first(None)
    test_vec_sum_fpga_transform_first_decoupled_interfaces(None)

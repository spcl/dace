# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vector addition with explicit dataflow. Computes Z += X + Y
Can be used for simple vectorization test
"""

import dace
from dace.fpga_testing import fpga_test
from dace.fpga_testing import xilinx_test
import numpy as np
import argparse
from dace.transformation.dataflow import Vectorization
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

SIZE = 64


@dace.program
def tensoradd_kernel(A: dace.float32[M, N, K], B: dace.float32[M, N, K],
                     C: dace.float32[M, N, K]):
    C[:] = A + B


@dace.program
def add_1_kernel(A: dace.float32[SIZE], B: dace.float32[SIZE]):
    B[:] = A + 1.0


@dace.program
def add_1_kernel_sym(A: dace.float32[N], B: dace.float32[N]):
    B[:] = A + 1.0


@dace.program
def matadd_kernel_sym(A: dace.float32[M, N], B: dace.float32[M, N],
                      C: dace.float32[M, N]):
    C[:] = A + B


@dace.program
def matadd_kernel(A: dace.float32[SIZE, SIZE], B: dace.float32[SIZE, SIZE],
                  C: dace.float32[SIZE, SIZE]):
    C[:] = A + B


@dace.program
def two_maps_kernel_legal(A: dace.float32[N], B: dace.float32[N],
                          C: dace.float32[N], D: dace.float32[N],
                          E: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_a << A[i]
        in_b << B[i]
        out >> D[i]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N]):
        in_b << B[i]
        in_c << C[i]
        out >> E[i]
        out = in_b + in_c


@dace.program
def two_maps_kernel_illegal(A: dace.float32[N], B: dace.float32[N],
                            C: dace.float32[N], D: dace.float32[N],
                            E: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_a << A[i]
        in_b << B[i]
        out >> D[i]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N:2]):
        in_b << B[i]
        in_c << C[i]
        out >> E[i]
        out = in_b + in_c


@dace.program
def vec_sum_kernel(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        in_x << x[i]
        in_y << y[i]
        in_z << z[i]
        out >> z[i]

        out = in_x + in_y + in_z


def vec_two_maps(strided_map):
    N.set(24)
    A = np.random.rand(N.get()).astype(dace.float32.type)
    B = np.random.rand(N.get()).astype(dace.float32.type)
    C = np.random.rand(N.get()).astype(dace.float32.type)
    D = np.random.rand(N.get()).astype(dace.float32.type)
    E = np.random.rand(N.get()).astype(dace.float32.type)

    D_exp = A + B
    E_exp = B + C

    sdfg: dace.SDFG = two_maps_kernel_legal.to_sdfg()

    assert sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG]) == 2

    assert sdfg.apply_transformations_repeated(
        Vectorization,
        options={
            'vector_len': 2,
            'target': dace.ScheduleType.FPGA_Device,
            'strided_map': strided_map
        }) == 1

    sdfg(A=A, B=B, C=C, D=D, E=E, N=N)

    assert np.allclose(D, D_exp)
    assert np.allclose(E, E_exp)

    return sdfg


def vec_sum(vectorize_first: bool, strided_map: bool):

    N.set(24)

    # Initialize arrays: X, Y and Z
    X = np.random.rand(N.get()).astype(dace.float32.type)
    Y = np.random.rand(N.get()).astype(dace.float32.type)
    Z = np.random.rand(N.get()).astype(dace.float32.type)

    Z_exp = X + Y + Z

    sdfg = vec_sum_kernel.to_sdfg()

    if vectorize_first:
        transformations = [
            dace.transformation.dataflow.vectorization.Vectorization,
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG
        ]
        transformation_options = [{
            "target": dace.ScheduleType.FPGA_Device,
            'strided_map': strided_map
        }, {}]
    else:
        transformations = [
            dace.transformation.interstate.fpga_transform_sdfg.
            FPGATransformSDFG,
            dace.transformation.dataflow.vectorization.Vectorization
        ]
        transformation_options = [{}, {
            "target": dace.ScheduleType.FPGA_Device,
            'strided_map': strided_map
        }]

    assert sdfg.apply_transformations(transformations,
                                      transformation_options) == 2

    sdfg(x=X, y=Y, z=Z, N=N)

    diff = np.linalg.norm(Z_exp - Z) / N.get()
    if diff > 1e-5:
        raise ValueError("Difference: {}".format(diff))

    return sdfg


def test_vec_two_maps_illegal():
    sdfg = two_maps_kernel_illegal.to_sdfg()

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                      }) == 0


def vec_matadd(strided_map):
    sdfg: dace.SDFG = matadd_kernel.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    # Run verification
    A = np.random.rand(SIZE, SIZE).astype(np.float32)
    B = np.random.rand(SIZE, SIZE).astype(np.float32)
    C = np.random.rand(SIZE, SIZE).astype(np.float32)

    sdfg(A=A, B=B, C=C)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


def vec_matadd_sym(strided_map):
    sdfg: dace.SDFG = matadd_kernel_sym.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    # Run verification
    A = np.random.rand(SIZE, SIZE).astype(np.float32)
    B = np.random.rand(SIZE, SIZE).astype(np.float32)
    C = np.random.rand(SIZE, SIZE).astype(np.float32)

    sdfg(A=A, B=B, C=C, N=SIZE, M=SIZE)

    diff = np.linalg.norm(C - (A + B))
    assert diff <= 1e-5

    return sdfg


def vec_add_1(strided_map):
    sdfg: dace.SDFG = add_1_kernel.to_sdfg()

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    A = np.random.rand(SIZE).astype(np.float32)
    B = np.random.rand(SIZE).astype(np.float32)

    sdfg(A=A, B=B)

    assert all(B == A + 1)

    return sdfg


def vec_add_1_sym(strided_map):
    sdfg: dace.SDFG = add_1_kernel_sym.to_sdfg()

    sdfg.apply_transformations([
        FPGATransformSDFG,
        InlineSDFG,
    ])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    A = np.random.rand(SIZE).astype(np.float32)
    B = np.random.rand(SIZE).astype(np.float32)

    sdfg(A=A, B=B, N=SIZE)

    assert all(B == A + 1)

    return sdfg


def tensor_add(strided_map):
    # Make SDFG
    sdfg: dace.SDFG = tensoradd_kernel.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target':
                                          dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    # Run verification
    A = np.random.rand(SIZE, SIZE, SIZE).astype(np.float32)
    B = np.random.rand(SIZE, SIZE, SIZE).astype(np.float32)
    C = np.random.rand(SIZE, SIZE, SIZE).astype(np.float32)

    sdfg(A=A, B=B, C=C, M=SIZE, K=SIZE, N=SIZE)

    diff = np.linalg.norm(C - (A + B))

    assert diff <= 1e-5

    return sdfg


@fpga_test()
def test_vec_two_maps_strided():
    return vec_two_maps(True)


@fpga_test()
def test_vec_two_maps_non_strided():
    return vec_two_maps(False)


@fpga_test()
def test_vec_sum_vectorize_first_strided():
    return vec_sum(True, True)


@fpga_test()
def test_vec_sum_vectorize_first_non_strided():
    return vec_sum(True, False)


@fpga_test()
def test_vec_sum_fpga_transform_first_strided():
    return vec_sum(False, True)


@fpga_test()
def test_vec_sum_fpga_transform_first_non_strided():
    return vec_sum(False, False)


@xilinx_test()
def test_vec_matadd_stride():
    return vec_matadd(True)


@xilinx_test()
def test_vec_matadd_non_stride():
    return vec_matadd(False)


@xilinx_test()
def test_vec_matadd_stride_sym():
    return vec_matadd_sym(True)


@xilinx_test()
def test_vec_matadd_non_stride_sym():
    return vec_matadd_sym(False)


@fpga_test()
def test_vec_add_1_stride():
    return vec_add_1(True)


@fpga_test()
def test_vec_add_1_non_stride():
    return vec_add_1(False)


@fpga_test()
def test_vec_add_1_stride_sym():
    return vec_add_1_sym(True)


@fpga_test()
def test_vec_add_1_non_stride_sym():
    return vec_add_1_sym(False)


@xilinx_test()
def test_vec_tensor_add_stride():
    return tensor_add(True)


@xilinx_test()
def test_vec_tensor_add_non_stride():
    return tensor_add(False)


if __name__ == "__main__":
    # test_vec_add_1_stride(None)
    # test_vec_add_1_non_stride(None)
    # test_vec_add_1_stride_sym(None)
    # test_vec_add_1_non_stride_sym(None)

    # test_vec_two_maps_strided(None)
    # test_vec_two_maps_non_strided(None)
    # test_vec_two_maps_illegal()

    # test_vec_matadd_stride(None)
    # test_vec_matadd_non_stride(None)

    # test_vec_matadd_stride_sym(None)
    # test_vec_matadd_non_stride_sym(None)

    # test_vec_sum_vectorize_first_strided(None)
    # test_vec_sum_vectorize_first_non_strided(None)
    # test_vec_sum_fpga_transform_first_strided(None)
    # test_vec_sum_fpga_transform_first_non_strided(None)

    test_vec_tensor_add_stride(None)
    test_vec_tensor_add_non_stride(None)

    # TODO: Add more tests

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Vector addition with explicit dataflow. Computes Z += X + Y
Can be used for simple vectorization test
"""

import dace
from dace import sdfg
from dace.fpga_testing import fpga_test
from dace.fpga_testing import xilinx_test
import numpy as np
import argparse
from dace.transformation.dataflow import Vectorization, MapExpansion
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from tests.fpga.streaming_memory_test import matadd_multistream
from dace.transformation.auto.fpga import fpga_rr_interleave_containers_to_banks

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

SIZE = 64


@dace.program
def matmul_vec_kernel(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


@dace.program
def maporder_vec_kernel(A: dace.float32[N, N, N], B: dace.float32[N, N, N], C: dace.float32[N, N, N],
                        D: dace.float32[N, N, N], E: dace.float32[N, N, N], F: dace.float32[N, N,
                                                                                            N], G: dace.float32[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            in_A << A[i, j, 0]  # No
            in_B << B[i, 0, j]  # Yes
            in_C << C[0, i, j]  # Yes
            in_D << D[j, i, 0]  # No
            in_E << E[j, 0, i]  # No
            in_F << F[0, j, i]  # No
            out >> G[i, j]  # Yes

            out = in_A + in_B + in_C + in_D + in_E + in_F


@dace.program
def vecadd_1_non_appl_1_kernel(A: dace.float32[N], B: dace.float32[N]):
    for i in dace.map[0:N:2]:
        with dace.tasklet:
            in_A << A[i]
            out >> B[i]
            out = in_A + 1.0


@dace.program
def matadd_bad_stride_kernel(A: dace.float32[SIZE + 1, SIZE + 1], B: dace.float32[SIZE + 1, SIZE + 1],
                             C: dace.float32[SIZE + 1, SIZE + 1]):
    C[:] = A + B


@dace.program
def vecadd_1_non_appl_0_kernel(A: dace.float32[N], B: dace.float32[N]):
    for i in dace.map[0:61]:
        with dace.tasklet:
            in_A << A[i]
            out >> B[i]
            out = in_A + 1.0


@dace.program
def matadd_multi_kernel(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N], D: dace.float32[M, N]):
    C[:] = A + B
    D[:] = A - B


@dace.program
def tensoradd_kernel(A: dace.float32[M, N, K], B: dace.float32[M, N, K], C: dace.float32[M, N, K]):
    C[:] = A + B


@dace.program
def add_1_kernel(A: dace.float32[SIZE], B: dace.float32[SIZE]):
    B[:] = A + 1.0


@dace.program
def add_1_kernel_sym(A: dace.float32[N], B: dace.float32[N]):
    B[:] = A + 1.0


@dace.program
def matadd_kernel_sym(A: dace.float32[M, N], B: dace.float32[M, N], C: dace.float32[M, N]):
    C[:] = A + B


@dace.program
def matadd_kernel(A: dace.float32[SIZE, SIZE], B: dace.float32[SIZE, SIZE], C: dace.float32[SIZE, SIZE]):
    C[:] = A + B


@dace.program
def two_maps_kernel_legal(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N],
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
def two_maps_nested_kernel_legal(A: dace.float32[N, M], B: dace.float32[N, M], C: dace.float32[N, M],
                                 D: dace.float32[N, M], E: dace.float32[N, M]):
    @dace.map
    def sum(i: _[0:N], j: _[0:M]):
        in_a << A[i, j]
        in_b << B[i, j]
        out >> D[i, j]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N], j: _[0:M]):
        in_b << B[i, j]
        in_c << C[i, j]
        out >> E[i, j]
        out = in_b + in_c


@dace.program
def two_maps_kernel_nested_illegal(A: dace.float32[N, M], B: dace.float32[N, M], C: dace.float32[N, M],
                                   D: dace.float32[N, M], E: dace.float32[N, M]):
    @dace.map
    def sum(i: _[0:N], j: _[0:M]):
        in_a << A[i, j]
        in_b << B[i, j]
        out >> D[i, j]
        out = in_a + in_b

    @dace.map
    def sum(i: _[0:N:2], j: _[0:M:2]):
        in_b << B[i, j]
        in_c << C[i, j]
        out >> E[i, j]
        out = in_b + in_c


@dace.program
def two_maps_kernel_illegal(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N],
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

    assert sdfg.apply_transformations_repeated(Vectorization,
                                               options={
                                                   'vector_len': 2,
                                                   'target': dace.ScheduleType.FPGA_Device,
                                                   'strided_map': strided_map
                                               }) == 1

    sdfg(A=A, B=B, C=C, D=D, E=E, N=N)

    assert np.allclose(D, D_exp)
    assert np.allclose(E, E_exp)

    return sdfg


def vec_two_maps_nested(strided_map):
    N.set(24)
    M.set(24)
    A = np.random.rand(N.get(), M.get()).astype(dace.float32.type)
    B = np.random.rand(N.get(), M.get()).astype(dace.float32.type)
    C = np.random.rand(N.get(), M.get()).astype(dace.float32.type)
    D = np.random.rand(N.get(), M.get()).astype(dace.float32.type)
    E = np.random.rand(N.get(), M.get()).astype(dace.float32.type)

    D_exp = A + B
    E_exp = B + C

    sdfg: dace.SDFG = two_maps_nested_kernel_legal.to_sdfg()

    assert sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG]) == 2

    assert sdfg.apply_transformations_repeated(MapExpansion) == 2

    assert sdfg.apply_transformations_repeated(Vectorization,
                                               options={
                                                   'vector_len': 2,
                                                   'target': dace.ScheduleType.FPGA_Device,
                                                   'strided_map': strided_map
                                               },
                                               print_report=True) == 1

    sdfg(A=A, B=B, C=C, D=D, E=E, N=N, M=M)

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
        transformation_options = [{"target": dace.ScheduleType.FPGA_Device, 'strided_map': strided_map}, {}]
    else:
        transformations = [
            dace.transformation.interstate.fpga_transform_sdfg.FPGATransformSDFG,
            dace.transformation.dataflow.vectorization.Vectorization
        ]
        transformation_options = [{}, {"target": dace.ScheduleType.FPGA_Device, 'strided_map': strided_map}]

    assert sdfg.apply_transformations(transformations, transformation_options) == 2

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
                                          'target': dace.ScheduleType.FPGA_Device,
                                      }) == 0


def test_vec_two_maps_nested_illegal():
    sdfg = two_maps_kernel_nested_illegal.to_sdfg()

    assert sdfg.apply_transformations_repeated(MapExpansion) == 2

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target': dace.ScheduleType.FPGA_Device,
                                      }) == 0


def vec_matadd(strided_map):
    sdfg: dace.SDFG = matadd_kernel.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target': dace.ScheduleType.FPGA_Device,
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
                                          'target': dace.ScheduleType.FPGA_Device,
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
                                          'target': dace.ScheduleType.FPGA_Device,
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
                                          'target': dace.ScheduleType.FPGA_Device,
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
                                          'target': dace.ScheduleType.FPGA_Device,
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


def vec_matadd_multi(strided_map):
    # Make SDFG
    sdfg: dace.SDFG = matadd_multi_kernel.to_sdfg()
    # Transform
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target': dace.ScheduleType.FPGA_Device,
                                          'strided_map': strided_map
                                      }) == 1

    # Run verification
    A = np.random.rand(SIZE, SIZE).astype(np.float32)
    B = np.random.rand(SIZE, SIZE).astype(np.float32)
    C = np.random.rand(SIZE, SIZE).astype(np.float32)
    D = np.random.rand(SIZE, SIZE).astype(np.float32)

    sdfg(A=A, B=B, C=C, D=D, M=SIZE, N=SIZE)

    diff1 = np.linalg.norm(C - (A + B))
    diff2 = np.linalg.norm(D - (A - B))
    assert diff1 <= 1e-5 and diff2 <= 1e-5

    return sdfg


def test_vec_not_applicable():

    sdfg2: dace.SDFG = matadd_bad_stride_kernel.to_sdfg()
    sdfg2.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg2.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': True
                                       }) == 0

    assert sdfg2.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': False
                                       }) == 0

    sdfg3: dace.SDFG = vecadd_1_non_appl_0_kernel.to_sdfg()
    sdfg3.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg3.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': False
                                       }) == 0

    assert sdfg3.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': True
                                       }) == 0

    sdfg4: dace.SDFG = vecadd_1_non_appl_1_kernel.to_sdfg()
    sdfg4.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg4.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': False
                                       }) == 0

    assert sdfg4.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': True
                                       }) == 0

    sdfg5: dace.SDFG = maporder_vec_kernel.to_sdfg()

    sdfg5.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg5.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': False
                                       }) == 0

    assert sdfg5.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': True
                                       }) == 0

    sdfg6: dace.SDFG = matmul_vec_kernel.to_sdfg()

    sdfg6.apply_transformations([FPGATransformSDFG, InlineSDFG])

    assert sdfg6.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': False
                                       }) == 0

    assert sdfg6.apply_transformations(Vectorization,
                                       options={
                                           'vector_len': 2,
                                           'target': dace.ScheduleType.FPGA_Device,
                                           'strided_map': True
                                       }) == 0


@fpga_test()
def test_vec_two_maps_strided():
    return vec_two_maps(True)


@fpga_test()
def test_vec_two_maps_non_strided():
    return vec_two_maps(False)


@xilinx_test()
def test_vec_two_maps_nested_strided():
    return vec_two_maps_nested(True)


@xilinx_test()
def test_vec_two_maps_nested_non_strided():
    return vec_two_maps_nested(False)


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


@xilinx_test()
def test_vec_matadd_multi_non_stride():
    return vec_matadd_multi(False)


@xilinx_test()
def test_vec_matadd_multi_stride():
    return vec_matadd_multi(True)


@dace.program
def bicg(A: dace.float32[N, M], p: dace.float32[M], r: dace.float32[N]):
    return r @ A, A @ p


@xilinx_test()
def test_vec_bicg():

    A = np.random.rand(SIZE, SIZE).astype(np.float32)
    p = np.random.rand(SIZE).astype(np.float32)
    r = np.random.rand(SIZE).astype(np.float32)

    # Parse SDFG and apply FPGA friendly optimization
    sdfg = bicg.to_sdfg(simplify=True)

    applied = sdfg.apply_transformations([FPGATransformSDFG])
    assert applied == 1

    fpga_rr_interleave_containers_to_banks(sdfg, num_banks=2)

    # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
    from dace.libraries.blas import Gemv
    Gemv.default_implementation = "FPGA_Accumulate"
    sdfg.expand_library_nodes()
    sm_applied = sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    assert sm_applied == 3  # 3 inlines

    sdfg.apply_transformations_repeated(Vectorization,
                                        options={
                                            'vector_len': 2,
                                            'target': dace.ScheduleType.FPGA_Device,
                                            'strided_map': True
                                        },
                                        print_report=True)

    # specialize the SDFG (needed by the GEMV expansion)
    sdfg.specialize(dict(M=SIZE, N=SIZE))

    res0, res1 = sdfg(A=A, p=p, r=r)

    # Compute ground truth and Validate result
    res0_ref, res1_ref = bicg.f(A, p, r)

    assert np.allclose(res0_ref, res0)
    assert np.allclose(res1, res1_ref)

    return sdfg


if __name__ == "__main__":
    test_vec_add_1_stride(None)
    test_vec_add_1_non_stride(None)
    test_vec_add_1_stride_sym(None)
    test_vec_add_1_non_stride_sym(None)

    test_vec_two_maps_strided(None)
    test_vec_two_maps_non_strided(None)
    test_vec_two_maps_illegal()

    test_vec_two_maps_nested_strided(None)
    test_vec_two_maps_nested_non_strided(None)
    test_vec_two_maps_nested_illegal()

    test_vec_matadd_stride(None)
    test_vec_matadd_non_stride(None)

    test_vec_matadd_stride_sym(None)
    test_vec_matadd_non_stride_sym(None)

    test_vec_sum_vectorize_first_strided(None)
    test_vec_sum_vectorize_first_non_strided(None)
    test_vec_sum_fpga_transform_first_strided(None)
    test_vec_sum_fpga_transform_first_non_strided(None)

    test_vec_tensor_add_stride(None)
    test_vec_tensor_add_non_stride(None)

    test_vec_matadd_multi_non_stride(None)
    test_vec_matadd_multi_stride(None)

    test_vec_not_applicable()

    test_vec_bicg(None)

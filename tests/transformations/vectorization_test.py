# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import Vectorization
from numpy.core.numeric import allclose, isclose

N = dace.symbol('N')


@dace.program
def tovec(A: dace.float64[20]):
    return A + A

@dace.program
def tovec2(A: dace.float64[20]):
    return A + A


def copy_kernel(type1=dace.float32, type2=dace.float32):
    @dace.program
    def copy(A: type1[N], B: type2[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    return copy


@dace.program
def matrix_copy(A: dace.float32[N, N], B: dace.float32[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]
            b = a


@dace.program
def tovec_sym(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    @dace.map
    def sum(i: _[0:N]):
        xx << x[i]
        yy << y[i]
        zz << z[i]
        out >> z[i]

        out = xx + yy + zz


@dace.program
def tovec_uneven(A: dace.float64[N + 2]):
    for i in dace.map[1:N + 1]:
        with dace.tasklet:
            a << A[i]
            b >> A[i]
            b = a + a


@dace.program
def diag_stride(A: dace.float32[N, N], B: dace.float32[N, N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i, i]
            b >> B[i, i]
            b = a


def test_vectorization():
    sdfg: dace.SDFG = tovec.to_sdfg()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len': 2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B, A * 2)


def test_basic_stride():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)
    sdfg(A=A, B=B, N=N.get())
    assert allclose(A, B)


def test_basic_stride_vec2():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization, {"vector_len": 2}) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)
    sdfg(A=A, B=B, N=N.get())
    assert allclose(A, B)


def test_basic_stride_matrix():

    sdfg = matrix_copy.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


def test_basic_stride_non_strided_map():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization,
                                      {"strided_map": False}) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


def test_basic_stride_matrix_non_strided_map():

    sdfg = matrix_copy.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization,
                                      {"strided_map": False}) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


def test_wrong_targets():
    sdfg: dace.SDFG = tovec.to_sdfg()

    wrong_targets = [
        dace.ScheduleType.Sequential,
        dace.ScheduleType.MPI,
        dace.ScheduleType.CPU_Multicore,
        dace.ScheduleType.Unrolled,
        dace.ScheduleType.GPU_Default,
        dace.ScheduleType.GPU_Device,
        dace.ScheduleType.GPU_ThreadBlock,
        dace.ScheduleType.GPU_ThreadBlock_Dynamic,
        dace.ScheduleType.GPU_Persistent,
    ]

    for t in wrong_targets:

        assert sdfg.apply_transformations(Vectorization, {"target": t}) == 0


def test_irregular_stride():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N * N]:
            with dace.tasklet:
                a << A[i * i]
                b >> B[i * i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # [i * i] has a stride of 2i + 1 which is not constant (cannot be vectorized)
    assert sdfg.apply_transformations(Vectorization) == 0


def test_diagonal_stride():

    sdfg = diag_stride.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 0

# def test_supported_wcr_sum():
#     @dace.program
#     def program(A: dace.float32[N], B: dace.float32[1]):
#         for i in dace.map[0:N]:
#             with dace.tasklet:
#                 a << A[i]
#                 b >> B(-1, lambda x, y: x + y)[0]
#                 b = a

#     sdfg = program.to_sdfg(strict=True)
#     assert sdfg.apply_transformations(Vectorization) == 1

#     N.set(64)
#     A = np.random.rand(N.get()).astype(np.float32)
#     B = np.zeros(1).astype(np.float32)
#     sdfg(A=A, B=B, N=N.get())
#     assert allclose(np.sum(A), B)

# def test_supported_wcr_max():
#     @dace.program
#     def program(A: dace.float32[N], B: dace.float32[1]):
#         for i in dace.map[0:N]:
#             with dace.tasklet:
#                 a << A[i]
#                 b >> B(-1, lambda x, y: max(x,y))[0]
#                 b = a

#     sdfg = program.to_sdfg(strict=True)
#     assert sdfg.apply_transformations(Vectorization) == 1

#     N.set(64)
#     A = np.random.rand(N.get()).astype(np.float32)
#     B = np.zeros(1).astype(np.float32)
#     sdfg(A=A, B=B, N=N.get())
#     assert allclose(np.max(A), B)

# def test_supported_wcr_min():
#     @dace.program
#     def program(A: dace.float32[N], B: dace.float32[1]):
#         for i in dace.map[0:N]:
#             with dace.tasklet:
#                 a << A[i]
#                 b >> B(-1, lambda x, y: min(x,y))[0]
#                 b = a

#     sdfg = program.to_sdfg(strict=True)
#     assert sdfg.apply_transformations(Vectorization) == 1

#     N.set(64)
#     A = np.random.rand(N.get()).astype(np.float32)
#     B = np.zeros(1).astype(np.float32)
#     sdfg(A=A, B=B, N=N.get())
#     assert allclose(np.min(A), B)


def test_unsupported_wcr_ptr():
    @dace.program
    def program(A: dace.pointer(dace.float32)[N],
                B: dace.pointer(dace.float32)[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 0


def test_unsupported_wcr_vec():
    @dace.program
    def program(A: dace.vector(dace.float32, 4)[N],
                B: dace.vector(dace.float32, 4)[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 0


def test_vectorization_uneven():
    sdfg: dace.SDFG = tovec_uneven.to_sdfg()

    A = np.ones([22], np.float64)
    result = np.array([1.] + [2.] * 20 + [1.], dtype=np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)

    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations(Vectorization, options={'vector_len': 2}) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code

    A = np.ones([22], np.float64)
    sdfg(A=A, N=20)
    assert np.allclose(A, result)


def test_vectorization_postamble():
    sdfg: dace.SDFG = tovec_sym.to_sdfg()
    sdfg.coarsen_dataflow()
    assert sdfg.apply_transformations(Vectorization) == 1
    assert 'vec<float, 4>' in sdfg.generate_code()[0].code
    csdfg = sdfg.compile()

    for N in range(24, 29):
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        z = np.random.rand(N).astype(np.float32)
        expected = x + y + z

        csdfg(x=x, y=y, z=z, N=N)
        assert np.allclose(z, expected)


def test_preamble():

    N.set(24)

    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[3:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization) == 1

    A = np.ndarray([N.get()], dtype=np.float32)
    B = np.ndarray([N.get()], dtype=np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert np.allclose(A[3:N.get()], B[3:N.get()])


def test_propagate_parent_stride():
    sdfg: dace.SDFG = tovec2.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target': dace.ScheduleType.FPGA_Device
                                      }) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B.reshape(20), A * 2)
    return sdfg

def test_propagate_parent_non_stride():
    sdfg: dace.SDFG = tovec.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization,
                                      options={
                                          'vector_len': 2,
                                          'target': dace.ScheduleType.FPGA_Device,
                                          'strided_map': False
                                      }) == 1
    assert 'vec<double, 2>' in sdfg.generate_code()[0].code
    A = np.random.rand(20)
    B = sdfg(A=A)
    assert np.allclose(B.reshape(20), A * 2)
    return sdfg


def test_supported_types():

    types = [
        # np.bool_,
        # np.int8,
        np.int16,
        np.int32,
        # np.int64,
        np.intc,
        np.uint8,
        np.uint16,
        np.uint32,
        # np.uint64,
        np.uintc,
        # np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]

    for t in types:

        sdfg = copy_kernel(dace.DTYPE_TO_TYPECLASS[t],
                           dace.DTYPE_TO_TYPECLASS[t]).to_sdfg(strict=True)
        assert sdfg.apply_transformations(Vectorization, {'vector_len': 2}) == 1

        N.set(64)

        A = np.random.rand(N.get()).astype(t)
        B = np.random.rand(N.get()).astype(t)

        sdfg(A=A, B=B, N=N.get())

        assert allclose(A, B)


if __name__ == '__main__':
    test_vectorization()
    test_basic_stride()
    test_basic_stride_vec2()
    test_basic_stride_matrix()
    test_basic_stride_non_strided_map()
    test_basic_stride_matrix_non_strided_map()
    test_wrong_targets()
    test_irregular_stride()
    test_diagonal_stride()
    # test_supported_wcr_sum()
    # test_supported_wcr_min()
    # test_supported_wcr_max()
    test_unsupported_wcr_ptr()
    test_unsupported_wcr_vec()
    test_vectorization_uneven()
    test_vectorization_postamble()
    test_preamble()
    test_propagate_parent_stride()
    test_propagate_parent_non_stride()
    test_supported_types()

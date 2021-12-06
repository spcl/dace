# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numpy.core.numeric import allclose
from dace.sdfg.graph import NodeNotFoundError
import dace
import numpy as np
from dace.transformation.dataflow.vectorization import Vectorization
from dace import SDFG
import dace.dtypes as dtypes
import pytest

N = dace.symbol('N')


def find_connector_by_name(sdfg: SDFG, name: str):
    """
    Utility function to obtain the type of a connector by its name
    """
    for node, _ in sdfg.start_state.all_nodes_recursive():
        if name in node.in_connectors:
            return node.in_connectors[name]
        elif name in node.out_connectors:
            return node.out_connectors[name]

    raise RuntimeError(f'Could not find connector "{name}"')


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
def diag_stride(A: dace.float32[N, N], B: dace.float32[N, N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i, i]
            b >> B[i, i]
            b = a


@pytest.mark.sve
def test_basic_stride():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


@pytest.mark.sve
def test_basic_stride_vec8():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
        "vector_len": 8
    }) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


@pytest.mark.sve
def test_basic_stride_matrix():

    sdfg = matrix_copy.to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


@pytest.mark.sve
def test_basic_stride_non_strided_map():

    sdfg = copy_kernel().to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
        "strided_map": False
    }) == 1

    N.set(64)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


@pytest.mark.sve
def test_basic_stride_matrix_non_strided_map():

    sdfg = matrix_copy.to_sdfg(strict=True)
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
        "strided_map": False
    }) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert allclose(A, B)


@pytest.mark.sve
def test_supported_types():

    types = [
        np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
        np.uint64, np.float16, np.float32, np.float64
    ]

    for t in types:

        sdfg = copy_kernel(dace.DTYPE_TO_TYPECLASS[t],
                           dace.DTYPE_TO_TYPECLASS[t]).to_sdfg(strict=True)
        assert sdfg.apply_transformations(
            Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

        N.set(64)

        A = np.random.rand(N.get()).astype(t)
        B = np.random.rand(N.get()).astype(t)

        sdfg(A=A, B=B, N=N.get())

        assert allclose(A, B)


def test_multiple_bit_widths():
    sdfg = copy_kernel(dace.float32, dace.float64).to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


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
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


@pytest.mark.sve
def test_diagonal_stride():

    sdfg = diag_stride.to_sdfg(strict=True)
    # [i, i] has a stride of N + 1, so it is perfectly fine
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())
    assert allclose(A, B)


@pytest.mark.sve
def test_diagonal_stride_non_strided_map():

    sdfg = diag_stride.to_sdfg(strict=True)
    # [i, i] has a stride of N + 1, so it is perfectly fine
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
        "strided_map": False
    }) == 1

    N.set(64)

    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    B = np.random.rand(N.get(), N.get()).astype(np.float32)

    sdfg(A=A, B=B, N=N.get())
    assert allclose(A, B)


def test_unsupported_types():

    sdfg = copy_kernel(dace.complex64, dace.complex64).to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0

    sdfg = copy_kernel(dace.float32, dace.complex64).to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0

    sdfg = copy_kernel(dace.vector(dace.float32, 4),
                       dace.vector(dace.float32, 4)).to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0

    sdfg = copy_kernel(dace.pointer(dace.float32),
                       dace.pointer(dace.float32)).to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


@pytest.mark.sve
def test_supported_wcr_sum():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x + y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)
    sdfg(A=A, B=B, N=N.get())
    assert allclose(np.sum(A), B)


@pytest.mark.sve
def test_supported_wcr_min():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: min(x, y))[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)
    sdfg(A=A, B=B, N=N.get())
    assert allclose(np.min(A), B)


@pytest.mark.sve
def test_supported_wcr_max():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: max(x, y))[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1

    N.set(64)
    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)
    sdfg(A=A, B=B, N=N.get())
    assert allclose(np.max(A), B)


def test_unsupported_wcr():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B(-1, lambda x, y: x * y)[0]
                b = a

    sdfg = program.to_sdfg(strict=True)
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


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
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


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
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


def test_first_level_vectorization():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i, j in dace.map[0:N, 0:N]:
            with dace.tasklet:
                a_scal << A[i]
                a_vec << A[j]
                b >> B[j]
                b = a_vec

    sdfg = program.to_sdfg(strict=True)
    sdfg.apply_transformations(Vectorization,
                               {"target": dace.ScheduleType.SVE_Map})

    # i is constant in the vectorized map
    assert not isinstance(find_connector_by_name(sdfg, 'a_scal'), dtypes.vector)
    # j is the innermost param
    assert isinstance(find_connector_by_name(sdfg, 'a_vec'), dtypes.vector)


def test_stream_push():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_out = dace.define_stream(dace.float32, N)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> S_out(-1)
                b = a
        S_out >> B

    sdfg = program.to_sdfg(strict=True)
    # Stream push is possible
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 1


def test_stream_pop():
    @dace.program(dace.float32[N], dace.float32[N])
    def program(A, B):
        S_in = dace.define_stream(dace.float32, N)
        S_in << A
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << S_in(-1)
                b >> B[i]
                b = a

    sdfg = program.to_sdfg(strict=True)
    # Stream pop is not implemented yet
    assert sdfg.apply_transformations(
        Vectorization, {"target": dace.ScheduleType.SVE_Map}) == 0


@pytest.mark.sve
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
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
    }) == 1

    A = np.ndarray([N.get()], dtype=np.float32)
    B = np.ndarray([N.get()], dtype=np.float32)

    sdfg(A=A, B=B, N=N.get())

    assert np.allclose(A[3:N.get()], B[3:N.get()])


@pytest.mark.sve
def test_postamble():
    @dace.program
    def program(A: dace.float32[N], B: dace.float32[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = program.to_sdfg()
    assert sdfg.apply_transformations(Vectorization, {
        "target": dace.ScheduleType.SVE_Map,
    }) == 1

    n = 26

    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)

    sdfg(A=x, B=y, N=n)
    assert np.allclose(x, y)


if __name__ == '__main__':
    test_basic_stride()
    test_basic_stride_vec8()
    test_basic_stride_matrix()
    test_basic_stride_non_strided_map()
    test_basic_stride_matrix_non_strided_map()
    test_supported_types()
    test_irregular_stride()
    test_diagonal_stride()
    test_diagonal_stride_non_strided_map()
    test_unsupported_types()
    test_supported_wcr_sum()
    test_supported_wcr_min()
    test_supported_wcr_max()
    test_unsupported_wcr()
    test_unsupported_wcr_vec()
    test_unsupported_wcr_ptr()
    test_first_level_vectorization()
    test_stream_push()
    test_stream_pop()
    test_multiple_bit_widths()
    test_preamble()
    test_postamble()

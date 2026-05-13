# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy
import pytest
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N,
    ssym,
    X,
    Y,
)


@dace.program
def vecscale_unit_stride(src: dace.float64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i] * scale


@dace.program
def gather_load(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[idx[i]] * scale


@dace.program
def gather_load_matrix_specialized(A: dace.float32[4, 8192], B: dace.int32[4, 8192], C: dace.float32[4, 8192]):
    for i, j in dace.map[0:4:1, 0:8192:1]:
        C[i, j] = A[i, B[i, j]] * 2.0


@dace.program
def strided_load_stride_2(src: dace.float64[2 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 2] * scale


@dace.program
def strided_load_stride_ssym(src: dace.float64[ssym * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * ssym] * scale


@dace.program
def strided_load_stride_3(src: dace.float64[3 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 3] * scale


@dace.program
def strided_load_stride_4(src: dace.float64[4 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 4] * scale


@dace.program
def strided_load_stride_5(src: dace.float64[5 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 5] * scale


@dace.program
def strided_load_stride_6(src: dace.float64[6 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 6] * scale


@dace.program
def strided_load_stride_7(src: dace.float64[7 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 7] * scale


@dace.program
def strided_load_stride_8(src: dace.float64[8 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 8] * scale


@dace.program
def strided_load_stride_16(src: dace.float64[16 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 16] * scale


@dace.program
def scatter_store(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[idx[i]] = src[i] * scale


@dace.program
def strided_store_stride_2(src: dace.float64[N], dst: dace.float64[2 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 2] = src[i] * scale


@dace.program
def strided_store_stride_ssym(src: dace.float64[N], dst: dace.float64[ssym * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * ssym] = src[i] * scale


@dace.program
def strided_store_stride_3(src: dace.float64[N], dst: dace.float64[3 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 3] = src[i] * scale


@dace.program
def strided_store_stride_4(src: dace.float64[N], dst: dace.float64[4 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 4] = src[i] * scale


@dace.program
def strided_store_stride_5(src: dace.float64[N], dst: dace.float64[5 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 5] = src[i] * scale


@dace.program
def strided_store_stride_6(src: dace.float64[N], dst: dace.float64[6 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 6] = src[i] * scale


@dace.program
def strided_store_stride_7(src: dace.float64[N], dst: dace.float64[7 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 7] = src[i] * scale


@dace.program
def strided_store_stride_8(src: dace.float64[N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 8] = src[i] * scale


@dace.program
def strided_store_stride_16(src: dace.float64[N], dst: dace.float64[16 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 16] = src[i] * scale


def test_vecscale_unit_stride():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=vecscale_unit_stride,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="vecscale_unit_stride",
    )


def test_gather_load():
    N = 64
    src = numpy.random.random(N)
    idx = numpy.random.permutation(N).astype(numpy.int64)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=gather_load,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="gather_load",
    )


def test_gather_load_matrix_specialized():
    Y_val = 4
    X_val = 8192
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)  # Random float32 values
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)  # Random indices in [0, 8192)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)  # Output array initialized to zeros

    run_vectorization_test(
        dace_func=gather_load_matrix_specialized,
        arrays={
            "A": A,
            "B": B,
            "C": C
        },
        params={},
        vector_width=32,
        sdfg_name="gather_load_matrix_specialized",
    )


def test_strided_load_stride_2():
    N = 64
    src = numpy.random.random(2 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_2",
    )


def test_strided_load_stride_ssym():
    N = 64
    _ssym = 2
    src = numpy.random.random(_ssym * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        sdfg_name="strided_load_stride_ssym",
    )


def test_strided_load_stride_3():
    N = 64
    src = numpy.random.random(3 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_3,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_3",
    )


def test_strided_load_stride_4():
    N = 64
    src = numpy.random.random(4 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_4,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        insert_copies=False,
        sdfg_name="strided_load_stride_4",
    )


def test_strided_load_stride_5():
    N = 64
    src = numpy.random.random(5 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_5,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_5",
    )


def test_strided_load_stride_6():
    N = 64
    src = numpy.random.random(6 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_6,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        insert_copies=True,
        fuse_overlapping_loads=True,
        sdfg_name="strided_load_stride_6",
    )


def test_strided_load_stride_7():
    N = 64
    src = numpy.random.random(7 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_7,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_7",
    )


def test_strided_load_stride_8():
    N = 64
    src = numpy.random.random(8 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_8,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_8",
    )


def test_strided_load_stride_16():
    N = 64
    src = numpy.random.random(16 * N)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=strided_load_stride_16,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_load_stride_16",
    )


def test_scatter_store():
    N = 64
    src = numpy.random.random(N)
    idx = numpy.random.permutation(N).astype(numpy.int64)
    dst = numpy.zeros(N)
    run_vectorization_test(
        dace_func=scatter_store,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="scatter_store",
    )


def test_strided_store_stride_2():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_2",
    )


def test_strided_store_stride_ssym():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    _ssym = numpy.int64(2)
    run_vectorization_test(
        dace_func=strided_store_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        sdfg_name="strided_store_stride_ssym",
    )


def test_strided_store_stride_3():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(3 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_3,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        insert_copies=True,
        fuse_overlapping_loads=True,
        sdfg_name="strided_store_stride_3",
    )


def test_strided_store_stride_4():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(4 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_4,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_4",
    )


def test_strided_store_stride_5():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(5 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_5,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_5",
    )


def test_strided_store_stride_6():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(6 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_6,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_6",
    )


def test_strided_store_stride_7():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(7 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_7,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_7",
    )


def test_strided_store_stride_8():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(8 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_8,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_8",
    )


def test_strided_store_stride_16():
    N = 64
    src = numpy.random.random(N)
    dst = numpy.zeros(16 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_16,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="strided_store_stride_16",
    )


def test_strided_store_stride_ssym():
    N = 64
    _ssym = 2
    src = numpy.random.random(N)
    dst = numpy.zeros(2 * N)
    run_vectorization_test(
        dace_func=strided_store_stride_ssym,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        sdfg_name="strided_store_stride_ssym",
    )


@dace.program
def nested_matrix_gather_load(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X], scale: dace.float32):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * scale


@dace.program
def nested_matrix_gather_load_specialized(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X]):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * 2.0


def test_nested_matrix_gather_load():
    X_val = 32
    Y_val = 32
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)
    run_vectorization_test(
        dace_func=nested_matrix_gather_load,
        arrays={
            "A": A,
            "B": B,
            "C": C,
        },
        params={
            "X": X_val,
            "Y": Y_val,
            "scale": 2.0
        },
        vector_width=8,
        sdfg_name="nested_matrix_gather_load",
    )


def test_nested_matrix_gather_load_specialized():
    X_val = 32
    Y_val = 32
    A = numpy.random.rand(Y_val, X_val).astype(numpy.float32)
    B = numpy.random.randint(0, X_val, size=(Y_val, X_val), dtype=numpy.int32)
    C = numpy.zeros((Y_val, X_val), dtype=numpy.float32)
    run_vectorization_test(
        dace_func=nested_matrix_gather_load_specialized,
        arrays={
            "A": A,
            "B": B,
            "C": C,
        },
        params={
            "X": X_val,
            "Y": Y_val,
        },
        vector_width=8,
        sdfg_name="nested_matrix_gather_load_specialized",
    )


# Diagonal access patterns: the index used in the inner map appears in MULTIPLE
# array dimensions, so the access is neither contiguous nor a pure index-array
# gather. The natural lowering is a gather (read) / scatter (write) over
# A.strides[0]*i + A.strides[1]*i = (sum_of_strides)*i.


@dace.program
def diagonal_gather_load(A: dace.float64[N, N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = A[i, i] * scale


@dace.program
def diagonal_scatter_store(src: dace.float64[N], A: dace.float64[N, N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        A[i, i] = src[i] * scale


@dace.program
def gather_load_2i_i(A: dace.float64[2 * N, N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = A[2 * i, i] * scale


@dace.program
def scatter_store_2i_i(src: dace.float64[N], A: dace.float64[2 * N, N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        A[2 * i, i] = src[i] * scale


@dace.program
def gather_load_i_2i(A: dace.float64[N, 2 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = A[i, 2 * i] * scale


@dace.program
def scatter_store_i_2i(src: dace.float64[N], A: dace.float64[N, 2 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        A[i, 2 * i] = src[i] * scale


def test_diagonal_gather_load():
    N_val = 64
    A = numpy.random.rand(N_val, N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=diagonal_gather_load,
        arrays={
            "A": A,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="diagonal_gather_load",
    )


def test_diagonal_scatter_store():
    N_val = 64
    src = numpy.random.rand(N_val)
    A = numpy.zeros((N_val, N_val))
    run_vectorization_test(
        dace_func=diagonal_scatter_store,
        arrays={
            "src": src,
            "A": A
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="diagonal_scatter_store",
    )


def test_gather_load_2i_i():
    N_val = 64
    A = numpy.random.rand(2 * N_val, N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=gather_load_2i_i,
        arrays={
            "A": A,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="gather_load_2i_i",
    )


def test_scatter_store_2i_i():
    N_val = 64
    src = numpy.random.rand(N_val)
    A = numpy.zeros((2 * N_val, N_val))
    run_vectorization_test(
        dace_func=scatter_store_2i_i,
        arrays={
            "src": src,
            "A": A
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="scatter_store_2i_i",
    )


def test_gather_load_i_2i():
    N_val = 64
    A = numpy.random.rand(N_val, 2 * N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=gather_load_i_2i,
        arrays={
            "A": A,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="gather_load_i_2i",
    )


def test_scatter_store_i_2i():
    N_val = 64
    src = numpy.random.rand(N_val)
    A = numpy.zeros((N_val, 2 * N_val))
    run_vectorization_test(
        dace_func=scatter_store_i_2i,
        arrays={
            "src": src,
            "A": A
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="scatter_store_i_2i",
    )


# Halve-index access pattern: ``c[i // 2]`` — each pair of consecutive
# iterations shares a source element. Lowering candidate for the
# ``multiplex`` pattern detector (utils/multiplex.py).
#
# Kernel shape borrowed from TSVC s4117:
#   a[i] = b[i] + c[i // 2] * d[i]
# Plus an isolated form ``dst[i] = src[i // 2]`` to exercise the pattern
# on its own without the surrounding arithmetic.


@dace.program
def halve_index_gather(src: dace.float64[N], dst: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i // 2]


@dace.program
def halve_index_s4117(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        a[i] = b[i] + c[i // 2] * d[i]


def test_halve_index_gather():
    N_val = 64
    src = numpy.random.rand(N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=halve_index_gather,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val
        },
        vector_width=8,
        sdfg_name="halve_index_gather",
    )


def test_halve_index_s4117():
    N_val = 64
    a = numpy.zeros(N_val)
    b = numpy.random.rand(N_val)
    c = numpy.random.rand(N_val)
    d = numpy.random.rand(N_val)
    run_vectorization_test(
        dace_func=halve_index_s4117,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "N": N_val
        },
        vector_width=8,
        sdfg_name="halve_index_s4117",
    )


# fp32 variants: one test per detect_*.py file exercising the templated
# runtime-length intrinsics (gather/scatter/strided_load/strided_store) on a
# non-double element type. The runtime header's scalar fallback runs (the
# AVX-512 / SVE paths are gated on ``T == double`` via ``if constexpr``) but
# the templated form must still compile and produce correct results.


@dace.program
def gather_load_fp32(src: dace.float32[N], idx: dace.int64[N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = src[idx[i]] * scale


@dace.program
def scatter_store_fp32(src: dace.float32[N], idx: dace.int64[N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[idx[i]] = src[i] * scale


@dace.program
def strided_load_stride_2_fp32(src: dace.float32[2 * N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 2] * scale


@dace.program
def strided_store_stride_2_fp32(src: dace.float32[N], dst: dace.float32[2 * N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i * 2] = src[i] * scale


@dace.program
def diagonal_gather_load_fp32(A: dace.float32[N, N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = A[i, i] * scale


@dace.program
def diagonal_scatter_store_fp32(src: dace.float32[N], A: dace.float32[N, N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        A[i, i] = src[i] * scale


def test_gather_load_fp32():
    N_val = 64
    src = numpy.random.rand(N_val).astype(numpy.float32)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=gather_load_fp32,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="gather_load_fp32",
    )


def test_scatter_store_fp32():
    N_val = 64
    src = numpy.random.rand(N_val).astype(numpy.float32)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=scatter_store_fp32,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="scatter_store_fp32",
    )


def test_strided_load_stride_2_fp32():
    N_val = 64
    src = numpy.random.rand(2 * N_val).astype(numpy.float32)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=strided_load_stride_2_fp32,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="strided_load_stride_2_fp32",
    )


def test_strided_store_stride_2_fp32():
    N_val = 64
    src = numpy.random.rand(N_val).astype(numpy.float32)
    dst = numpy.zeros(2 * N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=strided_store_stride_2_fp32,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="strided_store_stride_2_fp32",
    )


def test_diagonal_gather_load_fp32():
    N_val = 64
    A = numpy.random.rand(N_val, N_val).astype(numpy.float32)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=diagonal_gather_load_fp32,
        arrays={
            "A": A,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="diagonal_gather_load_fp32",
    )


def test_diagonal_scatter_store_fp32():
    N_val = 64
    src = numpy.random.rand(N_val).astype(numpy.float32)
    A = numpy.zeros((N_val, N_val), dtype=numpy.float32)
    run_vectorization_test(
        dace_func=diagonal_scatter_store_fp32,
        arrays={
            "src": src,
            "A": A
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name="diagonal_scatter_store_fp32",
    )


# Non-divisible-N tests for gather / scatter / strided. Locked decision
# (2026-05-14, Option B in the plan): masked-remainder + indirection
# requires intrinsic collapse + a verifier post-detect — landed as part of
# R2.
#
# - ``remainder_strategy="scalar"``: main map vectorizes on the W-aligned
#   head (W*floor(N/W) elements); postamble is W=1 sequential. Gather and
#   scatter work today because the postamble is a plain scalar tasklet
#   (no _packed node, no lane fanout to mis-handle). Strided load / store
#   currently raise ``NotImplementedError`` from the defensive guard in
#   ``_process_edges`` (subset volume > vector_width on the strided-but-
#   classified-vectorizable case under scalar mode) — todo #11
#   (scalar-postamble investigation).
# - ``remainder_strategy="masked"``: R2 (this slice). Requires
#   ``lower_to_intrinsics=True`` per the locked decision: the per-lane
#   memlet reads in an uncollapsed fan would fault on inactive lanes.
#   The verifier in ``utils/iteration.assert_no_lane_memlet_reads``
#   enforces this contract post-collapse.
#
# N=17, W=8 ⇒ main covers 16, remainder covers 1.

@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_gather_load_nondiv(remainder_strategy):
    # N=22 exercises remainder R=6; the previous N=17 case used R=1 which
    # happens to land on a single OMP iteration writing one element by
    # accident, masking the architectural gap (verified end-to-end against
    # the scalar reference).
    N_val = 22
    src = numpy.random.rand(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=gather_load,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name=f"gather_load_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_scatter_store_nondiv(remainder_strategy):
    N_val = 22
    src = numpy.random.rand(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=scatter_store,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name=f"scatter_store_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_strided_load_stride_2_nondiv(remainder_strategy):
    N_val = 17
    src = numpy.random.rand(2 * N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=strided_load_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name=f"strided_load_stride_2_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_strided_store_stride_2_nondiv(remainder_strategy):
    N_val = 17
    src = numpy.random.rand(N_val)
    dst = numpy.zeros(2 * N_val)
    run_vectorization_test(
        dace_func=strided_store_stride_2,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name=f"strided_store_stride_2_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


# Non-divisible-N coverage for strides 3 / 4 / 5 / 8. Stride 2 already
# exercised above (test_strided_*_stride_2_nondiv). The strided-inside-
# NSDFG path (Slice 1 ``_setup_strided_inside_nsdfg``) handles the
# ``bbox = (W-1)*stride + 1`` formula for any integer stride that
# divides cleanly.
@pytest.mark.parametrize("stride", [3, 4, 5, 8])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_strided_load_stride_n_nondiv(stride, remainder_strategy):
    N_val = 22  # non-divisible by W=8, R=6 lanes active in the remainder
    src = numpy.random.rand(stride * N_val)
    dst = numpy.zeros(N_val)
    prog = {3: strided_load_stride_3, 4: strided_load_stride_4,
            5: strided_load_stride_5, 8: strided_load_stride_8}[stride]
    run_vectorization_test(
        dace_func=prog,
        arrays={"src": src, "dst": dst},
        params={"N": N_val, "scale": 1.5},
        vector_width=8,
        sdfg_name=f"sl_stride_{stride}_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


@pytest.mark.parametrize("stride", [3, 4, 5, 8])
@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_strided_store_stride_n_nondiv(stride, remainder_strategy):
    N_val = 22
    src = numpy.random.rand(N_val)
    dst = numpy.zeros(stride * N_val)
    prog = {3: strided_store_stride_3, 4: strided_store_stride_4,
            5: strided_store_stride_5, 8: strided_store_stride_8}[stride]
    run_vectorization_test(
        dace_func=prog,
        arrays={"src": src, "dst": dst},
        params={"N": N_val, "scale": 1.5},
        vector_width=8,
        sdfg_name=f"ss_stride_{stride}_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )

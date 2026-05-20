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
def strided_load_stride_ssym(src: dace.float64[ssym * 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = src[i * ssym] * scale


@dace.program
def strided_load_stride_3(src: dace.float64[3 * N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 3] * scale


@dace.program
def scatter_store(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[idx[i]] = src[i] * scale


@dace.program
def strided_store_stride_2(src: dace.float64[N], dst: dace.float64[2 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 2] = src[i] * scale


@dace.program
def strided_store_stride_ssym(src: dace.float64[8 * N], dst: dace.float64[ssym * 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i * ssym] = src[i] * scale


@dace.program
def strided_store_stride_3(src: dace.float64[N], dst: dace.float64[3 * N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i * 3] = src[i] * scale


def test_vecscale_unit_stride(emission_style):
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
        emission_style=emission_style,
    )


def test_gather_load(emission_style):
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
        emission_style=emission_style,
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


def test_strided_load_stride_2(emission_style):
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
        emission_style=emission_style,
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N // 8,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        sdfg_name="strided_load_stride_ssym",
    )


def test_strided_load_stride_3(emission_style):
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
        emission_style=emission_style,
    )


def test_scatter_store(emission_style):
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
        emission_style=emission_style,
    )


def test_strided_store_stride_2(emission_style):
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
        emission_style=emission_style,
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N // 8,
            "scale": 1.5,
            "ssym": _ssym
        },
        vector_width=8,
        sdfg_name="strided_store_stride_ssym",
    )


def test_strided_store_stride_3(emission_style):
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
        emission_style=emission_style,
        fuse_overlapping_loads=True,
        sdfg_name="strided_store_stride_3",
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N // 8,
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


# Diagonal A[i,i]: map index in multiple dims; gather/scatter over
# (sum_of_strides)*i.


@dace.program
def diagonal_gather_load(A: dace.float64[N, N], dst: dace.float64[N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        dst[i] = A[i, i] * scale


@dace.program
def diagonal_scatter_store(src: dace.float64[N], A: dace.float64[N, N], scale: dace.float64):
    for i, in dace.map[0:N:1]:
        A[i, i] = src[i] * scale


@dace.program
def gather_load_2i_i(A: dace.float64[2 * 8 * N, 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = A[2 * i, i] * scale


@dace.program
def scatter_store_2i_i(src: dace.float64[8 * N], A: dace.float64[2 * 8 * N, 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        A[2 * i, i] = src[i] * scale


@dace.program
def gather_load_i_2i(A: dace.float64[8 * N, 2 * 8 * N], dst: dace.float64[8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
        dst[i] = A[i, 2 * i] * scale


@dace.program
def scatter_store_i_2i(src: dace.float64[8 * N], A: dace.float64[8 * N, 2 * 8 * N], scale: dace.float64):
    for i, in dace.map[0:8 * N:1]:
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N_val // 8,
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N_val // 8,
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N_val // 8,
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
            # kernel iterates 0:8*N — pass N=tiles so 8*N == array size,
            # and the trip is provably divisible by W=8 (no remainder).
            "N": N_val // 8,
            "scale": 1.5
        },
        vector_width=8,
        sdfg_name="scatter_store_i_2i",
    )


# Halve-index ``c[i // 2]`` (multiplex pattern): isolated form plus
# TSVC s4117 ``a[i] = b[i] + c[i // 2] * d[i]``.


@dace.program
def halve_index_gather(src: dace.float64[N], dst: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i // 2]


@dace.program
def halve_index_s4117(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    for i, in dace.map[0:N:1]:
        a[i] = b[i] + c[i // 2] * d[i]


def test_halve_index_gather(emission_style):
    N_val = 64
    src = numpy.random.rand(N_val)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=halve_index_gather,
        arrays={
            "src": src,
            "dst": dst
        },
        params={"N": N_val},
        vector_width=8,
        sdfg_name="halve_index_gather",
        emission_style=emission_style,
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
        params={"N": N_val},
        vector_width=8,
        sdfg_name="halve_index_s4117",
    )


# fp32 variants: templated runtime intrinsics must compile and be
# correct on a non-double element type (scalar fallback path).


@dace.program
def gather_load_fp32(src: dace.float32[N], idx: dace.int64[N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = src[idx[i]] * scale


@dace.program
def strided_load_stride_2_fp32(src: dace.float32[2 * N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 2] * scale


def test_gather_load_fp32(emission_style):
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
        emission_style=emission_style,
    )


def test_strided_load_stride_2_fp32(emission_style):
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
        emission_style=emission_style,
    )


# Non-divisible-N gather/scatter/strided. "scalar": W-aligned head +
# W=1 sequential postamble. "masked": needs lower_to_intrinsics=True
# (per-lane fan must collapse or it faults on inactive lanes).


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_gather_load_nondiv(remainder_strategy):
    # N=22 ⇒ remainder R=6 (R=1 would mask the gap).
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


# fp32 strided, divisible and non-divisible N, scalar/masked.


@dace.program
def strided_load_stride_2_fp32_nondiv(src: dace.float32[2 * N], dst: dace.float32[N], scale: dace.float32):
    for i, in dace.map[0:N:1]:
        dst[i] = src[i * 2] * scale


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_strided_load_fp32_stride_2_nondiv(remainder_strategy):
    N_val = 22
    src = numpy.random.rand(2 * N_val).astype(numpy.float32)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    run_vectorization_test(
        dace_func=strided_load_stride_2_fp32_nondiv,
        arrays={
            "src": src,
            "dst": dst
        },
        params={
            "N": N_val,
            "scale": numpy.float32(1.5)
        },
        vector_width=8,
        sdfg_name=f"sl_fp32_s2_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )


# Multi-dim strided (diagonal A[i,i]) under the NSDFG-wrapped
# scalar/masked remainder path (linearised-stride strided_load/store).


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_diagonal_gather_load_masked(remainder_strategy):
    N_val = 22
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
        sdfg_name=f"diag_gather_masked_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=True,
    )


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_diagonal_scatter_store_masked(remainder_strategy):
    N_val = 22
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
        sdfg_name=f"diag_scatter_masked_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=True,
    )


@dace.program
def strided_through_nsdfg(a: dace.float64[N], b: dace.float64[2 * N + 8]):
    # Constant-stride gather b[2*i] through the P1-nested NSDFG body
    # (strided/packed handler; A1.2 dropped the over-conservative raise).
    for i in dace.map[0:N]:
        a[i] = b[2 * i] + 1.0


def test_strided_through_nsdfg(remainder_strategy, branch_mode):
    n = 24
    a = numpy.zeros(n, dtype=numpy.float64)
    b = numpy.random.rand(2 * n + 8).astype(numpy.float64)
    run_vectorization_test(
        dace_func=strided_through_nsdfg,
        arrays={"a": a, "b": b},
        params={"N": n},
        sdfg_name="strided_through_nsdfg",
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )


# collapse_laneid_index_loads knob: the laneid index fan collapses to a
# direct ``_idx`` index-slice read; laneid symbols/ISE drop. Default OFF.


def _assert_laneid_fan_collapsed(vec_sdfg):
    """No laneid symbol / interstate assignment survives, and every gather
    tasklet reads its indices through an ``_idx`` connector."""
    from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme

    residual_syms = [
        s for sd in vec_sdfg.all_sdfgs_recursive() for s in sd.symbols if LaneIdScheme.is_laneid(s)
    ]
    assert not residual_syms, f"laneid symbols survived the collapse: {residual_syms}"
    residual_ise = [
        k for sd in vec_sdfg.all_sdfgs_recursive() for e in sd.edges() for k in (e.data.assignments or {})
        if LaneIdScheme.is_laneid(k)
    ]
    assert not residual_ise, f"laneid interstate-edge assignments survived: {residual_ise}"
    fan_tasklets = [
        n for n, _ in vec_sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and ("gather" in n.label or "scatter" in n.label)
    ]
    assert fan_tasklets, "expected at least one collapsed gather/scatter tasklet"
    for t in fan_tasklets:
        assert "_idx" in t.in_connectors, (f"{t.label} did not get an _idx connector; "
                                           f"in_connectors={set(t.in_connectors)}")
        assert "_laneid_" not in t.code.as_string, (f"{t.label} still references a laneid "
                                                    f"symbol: {t.code.as_string!r}")


def test_gather_load_collapse_laneid_structural():
    N_val = 64
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
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
        sdfg_name="gather_load_collapse_laneid",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


@dace.program
def gather_load_i32(src: dace.float64[N], idx: dace.int32[N], dst: dace.float64[N]):
    for i in dace.map[0:N]:
        dst[i] = src[idx[i]] + 1.0


def test_gather_collapse_laneid_noncontig_values():
    # Non-contiguous gather: the index *values* hit every 2nd source
    # element (a strided access pattern). The index table itself is read
    # as a contiguous slice, so the laneid fan still collapses; the
    # gather intrinsic does the non-contiguous source access.
    N_val = 64
    src = numpy.random.random(N_val)
    idx = ((numpy.arange(N_val) * 2) % N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
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
        sdfg_name="gather_collapse_laneid_noncontig",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_gather_collapse_laneid_int32_idx():
    # int32 index array exercises the conversion variant: a local int64
    # buffer is filled from _idx (element-width conversion the runtime
    # signature requires), still no laneid symbols.
    N_val = 64
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int32)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
        dace_func=gather_load_i32,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_collapse_laneid_i32",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)
    conv = [
        n for n, _ in vec_sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and "gather" in n.label and "__vec_lane_idx" in n.code.as_string
    ]
    assert conv, "int32 idx must use the int64-conversion gather variant"


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_gather_load_collapse_laneid_nondiv(remainder_strategy):
    # Non-divisible N (R=6): main + masked-remainder gather under the knob.
    N_val = 22
    src = numpy.random.rand(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
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
        sdfg_name=f"gather_load_collapse_laneid_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


# Scatter side. For-loop scatter needs loop_to_map_permissive=True
# (data-dependent write index; permutation is conflict-free here).


@dace.program
def scatter_loop_stencil(src: dace.float64[N], idx: dace.int64[N], dst: dace.float64[N]):
    for i in range(N):
        dst[idx[i]] = src[i] + 1.0


def test_scatter_store_collapse_laneid_structural():
    N_val = 64
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
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
        sdfg_name="scatter_store_collapse_laneid",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


@pytest.mark.parametrize("remainder_strategy", ["scalar", "masked"])
def test_scatter_store_collapse_laneid_nondiv(remainder_strategy):
    N_val = 22
    src = numpy.random.rand(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
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
        sdfg_name=f"scatter_store_collapse_laneid_nondiv_{remainder_strategy}",
        remainder_strategy=remainder_strategy,
        lower_to_intrinsics=(remainder_strategy == "masked"),
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_scatter_loop_stencil_collapse_laneid():
    # for-loop scatter: laneid fan over a symbolic slice idx[tile_i:+W].
    N_val = 64
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
        dace_func=scatter_loop_stencil,
        arrays={
            "src": src,
            "idx": idx,
            "dst": dst
        },
        params={"N": N_val},
        vector_width=8,
        sdfg_name="scatter_loop_stencil_collapse_laneid",
        collapse_laneid_index_loads=True,
        loop_to_map_permissive=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


# Edge cases for collapse_laneid_index_loads not covered above.


@dace.program
def gather_fp32_data(src: dace.float32[N], idx: dace.int64[N], dst: dace.float32[N]):
    for i in dace.map[0:N]:
        dst[i] = src[idx[i]] + 1.0


@dace.program
def two_gathers(a: dace.float64[N], ia: dace.int64[N], b: dace.float64[N], ib: dace.int64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[ia[i]] + b[ib[i]]


@dace.program
def no_indirection(src: dace.float64[N], dst: dace.float64[N], scale: dace.float64):
    for i in dace.map[0:N]:
        dst[i] = src[i] * scale


def test_gather_collapse_laneid_vw4():
    # vector_width != 8: recognition lane range 0..W-1, W=4.
    N_val = 64
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
        dace_func=gather_load,
        arrays={"src": src, "idx": idx, "dst": dst},
        params={"N": N_val, "scale": 1.5},
        vector_width=4,
        sdfg_name="gather_collapse_laneid_vw4",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_gather_collapse_laneid_fp32_data():
    # fp32 data array + int64 idx: direct-pass gather<float>.
    N_val = 64
    src = numpy.random.rand(N_val).astype(numpy.float32)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val, dtype=numpy.float32)
    vec_sdfg = run_vectorization_test(
        dace_func=gather_fp32_data,
        arrays={"src": src, "idx": idx, "dst": dst},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_collapse_laneid_fp32data",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_collapse_laneid_two_independent_gathers():
    # Two index tables in one map: each laneid fan collapses on its own.
    N_val = 64
    a = numpy.random.rand(N_val)
    b = numpy.random.rand(N_val)
    ia = numpy.random.permutation(N_val).astype(numpy.int64)
    ib = numpy.random.permutation(N_val).astype(numpy.int64)
    c = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
        dace_func=two_gathers,
        arrays={"a": a, "ia": ia, "b": b, "ib": ib, "c": c},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="collapse_laneid_two_gathers",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_collapse_laneid_noop_without_indirection():
    # Knob ON but no gather/scatter: must be inert and still correct.
    N_val = 64
    src = numpy.random.random(N_val)
    dst = numpy.zeros(N_val)
    vec_sdfg = run_vectorization_test(
        dace_func=no_indirection,
        arrays={"src": src, "dst": dst},
        params={"N": N_val, "scale": 1.5},
        vector_width=8,
        sdfg_name="collapse_laneid_noop",
        collapse_laneid_index_loads=True,
    )
    fan = [
        n for n, _ in vec_sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.Tasklet) and ("gather" in n.label or "scatter" in n.label)
    ]
    assert not fan, f"knob must be inert without indirection, found {[t.label for t in fan]}"


def test_gather_collapse_laneid_small_n():
    # N < W: no main vector gather, only the scalar remainder path.
    N_val = 5
    src = numpy.random.random(N_val)
    idx = numpy.random.permutation(N_val).astype(numpy.int64)
    dst = numpy.zeros(N_val)
    run_vectorization_test(
        dace_func=gather_load,
        arrays={"src": src, "idx": idx, "dst": dst},
        params={"N": N_val, "scale": 1.5},
        vector_width=8,
        sdfg_name="gather_collapse_laneid_smalln",
        collapse_laneid_index_loads=True,
    )


# Strided index-table access b[idx[c*i]] under the knob. The boundary
# window is the contiguous bbox holding every touched index element; the
# collapse emits a *strided* gather (_idx[l*c]) so lane l reads idx[c*l].
# Knob-ON only: knob-OFF b[idx[c*i]] is a separate pre-existing lane-fan
# bug (project_yakup_dev_strided_index_table_bug) — not asserted here.


@dace.program
def gather_strided_index_2(a: dace.float64[N], b: dace.float64[2 * N], idx: dace.int64[2 * N]):
    for i in dace.map[0:N]:
        a[i] = b[idx[2 * i]] + 1.0


@dace.program
def gather_strided_index_3(a: dace.float64[N], b: dace.float64[3 * N], idx: dace.int64[3 * N]):
    for i in dace.map[0:N]:
        a[i] = b[idx[3 * i]] + 1.0


def test_gather_collapse_laneid_strided_index_2():
    N_val = 64
    a = numpy.zeros(N_val)
    b = numpy.random.rand(2 * N_val)
    idx = numpy.random.permutation(2 * N_val).astype(numpy.int64)
    vec_sdfg = run_vectorization_test(
        dace_func=gather_strided_index_2,
        arrays={"a": a, "b": b, "idx": idx},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_collapse_laneid_stridedidx2",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


def test_gather_collapse_laneid_strided_index_3():
    N_val = 48
    a = numpy.zeros(N_val)
    b = numpy.random.rand(3 * N_val)
    idx = numpy.random.permutation(3 * N_val).astype(numpy.int64)
    vec_sdfg = run_vectorization_test(
        dace_func=gather_strided_index_3,
        arrays={"a": a, "b": b, "idx": idx},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_collapse_laneid_stridedidx3",
        collapse_laneid_index_loads=True,
    )
    _assert_laneid_fan_collapsed(vec_sdfg)


# Knob-OFF (no collapse): the per-lane laneid fan must itself read
# idx[c*i] correctly (lane k -> view[c*k]); e2e vs the unvectorized
# reference. This is the regression guard for the pre-existing bug
# where the fan read view[k] and dropped the c factor.


def test_gather_strided_index_2_knob_off():
    """Knob-OFF ``b[idx[2*i]]``: the uncollapsed laneid fan must read
    ``view[2*k]`` (e2e vs the unvectorized reference)."""
    N_val = 64
    a = numpy.zeros(N_val)
    b = numpy.random.rand(2 * N_val)
    idx = numpy.random.permutation(2 * N_val).astype(numpy.int64)
    run_vectorization_test(
        dace_func=gather_strided_index_2,
        arrays={"a": a, "b": b, "idx": idx},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_strided_index_2_knoboff",
        collapse_laneid_index_loads=False,
    )


def test_gather_strided_index_3_knob_off():
    """Knob-OFF ``b[idx[3*i]]``: the uncollapsed laneid fan must read
    ``view[3*k]`` (e2e vs the unvectorized reference)."""
    N_val = 48
    a = numpy.zeros(N_val)
    b = numpy.random.rand(3 * N_val)
    idx = numpy.random.permutation(3 * N_val).astype(numpy.int64)
    run_vectorization_test(
        dace_func=gather_strided_index_3,
        arrays={"a": a, "b": b, "idx": idx},
        params={"N": N_val},
        vector_width=8,
        sdfg_name="gather_strided_index_3_knoboff",
        collapse_laneid_index_loads=False,
    )

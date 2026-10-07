# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests the properties of the stochastically rounded float type"""

import dace
import numpy as np
from collections import Counter

from dace import Memlet
from dace.libraries.linalg import Cholesky
import dace.libraries.blas as blas

N = 1000
M = 5
EPS = 0.05
TYPE = dace.float32sr
NP_TYPE = np.float32


@dace.program
def dace_test_add(A: TYPE[N], B: TYPE[N], OUT: TYPE[N]):
    for i in range(N):
        OUT[i] = A[i] + B[i]


def test_add():
    a = 0.54019
    b = 1.9e-04
    fn = dace_test_add
    theoretical_prob = 0.66799

    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    fn(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    print(count)
    print(f"empirical_prob: {emp_prob}")
    print(f"theoretical_prob: {theoretical_prob}")

    assert abs(emp_prob - theoretical_prob) < EPS


@dace.program
def dace_test_sub(A: TYPE[N], B: TYPE[N], OUT: TYPE[N]):
    for i in range(N):
        OUT[i] = A[i] - B[i]


def test_sub():
    a = 0.54019
    b = 1.9e-04
    fn = dace_test_sub
    theoretical_prob = 0.32891

    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    fn(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    print(count)
    print(f"empirical_prob: {emp_prob}")
    print(f"theoretical_prob: {theoretical_prob}")

    assert abs(emp_prob - theoretical_prob) < EPS


def test_sub_exact_rep():
    a = 0.5
    b = 0.25
    fn = dace_test_sub
    theoretical_prob = 1

    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    fn(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    print(count)
    print(f"empirical_prob: {emp_prob}")
    print(f"theoretical_prob: {theoretical_prob}")

    assert abs(emp_prob - theoretical_prob) < EPS


@dace.program
def dace_test_mult(A: TYPE[N], B: TYPE[N], OUT: TYPE[N]):
    for i in range(N):
        OUT[i] = A[i] * B[i]


def test_mult():
    a = 0.54019
    b = 1.9e-04
    fn = dace_test_mult
    theoretical_prob = 0.14345

    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    fn(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    print(count)
    print(f"empirical_prob: {emp_prob}")
    print(f"theoretical_prob: {theoretical_prob}")

    assert abs(emp_prob - theoretical_prob) < EPS


@dace.program
def dace_test_div(A: TYPE[N], B: TYPE[N], OUT: TYPE[N]):
    for i in range(N):
        OUT[i] = A[i] / B[i]


def test_div():
    a = 0.54019
    b = 1.9e-04
    fn = dace_test_div
    theoretical_prob = 0.38520

    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    fn(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    print(count)
    print(f"empirical_prob: {emp_prob}")
    print(f"theoretical_prob: {theoretical_prob}")

    assert abs(emp_prob - theoretical_prob) < EPS


@dace.program
def dace_test_dot_runs(A: TYPE[M], B: TYPE[M], OUT: TYPE[1]):
    OUT[0] = np.dot(A, B)


def test_dot_runs():
    a = 2
    b = 1.01
    fn = dace_test_dot_runs

    A = np.array([a] * M, dtype=NP_TYPE)
    B = np.array([b] * M, dtype=NP_TYPE)
    OUT = np.zeros(1, dtype=NP_TYPE)

    fn(A, B, OUT)

    expected = np.dot(A, B)
    print(expected)
    print(OUT[0])
    assert abs(expected - OUT[0] < EPS)


@dace.program
def dace_test_matrix_mult_runs(A: TYPE[M, M], B: TYPE[M, M], OUT: TYPE[M, M]):
    OUT[:] = A @ B


def test_matrix_mult_runs():
    a = 3.14
    b = 2.71
    fn = dace_test_matrix_mult_runs

    A = np.array([a] * M * M, dtype=NP_TYPE)
    B = np.array([b] * M * M, dtype=NP_TYPE)
    OUT = np.zeros((M, M), dtype=NP_TYPE)

    fn(A, B, OUT)

    A = A.reshape((M, M))
    B = B.reshape((M, M))
    expected = A @ B
    assert np.allclose(expected, OUT, rtol=1e-5, atol=1e-6)


def dace_test_gemv(implementation: str = "pure"):
    """Helper to create a GEMV SDFG with float32sr"""

    m_size = M
    n_size = M

    sdfg = dace.SDFG(f"gemv_{implementation}_float32sr")
    state = sdfg.add_state("gemv_compute")

    sdfg.add_array("A", shape=[m_size, n_size], dtype=TYPE)
    sdfg.add_array("x", shape=[n_size], dtype=TYPE)
    sdfg.add_array("y", shape=[m_size], dtype=TYPE)

    A = state.add_read("A")
    x = state.add_read("x")
    y = state.add_write("y")

    gemv_node = blas.Gemv("gemv", transA=False, alpha=1.0, beta=0.0)
    gemv_node.implementation = implementation

    state.add_memlet_path(A, gemv_node, dst_conn="_A", memlet=Memlet(f"A[0:{m_size}, 0:{n_size}]"))
    state.add_memlet_path(x, gemv_node, dst_conn="_x", memlet=Memlet(f"x[0:{n_size}]"))
    state.add_memlet_path(gemv_node, y, src_conn="_y", memlet=Memlet(f"y[0:{m_size}]"))

    return sdfg


def test_gemv():
    """Test GEMV (matrix-vector multiplication) with float32sr"""
    sdfg = dace_test_gemv()

    A_arr = np.full((M, M), 2.5, dtype=NP_TYPE)
    x_arr = np.full(M, 1.5, dtype=NP_TYPE)
    y_arr = np.zeros(M, dtype=NP_TYPE)

    sdfg(A=A_arr, x=x_arr, y=y_arr)

    expected = A_arr @ x_arr
    assert np.allclose(expected, y_arr, rtol=1e-5, atol=1e-6)


def dace_test_cholesky(implementation: str = "OpenBLAS"):
    """Helper to create a Cholesky SDFG with float32sr."""
    size = M
    sdfg = dace.SDFG(f"cholesky_{implementation}")
    state = sdfg.add_state("cholesky_compute")

    inp = sdfg.add_array("xin", [size, size], dtype=TYPE)
    out = sdfg.add_array("xout", [size, size], dtype=TYPE)

    xin = state.add_read("xin")
    xout = state.add_write("xout")

    cholesky_node = Cholesky("cholesky", lower=True)
    cholesky_node.implementation = implementation

    state.add_memlet_path(xin, cholesky_node, dst_conn="_a", memlet=Memlet.from_array(*inp))
    state.add_memlet_path(cholesky_node, xout, src_conn="_b", memlet=Memlet.from_array(*out))

    return sdfg


def test_cholesky():
    """Test Cholesky library function using OpenBLAS expansion"""
    sdfg = dace_test_cholesky()
    size = M

    rng = np.random.default_rng(42)
    A = rng.random((size, size), dtype=NP_TYPE)
    A = (0.5 * (A @ A.T)).copy()
    B = np.zeros((size, size), dtype=NP_TYPE)

    sdfg(xin=A, xout=B)
    assert np.allclose(A, B @ B.T, rtol=1e-4, atol=1e-5)


@dace.program
def dace_test_assignment_properties(OUT: TYPE[N], c: dace.float64):
    for i in range(N):
        OUT[i] = c


@dace.program
def dace_test_init_properties(C: TYPE[N], OUT: TYPE[N]):
    for i in range(N):
        OUT[i] = C[i]


@dace.program
def dace_test_mixed_addition_properties(A: TYPE[N], B: dace.float64[N], OUT):
    for i in range(N):
        OUT[i] = A[i] + B[i]


@dace.program
def dace_test_single_mixed_addition_properties(A: TYPE[N], B: dace.float32[N], OUT):
    for i in range(N):
        OUT[i] = A[i] + B[i]


def test_init_properties():
    """ Init of a SR with same prec value should be deterministic """
    c = 1.5401030001
    C = np.array([c] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    dace_test_init_properties(C, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    assert emp_prob == 1


def test_assignment_properties():
    """ Check that doubles are rounded when assigned """
    c = 1.5401030001
    OUT = np.zeros(N, dtype=NP_TYPE)

    dace_test_assignment_properties(OUT, c)

    count = Counter(OUT)
    _, _, prob_lower, _ = calc_bounds(c)
    # Lower outcome is the smaller of the two float32 values
    keys_sorted = sorted(count.keys())
    freq_lower = count[keys_sorted[0]] / N if keys_sorted else 0

    print(count)
    assert abs(freq_lower - prob_lower) < 0.05


def test_assignment_properties_exact_rep():
    """ Check that doubles are exactly represented when assigned """
    c = 1.5
    OUT = np.zeros(N, dtype=NP_TYPE)

    dace_test_assignment_properties(OUT, c)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    _, _, _, theoretical_prob = calc_bounds(c)

    print(count)
    assert emp_prob == theoretical_prob


def test_mixed_addition_properties():
    """ Check that SR single prec is upcast to double for mixed calculations"""
    a = 1.5401030001
    b = 1.9e-07
    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=np.float64)
    OUT = np.zeros(N, dtype=NP_TYPE)

    dace_test_mixed_addition_properties(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    theoretical_prob = 1

    print(f"Empirical count: {count}")
    print(f"Theoretical prob: {theoretical_prob}")
    assert emp_prob == theoretical_prob


def test_single_mixed_addition_properties():
    """ Check that SR single prec is re-cast to RTN for mixed calculations"""
    a = 1.5401030001
    b = 1.9e-07
    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    dace_test_single_mixed_addition_properties(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    theoretical_prob = 1

    print(f"Empirical count: {count}")
    print(f"Theoretical prob: {theoretical_prob}")
    assert emp_prob == theoretical_prob


EXCESS_MASK_F32 = 0x1FFFFFFF
MASK_F64_TRUNCATE_TO_F32 = (1 << 64) - 1 - EXCESS_MASK_F32


def _double_exactly_representable_as_float32(x: float) -> bool:
    bits = np.array([x], dtype=np.float64).view(np.uint64)[0]
    return (bits & EXCESS_MASK_F32) == 0


def _float32_neighbors_from_bits(x: float):
    bits = np.array([x], dtype=np.float64).view(np.uint64)[0]
    bits_lower = bits & MASK_F64_TRUNCATE_TO_F32
    lower_f64 = np.array([bits_lower], dtype=np.uint64).view(np.float64)[0]
    lower_f32 = np.float32(lower_f64)
    upper_f32 = np.nextafter(lower_f32, np.float32(np.inf))
    upper_f64 = np.float64(upper_f32)
    return lower_f64, upper_f64


def calc_bounds(higher_prec_val):
    if _double_exactly_representable_as_float32(higher_prec_val):
        print("Exactly represented")
        return 0, 0, 1, 1

    lower_f64, upper_f64 = _float32_neighbors_from_bits(higher_prec_val)
    lower_diff = abs(higher_prec_val - lower_f64)
    upper_diff = abs(upper_f64 - higher_prec_val)
    lower, upper = np.float32(lower_f64), np.float32(upper_f64)

    print(f"lower: {lower}, upper: {upper}")
    print(f" C: {higher_prec_val}")
    print(f"LF: {lower_diff}")
    print(f"UF: {upper_diff}")
    total = lower_diff + upper_diff
    if total == 0:
        return 0, 0, 1, 1

    prob_lower = upper_diff / total
    theoretical_prob = max(prob_lower, 1 - prob_lower)

    return lower_diff, upper_diff, prob_lower, theoretical_prob


def test_serialization():
    from dace.dtypes import json_to_typeclass
    restored = json_to_typeclass("float32sr")
    assert restored is dace.float32sr

    ptr_type = dace.pointer(dace.float32sr)
    ptr_json = dace.serialize.to_json(ptr_type)
    ptr_restored = dace.serialize.from_json(ptr_json)
    assert ptr_restored == ptr_type


if __name__ == "__main__":
    test_add()
    test_sub()
    test_sub_exact_rep()
    test_mult()
    test_div()
    test_dot_runs()
    test_init_properties()
    test_assignment_properties()
    test_assignment_properties_exact_rep()
    test_mixed_addition_properties()
    test_single_mixed_addition_properties()
    test_serialization()

    test_matrix_mult_runs()
    test_gemv()
    test_cholesky()

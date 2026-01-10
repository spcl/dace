# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests the properties of the stochastically rounded float type"""

import ctypes
import dace
import numpy as np
import pytest
from collections import Counter

N = 1000
M = 5
EPS = 0.05
USE_GPU = False
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
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

    if USE_GPU:
        sdfg = fn.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
        fn(A, B, OUT)

    A = A.reshape((M, M))
    B = B.reshape((M, M))
    expected = A @ B
    assert np.array_equal(expected, OUT)


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

    if USE_GPU:
        sdfg = dace_test_init_properties.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(C=C, OUT=OUT)
    else:
        dace_test_init_properties(C, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N

    assert emp_prob == 1


def test_assignment_properties():
    """ Check that doubles are rounded when assigned """
    c = 1.5401030001
    OUT = np.zeros(N, dtype=NP_TYPE)

    if USE_GPU:
        sdfg = dace_test_assignment_properties.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(OUT=OUT, c=c)
    else:
        dace_test_assignment_properties(OUT, c)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    _, _, theoretical_prob = calc_bounds(c, NP_TYPE)

    print(count)
    assert abs(emp_prob - theoretical_prob) < 0.05


def test_mixed_addition_properties():
    """ Check that SR single prec is upcast to double for mixed calculations"""
    a = 1.5401030001
    b = 1.9e-07
    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=np.float64)
    OUT = np.zeros(N, dtype=NP_TYPE)

    if USE_GPU:
        sdfg = dace_test_mixed_addition_properties.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
        dace_test_mixed_addition_properties(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    theoretical_prob = 1

    print(f"Empircal count: {count}")
    print(f"Theoretical prob: {theoretical_prob}")
    assert abs(emp_prob - theoretical_prob) < 0.05


def test_single_mixed_addition_properties():
    """ Check that SR single prec is re-cast to RTN for mixed calculations"""
    a = 1.5401030001
    b = 1.9e-07
    A = np.array([a] * N, dtype=NP_TYPE)
    B = np.array([b] * N, dtype=NP_TYPE)
    OUT = np.zeros(N, dtype=NP_TYPE)

    if USE_GPU:
        sdfg = dace_test_single_mixed_addition_properties.to_sdfg()
        sdfg.apply_gpu_transformations()
        sdfg(A=A, B=B, OUT=OUT)
    else:
        dace_test_single_mixed_addition_properties(A, B, OUT)

    count = Counter(OUT)
    emp_prob = count[max(count.keys())] / N
    theoretical_prob = 1

    print(f"Empircal count: {count}")
    print(f"Theoretical prob: {theoretical_prob}")
    assert abs(emp_prob - theoretical_prob) < 0.05


def calc_bounds(higher_prec_val, np_type):
    rounded_val = np_type(higher_prec_val)

    if rounded_val == higher_prec_val:
        print("Exactly represented")
        return 0, 0, 1

    lower = np.nextafter(rounded_val, -np.inf, dtype=np_type)
    upper = np.nextafter(rounded_val, np.inf, dtype=np_type)

    if rounded_val < higher_prec_val:
        lower = rounded_val
    else:
        upper = rounded_val

    print(f"lower: {lower}, upper: {upper}")

    print(f" C: {higher_prec_val}")
    lower_diff = abs(higher_prec_val - lower)
    print(f"LF: {lower_diff}")
    upper_diff = abs(upper - higher_prec_val)
    print(f"UF: {upper_diff}")
    theoretical_prob = lower_diff / (lower_diff + upper_diff)

    return lower_diff, upper_diff, theoretical_prob


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

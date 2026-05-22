# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import numpy
import pytest
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    ReplaceSTDExpWithDaCeExp,
    ReplaceSTDLogWithDaCeLog,
    ReplaceSTDPowWithDaCePow,
)
from math import log, exp, pow  # noqa: A004 — used inside @dace.program bodies

from tests.passes.vectorization.helpers.harness import (
    run_vectorization_test,
    N,
    S,
)

# Core elementwise/op kernels — also exercise the K-dim tile-op config.
pytestmark = pytest.mark.tile_nodes


@dace.program
def add_sq_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def abs_op(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = abs(A[i, j])


@dace.program
def unary_symbol(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = -999


@dace.program
def add_int(A: dace.int64[N, N], B: dace.int64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def add_mixed_types(A: dace.int64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def add_int_scalars(A: dace.int64[N, N], B: dace.int64[N, N], c1: dace.int64, c2: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c3 = c1 * c2
        A[i, j] = A[i, j] + B[i, j] + c3
    for i, j in dace.map[0:N, 0:N]:
        c4 = c1 + c2
        B[i, j] = 3 * B[i, j] * 2 + 20 + c4


@dace.program
def vsubs_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - B[i, j]


@dace.program
def vexp_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = math.exp(A[i, j])


@dace.program
def vsubs_two_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = B[i, j] - A[i, j]


@dace.program
def v_const_subs_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] - 1.0


@dace.program
def v_const_subs_two_cpu(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = 1.0 - A[i, j]


@dace.program
def memset(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = 0.0


@dace.program
def v_const_subs_4d(A: dace.float64[N, N, N, N]):
    for i, j, k, m in dace.map[0:N, 0:N, 0:N, 0:N]:
        A[i, j, k, m] = 1.0 - A[i, j, k, m]


@dace.program
def v_const_subs_4d_indirect_access(A: dace.float64[N, N, N, N], c: dace.int64):
    for i, j, k in dace.map[0:N, 0:N, 0:N]:
        A[c + 1, i, j, k] = 1.0 - A[c + 1, i, j, k]


@dace.program
def memset_4d(A: dace.float64[N, N, N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            for k in dace.map[0:N]:
                for m in dace.map[0:N]:
                    A[i, j, k, m] = 0.0


def test_memset_4d(remainder_strategy):
    N = 8
    A = numpy.random.random((N, N, N, N))
    c = 0.8

    run_vectorization_test(dace_func=memset_4d,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                           },
                           vector_width=8,
                           sdfg_name="memset_4d",
                           remainder_strategy=remainder_strategy)


def test_v_const_subs_4d(remainder_strategy):
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(dace_func=v_const_subs_4d,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                           },
                           vector_width=8,
                           sdfg_name="v_const_subs_4d",
                           remainder_strategy=remainder_strategy)


def test_v_const_subs_4d_indirect_access(remainder_strategy):
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(dace_func=v_const_subs_4d_indirect_access,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                               'c': 0,
                           },
                           vector_width=8,
                           sdfg_name="v_const_subs_4d_indirect_access",
                           remainder_strategy=remainder_strategy)


def foo(A):
    A = 0.0


@dace.program
def nested_memset(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            foo(A[i, j])


@dace.program
def max_with_constant(A: dace.float64[N, N], c: dace.float64):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = max(c, A[i, j])


@dace.program
def max_with_constant_reversed_order(A: dace.float64[N, N], c: dace.float64):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            A[i, j] = max(A[i, j], c)


def test_max_with_constant(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(dace_func=max_with_constant,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                               'c': c
                           },
                           vector_width=8,
                           sdfg_name="max_with_constant",
                           remainder_strategy=remainder_strategy)


def test_max_with_constant_reversed_order(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(dace_func=max_with_constant_reversed_order,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                               'c': c
                           },
                           vector_width=8,
                           sdfg_name="max_with_constant_reversed_order",
                           remainder_strategy=remainder_strategy)


def test_vsubs_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(dace_func=vsubs_cpu,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="vsubs_one",
                           remainder_strategy=remainder_strategy)


def test_memset(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="memset",
                           exact=0.0,
                           remainder_strategy=remainder_strategy)


def test_memset_with_fuse_and_copyin_enabled(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           exact=0.0,
                           remainder_strategy=remainder_strategy)


def test_nested_memset_with_fuse_and_copyin_enabled(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=nested_memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="nested_memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           simplify=False,
                           exact=0.0,
                           remainder_strategy=remainder_strategy)


def test_vexp_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=vexp_cpu,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="vexp_one",
                           remainder_strategy=remainder_strategy)


def test_vsubs_two_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(dace_func=vsubs_two_cpu,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="vsubs_two",
                           remainder_strategy=remainder_strategy)


def test_v_const_subs_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=v_const_subs_cpu,
                           arrays={'A': A},
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="v_const_subs_one",
                           remainder_strategy=remainder_strategy)


def test_v_const_subs_two_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=v_const_subs_two_cpu,
                           arrays={'A': A},
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="v_const_subs_two",
                           remainder_strategy=remainder_strategy)


def test_simple_cpu(remainder_strategy, vectorize_config):
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=add_sq_cpu,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           sdfg_name="simple_cpu",
                           remainder_strategy=remainder_strategy,
                           vectorize_config=vectorize_config)


@dace.program
def add_unary_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c2 = -c
        c3 = B[i, j] + c2
        A[i, j] = A[i, j] + c3


@dace.program
def add_scalar_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c1: dace.float64, c2: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c3 = -c1
        c4 = c3 * c2
        c5 = B[i, j] + c4
        A[i, j] = A[i, j] + c5


def test_vadd_with_unary_scalar_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c = numpy.float64(0.5)

    run_vectorization_test(dace_func=add_unary_scalar_cpu,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'N': N,
                               'c': c
                           },
                           vector_width=8,
                           sdfg_name="add_unary_scalar_cpu",
                           remainder_strategy=remainder_strategy)


def test_vabs(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=abs_op,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                           },
                           vector_width=8,
                           sdfg_name="abs_op",
                           remainder_strategy=remainder_strategy)


def test_unary_symbol(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=unary_symbol,
                           arrays={
                               'A': A,
                           },
                           params={
                               'N': N,
                           },
                           vector_width=8,
                           sdfg_name="unary_symbol",
                           remainder_strategy=remainder_strategy)


def test_vadd_with_scalar_scalar_cpu(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c1 = numpy.float64(0.5)
    c2 = numpy.float64(0.7)

    run_vectorization_test(dace_func=add_scalar_scalar_cpu,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'N': N,
                               'c1': c1,
                               'c2': c2
                           },
                           vector_width=8,
                           sdfg_name="add_scalar_scalar_cpu",
                           remainder_strategy=remainder_strategy)


def test_vadd_int(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)

    run_vectorization_test(dace_func=add_int,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="add_int",
                           remainder_strategy=remainder_strategy)


def test_vadd_with_different_types(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.float64)

    run_vectorization_test(dace_func=add_mixed_types,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': N},
                           vector_width=8,
                           sdfg_name="add_mixed_types",
                           remainder_strategy=remainder_strategy)


def test_vadd_with_scalars_int(remainder_strategy):
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c1 = numpy.int64(5)
    c2 = numpy.int64(7)

    run_vectorization_test(dace_func=add_int_scalars,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={
                               'N': N,
                               'c1': c1,
                               'c2': c2
                           },
                           vector_width=8,
                           sdfg_name="add_int_scalars",
                           remainder_strategy=remainder_strategy)


@dace.program
def log_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = log(A[i])


@dace.program
def exp_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = exp(A[i])


@dace.program
def pow_implementations(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = pow(A[i], 3.3)


def test_log(remainder_strategy, emission_style):
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    log_implementations_std_sdfg = log_implementations.to_sdfg()
    ReplaceSTDLogWithDaCeLog().apply_pass(log_implementations_std_sdfg, {})

    run_vectorization_test(dace_func=log_implementations_std_sdfg,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           sdfg_name=f"test_log",
                           from_sdfg=True,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style)


def test_exp(remainder_strategy, emission_style):
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    exp_implementations_std_sdfg = exp_implementations.to_sdfg()
    ReplaceSTDExpWithDaCeExp().apply_pass(exp_implementations_std_sdfg, {})

    run_vectorization_test(dace_func=exp_implementations_std_sdfg,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           sdfg_name=f"test_exp",
                           from_sdfg=True,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style)


def test_pow(remainder_strategy, emission_style):
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    pow_implementations_std_sdfg = pow_implementations.to_sdfg()
    ReplaceSTDPowWithDaCePow().apply_pass(pow_implementations_std_sdfg, {})

    run_vectorization_test(dace_func=pow_implementations_std_sdfg,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           sdfg_name=f"test_pow",
                           from_sdfg=True,
                           remainder_strategy=remainder_strategy,
                           emission_style=emission_style)


@dace.program
def dace_s000(A: dace.float64[S], B: dace.float64[S]):
    for nl in range(2):
        for i in dace.map[0:S:1]:
            A[i] = B[i] + 1.0

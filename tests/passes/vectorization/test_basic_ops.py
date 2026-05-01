# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import math
import copy
from typing import Tuple
import dace
import pytest
import numpy
from dace import InterstateEdge
from dace import Union
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    ReplaceSTDExpWithDaCeExp, ReplaceSTDLogWithDaCeLog, ReplaceSTDPowWithDaCePow,
)
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N, S, S1, S2, klev, kidia, kfdia, n, m, nnz,
    KLON, KLEV, NCLDQL, NCLDQI, ssym, X, Y, C,
    log, exp, pow,
    _get_disjoint_chain_sdfg, _get_disjoint_chain_sdfg_two,
    _get_cloudsc_snippet_three, _get_cloudsc_snippet_four,
    _get_map_inside_nested_map,
    _get_dependency_edge_to_unary_symbol_sdfg,
    _get_unstructured_access_cloudsc_sdfg,
)

@dace.program
def vadds_gpu(A: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[N, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N] @ dace.dtypes.ScheduleType.GPU_Device:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vadds_cpu(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j]**2.0 + 2.0


@dace.program
def vabs(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = abs(A[i, j])


@dace.program
def unary_symbol(A: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = -999


@dace.program
def vadd_int(A: dace.int64[N, N], B: dace.int64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def vadd_with_different_types(A: dace.int64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 3 * B[i, j] * 2 + 20


@dace.program
def vadd_int_with_scalars(A: dace.int64[N, N], B: dace.int64[N, N], c1: dace.int64, c2: dace.int64):
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


def test_memset_4d():
    N = 8
    A = numpy.random.random((N, N, N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=memset_4d,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="memset_4d",
    )


def test_v_const_subs_4d():
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(
        dace_func=v_const_subs_4d,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_4d",
    )


def test_v_const_subs_4d_indirect_access():
    N = 8
    A = numpy.random.random((N, N, N, N))

    run_vectorization_test(
        dace_func=v_const_subs_4d_indirect_access,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': 0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_4d_indirect_access",
    )


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


def test_max_with_constant():
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=max_with_constant,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="max_with_constant",
    )


def test_max_with_constant_reversed_order():
    N = 64
    A = numpy.random.random((N, N))
    c = 0.8

    run_vectorization_test(
        dace_func=max_with_constant_reversed_order,
        arrays={
            'A': A,
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="max_with_constant_reversed_order",
    )


def test_vsubs_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_one",
    )


def test_memset():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="memset",
                           exact=0.0)


def test_memset_with_fuse_and_copyin_enabled():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           exact=0.0)


def test_nested_memset_with_fuse_and_copyin_enabled():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(dace_func=nested_memset,
                           arrays={
                               'A': A,
                           },
                           params={'N': N},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name="nested_memset_with_fuse_and_copy_in_enabled",
                           fuse_overlapping_loads=True,
                           insert_copies=True,
                           simplify=False,
                           exact=0.0)


def test_vexp_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vexp_cpu,
        arrays={
            'A': A,
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vexp_one",
    )


def test_vsubs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vsubs_two_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vsubs_two",
    )


def test_v_const_subs_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="v_const_subs_one",
    )


def test_v_const_subs_two_cpu():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=v_const_subs_two_cpu,
        arrays={'A': A},
        params={'N': N},
        vector_width=8,
        sdfg_name="v_const_subs_two",
        save_sdfgs=True,
    )


def test_simple_cpu():
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(
        dace_func=vadds_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': 64},
        vector_width=4,
        sdfg_name="simple_cpu",
        save_sdfgs=True,
    )


@dace.program
def vadd_with_unary_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c2 = -c
        c3 = B[i, j] + c2
        A[i, j] = A[i, j] + c3


@dace.program
def vadd_with_scalar_scalar_cpu(A: dace.float64[N, N], B: dace.float64[N, N], c1: dace.float64, c2: dace.float64):
    for i, j in dace.map[0:N, 0:N]:
        c3 = -c1
        c4 = c3 * c2
        c5 = B[i, j] + c4
        A[i, j] = A[i, j] + c5


def test_vadd_with_unary_scalar_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c = numpy.float64(0.5)

    run_vectorization_test(
        dace_func=vadd_with_unary_scalar_cpu,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c': c
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_with_unary_scalar_cpu",
    )


def test_vabs():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=vabs,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vabs",
    )


def test_unary_symbol():
    N = 64
    A = numpy.random.random((N, N))

    run_vectorization_test(
        dace_func=unary_symbol,
        arrays={
            'A': A,
        },
        params={
            'N': N,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="unary_symbol",
    )


def test_vadd_with_scalar_scalar_cpu():
    N = 64
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    c1 = numpy.float64(0.5)
    c2 = numpy.float64(0.7)

    run_vectorization_test(
        dace_func=vadd_with_scalar_scalar_cpu,
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
        save_sdfgs=True,
        sdfg_name="vadd_with_scalar_scalar_cpu",
    )


def test_vadd_int():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)

    run_vectorization_test(
        dace_func=vadd_int,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_int",
    )


def test_vadd_with_different_types():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.float64)

    run_vectorization_test(
        dace_func=vadd_with_different_types,
        arrays={
            'A': A,
            'B': B
        },
        params={'N': N},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="vadd_with_different_types",
    )


def test_vadd_with_scalars_int():
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c1 = numpy.int64(5)
    c2 = numpy.int64(7)

    run_vectorization_test(
        dace_func=vadd_int_with_scalars,
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
        save_sdfgs=True,
        sdfg_name="vadd_int_with_scalars",
    )


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


def test_log():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    log_implementations_std_sdfg = log_implementations.to_sdfg()
    ReplaceSTDLogWithDaCeLog().apply_pass(log_implementations_std_sdfg, {})

    run_vectorization_test(
        dace_func=log_implementations_std_sdfg,
        arrays={
            "A": A,
            "B": B
        },
        params={"S": _S},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"test_log",
        from_sdfg=True,
    )


def test_exp():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    exp_implementations_std_sdfg = exp_implementations.to_sdfg()
    ReplaceSTDExpWithDaCeExp().apply_pass(exp_implementations_std_sdfg, {})

    run_vectorization_test(
        dace_func=exp_implementations_std_sdfg,
        arrays={
            "A": A,
            "B": B
        },
        params={"S": _S},
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"test_exp",
        from_sdfg=True,
    )


@pytest.mark.skip
def test_pow():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    # Baseline SDFG
    pow_implementations_std_sdfg = pow_implementations.to_sdfg()
    ReplaceSTDPowWithDaCePow().apply_pass(pow_implementations_std_sdfg, {})

    run_vectorization_test(dace_func=pow_implementations,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=f"test_pow",
                           from_sdfg=True)


@dace.program
def dace_s000(A: dace.float64[S], B: dace.float64[S]):
    for nl in range(2):
        for i in dace.map[0:S:1]:
            A[i] = B[i] + 1.0


def test_s000():
    # Create test arrays
    _S = 64
    A = numpy.random.uniform(0.2, 1.2, size=(_S, ))
    B = numpy.abs(numpy.random.randn(_S, )) * 1e-4

    run_vectorization_test(dace_func=dace_s000,
                           arrays={
                               "A": A,
                               "B": B
                           },
                           params={"S": _S},
                           vector_width=8,
                           save_sdfgs=True,
                           sdfg_name=f"s000",
                           from_sdfg=False)


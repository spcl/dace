# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import warnings
import itertools
import sys
import dace
import random
import numpy as np
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
from dace.libraries.blas import Gemm

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


def create_gemm_sdfg(dtype, A_shape, B_shape, C_shape, Y_shape, transA, transB,
                     alpha, beta, implementation):

    sdfg = dace.SDFG("gemm")
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", B_shape, dtype)
    Y, Y_arr = sdfg.add_array("Y", Y_shape, dtype)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wY = state.add_write("Y")

    tasklet = Gemm('_Gemm_',
                   transA=transA,
                   transB=transB,
                   alpha=alpha,
                   beta=beta)
    tasklet.implementation = implementation
    state.add_node(tasklet)

    state.add_edge(rA, None, tasklet, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, tasklet, '_b', dace.Memlet.from_array(B, B_arr))
    state.add_edge(tasklet, '_c', wY, None, dace.Memlet.from_array(Y, Y_arr))

    return sdfg


def run_test(implementation,
             M=25,
             N=24,
             K=23,
             complex=False,
             transA=False,
             transB=False,
             alpha=1.0,
             beta=1.0,
             C_shape=["M", "N"]):
    if C_shape is not None:
        replace_map = dict(M=M, N=N)
        C_shape = [s if isinstance(s, int) else replace_map[s] for s in C_shape]

    # shape of the transposed arrays
    A_shape = trans_A_shape = [M, K]
    B_shape = trans_B_shape = [K, N]
    Y_shape = [M, N]

    # A_shape is the actual shape of A
    if transA:
        A_shape = list(reversed(trans_A_shape))

    if transB:
        B_shape = list(reversed(trans_B_shape))

    print('Matrix multiplication {}x{}x{}'.format(M, K, N))

    np_dtype = np.complex64 if complex else np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    B = np.random.rand(*B_shape).astype(np_dtype)
    if C_shape is not None:
        C = np.random.rand(*C_shape).astype(np_dtype)
    else:
        C = None
    Y = np.zeros(Y_shape, dtype=np_dtype)

    def numpy_gemm(A, B, C, transA, transB, alpha, beta):
        A_t = np.transpose(A) if transA else A
        B_t = np.transpose(B) if transB else B
        if C is not None:
            return alpha * (A_t @ B_t) + (beta * C)
        else:
            return alpha * (A_t @ B_t)

    Y_regression = numpy_gemm(A, B, C, transA, transB, alpha, beta)

    sdfg = create_gemm_sdfg(dace.complex64 if complex else dace.float32,
                            A_shape, B_shape, C_shape, Y_shape, transA, transB,
                            alpha, beta, implementation)

    if C_shape is not None:
        Y[:] = C
        sdfg(A=A, B=B, Y=Y)
    else:
        sdfg(A=A, B=B, Y=Y)

    diff = np.linalg.norm(Y_regression - Y) / (M * N)
    print("Difference:", diff)
    assert diff <= 1e-5


@pytest.mark.parametrize(
    ('implementation', ),
    [
        ('pure', ),
        ('MKL', ),
        # pytest.param('cuBLAS', marks=pytest.mark.gpu)])
    ])
def test_library_gemm(implementation):
    param_grid_trans = dict(
        transA=[True, False],
        transB=[True, False],
    )
    param_grid_scalars = dict(
        alpha=[1.0, 0.0, random.random()],
        beta=[1.0, 0.0, random.random()],
    )
    param_grid_complex = dict(
        complex=[True],
        alpha=[random.random(),
               complex(random.random(), random.random())],
        beta=[random.random(),
              complex(random.random(), random.random())],
    )

    param_grid_broadcast_C = dict(
        alpha=[random.random()],
        beta=[random.random()],
        C_shape=[None, ["M", "N"], ["M", 1], ["N"], [1, "N"]],
    )

    param_grids = [
        param_grid_trans, param_grid_scalars, param_grid_complex,
        param_grid_broadcast_C
    ]

    def params_generator(grid):
        keys, values = zip(*grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            yield params

    print("Testing implementation {}...".format(implementation))
    try:
        for param_grid in param_grids:
            for params in params_generator(param_grid):
                print("Testing params:", params)
                run_test(implementation, **params)
    except (CompilerConfigurationError, CompilationError):
        warnings.warn(
            "Configuration/compilation failed, library missing or "
            "misconfigured, skipping test for {}.".format(implementation))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_library_gemm('cuBLAS')
    test_library_gemm('pure')
    test_library_gemm('MKL')

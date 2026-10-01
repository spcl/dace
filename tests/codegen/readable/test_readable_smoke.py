# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Small kernels run under the legacy and the readable generator must give bit-identical results."""
import numpy as np
import pytest

import dace
from tests.codegen.readable.conftest import assert_bit_exact

N, M, K = (dace.symbol(s) for s in ('N', 'M', 'K'))


@dace.program
def elementwise(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
    C[:] = A + B


@dace.program
def reduction(A: dace.float64[N], s: dace.float64[1]):
    s[0] = np.sum(A)


@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = A @ B


@dace.program
def jacobi(A: dace.float64[N], B: dace.float64[N]):
    B[1:N - 1] = 0.33 * (A[0:N - 2] + A[1:N - 1] + A[2:N])


@dace.program
def transient(A: dace.float64[N], B: dace.float64[N]):
    tmp = A + 1.0
    B[:] = tmp * 2.0


def inputs(**arrays):
    return arrays


CASES = [
    pytest.param(elementwise,
                 inputs(A=np.random.rand(6, 8), B=np.random.rand(6, 8), C=np.zeros((6, 8)), M=6, N=8),
                 id="elementwise"),
    pytest.param(reduction, inputs(A=np.random.rand(64), s=np.zeros(1), N=64), id="reduction_with_conflict_resolution"),
    pytest.param(matmul,
                 inputs(A=np.random.rand(6, 5), B=np.random.rand(5, 7), C=np.zeros((6, 7)), M=6, K=5, N=7),
                 id="matmul_library_node"),
    pytest.param(jacobi, inputs(A=np.random.rand(32), B=np.zeros(32), N=32), id="stencil"),
    pytest.param(transient, inputs(A=np.random.rand(20), B=np.zeros(20), N=20), id="transient"),
]


@pytest.mark.parametrize("program, arrays", CASES)
def test_the_readable_generator_matches_legacy(program, arrays):

    def build(name):
        sdfg = program.to_sdfg(simplify=True)
        sdfg.name = name
        return sdfg

    assert_bit_exact(build, program.name, arrays)

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.libraries.blas as blas
from dace.transformation.dataflow import RedundantSecondArray
from dace.codegen.exceptions import CompilationError, CompilerConfigurationError
import numpy as np
import pytest

M = dace.symbol('M')
N = dace.symbol('N')


@pytest.mark.parametrize(('implementation', ),
                         [('pure', ), ('MKL', ), ('OpenBLAS', ),
                          pytest.param('cuBLAS', marks=pytest.mark.gpu)])
def test_gemv_strided(implementation):
    @dace.program
    def gemv(A: dace.float64[M, N], x: dace.float64[N, N]):
        return A @ x[:, 1]

    A = np.random.rand(20, 30)
    x = np.random.rand(30, 30)
    reference = A @ x[:, 1]
    sdfg = gemv.to_sdfg()
    sdfg.name = f'{sdfg.name}_{implementation}'
    if implementation == 'cuBLAS':
        sdfg.apply_gpu_transformations()
        sdfg.apply_transformations_repeated(RedundantSecondArray)

    blas.default_implementation = implementation
    try:
        daceres = sdfg(A=A, x=x, M=20, N=30)
    except (CompilationError, CompilerConfigurationError):
        print('Failed to compile, skipping')
        blas.default_implementation = None
        return

    blas.default_implementation = None
    assert np.allclose(daceres, reference)


if __name__ == '__main__':
    test_gemv_strided('pure')
    test_gemv_strided('MKL')
    test_gemv_strided('OpenBLAS')
    test_gemv_strided('cuBLAS')

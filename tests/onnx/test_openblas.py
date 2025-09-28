"""
Test that openblas is found and automatically selected on a matrix multiplication

If this fails, CPU CI will be very slow.
"""

import pytest
import dace
import dace.libraries.blas as blas

import numpy as np


@pytest.mark.onnx
def test_openblas_is_installed():
    assert blas.environments.OpenBLAS.is_installed()


@pytest.mark.onnx
def test_openblas_compiles():
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    C = np.random.rand(2, 4)

    blas.default_implementation = 'OpenBLAS'

    @dace.program
    def prog(A, B, C):
        C[:] = A @ B

    prog(A, B, C)

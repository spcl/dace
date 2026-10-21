# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Complex arithmetic in GPU kernels, which needs ``dace::complex64`` and ``dace::complex128`` on every GPU backend."""

import numpy as np
import pytest

import dace


@dace.program
def complex_kernel(a: dace.complex128[64], b: dace.complex64[64]):
    for i in dace.map[0:64]:
        a[i] = a[i] * (1.0 + 2.0j)
        b[i] = b[i] + b[i]


@pytest.mark.gpu
def test_complex_arithmetic_in_a_gpu_kernel():
    sdfg = complex_kernel.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    a = (np.arange(64) + 1j * np.arange(64)).astype(np.complex128)
    b = (np.arange(64) - 1j).astype(np.complex64)
    expected_a, expected_b = a * (1.0 + 2.0j), b + b
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b)


if __name__ == "__main__":
    test_complex_arithmetic_in_a_gpu_kernel()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Complex arithmetic and math (``conj``, ``exp``) in GPU kernels, which need ``dace::complex64`` and
``dace::complex128`` and their math overloads to agree on every GPU backend."""

import numpy as np
import pytest

import dace


@dace.program
def complex_kernel(a: dace.complex128[64], b: dace.complex64[64]):
    for i in dace.map[0:64]:
        a[i] = np.conj(a[i]) * (1.0 + 2.0j)
        b[i] = np.exp(b[i]) + b[i]


@pytest.mark.gpu
def test_complex_arithmetic_in_a_gpu_kernel():
    sdfg = complex_kernel.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    a = (np.arange(64) + 1j * np.arange(64)).astype(np.complex128)
    b = (np.arange(64) - 1j).astype(np.complex64)
    expected_a, expected_b = np.conj(a) * (1.0 + 2.0j), np.exp(b) + b
    sdfg(a=a, b=b)
    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b, rtol=1e-6)


if __name__ == "__main__":
    test_complex_arithmetic_in_a_gpu_kernel()

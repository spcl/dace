# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU tests for cuFFT lowering of :class:`dace.libraries.fft.nodes.FFT`.

Extends the basic 1-D cuFFT coverage in :file:`fft_test.py` with:

* 2-D complex128 forward FFT,
* 3-D complex128 forward FFT,
* 1-D complex64 (single-precision) forward + inverse round trip.

All tests run only when the ``gpu`` marker is selected.
"""
import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib


def _expand_with(backend, nodes_to_set):
    """Context-manager-style: temporarily switch the FFT default implementation."""
    previous = {n: n.default_implementation for n in nodes_to_set}
    for n in nodes_to_set:
        n.default_implementation = backend
    return previous


def _restore(previous):
    """Restore previous default_implementation choices."""
    for node, impl in previous.items():
        node.default_implementation = impl


@pytest.mark.gpu
def test_cufft_2d():
    """2-D complex128 forward FFT through cuFFT (true N-D FFT, matches ``np.fft.fftn``)."""
    M, N = 32, 48

    @dace.program
    def fft2d(x: dace.complex128[M, N]):
        return np.fft.fftn(x)

    sdfg = fft2d.to_sdfg()
    sdfg.apply_gpu_transformations()
    prev = _expand_with('cuFFT', [fftlib.FFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    y = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fftn(x), rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_cufft_3d():
    """3-D complex128 forward FFT through cuFFT (true N-D FFT, matches ``np.fft.fftn``)."""
    L, M, N = 8, 12, 16

    @dace.program
    def fft3d(x: dace.complex128[L, M, N]):
        return np.fft.fftn(x)

    sdfg = fft3d.to_sdfg()
    sdfg.apply_gpu_transformations()
    prev = _expand_with('cuFFT', [fftlib.FFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(8)
    x = rng.standard_normal((L, M, N)) + 1j * rng.standard_normal((L, M, N))
    y = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fftn(x), rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_cufft_complex64_roundtrip():
    """1-D complex64 forward+inverse round trip through cuFFT (CUFFT_C2C)."""
    N = 128

    @dace.program
    def roundtrip(x: dace.complex64[N]):
        y = np.fft.fft(x)
        return np.fft.ifft(y, norm='forward')

    sdfg = roundtrip.to_sdfg()
    sdfg.apply_gpu_transformations()
    prev = _expand_with('cuFFT', [fftlib.FFT, fftlib.IFFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(9)
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    z = sdfg(x.copy())
    # ``norm='forward'`` means the IFFT is un-normalised, so the round trip
    # returns to ``N * x`` -- compare element-wise after scaling.
    np.testing.assert_allclose(z / N, x, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    test_cufft_2d()
    test_cufft_3d()
    test_cufft_complex64_roundtrip()
    print('cuFFT extended GPU tests PASS')

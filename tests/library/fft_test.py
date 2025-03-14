# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np

import dace


@pytest.mark.parametrize('symbolic', (False, True))
def test_fft(symbolic):
    if symbolic:
        N = dace.symbol('N')
    else:
        N = 21

    @dace.program
    def tester(x: dace.complex128[N]):
        return np.fft.fft(x)

    a = np.random.rand(21) + 1j * np.random.rand(21)
    b = tester(a)
    assert np.allclose(b, np.fft.fft(a))


def test_fft_r2c():
    """
    Tests implicit conversion to complex types
    """

    @dace.program
    def tester(x: dace.float32[20]):
        return np.fft.fft(x)

    a = np.random.rand(20).astype(np.float32)
    b = tester(a)
    assert b.dtype == np.complex64
    assert np.allclose(b, np.fft.fft(a), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('norm', ('backward', 'forward', 'ortho'))
def test_ifft(norm):

    @dace.program
    def tester(x: dace.complex128[21]):
        return np.fft.ifft(x, norm=norm)

    a = np.random.rand(21) + 1j * np.random.rand(21)
    b = tester(a)
    assert np.allclose(b, np.fft.ifft(a, norm=norm))


@pytest.mark.gpu
def test_cufft():
    import dace.libraries.fft as fftlib

    @dace.program
    def tester(x: dace.complex128[210]):
        return np.fft.fft(x)

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    fftlib.FFT.default_implementation = 'cuFFT'
    sdfg.expand_library_nodes()
    fftlib.FFT.default_implementation = 'pure'

    a = np.random.rand(210) + 1j * np.random.rand(210)
    b = sdfg(a)
    assert np.allclose(b, np.fft.fft(a))


@pytest.mark.gpu
def test_cufft_twoplans():
    import dace.libraries.fft as fftlib

    @dace.program
    def tester(x: dace.complex128[210], y: dace.complex64[19]):
        return np.fft.fft(x), np.fft.ifft(y, norm='forward')

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    fftlib.FFT.default_implementation = 'cuFFT'
    fftlib.IFFT.default_implementation = 'cuFFT'
    sdfg.expand_library_nodes()
    fftlib.FFT.default_implementation = 'pure'
    fftlib.IFFT.default_implementation = 'pure'

    a = np.random.rand(210) + 1j * np.random.rand(210)
    b = (np.random.rand(19) + 1j * np.random.rand(19)).astype(np.complex64)
    c, d = sdfg(a, b)
    assert np.allclose(c, np.fft.fft(a), rtol=1e-3, atol=1e-5)
    assert np.allclose(d, np.fft.ifft(b, norm='forward'), rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    test_fft(False)
    test_fft(True)
    test_fft_r2c()
    test_ifft('backward')
    test_ifft('forward')
    test_ifft('ortho')
    test_cufft()
    test_cufft_twoplans()

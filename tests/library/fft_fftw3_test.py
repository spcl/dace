# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit test for the FFTW3 lowering of :class:`dace.libraries.fft.nodes.FFT`.

Drives a forward + inverse round-trip through the FFTW3 expansion and
compares against numpy's reference. The marker ``fftw`` opts the test
into the CI step that installs ``libfftw3-dev``.
"""
import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib


@pytest.mark.fftw
def test_fft_fftw3_lowering():
    """One test exercising the FFTW3 backend for both FFT and IFFT."""
    N = 256

    @dace.program
    def fft_then_ifft(x: dace.complex128[N]):
        y = np.fft.fft(x)
        z = np.fft.ifft(y, norm='forward')
        return y, z

    sdfg = fft_then_ifft.to_sdfg()
    fftlib.FFT.default_implementation = 'FFTW3'
    fftlib.IFFT.default_implementation = 'FFTW3'
    try:
        sdfg.expand_library_nodes()
    finally:
        fftlib.FFT.default_implementation = 'pure'
        fftlib.IFFT.default_implementation = 'pure'

    rng = np.random.default_rng(42)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y, z = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x), rtol=1e-12, atol=1e-12)
    # ``norm='forward'`` leaves the IFFT un-normalised, matching FFTW's
    # backward convention. The forward+backward round trip therefore
    # scales by ``N``, so divide out before comparing.
    np.testing.assert_allclose(z / N, x, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    test_fft_fftw3_lowering()
    print('FFTW3 FFT lowering test PASS')

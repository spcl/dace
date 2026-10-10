# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the **native pure** N-D / axis-batched DFT expansion.

The pure (library-free) expansion previously handled only rank-1 inputs and
raised ``NotImplementedError`` on anything higher.  It now lowers multi-dim
inputs the same way the cuFFT / FFTW3 backends do: ``axis is None`` is a true
N-D ``fftn`` (separable batched 1-D DFTs, one per axis); a set ``axis`` is a
single batched 1-D DFT along that axis.  These tests pin that against numpy
*without* needing an external FFT library (unlike ``fft_axis_test.py`` which is
FFTW3-gated).
"""

import numpy as np
import pytest

import dace


# ---------------------------------------------------------------------------
# Full N-D (axis=None) via the numpy frontend (fftn / ifftn)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(8, 12), (4, 6, 5)])
def test_pure_fftn(shape):

    @dace.program
    def tester(x: dace.complex128[tuple(shape)]):
        return np.fft.fftn(x)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fftn(x), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Batched 1-D (axis=k) via the numpy frontend (fft(x, axis=k))
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((8, 12), 0),
        ((8, 12), -1),
        ((4, 6, 5), 1),
        ((4, 6, 5), -1),
    ],
)
def test_pure_fft_axis(shape, axis):

    @dace.program
    def tester(x: dace.complex128[tuple(shape)]):
        return np.fft.fft(x, axis=axis)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=axis), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# In-place (input array IS the output array) -- the Quantum ESPRESSO pattern
# where ``invfft(f, dfft)`` transforms ``f`` in place.  The frontend always
# allocates a fresh output, so drive the lib node directly to exercise the
# alias-decoupling copy in the builder.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Symbolic dimensions -- the shape is only known at call time.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The rank-1 path must stay byte-identical (still routes to dft_explicit).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for shape in [(8, 12), (4, 6, 5)]:
        test_pure_fftn(shape)
    for shape, axis in [((8, 12), 0), ((8, 12), -1), ((4, 6, 5), 1), ((4, 6, 5), -1)]:
        test_pure_fft_axis(shape, axis)

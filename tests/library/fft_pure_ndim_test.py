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
import dace.libraries.fft as fftlib
from dace.libraries.fft.algorithms import dft
from dace.libraries.fft.nodes import FFT, IFFT


# ---------------------------------------------------------------------------
# Full N-D (axis=None) via the numpy frontend (fftn / ifftn)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(8, 12), (4, 6, 5)])
def test_pure_fftn(shape):

    @dace.program
    def tester(x: dace.complex128[tuple(shape)]):
        return np.fft.fftn(x)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fftn(x), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('norm', ['backward', 'forward', 'ortho'])
def test_pure_ifftn_2d(norm):
    shape = (8, 12)

    @dace.program
    def tester(x: dace.complex128[8, 12]):
        return np.fft.ifftn(x, norm=norm)

    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.ifftn(x, norm=norm), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Batched 1-D (axis=k) via the numpy frontend (fft(x, axis=k))
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape,axis', [
    ((8, 12), 0),
    ((8, 12), -1),
    ((4, 6, 5), 1),
    ((4, 6, 5), -1),
])
def test_pure_fft_axis(shape, axis):

    @dace.program
    def tester(x: dace.complex128[tuple(shape)]):
        return np.fft.fft(x, axis=axis)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=axis), rtol=1e-10, atol=1e-10)


def test_pure_ifft_axis_inverse():
    shape, axis = (4, 6, 5), 0

    @dace.program
    def tester(x: dace.complex128[4, 6, 5]):
        return np.fft.ifft(x, axis=axis)

    rng = np.random.default_rng(3)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.ifft(x, axis=axis), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# In-place (input array IS the output array) -- the Quantum ESPRESSO pattern
# where ``invfft(f, dfft)`` transforms ``f`` in place.  The frontend always
# allocates a fresh output, so drive the lib node directly to exercise the
# alias-decoupling copy in the builder.
# ---------------------------------------------------------------------------
def test_pure_fftn_inplace():
    shape = (6, 5)
    sdfg = dace.SDFG('inplace_fftn')
    sdfg.add_array('buf', shape, dace.complex128)
    state = sdfg.add_state()
    rnode = state.add_read('buf')
    wnode = state.add_write('buf')
    node = FFT('fft')  # axis=None -> full 2-D fftn
    state.add_node(node)
    state.add_edge(rnode, None, node, '_inp', dace.Memlet.from_array('buf', sdfg.arrays['buf']))
    state.add_edge(node, '_out', wnode, None, dace.Memlet.from_array('buf', sdfg.arrays['buf']))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(4)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    buf = x.copy()
    sdfg(buf=buf)
    np.testing.assert_allclose(buf, np.fft.fftn(x), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Symbolic dimensions -- the shape is only known at call time.
# ---------------------------------------------------------------------------
def test_pure_fftn_symbolic():
    M, N = dace.symbol('M'), dace.symbol('N')

    @dace.program
    def tester(x: dace.complex128[M, N]):
        return np.fft.fftn(x)

    rng = np.random.default_rng(5)
    x = (rng.standard_normal((7, 9)) + 1j * rng.standard_normal((7, 9))).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fftn(x), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# The rank-1 path must stay byte-identical (still routes to dft_explicit).
# ---------------------------------------------------------------------------
def test_pure_rank1_unchanged():

    @dace.program
    def tester(x: dace.complex128[21]):
        return np.fft.fft(x)

    rng = np.random.default_rng(6)
    x = (rng.standard_normal(21) + 1j * rng.standard_normal(21)).astype(np.complex128)
    y = tester(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x), rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    test_pure_fftn((8, 12))
    test_pure_fftn((4, 6, 5))
    for nrm in ('backward', 'forward', 'ortho'):
        test_pure_ifftn_2d(nrm)
    for sh, ax in (((8, 12), 0), ((8, 12), -1), ((4, 6, 5), 1), ((4, 6, 5), -1)):
        test_pure_fft_axis(sh, ax)
    test_pure_ifft_axis_inverse()
    test_pure_fftn_inplace()
    test_pure_fftn_symbolic()
    test_pure_rank1_unchanged()
    print('pure N-D FFT tests PASS')

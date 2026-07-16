# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the axis-aware FFT lowering.

Drives ``np.fft.fft(x, axis=k)`` for k in {0, -1} on 2-D and 3-D
complex128 inputs through the FFT lib node's FFTW3 expansion and
compares against numpy.  Axis=None (full N-D) is already covered by
``fft_test.py``.
"""
import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib


def _expand(backend, nodes_to_set):
    prev = {n: n.default_implementation for n in nodes_to_set}
    for n in nodes_to_set:
        n.default_implementation = backend
    return prev


def _restore(prev):
    for n, impl in prev.items():
        n.default_implementation = impl


@pytest.mark.fftw
def test_fft_axis_last_2d_fftw3():
    M, N = 8, 32

    @dace.program
    def fft_axis_last(x: dace.complex128[M, N]):
        return np.fft.fft(x, axis=-1)

    sdfg = fft_axis_last.to_sdfg()
    prev = _expand('FFTW3', [fftlib.FFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))).astype(np.complex128)
    y = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=-1), rtol=1e-12, atol=1e-12)


@pytest.mark.fftw
def test_fft_axis_first_2d_fftw3():
    M, N = 16, 24

    @dace.program
    def fft_axis_first(x: dace.complex128[M, N]):
        return np.fft.fft(x, axis=0)

    sdfg = fft_axis_first.to_sdfg()
    prev = _expand('FFTW3', [fftlib.FFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(1)
    x = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))).astype(np.complex128)
    y = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=0), rtol=1e-12, atol=1e-12)


@pytest.mark.fftw
def test_fft_axis_last_3d_fftw3():
    L, M, N = 4, 6, 8

    @dace.program
    def fft_axis_last_3d(x: dace.complex128[L, M, N]):
        return np.fft.fft(x, axis=-1)

    sdfg = fft_axis_last_3d.to_sdfg()
    prev = _expand('FFTW3', [fftlib.FFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal((L, M, N)) + 1j * rng.standard_normal((L, M, N))).astype(np.complex128)
    y = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=-1), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    test_fft_axis_last_2d_fftw3()
    test_fft_axis_first_2d_fftw3()
    test_fft_axis_last_3d_fftw3()
    print('FFT axis FFTW3 lowering tests PASS')

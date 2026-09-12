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


@pytest.mark.fftw
@pytest.mark.parametrize('shape,axes', [
    ((3, 4, 5, 2), (1, 2, 3)),
    ((4, 5, 6), (0, 2)),
    ((4, 5, 6), (-1, -3)),
    ((3, 4, 5, 2), (0, 1, 2, 3)),
])
@pytest.mark.parametrize('dtype,dace_type', [(np.complex128, dace.complex128), (np.complex64, dace.complex64)])
def test_fftn_and_ifftn_over_any_axes_through_fftw3_match_numpy(shape, axes, dtype, dace_type):
    """Axes that are neither every axis of a rank 1-3 array nor a single leading or trailing one go through
    ``fftw_plan_guru_dft``, which steps each transformed and batch axis by the descriptor strides."""

    @dace.program
    def tester(x: dace_type[tuple(shape)]):
        return np.fft.fftn(x, axes=axes), np.fft.ifftn(x, axes=axes, norm='forward')

    sdfg = tester.to_sdfg()
    prev = _expand('FFTW3', [fftlib.FFT, fftlib.IFFT])
    try:
        sdfg.expand_library_nodes()
    finally:
        _restore(prev)
    labels = sorted(n.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert 'fftw3_fft' in labels and 'fftw3_ifft' in labels, labels

    rng = np.random.default_rng(3)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    forward, inverse = sdfg(x.copy())
    tol = 1e-4 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(forward, np.fft.fftn(x, axes=axes), rtol=tol, atol=tol)
    np.testing.assert_allclose(inverse, np.fft.ifftn(x, axes=axes, norm='forward'), rtol=tol, atol=tol)


if __name__ == '__main__':
    test_fft_axis_last_2d_fftw3()
    test_fft_axis_first_2d_fftw3()
    test_fft_axis_last_3d_fftw3()
    print('FFT axis FFTW3 lowering tests PASS')

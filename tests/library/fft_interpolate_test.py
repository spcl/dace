# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Correctness tests for the :class:`FFTInterpolate` lib node's pure expansion.

The pure expansion composes FFT -> pad/truncate spectrum -> IFFT.  The
test feeds a smooth signal sampled on the source grid through the lib
node and verifies the output matches a numpy reference produced by the
same pipeline (FFT in numpy, zero-pad / truncate the spectrum, IFFT).
"""
import numpy as np

import dace
from dace.libraries.fft.nodes import FFTInterpolate


def _numpy_reference(x, nout, dtype_kind):
    """Reference: FFT -> symmetric-split pad / truncate -> IFFT, scaled by nout/nin.

    Standard FFT zero-padding convention: keep the first ``ceil(n/2)`` and
    last ``floor(n/2)`` frequency bins.
    """
    nin = x.size
    if dtype_kind == 'real':
        x = x.astype(np.complex128)
    spec_in = np.fft.fft(x)
    spec_out = np.zeros(nout, dtype=np.complex128)
    if nin <= nout:
        low_n = (nin + 1) // 2
        high_n = nin // 2
    else:
        low_n = (nout + 1) // 2
        high_n = nout // 2
    if low_n > 0:
        spec_out[:low_n] = spec_in[:low_n]
    if high_n > 0:
        spec_out[nout - high_n:] = spec_in[nin - high_n:]
    y = np.fft.ifft(spec_out) * (nout / nin)
    return y.real if dtype_kind == 'real' else y


def _build_sdfg(nin, nout, dtype, dtype_kind):
    sdfg = dace.SDFG(f'fftinterp_{dtype_kind}_{nin}_{nout}')
    sdfg.add_array('v_in', [nin], dtype)
    sdfg.add_array('v_out', [nout], dtype)
    state = sdfg.add_state()
    node = FFTInterpolate('fft_interp', dtype_kind=dtype_kind)
    state.add_node(node)
    state.add_edge(state.add_read('v_in'), None, node, '_inp', dace.Memlet.from_array('v_in', sdfg.arrays['v_in']))
    state.add_edge(node, '_out', state.add_write('v_out'), None, dace.Memlet.from_array('v_out', sdfg.arrays['v_out']))
    sdfg.expand_library_nodes()
    return sdfg


def test_fft_interpolate_complex_upsample():
    """Upsample a smooth complex signal from N=16 to N=64."""
    nin, nout = 16, 64
    sdfg = _build_sdfg(nin, nout, dace.complex128, 'complex')
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(nin) + 1j * rng.standard_normal(nin)).astype(np.complex128)
    y = np.zeros(nout, dtype=np.complex128)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference(x, nout, 'complex'), rtol=1e-12, atol=1e-12)


def test_fft_interpolate_complex_downsample():
    """Downsample a smooth complex signal from N=64 to N=16."""
    nin, nout = 64, 16
    sdfg = _build_sdfg(nin, nout, dace.complex128, 'complex')
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(nin) + 1j * rng.standard_normal(nin)).astype(np.complex128)
    y = np.zeros(nout, dtype=np.complex128)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference(x, nout, 'complex'), rtol=1e-12, atol=1e-12)


def test_fft_interpolate_real_upsample():
    """Upsample a real-valued signal from N=16 to N=64."""
    nin, nout = 16, 64
    sdfg = _build_sdfg(nin, nout, dace.float64, 'real')
    rng = np.random.default_rng(2)
    x = rng.standard_normal(nin)
    y = np.zeros(nout, dtype=np.float64)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference(x, nout, 'real'), rtol=1e-12, atol=1e-12)


# --- Rank > 1 ------------------------------------------------------------


def _numpy_reference_nd(x, out_shape, dtype_kind):
    """N-D reference: FFTn -> per-axis symmetric pad/truncate -> IFFTn, scaled by prod(nout)/prod(nin)."""
    in_shape = x.shape
    rank = len(in_shape)
    if dtype_kind == 'real':
        x = x.astype(np.complex128)
    spec_in = np.fft.fftn(x)
    spec_out = np.zeros(out_shape, dtype=np.complex128)
    cuts = []
    for d in range(rank):
        smaller = min(in_shape[d], out_shape[d])
        cuts.append(((smaller + 1) // 2, smaller // 2))
    import itertools
    for combo in itertools.product(('low', 'high'), repeat=rank):
        in_slc, out_slc = [], []
        skip = False
        for part, (lo, hi), nin_d, nout_d in zip(combo, cuts, in_shape, out_shape):
            if part == 'low':
                if lo == 0:
                    skip = True
                    break
                in_slc.append(slice(0, lo))
                out_slc.append(slice(0, lo))
            else:
                if hi == 0:
                    skip = True
                    break
                in_slc.append(slice(nin_d - hi, nin_d))
                out_slc.append(slice(nout_d - hi, nout_d))
        if skip:
            continue
        spec_out[tuple(out_slc)] = spec_in[tuple(in_slc)]
    in_size = int(np.prod(in_shape))
    out_size = int(np.prod(out_shape))
    y = np.fft.ifftn(spec_out) * (out_size / in_size)
    return y.real if dtype_kind == 'real' else y


def _build_sdfg_nd(in_shape, out_shape, dtype, dtype_kind):
    sdfg = dace.SDFG(f'fftinterp_{dtype_kind}_nd')
    sdfg.add_array('v_in', list(in_shape), dtype)
    sdfg.add_array('v_out', list(out_shape), dtype)
    state = sdfg.add_state()
    node = FFTInterpolate('fft_interp', dtype_kind=dtype_kind)
    state.add_node(node)
    state.add_edge(state.add_read('v_in'), None, node, '_inp', dace.Memlet.from_array('v_in', sdfg.arrays['v_in']))
    state.add_edge(node, '_out', state.add_write('v_out'), None, dace.Memlet.from_array('v_out', sdfg.arrays['v_out']))
    sdfg.expand_library_nodes()
    return sdfg


def test_fft_interpolate_2d_complex_upsample():
    """2-D upsample (8,8) -> (16,16) complex."""
    in_shape, out_shape = (8, 8), (16, 16)
    sdfg = _build_sdfg_nd(in_shape, out_shape, dace.complex128, 'complex')
    rng = np.random.default_rng(10)
    x = (rng.standard_normal(in_shape) + 1j * rng.standard_normal(in_shape)).astype(np.complex128)
    y = np.zeros(out_shape, dtype=np.complex128)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference_nd(x, out_shape, 'complex'), rtol=1e-12, atol=1e-12)


def test_fft_interpolate_2d_complex_downsample():
    """2-D downsample (16,16) -> (8,8) complex."""
    in_shape, out_shape = (16, 16), (8, 8)
    sdfg = _build_sdfg_nd(in_shape, out_shape, dace.complex128, 'complex')
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(in_shape) + 1j * rng.standard_normal(in_shape)).astype(np.complex128)
    y = np.zeros(out_shape, dtype=np.complex128)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference_nd(x, out_shape, 'complex'), rtol=1e-12, atol=1e-12)


def test_fft_interpolate_3d_complex_upsample():
    """3-D upsample (4,6,8) -> (8,12,16) complex."""
    in_shape, out_shape = (4, 6, 8), (8, 12, 16)
    sdfg = _build_sdfg_nd(in_shape, out_shape, dace.complex128, 'complex')
    rng = np.random.default_rng(12)
    x = (rng.standard_normal(in_shape) + 1j * rng.standard_normal(in_shape)).astype(np.complex128)
    y = np.zeros(out_shape, dtype=np.complex128)
    sdfg(v_in=x, v_out=y)
    np.testing.assert_allclose(y, _numpy_reference_nd(x, out_shape, 'complex'), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    test_fft_interpolate_complex_upsample()
    test_fft_interpolate_complex_downsample()
    test_fft_interpolate_real_upsample()
    test_fft_interpolate_2d_complex_upsample()
    test_fft_interpolate_2d_complex_downsample()
    test_fft_interpolate_3d_complex_upsample()
    print('FFTInterpolate pure expansion tests PASS')

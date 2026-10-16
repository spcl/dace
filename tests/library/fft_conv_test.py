# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
FFT-based 7x7 convolutions of CNNs (like cuDNN's FFT algorithms) with the native Stockham FFT.

Both convolutions compute the cross-correlation of a CNN layer with "same" zero padding. The input is zero-padded to
a ``P x Q`` grid, with ``P >= H + 3`` and ``Q >= W + 3``, and filter tap ``(u, v)`` is placed at ``(6 - u, 6 - v)``,
so that the circular convolution of the padded input and filter is the CNN output shifted by 3 in both dimensions:
the output is the window ``[3, H + 3) x [3, W + 3)`` of the inverse transform.

The depthwise convolution (one filter per channel, e.g., in ConvNeXt blocks) multiplies each transform by the spectrum
of its channel's filter, which ``BlockLocalFFTFusion`` fuses with the zero-padded FFT (``fft2(x, s=...)``), the
inverse FFT and the window. The full convolution
sums the products of the spectra over the input channels for each output channel, which mixes transforms and is not
fused.
"""

import numpy as np
import pytest

import dace
from dace.libraries.fft import BlockLocalFFTFusion

N, C, K, H, W = 2, 4, 3, 26, 26
P, Q = 32, 32
R = 7  # Filter size
HALF = Q // 2 + 1


def _reference(x: np.ndarray, w: np.ndarray, depthwise: bool) -> np.ndarray:
    """The CNN cross-correlation with "same" zero padding, computed directly."""
    pad = R // 2
    padded = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_channels = x.shape[1] if depthwise else w.shape[0]
    result = np.zeros((x.shape[0], out_channels, H, W))
    for u in range(R):
        for v in range(R):
            window = padded[:, :, u : u + H, v : v + W]
            if depthwise:
                result += window * w[None, :, u, v, None, None]
            else:
                result += np.einsum("nchw,kc->nkhw", window, w[:, :, u, v])
    return result


def _depthwise_program(storage: dace.StorageType):

    @dace.program
    def depthwise_conv(
        x: dace.float32[N, C, H, W] @ storage, w: dace.float32[C, R, R] @ storage, y: dace.float32[N, C, H, W] @ storage
    ):
        # Spectra of the flipped filters, zero-padded to the grid
        wflip = np.ndarray((C, R, R), dtype=np.float32)
        for c, u, v in dace.map[0:C, 0:R, 0:R]:
            wflip[c, R - 1 - u, R - 1 - v] = w[c, u, v]
        wf = np.fft.fft2(wflip, s=(P, Q))

        # Zero-padded FFT -> multiplication by the spectrum of the channel's filter -> inverse FFT -> window
        X = np.fft.fft2(x, s=(P, Q))
        Y = np.empty_like(X)
        for n, c, i, j in dace.map[0:N, 0:C, 0:P, 0:Q]:
            Y[n, c, i, j] = X[n, c, i, j] * wf[c, i, j]
        yp = np.fft.ifft2(Y)
        for n, c, i, j in dace.map[0:N, 0:C, 0:H, 0:W]:
            y[n, c, i, j] = yp[n, c, i + 3, j + 3].real

    return depthwise_conv


def _full_program(storage: dace.StorageType):

    @dace.program
    def conv(
        x: dace.float32[N, C, H, W] @ storage,
        w: dace.float32[K, C, R, R] @ storage,
        y: dace.float32[N, K, H, W] @ storage,
    ):
        xp = np.zeros((N, C, P, Q), dtype=np.float32)
        xp[:, :, :H, :W] = x
        wp = np.zeros((K, C, P, Q), dtype=np.float32)
        for k, c, u, v in dace.map[0:K, 0:C, 0:R, 0:R]:
            wp[k, c, R - 1 - u, R - 1 - v] = w[k, c, u, v]

        # Half spectra of the real inputs and filters, and their products summed over the input channels
        X = np.fft.rfft2(xp)
        Wf = np.fft.rfft2(wp)
        Y = np.ndarray((N, K, P, HALF), dtype=np.complex64)
        for n, k, i, j in dace.map[0:N, 0:K, 0:P, 0:HALF]:
            acc = np.complex64(0)
            for c in range(C):
                acc = acc + X[n, c, i, j] * Wf[k, c, i, j]
            Y[n, k, i, j] = acc
        yp = np.fft.irfft2(Y, s=(P, Q))

        y[:] = yp[:, :, 3 : H + 3, 3 : W + 3]

    return conv


def _inputs(depthwise: bool):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((N, C, H, W)).astype(np.float32)
    w = rng.standard_normal((C, R, R) if depthwise else (K, C, R, R)).astype(np.float32)
    return x, w


def _run_depthwise(sdfg: dace.SDFG, module=np) -> None:
    x, w = _inputs(True)
    y = module.zeros((N, C, H, W), np.float32)
    sdfg(x=module.asarray(x), w=module.asarray(w), y=y)
    result = y if module is np else y.get()
    assert np.allclose(result, _reference(x, w, True), rtol=1e-4, atol=1e-4)


def _run_full(sdfg: dace.SDFG, module=np) -> None:
    x, w = _inputs(False)
    y = module.zeros((N, K, H, W), np.float32)
    sdfg(x=module.asarray(x), w=module.asarray(w), y=y)
    result = y if module is np else y.get()
    assert np.allclose(result, _reference(x, w, False), rtol=1e-4, atol=1e-3)


def _fuse_depthwise(sdfg: dace.SDFG) -> None:
    """
    Fuses the zero-padded FFT, the multiplication, the inverse FFT and the window into one kernel, so that neither
    the padded input, the spectra nor the whole result of the inverse FFT are arrays of the program.
    """
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(BlockLocalFFTFusion) == 1
    assert not {"X", "Y", "yp"} & set(sdfg.arrays)


def test_depthwise_conv_fused():
    """
    The depthwise convolution with the FFT, the multiplication and the inverse FFT fused into one map, in which each
    iteration transforms one channel of one image in buffers on the stack.
    """
    sdfg = _depthwise_program(dace.StorageType.Default).to_sdfg()
    _fuse_depthwise(sdfg)
    _run_depthwise(sdfg)


def test_full_conv():
    sdfg = _full_program(dace.StorageType.Default).to_sdfg()
    _run_full(sdfg)


@pytest.mark.gpu
def test_depthwise_conv_fused_gpu():
    """
    The depthwise convolution on a GPU, with one thread block per channel of an image in the fused kernel.
    """
    import cupy as cp  # CuPy is only installed on GPU machines

    sdfg = _depthwise_program(dace.StorageType.GPU_Global).to_sdfg()
    sdfg.apply_gpu_transformations()
    _fuse_depthwise(sdfg)
    _run_depthwise(sdfg, cp)


@pytest.mark.gpu
def test_full_conv_gpu():
    import cupy as cp  # CuPy is only installed on GPU machines

    sdfg = _full_program(dace.StorageType.GPU_Global).to_sdfg()
    sdfg.apply_gpu_transformations()
    _run_full(sdfg, cp)


if __name__ == "__main__":
    test_depthwise_conv_fused()
    test_full_conv()
    test_depthwise_conv_fused_gpu()
    test_full_conv_gpu()

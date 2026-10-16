# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace


@pytest.mark.parametrize("symbolic", (False, True))
def test_fft(symbolic):
    if symbolic:
        N = dace.symbol("N")
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


@pytest.mark.parametrize("norm", ("backward", "forward", "ortho"))
def test_ifft(norm):

    @dace.program
    def tester(x: dace.complex128[21]):
        return np.fft.ifft(x, norm=norm)

    a = np.random.rand(21) + 1j * np.random.rand(21)
    b = tester(a)
    assert np.allclose(b, np.fft.ifft(a, norm=norm))


@pytest.mark.parametrize("norm", ("backward", "forward", "ortho"))
@pytest.mark.parametrize("transform", ("fft", "ifft"))
def test_1d_fft_normalization_over_a_symbolic_extent(transform, norm):
    """A ``1/N`` or ``sqrt(1/N)`` factor over an integer extent symbol divides in floating point, never as C ints."""
    N = dace.symbol("N")

    @dace.program
    def forward(x: dace.complex128[N]):
        return np.fft.fft(x, norm=norm)

    @dace.program
    def inverse(x: dace.complex128[N]):
        return np.fft.ifft(x, norm=norm)

    rng = np.random.default_rng(7)
    a = rng.standard_normal(21) + 1j * rng.standard_normal(21)
    tester, reference = (forward, np.fft.fft) if transform == "fft" else (inverse, np.fft.ifft)
    np.testing.assert_allclose(tester(a.copy()), reference(a, norm=norm), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_fft(False)
    test_fft(True)
    test_fft_r2c()
    test_ifft("backward")
    test_ifft("forward")
    test_ifft("ortho")
    test_1d_fft_normalization_over_a_symbolic_extent("fft", "backward")
    test_1d_fft_normalization_over_a_symbolic_extent("fft", "forward")
    test_1d_fft_normalization_over_a_symbolic_extent("fft", "ortho")
    test_1d_fft_normalization_over_a_symbolic_extent("ifft", "backward")
    test_1d_fft_normalization_over_a_symbolic_extent("ifft", "forward")
    test_1d_fft_normalization_over_a_symbolic_extent("ifft", "ortho")

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


@pytest.mark.parametrize("extent", (7, 8))
@pytest.mark.parametrize("half_spectrum", (False, True))
def test_sample_frequencies_are_python_tasklets_matching_numpy(extent, half_spectrum):
    N = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def fftfreq_tester(out: dace.float64[N]):
        out[:] = np.fft.fftfreq(N, 0.5)

    @dace.program
    def rfftfreq_tester(out: dace.float64[N // 2 + 1]):
        out[:] = np.fft.rfftfreq(N, 0.5)

    program = rfftfreq_tester if half_spectrum else fftfreq_tester
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f"{sdfg.name}_{extent}"
    tasklets = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.Tasklet)]
    assert tasklets and all(node.language == dace.Language.Python for node in tasklets)

    expected = np.fft.rfftfreq(extent, 0.5) if half_spectrum else np.fft.fftfreq(extent, 0.5)
    out = np.zeros(expected.shape)
    sdfg(out=out, N=extent)
    assert np.allclose(out, expected)


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
    test_sample_frequencies_are_python_tasklets_matching_numpy(7, False)
    test_sample_frequencies_are_python_tasklets_matching_numpy(8, False)
    test_sample_frequencies_are_python_tasklets_matching_numpy(7, True)
    test_sample_frequencies_are_python_tasklets_matching_numpy(8, True)

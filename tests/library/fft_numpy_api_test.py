# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The numpy.fft functions the Python frontend lowers, each against numpy itself.

The real-input family (``rfft``, ``irfft``, ``rfftn``, ``irfftn``, ``rfft2``, ``irfft2``) keeps only the
non-negative frequencies of the last transformed axis; ``n`` and ``s`` crop or zero-pad the input first; the
shifts are rolls and the frequencies a map. Every transform case runs through ``pure`` and FFTW3, and on a GPU
through the backend's vendor library, without falling back.
"""

import warnings

import numpy as np
import pytest

import dace
from dace.codegen.common import get_gpu_backend
from dace.libraries.fft.nodes import FFT, IFFT

M, N = dace.symbol("M"), dace.symbol("N")

CPU_IMPLEMENTATIONS = [
    "pure",
    pytest.param("FFTW3", marks=pytest.mark.fftw),
    pytest.param("MKL", marks=pytest.mark.mkl),
]


def gpu_implementation() -> str:
    return "hipFFT" if get_gpu_backend() == "hip" else "cuFFT"


def run(program, implementation: str, gpu: bool = False, **arguments):
    """``program`` with every FFT / IFFT node on ``implementation``; a vendor fallback to ``pure`` fails the case."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"{sdfg.name}_{implementation}"
    if gpu:
        sdfg.apply_gpu_transformations()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (FFT, IFFT)):
            node.implementation = implementation
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".* cannot transform axes")
        return sdfg(**arguments)


def real_array(shape) -> np.ndarray:
    return np.random.default_rng(sum(shape)).standard_normal(shape)


def complex_array(shape) -> np.ndarray:
    rng = np.random.default_rng(sum(shape) + 1)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def axis_symmetric(mask: np.ndarray) -> np.ndarray:
    """``mask`` made equal at ``j`` and ``-j`` along axis 0, so that it keeps a half spectrum Hermitian.

    A multi-axis complex-to-real transform of a spectrum that is not Hermitian along its full axes is
    implementation-defined: FFTW and MKL's FFTW3 interface return different arrays for it.
    """
    return np.maximum(mask, mask[(-np.arange(mask.shape[0])) % mask.shape[0]])


@dace.program
def rfft_of_rows(x: dace.float64[M, N]):
    return np.fft.rfft(x)


@dace.program
def irfft_of_rows(x: dace.complex128[M, N]):
    return np.fft.irfft(x)


@dace.program
def irfft_to_an_odd_length(x: dace.complex128[M, N]):
    return np.fft.irfft(x, n=9, axis=1)


@dace.program
def rfftn_and_back(x: dace.float64[4, 6, 8]):
    return np.fft.irfftn(np.fft.rfftn(x), s=(4, 6, 8), axes=(0, 1, 2))


@dace.program
def rfft2_of_the_last_axes(x: dace.float64[3, 6, 10]):
    return np.fft.rfft2(x, norm="ortho")


@dace.program
def irfft2_over_leading_axes(x: dace.complex128[6, 5, 3]):
    return np.fft.irfft2(x, axes=(0, 1))


def check_real_family(implementation: str, gpu: bool = False):
    x, z = real_array((5, 12)), complex_array((5, 7))
    np.testing.assert_allclose(
        run(rfft_of_rows, implementation, gpu, x=x.copy(), M=5, N=12), np.fft.rfft(x), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        run(irfft_of_rows, implementation, gpu, x=z.copy(), M=5, N=7), np.fft.irfft(z), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        run(irfft_to_an_odd_length, implementation, gpu, x=z.copy(), M=5, N=7),
        np.fft.irfft(z, n=9, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    grid = real_array((4, 6, 8))
    np.testing.assert_allclose(run(rfftn_and_back, implementation, gpu, x=grid.copy()), grid, rtol=1e-12, atol=1e-12)
    volume = real_array((3, 6, 10))
    np.testing.assert_allclose(
        run(rfft2_of_the_last_axes, implementation, gpu, x=volume.copy()),
        np.fft.rfft2(volume, norm="ortho"),
        rtol=1e-12,
        atol=1e-10,
    )
    # The half spectrum of a real array: a random one is not Hermitian along axis 0 (see ``axis_symmetric``).
    spectrum = np.fft.rfft2(real_array((6, 8, 3)), axes=(0, 1))
    np.testing.assert_allclose(
        run(irfft2_over_leading_axes, implementation, gpu, x=spectrum.copy()),
        np.fft.irfft2(spectrum, axes=(0, 1)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_the_real_input_family_keeps_half_the_spectrum(implementation):
    check_real_family(implementation)


@pytest.mark.gpu
@pytest.mark.parametrize("vendor", [True, False])
def test_the_real_input_family_keeps_half_the_spectrum_on_the_gpu(vendor):
    check_real_family(gpu_implementation() if vendor else "pure", gpu=True)


def test_a_real_transform_outputs_the_halved_shape_and_real_type():
    sdfg = rfft_of_rows.to_sdfg(simplify=False)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, FFT))
    assert node.half_spectrum
    assert node.axes == [1]
    inverse = irfft_to_an_odd_length.to_sdfg(simplify=False)
    returned = inverse.arrays["__return"]
    assert returned.dtype == dace.float64 and returned.shape[1] == 9


def test_rfft_refuses_a_complex_input():

    @dace.program
    def rfft_of_a_complex_input(x: dace.complex128[8]):
        return np.fft.rfft(x)

    with pytest.raises(TypeError, match="real input"):
        rfft_of_a_complex_input.to_sdfg()


@dace.program
def padded_along_the_first_axis(x: dace.complex128[6, 8]):
    return np.fft.fft(x, n=11, axis=0)


@dace.program
def cropped_inverse(x: dace.complex128[6, 8]):
    return np.fft.ifft(x, n=5)


@dace.program
def padded_and_cropped_2d(x: dace.complex128[6, 8]):
    return np.fft.fft2(x, s=(4, 9), norm="ortho")


@dace.program
def padded_real_transform(x: dace.float64[6, 8]):
    return np.fft.rfftn(x, s=(7, 10), axes=(0, 1))


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_n_and_s_crop_or_zero_pad_the_input(implementation):
    x = complex_array((6, 8))
    np.testing.assert_allclose(
        run(padded_along_the_first_axis, implementation, x=x.copy()),
        np.fft.fft(x, n=11, axis=0),
        rtol=1e-12,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        run(cropped_inverse, implementation, x=x.copy()), np.fft.ifft(x, n=5), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        run(padded_and_cropped_2d, implementation, x=x.copy()),
        np.fft.fft2(x, s=(4, 9), norm="ortho"),
        rtol=1e-12,
        atol=1e-10,
    )
    r = real_array((6, 8))
    np.testing.assert_allclose(
        run(padded_real_transform, implementation, x=r.copy()),
        np.fft.rfftn(r, s=(7, 10), axes=(0, 1)),
        rtol=1e-12,
        atol=1e-10,
    )


@dace.program
def low_pass(x: dace.complex128[16, 12], mask: dace.float64[16, 12]):
    return np.fft.ifftn(np.fft.fftn(x) * mask)


@dace.program
def real_low_pass(x: dace.float64[16, 12], mask: dace.float64[16, 7]):
    return np.fft.irfftn(np.fft.rfftn(x) * mask, s=(16, 12), axes=(0, 1))


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_a_filter_between_a_transform_and_its_inverse(implementation):
    """The FFT -> elementwise filter -> IFFT chain: the filter is an ordinary map between two library nodes."""
    x, mask = complex_array((16, 12)), (real_array((16, 12)) > 0).astype(np.float64)
    np.testing.assert_allclose(
        run(low_pass, implementation, x=x.copy(), mask=mask),
        np.fft.ifftn(np.fft.fftn(x) * mask),
        rtol=1e-12,
        atol=1e-12,
    )
    r, half_mask = real_array((16, 12)), axis_symmetric((real_array((16, 7)) > 0).astype(np.float64))
    np.testing.assert_allclose(
        run(real_low_pass, implementation, x=r.copy(), mask=half_mask),
        np.fft.irfftn(np.fft.rfftn(r) * half_mask, s=(16, 12), axes=(0, 1)),
        rtol=1e-12,
        atol=1e-12,
    )


@dace.program
def shifted_both_ways(x: dace.complex128[5, 8]):
    return np.fft.fftshift(x), np.fft.ifftshift(x), np.fft.fftshift(x, axes=1), np.fft.ifftshift(np.fft.fftshift(x))


def test_fftshift_and_ifftshift_roll_by_half_of_each_axis():
    """An odd axis tells the two apart: ``fftshift`` rolls 5 by 2, ``ifftshift`` by -2."""
    x = complex_array((5, 8))
    forward, inverse, one_axis, round_trip = shifted_both_ways(x.copy())
    np.testing.assert_array_equal(forward, np.fft.fftshift(x))
    np.testing.assert_array_equal(inverse, np.fft.ifftshift(x))
    np.testing.assert_array_equal(one_axis, np.fft.fftshift(x, axes=1))
    np.testing.assert_array_equal(round_trip, x)


@dace.program
def frequencies_of_odd_and_even_lengths():
    return np.fft.fftfreq(7, 0.5), np.fft.fftfreq(8), np.fft.rfftfreq(7, 0.25), np.fft.rfftfreq(8)


def test_fftfreq_and_rfftfreq_match_numpy():
    got = frequencies_of_odd_and_even_lengths()
    want = (np.fft.fftfreq(7, 0.5), np.fft.fftfreq(8), np.fft.rfftfreq(7, 0.25), np.fft.rfftfreq(8))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-15, atol=0)


@dace.program
def fft_and_rfft_of_integers(x: dace.int64[8]):
    return np.fft.fft(x), np.fft.rfft(x)


@dace.program
def fft_of_booleans(x: dace.bool_[8]):
    return np.fft.fft(x)


@dace.program
def rfft_of_halves(x: dace.float16[8]):
    return np.fft.rfft(x)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_non_floating_inputs_promote_as_numpy_does(implementation):
    """numpy computes an integer or boolean input in ``complex128`` and a half-precision one in ``complex64``."""
    x = np.arange(8, dtype=np.int64) * 3 - 5
    full, half = run(fft_and_rfft_of_integers, implementation, x=x)
    assert full.dtype == np.complex128 and half.dtype == np.complex128
    np.testing.assert_allclose(full, np.fft.fft(x), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(half, np.fft.rfft(x), rtol=1e-12, atol=1e-10)
    flags = np.arange(8) % 3 == 0
    got = run(fft_of_booleans, implementation, x=flags)
    assert got.dtype == np.complex128
    np.testing.assert_allclose(got, np.fft.fft(flags), rtol=1e-12, atol=1e-12)
    halves = (np.arange(8) / 4 - 1).astype(np.float16)
    got = run(rfft_of_halves, implementation, x=halves)
    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, np.fft.rfft(halves), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

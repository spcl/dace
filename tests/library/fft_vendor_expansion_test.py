# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The FFT / IFFT library node against numpy, through every implementation that can run here.

``pure`` is the separable DFT every other expansion falls back to. ``FFTW3`` is the CPU vendor
call and ``cuFFT`` / ``hipFFT`` the GPU one, all three wrapped in the same nested SDFG: a cast for a
real input, the unnormalised complex-to-complex transform, then the ``norm`` scale. Each case pins
the numpy semantics that wrapper has to reproduce -- the axes, the three norms, both precisions, a
real input, a strided (leading-axis) transform and the in-place ``x[:] = fft(x)``.
"""
import warnings

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.common import get_gpu_backend
from dace.libraries.fft.environments.fftw3 import FFTW3
from dace.libraries.fft.nodes import FFT, IFFT
from dace.transformation.auto.auto_optimize import find_fast_library
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations

N = dace.symbol('N')

CPU_IMPLEMENTATIONS = ['pure', pytest.param('FFTW3', marks=pytest.mark.fftw)]


def gpu_implementation() -> str:
    """The vendor FFT of this build's GPU backend."""
    return 'hipFFT' if get_gpu_backend() == 'hip' else 'cuFFT'


def compile_with(program, implementation: str, gpu: bool = False) -> dace.SDFG:
    """``program`` as an SDFG whose every FFT / IFFT node takes ``implementation``."""
    sdfg = program.to_sdfg(simplify=True)
    if gpu:
        sdfg.apply_gpu_transformations()
    nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (FFT, IFFT))]
    assert nodes, 'the program lost its FFT library nodes'
    for node in nodes:
        node.implementation = implementation
    return sdfg


def run_without_fallback(sdfg: dace.SDFG, **arguments):
    """Run ``sdfg``; a vendor expansion falling back to ``pure`` warns, and that fails the case."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.* cannot transform axes')
        return sdfg(**arguments)


@dace.program
def poisson_round_trip(x: dace.float64[N, N, N]):
    """ls3df's reciprocal-space solve: a real ``fftn``, a scale, a normalised ``ifftn``, the real part."""
    return np.fft.ifftn(4.0 * np.fft.fftn(x)).real


@dace.program
def fft_1d_round_trip(x: dace.complex128[N], y: dace.complex128[N], z: dace.complex128[N]):
    """fft_1d: the forward transform and the round trip back, which must recover ``x``."""
    y[:] = np.fft.fft(x)
    z[:] = np.fft.ifft(y)


def rng_complex(shape, dtype=np.complex128) -> np.ndarray:
    rng = np.random.default_rng(sum(shape))
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def check_poisson_round_trip(implementation: str, gpu: bool = False):
    x = np.random.default_rng(1).standard_normal((12, 12, 12))
    got = run_without_fallback(compile_with(poisson_round_trip, implementation, gpu), x=x.copy(), N=12)
    np.testing.assert_allclose(got, np.fft.ifftn(4.0 * np.fft.fftn(x)).real, rtol=1e-12, atol=1e-12)


def check_fft_1d_round_trip(implementation: str, gpu: bool = False):
    x = rng_complex((1000, ))
    y, z = np.zeros_like(x), np.zeros_like(x)
    run_without_fallback(compile_with(fft_1d_round_trip, implementation, gpu), x=x.copy(), y=y, z=z, N=1000)
    np.testing.assert_allclose(y, np.fft.fft(x), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(z, x, rtol=1e-12, atol=1e-12)


def forward_and_inverse(axes, norm):
    """``fftn`` and ``ifftn`` of a rank-3 array over ``axes`` under ``norm``, as one program."""

    @dace.program
    def transforms(x: dace.complex128[6, 10, 8]):
        return np.fft.fftn(x, axes=axes, norm=norm), np.fft.ifftn(x, axes=axes, norm=norm)

    return transforms


def check_norms_and_axes(implementation: str, axes, gpu: bool = False):
    """``fftn`` / ``ifftn`` over ``axes`` of a rank-3 array under each of numpy's three norms."""
    for norm in ('backward', 'ortho', 'forward'):
        x = rng_complex((6, 10, 8))
        program = forward_and_inverse(axes, norm)
        forward, inverse = run_without_fallback(compile_with(program, implementation, gpu), x=x.copy())
        np.testing.assert_allclose(forward, np.fft.fftn(x, axes=axes, norm=norm), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inverse, np.fft.ifftn(x, axes=axes, norm=norm), rtol=1e-12, atol=1e-12)


def check_single_precision(implementation: str, gpu: bool = False):
    """complex64 stays complex64 through the transform and its normalisation."""

    @dace.program
    def transforms(x: dace.complex64[N]):
        return np.fft.fft(x), np.fft.ifft(x)

    x = rng_complex((256, ), np.complex64)
    forward, inverse = run_without_fallback(compile_with(transforms, implementation, gpu), x=x.copy(), N=256)
    assert forward.dtype == np.complex64 and inverse.dtype == np.complex64
    np.testing.assert_allclose(forward, np.fft.fft(x), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(inverse, np.fft.ifft(x), rtol=1e-4, atol=1e-5)


def check_leading_axis(implementation: str, gpu: bool = False):
    """``fft(x, axis=0)`` of a matrix: each transform steps by the row length, the batch by one."""

    @dace.program
    def transform(x: dace.complex128[16, 12]):
        return np.fft.fft(x, axis=0)

    x = rng_complex((16, 12))
    got = run_without_fallback(compile_with(transform, implementation, gpu), x=x.copy())
    np.testing.assert_allclose(got, np.fft.fft(x, axis=0), rtol=1e-12, atol=1e-12)


def check_in_place(implementation: str, gpu: bool = False):
    """``x[:] = fft(x)`` overwrites the operand it reads."""

    @dace.program
    def transform(x: dace.complex128[N]):
        x[:] = np.fft.fft(x)

    x = rng_complex((300, ))
    want = np.fft.fft(x)
    run_without_fallback(compile_with(transform, implementation, gpu), x=x, N=300)
    np.testing.assert_allclose(x, want, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_poisson_round_trip_of_a_real_grid(implementation):
    check_poisson_round_trip(implementation)


@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_1d_round_trip_over_a_symbolic_extent(implementation):
    check_fft_1d_round_trip(implementation)


@pytest.mark.parametrize('axes', [None, (1, 2), (0, 2), (2, 0)])
@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_norms_over_every_axis_set(implementation, axes):
    check_norms_and_axes(implementation, axes)


@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_single_precision(implementation):
    check_single_precision(implementation)


@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_leading_axis(implementation):
    check_leading_axis(implementation)


@pytest.mark.parametrize('implementation', CPU_IMPLEMENTATIONS)
def test_cpu_in_place(implementation):
    check_in_place(implementation)


@pytest.mark.fftw
def test_fftw3_falls_back_to_pure_for_a_repeated_axis():
    """numpy transforms a repeated axis twice; one FFTW plan cannot, so the expansion hands it to ``pure``."""

    @dace.program
    def transform(x: dace.complex128[8, 6]):
        return np.fft.fftn(x, axes=(1, 1))

    x = rng_complex((8, 6))
    sdfg = compile_with(transform, 'FFTW3')
    with pytest.warns(UserWarning, match='FFTW3 cannot transform axes'):
        sdfg.expand_library_nodes()
    np.testing.assert_allclose(sdfg(x=x.copy()), np.fft.fftn(x, axes=(1, 1)), rtol=1e-12, atol=1e-12)


@pytest.mark.fftw
def test_canonicalize_lowers_a_host_fft_to_fftw3():
    """The canonicalize finalize picks the vendor transform over the O(N^2) ``pure`` DFT."""
    assert FFTW3.is_installed()
    sdfg = poisson_round_trip.to_sdfg(simplify=True)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.CPU)
    picked = {n.implementation for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (FFT, IFFT))}
    assert picked == {'FFTW3'}, picked


@pytest.mark.parametrize('backend, vendor', [('cuda', 'cuFFT'), ('hip', 'hipFFT')])
def test_each_gpu_backend_prioritizes_its_own_fft(monkeypatch, backend, vendor):
    """Each GPU row names its vendor FFT, so neither backend lowers a host FFT to ``pure``."""
    monkeypatch.setattr('dace.codegen.common.get_gpu_backend', lambda: backend)
    priority = find_fast_library(dtypes.DeviceType.GPU)
    assert vendor in priority and priority.index(vendor) < priority.index('pure'), priority


@pytest.mark.parametrize('backend, vendor', [('cuda', 'cuFFT'), ('hip', 'hipFFT')])
def test_canonicalize_lowers_a_host_gpu_fft_to_the_backend_vendor(monkeypatch, backend, vendor):
    """On the GPU the canonicalize finalize picks the backend's vendor transform through the shared dialect."""
    monkeypatch.setattr('dace.codegen.common.get_gpu_backend', lambda: backend)
    sdfg = poisson_round_trip.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    picked = {n.implementation for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (FFT, IFFT))}
    assert picked == {vendor}, picked


@pytest.mark.gpu
def test_gpu_poisson_round_trip_of_a_real_grid():
    check_poisson_round_trip(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_1d_round_trip_over_a_symbolic_extent():
    check_fft_1d_round_trip(gpu_implementation(), gpu=True)


@pytest.mark.gpu
@pytest.mark.parametrize('axes', [None, (1, 2), (0, 1), (2, )])
def test_gpu_norms_over_every_batched_axis_block(axes):
    check_norms_and_axes(gpu_implementation(), axes, gpu=True)


@pytest.mark.gpu
def test_gpu_single_precision():
    check_single_precision(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_leading_axis():
    check_leading_axis(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_in_place():
    check_in_place(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_fft_falls_back_to_pure_for_a_middle_axis():
    """A middle axis leaves two batch strides, which no ``MakePlanMany`` layout expresses."""

    @dace.program
    def transform(x: dace.complex128[4, 6, 8]):
        return np.fft.fft(x, axis=1)

    x = rng_complex((4, 6, 8))
    sdfg = compile_with(transform, gpu_implementation(), gpu=True)
    with pytest.warns(UserWarning, match='cannot transform axes'):
        sdfg.expand_library_nodes()
    np.testing.assert_allclose(sdfg(x=x.copy()), np.fft.fft(x, axis=1), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

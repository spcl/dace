# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""N-dimensional transforms and transforms along one axis, through the pure and the FFTW3 expansion."""

import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib

rng = np.random.default_rng(0)


def complex_array(shape, order="C") -> np.ndarray:
    return np.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), order=order)


def expanded(program, implementation: str) -> dace.SDFG:
    sdfg = program.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (fftlib.FFT, fftlib.IFFT)):
            node.implementation = implementation
    sdfg.expand_library_nodes()
    return sdfg


@pytest.mark.parametrize("implementation", ["pure", "FFTW3"])
@pytest.mark.parametrize("shape", [(8, 12), (4, 6, 5)])
def test_fftn_and_back(implementation, shape):
    @dace.program
    def fftn_and_back(x: dace.complex128[shape]):
        return np.fft.ifftn(np.fft.fftn(x) * 2)

    x = complex_array(shape)
    np.testing.assert_allclose(expanded(fftn_and_back, implementation)(x), 2 * x)


@pytest.mark.parametrize("implementation", ["pure", "FFTW3"])
@pytest.mark.parametrize("shape,axis", [((8, 12), 0), ((4, 6, 5), 1), ((4, 6, 5), -1)])
def test_fft_along_an_axis(implementation, shape, axis):
    @dace.program
    def fft_along_axis(x: dace.complex128[shape]):
        return np.fft.fft(x, axis=axis)

    x = complex_array(shape)
    np.testing.assert_allclose(expanded(fft_along_axis, implementation)(x), np.fft.fft(x, axis=axis))


@pytest.mark.parametrize("implementation", ["pure", "FFTW3"])
def test_fft_of_a_fortran_ordered_array(implementation):
    """FFTW3 plans C-ordered arrays; a Fortran-ordered one goes through the pure expansion, which reads its strides."""
    sdfg = dace.SDFG(f"fft_fortran_order_{implementation}")
    sdfg.add_array("x", [6, 4], dace.complex128, strides=[1, 6])
    sdfg.add_array("y", [6, 4], dace.complex128, strides=[1, 6])
    state = sdfg.add_state()
    node = fftlib.FFT("fft", axis=0)
    node.implementation = implementation
    state.add_edge(state.add_read("x"), None, node, "_inp", dace.Memlet("x"))
    state.add_edge(node, "_out", state.add_write("y"), None, dace.Memlet("y"))
    x, y = complex_array((6, 4), "F"), np.zeros((6, 4), np.complex128, order="F")
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y, np.fft.fft(x, axis=0))


if __name__ == "__main__":
    for implementation in ("pure", "FFTW3"):
        for shape in ((8, 12), (4, 6, 5)):
            test_fftn_and_back(implementation, shape)
        for shape, axis in (((8, 12), 0), ((4, 6, 5), 1), ((4, 6, 5), -1)):
            test_fft_along_an_axis(implementation, shape, axis)
        test_fft_of_a_fortran_ordered_array(implementation)

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""N-dimensional transforms and transforms over a subset of the axes, against numpy.

Most cases run the default ``pure`` expansion, the separable DFT every vendor expansion falls back to; the
parametrized ones also run FFTW3.
"""

import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib
from dace.libraries.fft.nodes import FFT, IFFT

IMPLEMENTATIONS = ["pure", pytest.param("FFTW3", marks=pytest.mark.fftw)]


# Full N-D (axis=None) via the numpy frontend (fftn / ifftn)


# Batched 1-D (axis=k) via the numpy frontend (fft(x, axis=k))
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((8, 12), 0),
        ((8, 12), -1),
        ((4, 6, 5), 1),
        ((4, 6, 5), -1),
    ],
)
def test_pure_fft_axis(shape, axis):

    @dace.program
    def tester_pure_fft_axis(x: dace.complex128[tuple(shape)]):
        return np.fft.fft(x, axis=axis)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester_pure_fft_axis(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x, axis=axis), rtol=1e-10, atol=1e-10)


def test_pure_ifft_axis_inverse():
    shape, axis = (4, 6, 5), 0

    @dace.program
    def tester_pure_ifft_axis_inverse(x: dace.complex128[4, 6, 5]):
        return np.fft.ifft(x, axis=axis)

    rng = np.random.default_rng(3)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = tester_pure_ifft_axis_inverse(x.copy())
    np.testing.assert_allclose(y, np.fft.ifft(x, axis=axis), rtol=1e-10, atol=1e-10)


# In-place (input array IS the output array) -- the Quantum ESPRESSO pattern
# where ``invfft(f, dfft)`` transforms ``f`` in place.  The frontend always
# allocates a fresh output, so drive the lib node directly to exercise the
# alias-decoupling copy in the builder.
def test_pure_fftn_inplace():
    shape = (6, 5)
    sdfg = dace.SDFG("inplace_fftn")
    sdfg.add_array("buf", shape, dace.complex128)
    state = sdfg.add_state()
    rnode = state.add_read("buf")
    wnode = state.add_write("buf")
    node = FFT("fft")  # axes=None -> full 2-D fftn
    state.add_node(node)
    state.add_edge(rnode, None, node, "_inp", dace.Memlet.from_array("buf", sdfg.arrays["buf"]))
    state.add_edge(node, "_out", wnode, None, dace.Memlet.from_array("buf", sdfg.arrays["buf"]))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(4)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    buf = x.copy()
    sdfg(buf=buf)
    np.testing.assert_allclose(buf, np.fft.fftn(x), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("in_loop", [False, True])
def test_pure_factor_symbol_only_in_the_factor(in_loop):
    """A symbol that only the normalization factor names -- an SDFG symbol, or the iterator of the loop around the
    node -- still reaches the expansion: the scaling tasklet is C++, whose symbols nothing infers from its code."""
    from dace.sdfg.state import LoopRegion

    shape = (6, 5)
    sdfg = dace.SDFG(f"factor_symbol_fftn_{int(in_loop)}")
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_array("x", shape, dace.complex128)
    sdfg.add_array("y", shape, dace.complex128)
    if in_loop:
        loop = LoopRegion("passes", "it < 3", "it", "it = 2", "it = it + 1")
        sdfg.add_node(loop, is_start_block=True)
        state = loop.add_state("fft", is_start_block=True)
    else:
        state = sdfg.add_state()
    node = FFT("fft")
    node.factor = "1/(K*it)" if in_loop else "1/K"
    state.add_node(node)
    state.add_edge(state.add_read("x"), None, node, "_inp", dace.Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(node, "_out", state.add_write("y"), None, dace.Memlet.from_array("y", sdfg.arrays["y"]))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(11)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    y = np.zeros_like(x)
    sdfg(x=x, y=y, K=4)
    np.testing.assert_allclose(y, np.fft.fftn(x) / (4 * (2 if in_loop else 1)), rtol=1e-10, atol=1e-10)


# Symbolic dimensions -- the shape is only known at call time.


@pytest.mark.parametrize("norm", ["backward", "ortho"])
def test_pure_ifftn_symbolic_factor_is_emitted_in_floating_point(norm):
    """The tasklet carries the normalization as text, so the text is what decides integer versus floating division."""
    M, N = dace.symbol("M"), dace.symbol("N")

    @dace.program
    def tester_pure_ifftn_symbolic_factor_is_emitted_in_floating_point(x: dace.complex128[M, N]):
        return np.fft.ifftn(x, norm=norm)

    sdfg = tester_pure_ifftn_symbolic_factor_is_emitted_in_floating_point.to_sdfg(simplify=False)
    sdfg.expand_library_nodes()
    codes = [n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    scaled = [code for code in codes if "o *=" in code]
    assert scaled, f"no DFT tasklet carries the {norm} factor: {codes}"
    assert all("(1.0 * M)" in code and "(1.0 * N)" in code for code in scaled), scaled
    assert not any("1/(M*N)" in code.replace(" ", "") for code in scaled), scaled


# A subset of the axes (np.fft.fftn(x, axes=...)): unlisted axes are batch dimensions.
PARTIAL_AXES = [
    ((3, 4, 5, 2), (1, 2, 3)),
    ((4, 5, 6), (0, 2)),
    ((4, 5, 6), (-1, -3)),
    ((6, 5), (0, 0)),
]


def random_array(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def test_ifftn_of_fftn_over_a_subset_of_axes_returns_the_input():

    @dace.program
    def tester_ifftn_of_fftn_over_a_subset_of_axes_returns_the_input(x: dace.complex128[3, 4, 5, 2]):
        return np.fft.ifftn(np.fft.fftn(x, axes=(1, 2, 3)), axes=(1, 2, 3))

    x = random_array((3, 4, 5, 2), np.complex128, 11)
    np.testing.assert_allclose(
        tester_ifftn_of_fftn_over_a_subset_of_axes_returns_the_input(x.copy()), x, rtol=1e-10, atol=1e-10
    )


def test_fft2_and_ifft2_transform_the_last_two_axes_by_default():
    """Without a replacement the frontend falls back to a Python callback, which also matches numpy."""

    @dace.program
    def tester_fft2_and_ifft2_transform_the_last_two_axes_by_default(x: dace.complex128[4, 5, 6]):
        return np.fft.fft2(x), np.fft.ifft2(x, axes=(0, 2))

    sdfg = tester_fft2_and_ifft2_transform_the_last_two_axes_by_default.to_sdfg(simplify=False)
    got = sorted((type(n).__name__, n.axes) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (FFT, IFFT)))
    assert got == [("FFT", [1, 2]), ("IFFT", [0, 2])], got
    x = random_array((4, 5, 6), np.complex128, 12)
    forward, inverse = tester_fft2_and_ifft2_transform_the_last_two_axes_by_default(x.copy())
    np.testing.assert_allclose(forward, np.fft.fft2(x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(inverse, np.fft.ifft2(x, axes=(0, 2)), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "axes,want",
    [
        (None, None),
        ((1, 0), None),
        ((-2, -1), None),
        ((0,), [0]),
        ((-1, 0, -1), [1, 0, 1]),
    ],
)
def test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes(axes, want):
    """A full-array transform must keep the whole-array lowering (``plan_dft_{rank}d``, the rank-generic DFT)."""

    @dace.program
    def tester_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes(x: dace.complex128[4, 5]):
        return np.fft.fftn(x, axes=axes)

    sdfg = tester_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes.to_sdfg(simplify=False)
    got = [n.axes for n, _ in sdfg.all_nodes_recursive() if isinstance(n, FFT)]
    assert got == [want], got


@pytest.mark.parametrize("axes", [(), (2,), (-3,)])
def test_empty_or_out_of_range_axes_are_refused(axes):

    @dace.program
    def tester_empty_or_out_of_range_axes_are_refused(x: dace.complex128[4, 5]):
        return np.fft.fftn(x, axes=axes)

    with pytest.raises((ValueError, NotImplementedError)):
        tester_empty_or_out_of_range_axes_are_refused.to_sdfg(simplify=False)


def complex_array(shape, order="C") -> np.ndarray:
    rng = np.random.default_rng(sum(shape))
    return np.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), order=order)


def expanded(program, implementation: str) -> dace.SDFG:
    sdfg = program.to_sdfg()
    sdfg.name = f"{sdfg.name}_{implementation}"
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (fftlib.FFT, fftlib.IFFT)):
            node.implementation = implementation
    sdfg.expand_library_nodes()
    return sdfg


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", [(8, 12), (4, 6, 5)])
def test_fftn_and_back(implementation, shape):
    @dace.program
    def fftn_and_back(x: dace.complex128[shape]):
        return np.fft.ifftn(np.fft.fftn(x) * 2)

    x = complex_array(shape)
    np.testing.assert_allclose(expanded(fftn_and_back, implementation)(x), 2 * x)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("shape,axis", [((8, 12), 0), ((4, 6, 5), 1), ((4, 6, 5), -1)])
def test_fft_along_an_axis(implementation, shape, axis):
    @dace.program
    def fft_along_axis(x: dace.complex128[shape]):
        return np.fft.fft(x, axis=axis)

    x = complex_array(shape)
    np.testing.assert_allclose(expanded(fft_along_axis, implementation)(x), np.fft.fft(x, axis=axis))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fft_of_a_fortran_ordered_array(implementation):
    """FFTW3 plans C-ordered arrays; a Fortran-ordered one goes through the pure expansion, which reads its strides."""
    sdfg = dace.SDFG(f"fft_fortran_order_{implementation}")
    sdfg.add_array("x", [6, 4], dace.complex128, strides=[1, 6])
    sdfg.add_array("y", [6, 4], dace.complex128, strides=[1, 6])
    state = sdfg.add_state()
    node = fftlib.FFT("fft", axes=[0])
    node.implementation = implementation
    state.add_edge(state.add_read("x"), None, node, "_inp", dace.Memlet("x"))
    state.add_edge(node, "_out", state.add_write("y"), None, dace.Memlet("y"))
    x, y = complex_array((6, 4), "F"), np.zeros((6, 4), np.complex128, order="F")
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y, np.fft.fft(x, axis=0))


if __name__ == "__main__":
    test_pure_fft_axis((8, 12), 0)
    test_pure_fft_axis((8, 12), -1)
    test_pure_fft_axis((4, 6, 5), 1)
    test_pure_fft_axis((4, 6, 5), -1)
    test_pure_ifft_axis_inverse()
    test_pure_fftn_inplace()
    test_pure_ifftn_symbolic_factor_is_emitted_in_floating_point("backward")
    test_pure_ifftn_symbolic_factor_is_emitted_in_floating_point("ortho")
    test_ifftn_of_fftn_over_a_subset_of_axes_returns_the_input()
    test_fft2_and_ifft2_transform_the_last_two_axes_by_default()
    test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes(None, None)
    test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes((1, 0), None)
    test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes((-2, -1), None)
    test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes((0,), [0])
    test_only_a_transform_that_skips_an_axis_or_repeats_one_sets_the_node_axes((-1, 0, -1), [1, 0, 1])
    test_empty_or_out_of_range_axes_are_refused(())
    test_empty_or_out_of_range_axes_are_refused((2,))
    test_empty_or_out_of_range_axes_are_refused((-3,))
    test_fftn_and_back("pure", (8, 12))
    test_fftn_and_back("pure", (4, 6, 5))
    test_fftn_and_back("FFTW3", (8, 12))
    test_fftn_and_back("FFTW3", (4, 6, 5))
    test_fft_along_an_axis("pure", (8, 12), 0)
    test_fft_along_an_axis("pure", (4, 6, 5), 1)
    test_fft_along_an_axis("pure", (4, 6, 5), -1)
    test_fft_along_an_axis("FFTW3", (8, 12), 0)
    test_fft_along_an_axis("FFTW3", (4, 6, 5), 1)
    test_fft_along_an_axis("FFTW3", (4, 6, 5), -1)
    test_fft_of_a_fortran_ordered_array("pure")
    test_fft_of_a_fortran_ordered_array("FFTW3")

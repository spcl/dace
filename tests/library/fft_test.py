# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import re

import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib
from dace.libraries.fft import BlockLocalFFTFusion
from dace.libraries.fft.algorithms import stockham
from dace.transformation.dataflow import OTFMapFusion


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


@pytest.mark.gpu
def test_cufft():
    @dace.program
    def tester(x: dace.complex128[210]):
        return np.fft.fft(x)

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    fftlib.FFT.default_implementation = "cuFFT"
    sdfg.expand_library_nodes()
    fftlib.FFT.default_implementation = "pure"

    a = np.random.rand(210) + 1j * np.random.rand(210)
    b = sdfg(a)
    assert np.allclose(b, np.fft.fft(a))


@pytest.mark.gpu
def test_cufft_twoplans():
    @dace.program
    def tester(x: dace.complex128[210], y: dace.complex64[19]):
        return np.fft.fft(x), np.fft.ifft(y, norm="forward")

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    fftlib.FFT.default_implementation = "cuFFT"
    fftlib.IFFT.default_implementation = "cuFFT"
    sdfg.expand_library_nodes()
    fftlib.FFT.default_implementation = "pure"
    fftlib.IFFT.default_implementation = "pure"

    a = np.random.rand(210) + 1j * np.random.rand(210)
    b = (np.random.rand(19) + 1j * np.random.rand(19)).astype(np.complex64)
    c, d = sdfg(a, b)
    assert np.allclose(c, np.fft.fft(a), rtol=1e-3, atol=1e-5)
    assert np.allclose(d, np.fft.ifft(b, norm="forward"), rtol=1e-3, atol=1e-5)


def _random_complex(shape, dtype=np.complex128) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def _set_fft_implementation(sdfg: dace.SDFG, implementation: str) -> None:
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (fftlib.FFT, fftlib.IFFT)):
            node.implementation = implementation


def _stockham_stage_count(sdfg: dace.SDFG) -> int:
    """
    Counts the stage maps, over whole arrays or in the per-row (or per-column-block) kernels of the CPU.
    """
    return sum(
        1
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and re.fullmatch(r"stockham_(axis_\d+_)?stage_\d+_map", node.map.label)
    )


def test_stockham_factorize():
    assert stockham.factorize(1) == []
    assert stockham.factorize(4096) == [16, 16, 16]
    assert stockham.factorize(2**20) == [16] * 5
    assert stockham.factorize(60) == [3, 5, 4]
    assert stockham.factorize(2048) == [8, 16, 16]
    assert stockham.factorize(143) == [11, 13]
    assert stockham.factorize(17) == [17]
    assert stockham.factorize(8001) == [127, 3, 3, 7]
    assert stockham.factorize(0) is None


@pytest.mark.parametrize("n", (1, 2, 16, 21, 60, 143, 4096, 3 * 2**12, 8001))
def test_stockham_1d(n):
    @dace.program
    def tester(x: dace.complex128[n]):
        return np.fft.fft(x), np.fft.ifft(x), np.fft.fft(x, norm="ortho"), np.fft.ifft(x, norm="forward")

    sdfg = tester.to_sdfg()
    _set_fft_implementation(sdfg, "Stockham")
    a = _random_complex(n)
    results = sdfg(x=a)
    expected = (np.fft.fft(a), np.fft.ifft(a), np.fft.fft(a, norm="ortho"), np.fft.ifft(a, norm="forward"))
    for result, reference in zip(results, expected):
        assert np.allclose(result, reference)


def test_stockham_complex64():
    @dace.program
    def tester(x: dace.complex64[2**14]):
        return np.fft.fft(x), np.fft.ifft(x)

    a = _random_complex(2**14, np.complex64)
    b, c = tester(a)
    assert b.dtype == np.complex64 and c.dtype == np.complex64
    assert np.allclose(b, np.fft.fft(a), rtol=1e-4, atol=1e-3)
    assert np.allclose(c, np.fft.ifft(a), rtol=1e-4, atol=1e-6)


def test_stockham_r2c():
    @dace.program
    def tester(x: dace.float32[6, 64]):
        return np.fft.fft(x)

    a = np.random.default_rng(0).standard_normal((6, 64)).astype(np.float32)
    b = tester(a)
    assert b.dtype == np.complex64
    assert np.allclose(b, np.fft.fft(a), rtol=1e-4, atol=1e-4)


def test_stockham_nd():
    """
    Transforms of the inner axes are computed per row and those of the outer axes per block of eight columns, in
    padded buffers for the length-64 axis.
    """

    @dace.program
    def tester(x: dace.complex128[12, 64, 8]):
        return (
            np.fft.fft(x, axis=0),
            np.fft.fft(x, axis=1),
            np.fft.ifft(x),
            np.fft.fft2(x),
            np.fft.fftn(x, axes=(0, 2)),
            np.fft.ifftn(x),
        )

    a = _random_complex((12, 64, 8))
    expected = (
        np.fft.fft(a, axis=0),
        np.fft.fft(a, axis=1),
        np.fft.ifft(a),
        np.fft.fft2(a),
        np.fft.fftn(a, axes=(0, 2)),
        np.fft.ifftn(a),
    )
    for result, reference in zip(tester(a), expected):
        assert np.allclose(result, reference)


def test_stockham_groups():
    """
    Axes whose transforms do not fit into the buffers of a map iteration are computed in groups of stages, each one
    map over views of the arrays, with the twiddle factors of the group applied at its start.
    """
    assert stockham.group_lengths(2**16, 512) == [256, 256]
    assert stockham.group_lengths(2**20, 512) == [128, 128, 64]
    assert stockham.group_lengths(2 * 1031, 512) is None
    assert stockham.axis_groups((2**16,), 0, dace.complex128, {}) == [(1, 256, 256), (256, 256, 1)]
    assert stockham.axis_groups((4096,), 0, dace.complex128, {}) == [(1, 64, 64), (64, 64, 1)]

    @dace.program
    def tester(x: dace.complex128[2**16], c: dace.complex64[4096, 16], r: dace.float64[4, 3 * 2**12]):
        return np.fft.ifft(x, norm="ortho"), np.fft.fft(c, axis=0), np.fft.fft(r)

    sdfg = tester.to_sdfg()
    sdfg.expand_library_nodes()
    labels = [node.label for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)]
    assert sum(re.fullmatch(r"stockham_group_\d+", label) is not None for label in labels) == 6

    x, c = _random_complex(2**16), _random_complex((4096, 16), np.complex64)
    r = np.random.default_rng(0).random((4, 3 * 2**12))
    results = tester(x, c, r)
    assert np.allclose(results[0], np.fft.ifft(x, norm="ortho"))
    assert np.allclose(results[1], np.fft.fft(c, axis=0), rtol=1e-4, atol=1e-3)
    assert np.allclose(results[2], np.fft.fft(r))


def test_stockham_symbolic_batch():
    """
    A symbolic batch dimension keeps the buffers allocated per call, so calls with different sizes are correct.
    """
    B = dace.symbol("B")

    @dace.program
    def tester(x: dace.complex128[B, 64]):
        return np.fft.fft(x)

    csdfg = tester.to_sdfg().compile()
    for batch in (3, 17, 5):
        a = _random_complex((batch, 64))
        assert np.allclose(csdfg(x=a, B=batch), np.fft.fft(a))


def test_stockham_inside_map():
    """
    An FFT node inside a map transforms the subset it is connected to, with per-iteration buffers.
    """
    batch, n = 8, 4096
    sdfg = dace.SDFG("stockham_inside_map")
    sdfg.add_array("x", [batch, n], dace.complex128)
    sdfg.add_array("y", [batch, n], dace.complex128)
    state = sdfg.add_state()
    me, mx = state.add_map("rows", dict(i=f"0:{batch}"))
    node = fftlib.FFT("fft")
    node.axes = [1]
    state.add_memlet_path(state.add_read("x"), me, node, dst_conn="_inp", memlet=dace.Memlet(f"x[i, 0:{n}]"))
    state.add_memlet_path(node, mx, state.add_write("y"), src_conn="_out", memlet=dace.Memlet(f"y[i, 0:{n}]"))
    sdfg.expand_library_nodes()
    assert _stockham_stage_count(sdfg) == 3
    for desc in sdfg.arrays_recursive():
        if desc[1].startswith("__stockham"):
            assert desc[2].lifetime == dace.AllocationLifetime.Scope

    a = _random_complex((batch, n))
    b = np.empty_like(a)
    sdfg(x=a, y=b)
    assert np.allclose(b, np.fft.fft(a))


def test_stockham_dispatch():
    """
    The native expansion uses Stockham for fixed lengths (with a general-radix stage for a prime factor without a
    generated butterfly) and falls back to the DFT otherwise.
    """
    N = dace.symbol("N")

    @dace.program
    def fixed(x: dace.complex128[60]):
        return np.fft.fft(x)

    @dace.program
    def prime(x: dace.complex128[17]):
        return np.fft.fft(x)

    @dace.program
    def symbolic(x: dace.complex128[N]):
        return np.fft.fft(x)

    for program, stages in ((fixed, 3), (prime, 1), (symbolic, 0)):
        sdfg = program.to_sdfg()
        sdfg.expand_library_nodes()
        assert _stockham_stage_count(sdfg) == stages

    sdfg = symbolic.to_sdfg()
    _set_fft_implementation(sdfg, "Stockham")
    with pytest.raises(ValueError, match="Stockham"):
        sdfg.expand_library_nodes()


def test_stockham_gpu_stage_fusion():
    """
    On a GPU, a transform that does not fit into shared memory is a single state with one map per stage, which
    inlines and fuses with elementwise maps.
    """
    N = 4096
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(x: dace.complex128[N] @ G, w: dace.complex128[N] @ G, y: dace.complex128[N] @ G):
        y[:] = np.fft.ifft(np.fft.fft(x * 2.0) * w)

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    sdfg.expand_library_nodes()
    sdfg.simplify()
    assert len(list(sdfg.states())) == 1
    assert not any(isinstance(node, dace.nodes.NestedSDFG) for node, _ in sdfg.all_nodes_recursive())
    assert _stockham_stage_count(sdfg) == 6
    assert sdfg.apply_transformations_repeated(OTFMapFusion) == 2


def test_cufft_partial_axes():
    @dace.program
    def tester(x: dace.complex128[16, 32] @ dace.StorageType.GPU_Global):
        return np.fft.fft(x, axis=0)

    sdfg = tester.to_sdfg()
    _set_fft_implementation(sdfg, "cuFFT")
    with pytest.raises(NotImplementedError, match="all axes"):
        sdfg.expand_library_nodes()


def _block_local_kernel_count(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.label == "stockham_fft_block_local_block"
    )


def _stockham_gpu_program(dtype_dace):
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(
        x: dtype_dace[2**16] @ G,
        m: dtype_dace[24, 40] @ G,
        v: dtype_dace[8, 323] @ G,
        y: dtype_dace[2**16] @ G,
        z: dtype_dace[24, 40] @ G,
        w: dtype_dace[8, 323] @ G,
    ):
        y[:] = np.fft.ifft(np.fft.fft(x) * 2)
        z[:] = np.fft.fft2(m)
        w[:] = np.fft.ifft(v, norm="ortho")

    return tester


def test_stockham_gpu_expansion():
    """
    On a GPU, transforms that fit into shared memory and include the contiguous dimension are one kernel with a
    thread block per transform (here, the two-dimensional transform and the general-radix transforms of length
    ``17 * 19``), and larger ones have one kernel per stage.
    """
    sdfg = _stockham_gpu_program(dace.complex64).to_sdfg()
    sdfg.apply_gpu_transformations()
    sdfg.expand_library_nodes()
    assert _block_local_kernel_count(sdfg) == 2
    assert _stockham_stage_count(sdfg) == 2 * len(stockham.factorize(2**16))


def test_stockham_gpu_shared_memory_limit():
    """
    A transform is computed by one thread block if its two buffers fit into ``compiler.cuda.max_shared_memory``, beyond
    the static limit of 48 KiB (here, 64 KiB for 4096 complex64 elements), and by one kernel per stage otherwise.
    """
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(x: dace.complex64[8, 4096] @ G, y: dace.complex64[8, 4096] @ G):
        y[:] = np.fft.fft(x)

    for limit, block_local in ((64 * 1024, True), (48 * 1024, False)):
        with dace.config.set_temporary("compiler", "cuda", "max_shared_memory", value=limit):
            sdfg = tester.to_sdfg()
            sdfg.apply_gpu_transformations()
            sdfg.expand_library_nodes()
        assert _block_local_kernel_count(sdfg) == int(block_local)
        assert _stockham_stage_count(sdfg) == (0 if block_local else len(stockham.factorize(4096)))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_stockham_gpu(dtype):
    import cupy as cp  # CuPy is only installed on GPU machines

    sdfg = _stockham_gpu_program(dace.dtypes.dtype_to_typeclass(dtype)).to_sdfg()
    sdfg.apply_gpu_transformations()
    _set_fft_implementation(sdfg, "Stockham")
    x, m = cp.asarray(_random_complex(2**16, dtype)), cp.asarray(_random_complex((24, 40), dtype))
    v = cp.asarray(_random_complex((8, 323), dtype))
    y, z, w = cp.empty_like(x), cp.empty_like(m), cp.empty_like(v)
    sdfg(x=x, m=m, v=v, y=y, z=z, w=w)
    tolerance = 1e-4 if dtype == np.complex64 else 1e-10
    assert np.allclose(y.get(), 2 * x.get(), rtol=tolerance, atol=tolerance)
    assert np.allclose(z.get(), np.fft.fft2(m.get()), rtol=tolerance, atol=100 * tolerance)
    assert np.allclose(w.get(), np.fft.ifft(v.get(), norm="ortho"), rtol=tolerance, atol=tolerance)


def _fuse_block_local(sdfg: dace.SDFG) -> int:
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    return sdfg.apply_transformations_repeated(BlockLocalFFTFusion)


def test_block_local_fusion_structure():
    """
    FFT -> pointwise map -> IFFT becomes one kernel with one state per stage and synchronization between states.
    """
    B, N = 8, 64
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(x: dace.complex128[B, N] @ G, w: dace.complex128[N] @ G, y: dace.complex128[B, N] @ G):
        y[:] = np.fft.ifft(np.fft.fft(x) * w)

    sdfg = tester.to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    sdfg.validate()
    assert not any(isinstance(node, dace.nodes.LibraryNode) for node, _ in sdfg.all_nodes_recursive())
    blocks = [
        node
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.NestedSDFG) and node.sdfg.name == "fft_block_local"
    ]
    assert len(blocks) == 1
    num_states = len(list(blocks[0].sdfg.states()))
    assert num_states == 2 * len(stockham.plan_stages((N,), [0], {})) + 1
    shared = [desc for desc in blocks[0].sdfg.arrays.values() if desc.storage == dace.StorageType.GPU_Shared]
    assert len(shared) == 2

    cuda_code = next(code for code in sdfg.generate_code() if code.language == "cu").clean_code
    assert cuda_code.count("__syncthreads();") == num_states - 1


@pytest.mark.parametrize(
    "case", ("other_transform", "too_large", "spectrum_reused", "different_axes", "symbolic", "in_place")
)
def test_block_local_fusion_not_applicable(case):
    B, N = 8, (8192 if case == "too_large" else 32)
    if case == "symbolic":
        N = dace.symbol("N")
    G = dace.StorageType.GPU_Global
    ctype = dace.complex64

    if case == "other_transform":

        @dace.program
        def tester(x: ctype[B, N] @ G, y: ctype[B, N] @ G):
            X = np.fft.fft(x)
            Y = np.empty_like(X)
            for i, f in dace.map[0:B, 0:N]:
                Y[i, f] = X[i, f] + X[(i + 1) % B, f]
            y[:] = np.fft.ifft(Y)

    elif case == "spectrum_reused":

        @dace.program
        def tester(x: ctype[B, N] @ G, y: ctype[B, N] @ G, z: ctype[B, N] @ G):
            X = np.fft.fft(x)
            z[:] = X
            y[:] = np.fft.ifft(X * 2)

    elif case == "different_axes":

        @dace.program
        def tester(x: ctype[N, N] @ G, y: ctype[N, N] @ G):
            y[:] = np.fft.ifft(np.fft.fft(x, axis=1) * 2, axis=0)

    elif case == "in_place":

        @dace.program
        def tester(x: ctype[B, N] @ G):
            x[:] = np.fft.ifft(np.fft.fft(x) * 2)

    else:

        @dace.program
        def tester(x: ctype[B, N] @ G, y: ctype[B, N] @ G):
            y[:] = np.fft.ifft(np.fft.fft(x) * 2)

    assert _fuse_block_local(tester.to_sdfg()) == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,n", ((np.complex64, 2048), (np.complex64, 4096), (np.complex128, 1024), (np.complex128, 60))
)
def test_block_local_fusion_gpu(dtype, n):
    """
    A frequency-domain filter with a stencil along the transformed axis and orthonormal transforms. The buffers of 4096
    complex64 elements exceed the static shared memory of a thread block and use dynamic shared memory.
    """
    import cupy as cp  # CuPy is only installed on GPU machines

    B, N = 33, n
    G = dace.StorageType.GPU_Global
    ctype = dace.dtypes.dtype_to_typeclass(dtype)

    @dace.program
    def tester(x: ctype[B, N] @ G, w: ctype[N] @ G, y: ctype[B, N] @ G):
        X = np.fft.fft(x, norm="ortho")
        Y = np.empty_like(X)
        for i, f in dace.map[0:B, 0:N]:
            Y[i, f] = X[i, f] * w[f] + 0.5 * X[i, (f + 1) % N]
        y[:] = np.fft.ifft(Y, norm="ortho")

    sdfg = tester.to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    x, w = _random_complex((B, N), dtype), _random_complex(N, dtype)
    y = cp.empty((B, N), dtype)
    sdfg(x=cp.asarray(x), w=cp.asarray(w), y=y)

    X = np.fft.fft(x, norm="ortho")
    expected = np.fft.ifft(X * w + 0.5 * np.roll(X, -1, axis=1), norm="ortho")
    tolerance = 1e-4 if dtype == np.complex64 else 1e-10
    assert np.allclose(y.get(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
def test_block_local_fusion_fft2_gpu():
    """
    Two-dimensional transforms of the trailing axes, with a filter that depends on the batch index.
    """
    import cupy as cp  # CuPy is only installed on GPU machines

    B, H, W = 6, 16, 32
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(
        x: dace.complex128[B, H, W] @ G,
        w: dace.complex128[H, W] @ G,
        s: dace.float64[B] @ G,
        y: dace.complex128[B, H, W] @ G,
    ):
        X = np.fft.fft2(x)
        Y = np.empty_like(X)
        for b, i, j in dace.map[0:B, 0:H, 0:W]:
            Y[b, i, j] = X[b, i, j] * w[i, j] * s[b]
        y[:] = np.fft.ifft2(Y)

    sdfg = tester.to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    x, w, s = _random_complex((B, H, W)), _random_complex((H, W)), np.random.rand(B)
    y = cp.empty((B, H, W), np.complex128)
    sdfg(x=cp.asarray(x), w=cp.asarray(w), s=cp.asarray(s), y=y)
    expected = np.fft.ifft2(np.fft.fft2(x) * w * s[:, None, None])
    assert np.allclose(y.get(), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.gpu
def test_block_local_fusion_r2c_gpu():
    """
    A real input, transformed along the first axis.
    """
    import cupy as cp  # CuPy is only installed on GPU machines

    N, B = 256, 40
    G = dace.StorageType.GPU_Global

    @dace.program
    def tester(x: dace.float32[N, B] @ G, w: dace.complex64[N, B] @ G, y: dace.complex64[N, B] @ G):
        y[:] = np.fft.ifft(np.fft.fft(x, axis=0) * w, axis=0)

    sdfg = tester.to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    x, w = np.random.rand(N, B).astype(np.float32), _random_complex((N, B), np.complex64)
    y = cp.empty((N, B), np.complex64)
    sdfg(x=cp.asarray(x), w=cp.asarray(w), y=y)
    expected = np.fft.ifft(np.fft.fft(x, axis=0) * w, axis=0)
    assert np.allclose(y.get(), expected, rtol=1e-4, atol=1e-4)


def _band_pass_program(B: int, H: int, W: int, G: dace.StorageType = dace.StorageType.GPU_Global):
    """
    A radial band pass, written with a conditional that becomes a nested SDFG in the map.
    """

    @dace.program
    def band_pass(x: dace.complex128[B, H, W] @ G, lo: dace.float64, hi: dace.float64, y: dace.complex128[B, H, W] @ G):
        X = np.fft.fft2(x)
        Y = np.empty_like(X)
        for b, i, j in dace.map[0:B, 0:H, 0:W]:
            ki = min(i, H - i)
            kj = min(j, W - j)
            r = np.sqrt(ki * ki + kj * kj)
            if r >= lo and r < hi:
                Y[b, i, j] = X[b, i, j]
            else:
                Y[b, i, j] = 0
        y[:] = np.fft.ifft2(Y)

    return band_pass


def test_block_local_fusion_conditional():
    sdfg = _band_pass_program(5, 16, 32).to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    sdfg.validate()
    sdfg.generate_code()


@pytest.mark.gpu
def test_block_local_fusion_conditional_gpu():
    import cupy as cp  # CuPy is only installed on GPU machines

    B, H, W = 5, 32, 32
    sdfg = _band_pass_program(B, H, W).to_sdfg()
    assert _fuse_block_local(sdfg) == 1
    x = _random_complex((B, H, W))
    y = cp.empty((B, H, W), np.complex128)
    sdfg(x=cp.asarray(x), lo=3.0, hi=9.0, y=y)
    ki, kj = np.fft.fftfreq(H, 1 / H)[:, None], np.fft.fftfreq(W, 1 / W)[None, :]
    r = np.sqrt(ki**2 + kj**2)
    expected = np.fft.ifft2(np.fft.fft2(x) * ((r >= 3) & (r < 9)))
    assert np.allclose(y.get(), expected, rtol=1e-10, atol=1e-10)


def _fused_block(sdfg: dace.SDFG) -> dace.SDFG:
    """
    Returns the SDFG of an iteration of the only fused kernel.
    """
    (block,) = [
        node.sdfg
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.NestedSDFG) and node.sdfg.name == "fft_block_local"
    ]
    return block


@pytest.mark.parametrize("case", ("pointwise", "stencil", "general", "columns", "fft2", "conditional"))
def test_block_local_fusion_cpu(case):
    """
    On the CPU, each iteration of the fused map computes one transform (or eight adjacent columns) in stack buffers.
    """
    B, N = 33, (60 if case == "general" else 256)

    if case == "stencil":

        @dace.program
        def tester(x: dace.complex128[B, N], w: dace.complex128[N], y: dace.complex128[B, N]):
            X = np.fft.fft(x, norm="ortho")
            Y = np.empty_like(X)
            for i, f in dace.map[0:B, 0:N]:
                Y[i, f] = X[i, f] * w[f] + 0.5 * X[i, (f + 1) % N]
            y[:] = np.fft.ifft(Y, norm="ortho")

        x, w = _random_complex((B, N)), _random_complex(N)
        X = np.fft.fft(x, norm="ortho")
        expected = np.fft.ifft(X * w + 0.5 * np.roll(X, -1, axis=1), norm="ortho")
        arguments = dict(x=x, w=w)
    elif case == "columns":
        # A real input, transformed along the first axis: iterations compute eight adjacent columns
        B = 40

        @dace.program
        def tester(x: dace.float32[N, B], w: dace.complex64[N, B], y: dace.complex64[N, B]):
            y[:] = np.fft.ifft(np.fft.fft(x, axis=0) * w, axis=0)

        x, w = np.random.default_rng(0).random((N, B)).astype(np.float32), _random_complex((N, B), np.complex64)
        expected = np.fft.ifft(np.fft.fft(x, axis=0) * w, axis=0)
        arguments = dict(x=x, w=w)
    elif case == "fft2":
        B, H, W = 6, 16, 32

        @dace.program
        def tester(
            x: dace.complex128[B, H, W], w: dace.complex128[H, W], s: dace.float64[B], y: dace.complex128[B, H, W]
        ):
            X = np.fft.fft2(x)
            Y = np.empty_like(X)
            for b, i, j in dace.map[0:B, 0:H, 0:W]:
                Y[b, i, j] = X[b, i, j] * w[i, j] * s[b]
            y[:] = np.fft.ifft2(Y)

        x, w, s = _random_complex((B, H, W)), _random_complex((H, W)), np.random.default_rng(0).random(B)
        expected = np.fft.ifft2(np.fft.fft2(x) * w * s[:, None, None])
        arguments = dict(x=x, w=w, s=s)
    elif case == "conditional":
        B, H, W = 5, 32, 32
        tester = _band_pass_program(B, H, W, dace.StorageType.Default)
        x = _random_complex((B, H, W))
        ki, kj = np.fft.fftfreq(H, 1 / H)[:, None], np.fft.fftfreq(W, 1 / W)[None, :]
        r = np.sqrt(ki**2 + kj**2)
        expected = np.fft.ifft2(np.fft.fft2(x) * ((r >= 3) & (r < 9)))
        arguments = dict(x=x, lo=3.0, hi=9.0)
    else:

        @dace.program
        def tester(x: dace.complex128[B, N], w: dace.complex128[N], y: dace.complex128[B, N]):
            X = np.fft.fft(x)
            Y = np.empty_like(X)
            for i, f in dace.map[0:B, 0:N]:
                Y[i, f] = X[i, f] * w[f]
            y[:] = np.fft.ifft(Y)

        x, w = _random_complex((B, N)), _random_complex(N)
        expected = np.fft.ifft(np.fft.fft(x) * w)
        arguments = dict(x=x, w=w)

    sdfg = tester.to_sdfg()
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(BlockLocalFFTFusion) == 1
    sdfg.validate()
    assert not any(isinstance(node, dace.nodes.LibraryNode) for node, _ in sdfg.all_nodes_recursive())
    buffers = [desc for name, desc in _fused_block(sdfg).arrays.items() if name.startswith("__fft_block_buf")]
    assert len(buffers) == 2 and all(desc.storage == dace.StorageType.Register for desc in buffers)
    if case == "pointwise":
        # Split into padded rows, as only pointwise accesses are restated
        assert len(buffers[0].shape) == 3 and buffers[0].total_size > N
    elif case == "columns":
        assert buffers[0].shape[-1] == 8

    y = np.empty(expected.shape, expected.dtype.type if case != "columns" else np.complex64)
    sdfg(**arguments, y=y)
    tolerance = 1e-4 if case == "columns" else 1e-10
    assert np.allclose(y, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("case", ("too_large", "partial_columns"))
def test_block_local_fusion_cpu_not_applicable(case):
    """
    The columns of an iteration must fit into its stack buffers and span eight adjacent elements.
    """
    N, B = (4096, 256) if case == "too_large" else (64, 12)

    @dace.program
    def tester(x: dace.complex128[N, B], y: dace.complex128[N, B]):
        y[:] = np.fft.ifft(np.fft.fft(x, axis=0) * 2, axis=0)

    sdfg = tester.to_sdfg()
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated(BlockLocalFFTFusion) == 0


def _shaped_fft_program(G=dace.StorageType.Default):

    @dace.program
    def tester(
        x: dace.float32[3, 26, 26] @ G,
        z: dace.complex128[40] @ G,
        a: dace.complex64[3, 32, 32] @ G,
        b: dace.complex128[16] @ G,
        c: dace.complex64[3, 26, 48] @ G,
        d: dace.complex64[3, 26, 8192] @ G,
    ):
        a[:] = np.fft.fft2(x, s=(32, 32))
        b[:] = np.fft.ifft(z, n=16)
        c[:] = np.fft.fft(x, n=48, norm="ortho")
        d[:] = np.fft.fft(x, n=8192)

    return tester


def _check_shaped_fft(x: np.ndarray, z: np.ndarray, results: dict) -> None:
    expected = dict(
        a=np.fft.fft2(x, s=(32, 32)),
        b=np.fft.ifft(z, n=16),
        c=np.fft.fft(x, n=48, norm="ortho"),
        d=np.fft.fft(x, n=8192),
    )
    for name, value in expected.items():
        tolerance = 1e-12 if results[name].dtype == np.complex128 else 1e-5
        assert np.allclose(results[name], value, rtol=tolerance, atol=tolerance * np.abs(value).max()), name


def test_fft_shape():
    """
    Transforms of the lengths given by ``s`` (or ``n``): shorter inputs are zero-padded and longer ones truncated, and
    normalization uses the lengths of the transforms. A padded transform of 8192 elements is not local.
    """
    rng = np.random.default_rng(3)
    x, z = rng.random((3, 26, 26)).astype(np.float32), _random_complex(40)
    results = dict(
        a=np.zeros((3, 32, 32), np.complex64),
        b=np.zeros(16, np.complex128),
        c=np.zeros((3, 26, 48), np.complex64),
        d=np.zeros((3, 26, 8192), np.complex64),
    )
    _shaped_fft_program()(x=x, z=z, **results)
    _check_shaped_fft(x, z, results)


@pytest.mark.gpu
def test_fft_shape_gpu():
    """
    Zero-padded transforms on the GPU, in thread blocks (which pad the input in shared memory) and in global stages.
    """
    import cupy as cp  # CuPy is only installed on GPU machines

    sdfg = _shaped_fft_program(dace.StorageType.GPU_Global).to_sdfg()
    sdfg.apply_gpu_transformations()
    rng = np.random.default_rng(3)
    x, z = rng.random((3, 26, 26)).astype(np.float32), _random_complex(40)
    results = dict(
        a=cp.zeros((3, 32, 32), np.complex64),
        b=cp.zeros(16, np.complex128),
        c=cp.zeros((3, 26, 48), np.complex64),
        d=cp.zeros((3, 26, 8192), np.complex64),
    )
    sdfg(x=cp.asarray(x), z=cp.asarray(z), **results)
    _check_shaped_fft(x, z, {name: value.get() for name, value in results.items()})


def _real_step_count(sdfg: dace.SDFG) -> int:
    """
    Counts the steps that separate or combine the halves of a real transform of half length (see ``stockham_real``).
    """
    return sum(
        1
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.Tasklet) and node.label in ("stockham_real_separate", "stockham_real_combine")
    )


def _real_fft_program(G=dace.StorageType.Default):

    @dace.program
    def tester(
        x: dace.float64[16, 64] @ G,
        y: dace.complex128[16, 33] @ G,
        z: dace.float64[16, 64] @ G,
        u: dace.complex128[16, 33] @ G,
        v: dace.float64[16, 64] @ G,
        f: dace.float32[8, 31] @ G,
        g: dace.complex64[8, 16] @ G,
        h: dace.float32[8, 31] @ G,
        c: dace.float64[64, 16] @ G,
        d: dace.complex128[33, 16] @ G,
    ):
        y[:] = np.fft.rfft(x)
        z[:] = np.fft.irfft(y)
        u[:] = np.fft.rfft2(x, norm="ortho")
        v[:] = np.fft.irfftn(u, s=(16, 64), norm="ortho")
        g[:] = np.fft.rfft(f, axis=1)
        h[:] = np.fft.irfft(g, n=31)
        d[:] = np.fft.rfft(c, axis=0, norm="forward")

    return tester


def _real_fft_arguments(module=np):
    rng = np.random.default_rng(1)
    inputs = dict(x=rng.random((16, 64)), f=rng.random((8, 31)).astype(np.float32), c=rng.random((64, 16)))
    outputs = dict(
        y=((16, 33), np.complex128),
        z=((16, 64), np.float64),
        u=((16, 33), np.complex128),
        v=((16, 64), np.float64),
        g=((8, 16), np.complex64),
        h=((8, 31), np.float32),
        d=((33, 16), np.complex128),
    )
    arguments = {name: module.asarray(value) for name, value in inputs.items()}
    arguments.update({name: module.zeros(shape, dtype) for name, (shape, dtype) in outputs.items()})
    return inputs, arguments


def _check_real_fft(inputs: dict, results: dict) -> None:
    x, f, c = inputs["x"], inputs["f"], inputs["c"]
    expected = dict(
        y=np.fft.rfft(x),
        z=x,
        u=np.fft.rfft2(x, norm="ortho"),
        v=x,
        g=np.fft.rfft(f),
        h=f,
        d=np.fft.rfft(c, axis=0, norm="forward"),
    )
    for name, value in expected.items():
        tolerance = 1e-5 if results[name].dtype in (np.float32, np.complex64) else 1e-12
        assert np.allclose(results[name], value, rtol=tolerance, atol=tolerance * np.abs(value).max()), name


def test_real_fft():
    """
    R2C and C2R transforms of rows, columns and both axes, normalized in all directions, in single and double
    precision. Even lengths are complex transforms of half the length, and the odd length 31 is a complex transform of
    the whole length.
    """
    sdfg = _real_fft_program().to_sdfg()
    inputs, arguments = _real_fft_arguments()
    sdfg(**arguments)
    _check_real_fft(inputs, arguments)

    sdfg.expand_library_nodes()
    # One step each for rfft, irfft, rfft2, irfftn and the columns; the transforms of length 31 are complex
    assert _real_step_count(sdfg) == 5
    full_transforms = [
        name for _, name, desc in sdfg.arrays_recursive() if name == "__stockham_real_full" and desc.transient
    ]
    assert len(full_transforms) == 2


@pytest.mark.gpu
def test_real_fft_gpu():
    import cupy as cp  # CuPy is only installed on GPU machines

    sdfg = _real_fft_program(dace.StorageType.GPU_Global).to_sdfg()
    sdfg.apply_gpu_transformations()
    inputs, arguments = _real_fft_arguments(cp)
    sdfg(**arguments)
    _check_real_fft(inputs, {name: value.get() for name, value in arguments.items()})


if __name__ == "__main__":
    test_fft(False)
    test_fft(True)
    test_fft_r2c()
    test_real_fft()
    test_real_fft_gpu()
    test_fft_shape()
    test_fft_shape_gpu()
    test_ifft("backward")
    test_ifft("forward")
    test_ifft("ortho")
    test_cufft()
    test_cufft_twoplans()
    test_stockham_factorize()
    for n in (1, 2, 16, 21, 60, 143, 4096, 3 * 2**12, 8001):
        test_stockham_1d(n)
    test_stockham_complex64()
    test_stockham_r2c()
    test_stockham_nd()
    test_stockham_groups()
    test_stockham_symbolic_batch()
    test_stockham_inside_map()
    test_stockham_dispatch()
    test_stockham_gpu_stage_fusion()
    test_cufft_partial_axes()
    test_stockham_gpu_expansion()
    test_stockham_gpu_shared_memory_limit()
    test_stockham_gpu(np.complex64)
    test_stockham_gpu(np.complex128)
    test_block_local_fusion_structure()
    for case in ("other_transform", "too_large", "spectrum_reused", "different_axes", "symbolic", "in_place"):
        test_block_local_fusion_not_applicable(case)
    test_block_local_fusion_gpu(np.complex64, 2048)
    test_block_local_fusion_gpu(np.complex64, 4096)
    test_block_local_fusion_gpu(np.complex128, 1024)
    test_block_local_fusion_gpu(np.complex128, 60)
    test_block_local_fusion_fft2_gpu()
    test_block_local_fusion_r2c_gpu()
    test_block_local_fusion_conditional()
    test_block_local_fusion_conditional_gpu()
    for case in ("pointwise", "stencil", "general", "columns", "fft2", "conditional"):
        test_block_local_fusion_cpu(case)
    for case in ("too_large", "partial_columns"):
        test_block_local_fusion_cpu_not_applicable(case)

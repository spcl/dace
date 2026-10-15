# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
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
    return sum(
        1
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.label.startswith("stockham_stage")
    )


def test_stockham_factorize():
    assert stockham.factorize(1) == []
    assert stockham.factorize(4096) == [16, 16, 16]
    assert stockham.factorize(2**20) == [16] * 5
    assert stockham.factorize(60) == [4, 3, 5]
    assert stockham.factorize(143) == [11, 13]
    assert stockham.factorize(17) is None
    assert stockham.factorize(0) is None


@pytest.mark.parametrize("n", (1, 2, 16, 21, 60, 143, 4096, 3 * 2**12))
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
    @dace.program
    def tester(x: dace.complex128[12, 16, 10]):
        return (
            np.fft.fft(x, axis=0),
            np.fft.fft(x, axis=1),
            np.fft.ifft(x),
            np.fft.fft2(x),
            np.fft.fftn(x, axes=(0, 2)),
            np.fft.ifftn(x),
        )

    a = _random_complex((12, 16, 10))
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
    The native expansion uses Stockham for fixed supported lengths and falls back to the DFT otherwise.
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

    for program, stages in ((fixed, 3), (prime, 0), (symbolic, 0)):
        sdfg = program.to_sdfg()
        sdfg.expand_library_nodes()
        assert _stockham_stage_count(sdfg) == stages

    sdfg = prime.to_sdfg()
    _set_fft_implementation(sdfg, "Stockham")
    with pytest.raises(ValueError, match="Stockham"):
        sdfg.expand_library_nodes()


def test_stockham_single_state_fusion():
    """
    The expansion is a single state with one map per stage, which inlines and fuses with elementwise maps.
    """
    N = 4096

    @dace.program
    def tester(x: dace.complex128[N], w: dace.complex128[N]):
        return np.fft.ifft(np.fft.fft(x * 2.0) * w)

    sdfg = tester.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.simplify()
    assert len(list(sdfg.states())) == 1
    assert not any(isinstance(node, dace.nodes.NestedSDFG) for node, _ in sdfg.all_nodes_recursive())
    assert _stockham_stage_count(sdfg) == 6
    assert sdfg.apply_transformations_repeated(OTFMapFusion) == 2

    a, w = _random_complex(N), _random_complex(N)
    assert np.allclose(sdfg(x=a, w=w), np.fft.ifft(np.fft.fft(a * 2.0) * w))


def test_cufft_partial_axes():
    @dace.program
    def tester(x: dace.complex128[16, 32] @ dace.StorageType.GPU_Global):
        return np.fft.fft(x, axis=0)

    sdfg = tester.to_sdfg()
    _set_fft_implementation(sdfg, "cuFFT")
    with pytest.raises(NotImplementedError, match="all axes"):
        sdfg.expand_library_nodes()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_stockham_gpu(dtype):
    import cupy as cp  # CuPy is only installed on GPU machines

    dtype_dace = dace.dtypes.dtype_to_typeclass(dtype)

    @dace.program
    def tester(
        x: dtype_dace[2**16] @ dace.StorageType.GPU_Global,
        m: dtype_dace[24, 40] @ dace.StorageType.GPU_Global,
        y: dtype_dace[2**16] @ dace.StorageType.GPU_Global,
        z: dtype_dace[24, 40] @ dace.StorageType.GPU_Global,
    ):
        y[:] = np.fft.ifft(np.fft.fft(x) * 2)
        z[:] = np.fft.fft2(m)

    sdfg = tester.to_sdfg()
    sdfg.apply_gpu_transformations()
    _set_fft_implementation(sdfg, "Stockham")
    x, m = cp.asarray(_random_complex(2**16, dtype)), cp.asarray(_random_complex((24, 40), dtype))
    y, z = cp.empty_like(x), cp.empty_like(m)
    sdfg(x=x, m=m, y=y, z=z)
    tolerance = 1e-4 if dtype == np.complex64 else 1e-10
    assert np.allclose(y.get(), 2 * x.get(), rtol=tolerance, atol=tolerance)
    assert np.allclose(z.get(), np.fft.fft2(m.get()), rtol=tolerance, atol=100 * tolerance)


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
    B, N = 8, (4096 if case == "too_large" else 32)
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
@pytest.mark.parametrize("dtype,n", ((np.complex64, 2048), (np.complex128, 1024), (np.complex128, 60)))
def test_block_local_fusion_gpu(dtype, n):
    """
    A frequency-domain filter with a stencil along the transformed axis and orthonormal transforms.
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


if __name__ == "__main__":
    test_fft(False)
    test_fft(True)
    test_fft_r2c()
    test_ifft("backward")
    test_ifft("forward")
    test_ifft("ortho")
    test_cufft()
    test_cufft_twoplans()
    test_stockham_factorize()
    for n in (1, 2, 16, 21, 60, 143, 4096, 3 * 2**12):
        test_stockham_1d(n)
    test_stockham_complex64()
    test_stockham_r2c()
    test_stockham_nd()
    test_stockham_symbolic_batch()
    test_stockham_inside_map()
    test_stockham_dispatch()
    test_stockham_single_state_fusion()
    test_cufft_partial_axes()
    test_stockham_gpu(np.complex64)
    test_stockham_gpu(np.complex128)
    test_block_local_fusion_structure()
    for case in ("other_transform", "too_large", "spectrum_reused", "different_axes", "symbolic", "in_place"):
        test_block_local_fusion_not_applicable(case)
    test_block_local_fusion_gpu(np.complex64, 2048)
    test_block_local_fusion_gpu(np.complex128, 1024)
    test_block_local_fusion_gpu(np.complex128, 60)
    test_block_local_fusion_fft2_gpu()
    test_block_local_fusion_r2c_gpu()

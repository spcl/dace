# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeGPU`` — CUDA half2 (FP16x2) vectorization entry.

The lowering-selection + header contract is covered by
``passes/test_cuda_tile_lowering.py``; here we drive the full pipeline on a few
fp16 kernels and assert (a) the ``assume_even`` structural result (a single
strided ``0:N:W`` GPU_Device map, no mask), (b) the emitted CUDA calls the
``dace::tileops::tile_*`` contract inside the device TU, and -- when nvcc is
available -- (c) the generated code compiles and the fp16 arithmetic lowers to
native ``f16x2`` SIMD in the PTX.
"""
import os
import shutil

import pytest

import dace
from dace.dtypes import ScheduleType
from dace.transformation.interstate import LoopToMap
from dace.libraries.tileops import TileMaskGen, TileBinop
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import _TILE_NODE_TYPES
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

_HAS_NVCC = shutil.which("nvcc") is not None
N = dace.symbol("N")
TSTEPS = dace.symbol("TSTEPS")


@dace.program
def _add16(A: dace.float16[N, N], B: dace.float16[N, N], C: dace.float16[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def _jacobi2d16(A: dace.float16[N, N], B: dace.float16[N, N]):
    for i, j in dace.map[1:N - 1, 1:N - 1]:
        B[i, j] = dace.float16(0.2) * (A[i, j] + A[i, j - 1] + A[i, j + 1] + A[i + 1, j] + A[i - 1, j])


@dace.program
def _scale_const16(A: dace.float16[N], C: dace.float16[N]):
    # The canonical constant shape the frontend emits: a scalar cast then a binop.
    for i in dace.map[0:N]:
        tmp = dace.float16(0.5)
        C[i] = A[i] * tmp


@dace.program
def _vsum16(A: dace.float16[N], out: dace.float16[1]):
    s = dace.float16(0.0)
    for i in dace.map[0:N]:
        s += A[i]
    out[0] = s


@dace.program
def _heat3d16(A: dace.float16[N, N, N], B: dace.float16[N, N, N]):
    for t in range(1, TSTEPS):
        for i, j, k in dace.map[1:N - 1, 1:N - 1, 1:N - 1]:
            B[i, j, k] = dace.float16(0.125) * (A[i + 1, j, k] - dace.float16(2.0) * A[i, j, k] + A[i - 1, j, k]) \
                + dace.float16(0.125) * (A[i, j + 1, k] - dace.float16(2.0) * A[i, j, k] + A[i, j - 1, k]) \
                + dace.float16(0.125) * (A[i, j, k + 1] - dace.float16(2.0) * A[i, j, k] + A[i, j, k - 1]) + A[i, j, k]
        for i, j, k in dace.map[1:N - 1, 1:N - 1, 1:N - 1]:
            A[i, j, k] = dace.float16(0.125) * (B[i + 1, j, k] - dace.float16(2.0) * B[i, j, k] + B[i - 1, j, k]) \
                + dace.float16(0.125) * (B[i, j + 1, k] - dace.float16(2.0) * B[i, j, k] + B[i, j - 1, k]) \
                + dace.float16(0.125) * (B[i, j, k + 1] - dace.float16(2.0) * B[i, j, k] + B[i, j, k - 1]) + B[i, j, k]


def _prep(prog):
    """@dace.program -> simplify + LoopToMap + simplify (the caller-side recipe)."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    return sdfg


def _inner_maps(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


def test_assume_even_single_strided_gpu_map_no_mask():
    """``assume_even`` (the GPU default) emits ONE ``0:N:2`` GPU_Device map per
    original map -- no remainder split, no ``TileMaskGen`` (so no mismatched
    thread-block sizes on GPU)."""
    sdfg = _prep(_add16)
    VectorizeGPU().apply_pass(sdfg, {})
    maps = _inner_maps(sdfg)
    assert len(maps) == 1, f"assume_even must not split the map; got {len(maps)} maps"
    m = maps[0]
    assert m.map.schedule == ScheduleType.GPU_Device
    # innermost dim strided by the half2 width (2)
    assert str(m.map.range.ranges[-1][2]) == "2"
    assert not any(isinstance(n, TileMaskGen) for n, _ in sdfg.all_nodes_recursive()), \
        "assume_even must generate no iteration mask"


def test_deferred_tile_nodes_are_cuda_stamped():
    """By default the GPU pipeline does NOT expand the tile lib nodes: the SDFG
    returns with ``TileBinop`` / ``TileLoad`` present, each stamped with the
    ``CUDA`` ISA + ``cuda`` implementation, ready for a later
    ``expand_library_nodes()`` (or ``compile()``)."""
    sdfg = _prep(_add16)
    VectorizeGPU().apply_pass(sdfg, {})
    tiles = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, _TILE_NODE_TYPES)]
    assert tiles, "expected tile lib nodes to remain (deferred expansion)"
    for n in tiles:
        assert n.target_isa == "CUDA"
        assert n.implementation == "cuda"


def test_scalar_cast_constant_broadcasts():
    """The canonical constant-input shape ``tmp = float16(0.5); C = A * tmp`` -- a
    scalar cast feeding a binop -- vectorizes to a single ``TileBinop`` (the scalar
    is cast to the tile precision and broadcast into the tile), and compiles."""
    sdfg = _prep(_scale_const16)
    VectorizeGPU().apply_pass(sdfg, {})
    binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
    assert len(binops) == 1, f"expected one TileBinop for A*const; got {len(binops)}"
    sdfg.expand_library_nodes()
    # the constant stays at the input (fp16) precision -- no fp64 container leaked
    assert all(d.dtype != dace.float64 for d in sdfg.arrays.values())
    if _HAS_NVCC:
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        sdfg.compile()


def test_gpu_half2_emits_tile_ops_in_device_tu():
    """The device (``.cu``) TU calls the ``dace::tileops::tile_*`` contract on
    ``dace::float16`` tiles of width 2, and pulls in the cuda.h header (the
    ``'cuda'`` env key), so the half2 intrinsics are in scope on device."""
    sdfg = _prep(_add16)
    VectorizeGPU().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()  # default defers expansion; lower for codegen
    cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cu")
    assert "dace/tile_ops/cuda.h" in cu, "cuda.h not included in the device TU"
    assert "dace::tileops::tile_binop<dace::float16, 2" in cu
    assert "dace::tileops::tile_load<dace::float16, 2" in cu


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; compile check skipped")
@pytest.mark.parametrize("name,prog", [("add16", _add16), ("jacobi2d16", _jacobi2d16), ("heat3d16", _heat3d16)])
def test_gpu_half2_compiles(name, prog):
    """The half2 GPU vectorization of an elementwise / stencil fp16 kernel
    compiles end-to-end with nvcc (deferred expand -> compile)."""
    sdfg = _prep(prog)
    VectorizeGPU().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    sdfg.compile()  # raises CompilationError on failure


def test_gpu_reduction_uses_gpu_expansion():
    """A top-level fp16 reduction lifts to a ``Reduce`` node placed on the GPU
    (``GPU_Device`` schedule, ``GPU_Global`` input) so it selects a GPU expansion
    (``GPUAuto`` / cub) rather than the CPU horizontal fold -- and compiles."""
    from dace.libraries.standard.nodes.reduce import Reduce
    sdfg = _prep(_vsum16)
    VectorizeGPU().apply_pass(sdfg, {})
    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert reduces, "expected the sum to lift to a Reduce node"
    for r in reduces:
        assert r.schedule == ScheduleType.GPU_Device, f"Reduce not GPU-scheduled: {r.schedule}"
        assert r.implementation in ("GPUAuto", "CUDA (device)", "CUDA (block)"), \
            f"Reduce did not pick a GPU expansion: {r.implementation}"
    if _HAS_NVCC:
        sdfg.expand_library_nodes()
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        sdfg.compile()


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; PTX check skipped")
def test_gpu_half2_lowers_to_native_f16x2():
    """The fp16 add tile lowers to a native ``f16x2`` SIMD instruction (two lanes
    per op), not the scalar float fallback -- verified in the generated PTX."""
    import subprocess
    src = ("#include \"dace/dace.h\"\n#include \"dace/tile_ops/cuda.h\"\n"
           "__global__ void k(dace::float16* o, const dace::float16* a, const dace::float16* b) {\n"
           "  dace::tileops::tile_binop<dace::float16, 2, '+', false, false, false>(o, a, b, nullptr);\n}\n")
    inc = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")
    tmp = os.path.join(os.path.dirname(__file__), "_f16x2_probe.cu")
    ptx = tmp + ".ptx"
    try:
        with open(tmp, "w") as f:
            f.write(src)
        subprocess.run(["nvcc", "-I", inc, "-ptx", "-arch=sm_80", tmp, "-o", ptx], check=True, capture_output=True)
        assert "f16x2" in open(ptx).read(), "fp16 tile add did not lower to native f16x2 SIMD"
    finally:
        for f in (tmp, ptx):
            if os.path.exists(f):
                os.remove(f)


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; PTX check skipped")
def test_gpu_half2_reduce_lowers_to_native_f16x2():
    """The in-map fp16 horizontal reduce (``TileReduce`` -> ``dace::tileops::tile_reduce``)
    folds via native ``f16x2`` SIMD (two lanes per op) and returns a single ``__half`` --
    CUDA has no "reduce half2 -> half" intrinsic, so cuda.h composes one; verify the
    composed fold uses the packed half2 add/max in the PTX (not the scalar fallback)."""
    import subprocess
    src = ("#include \"dace/dace.h\"\n#include \"dace/tile_ops/cuda.h\"\n"
           "__global__ void k(dace::float16* o, const dace::float16* a) {\n"
           "  o[0] = dace::tileops::tile_reduce<dace::float16, 8, '+'>(a);\n"
           "  o[1] = dace::tileops::tile_reduce<dace::float16, 8, 'M'>(a);\n}\n")
    inc = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")
    tmp = os.path.join(os.path.dirname(__file__), "_f16x2_reduce_probe.cu")
    ptx = tmp + ".ptx"
    try:
        with open(tmp, "w") as f:
            f.write(src)
        subprocess.run(["nvcc", "-I", inc, "-ptx", "-arch=sm_80", tmp, "-o", ptx], check=True, capture_output=True)
        ptx_src = open(ptx).read()
        assert "add.f16x2" in ptx_src, "fp16 tile_reduce(+) did not fold via native f16x2 add"
        assert "max.f16x2" in ptx_src, "fp16 tile_reduce(max) did not fold via native f16x2 max"
    finally:
        for f in (tmp, ptx):
            if os.path.exists(f):
                os.remove(f)

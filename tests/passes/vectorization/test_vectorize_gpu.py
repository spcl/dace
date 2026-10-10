# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeGPU`` — CUDA half2 (FP16x2) vectorization entry.

The lowering-selection + header contract is covered by
``passes/test_cuda_tile_lowering.py``; here we drive the full pipeline on a few
fp16 kernels and assert (a) the ``assume_even`` structural result (a single
strided ``0:N:W`` GPU_Device map, no mask), (b) the emitted CUDA calls the
``dace::tileops::tile_*`` contract inside the device TU, and (c) the generated code compiles and the fp16 arithmetic lowers to the
backend's packed two-lane fp16 SIMD in the device assembly.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.codegen import common
from dace.dtypes import ScheduleType
from dace.libraries.tileops import TileBinop, TileMaskGen
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import TILE_NODE_TYPES
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

N = dace.symbol("N")
TSTEPS = dace.symbol("TSTEPS")


def _gpu_compiler_available() -> bool:
    """Whether the active GPU backend's compiler (nvcc or hipcc) is on the PATH."""
    return shutil.which("nvcc" if common.get_gpu_backend() == "cuda" else "hipcc") is not None


#: The backend's packed two-lane fp16 add / max instructions in device assembly (PTX for CUDA, AMDGPU for HIP).
PACKED_F16_ADD = {"cuda": "add.f16x2", "hip": "v_pk_add_f16"}
PACKED_F16_MAX = {"cuda": "max.f16x2", "hip": "v_pk_max_f16"}


def _device_assembly(tmp_path, body: str) -> str:
    """The device assembly the active GPU backend compiles ``body`` (a kernel using ``tile_ops/cuda.h``) to."""
    backend = common.get_gpu_backend()
    src = tmp_path / "probe.cu"
    out = tmp_path / "probe.s"
    src.write_text('#include "dace/dace.h"\n#include "dace/tile_ops/cuda.h"\n' + body)
    inc = str(Path(dace.__file__).parent / "runtime" / "include")
    if backend == "cuda":
        cmd = ["nvcc", "-I", inc, "-ptx", "-arch=sm_80", str(src), "-o", str(out)]
    else:
        arch = (dace.Config.get("compiler", "cuda", "hip_arch") or "gfx942").split(",")[0].strip()
        cmd = ["hipcc", "-I", inc, "-x", "hip", "-O3", "--cuda-device-only", "-S", f"--offload-arch={arch}"]
        cmd += [str(src), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out.read_text()


@dace.program
def _add16(A: dace.float16[N, N], B: dace.float16[N, N], C: dace.float16[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def _jacobi2d16(A: dace.float16[N, N], B: dace.float16[N, N]):
    for i, j in dace.map[1 : N - 1, 1 : N - 1]:
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
        for i, j, k in dace.map[1 : N - 1, 1 : N - 1, 1 : N - 1]:
            B[i, j, k] = (
                dace.float16(0.125) * (A[i + 1, j, k] - dace.float16(2.0) * A[i, j, k] + A[i - 1, j, k])
                + dace.float16(0.125) * (A[i, j + 1, k] - dace.float16(2.0) * A[i, j, k] + A[i, j - 1, k])
                + dace.float16(0.125) * (A[i, j, k + 1] - dace.float16(2.0) * A[i, j, k] + A[i, j, k - 1])
                + A[i, j, k]
            )
        for i, j, k in dace.map[1 : N - 1, 1 : N - 1, 1 : N - 1]:
            A[i, j, k] = (
                dace.float16(0.125) * (B[i + 1, j, k] - dace.float16(2.0) * B[i, j, k] + B[i - 1, j, k])
                + dace.float16(0.125) * (B[i, j + 1, k] - dace.float16(2.0) * B[i, j, k] + B[i, j - 1, k])
                + dace.float16(0.125) * (B[i, j, k + 1] - dace.float16(2.0) * B[i, j, k] + B[i, j, k - 1])
                + B[i, j, k]
            )


@dace.program
def where_bf16(x: dace.bfloat16[N], y: dace.bfloat16[N]):
    y[:] = np.where(x > 0, dace.bfloat16(0.0), x)


def _prep(prog):
    """@dace.program -> simplify + LoopToMap + simplify, THEN GPU-offload (the caller-side recipe).

    ``VectorizeGPU`` assumes an already-offloaded SDFG -- it does NOT offload / schedule itself --
    so the test mirrors the production canonicalize-GPU path (``finalize_for_target(sdfg, 'gpu')``
    -> :func:`offload_to_gpu`) that moves the SDFG onto the device (``GPU_Device`` maps, ``GPU_Global``
    storage, block sizes) BEFORE the vectorizer runs.
    """
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    return sdfg


def _inner_maps(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


def test_assume_even_single_strided_gpu_map_no_mask():
    """``assume_even=True`` emits ONE ``0:N:2`` GPU_Device map per original map --
    no remainder split, no ``TileMaskGen`` (so no mismatched thread-block sizes on
    GPU). ``assume_even`` is opt-in: the GPU K=1 default is ``branched_masked_tail``."""
    sdfg = _prep(_add16)
    VectorizeGPU(VectorizeConfig(widths=(2,), assume_even=True)).apply_pass(sdfg, {})
    maps = _inner_maps(sdfg)
    assert len(maps) == 1, f"assume_even must not split the map; got {len(maps)} maps"
    m = maps[0]
    assert m.map.schedule == ScheduleType.GPU_Device
    # innermost dim strided by the half2 width (2)
    assert str(m.map.range.ranges[-1][2]) == "2"
    assert not any(isinstance(n, TileMaskGen) for n, _ in sdfg.all_nodes_recursive()), (
        "assume_even must generate no iteration mask"
    )


def test_deferred_tile_nodes_are_cuda_stamped():
    """By default the GPU pipeline does NOT expand the tile lib nodes: the SDFG
    returns with ``TileBinop`` / ``TileGather`` present, each stamped with the
    ``CUDA`` ISA + ``cuda`` implementation, ready for a later
    ``expand_library_nodes()`` (or ``compile()``)."""
    sdfg = _prep(_add16)
    VectorizeGPU(VectorizeConfig(widths=(2,))).apply_pass(sdfg, {})
    tiles = [n for n, node_state in sdfg.all_nodes_recursive() if isinstance(n, TILE_NODE_TYPES)]
    assert tiles, "expected tile lib nodes to remain (deferred expansion)"
    for n in tiles:
        assert n.target_isa is ISA.CUDA
        assert n.implementation == "cuda"


def test_scalar_cast_constant_broadcasts():
    """The canonical constant-input shape ``tmp = float16(0.5); C = A * tmp`` -- a
    scalar cast feeding a binop -- vectorizes to one ``TileBinop`` per branch arm (the
    scalar is cast to the tile precision and broadcast into the tile), and compiles."""
    sdfg = _prep(_scale_const16)
    VectorizeGPU(VectorizeConfig(widths=(2,))).apply_pass(sdfg, {})
    binops = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileBinop)]
    # ONE per branch arm: the K=1 remainder default is ``branched_masked_tail`` (640c9e8d9), whose
    # else-arm is a MASKED tile body rather than a scalar lane loop, so the multiply is a tile op in
    # both arms. The cast is still broadcast, not re-materialized -- that would show up as a third.
    assert len(binops) == 2, f"expected one TileBinop per branch arm for A*const; got {len(binops)}"
    sdfg.expand_library_nodes()
    # the constant stays at the input (fp16) precision -- no fp64 container leaked
    assert all(d.dtype != dace.float64 for d in sdfg.arrays.values())
    if _gpu_compiler_available():
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        sdfg.compile()


def test_a_narrow_float_outside_numpy_s_hierarchy_still_vectorizes():
    """``bfloat16`` is an ml_dtypes scalar, not an ``np.floating`` subtype, so the tile nodes'
    promotion check read it as neither integer nor float and refused every conversion off it --
    including the ``-> bool`` a comparison's result IS. ``np.where(x > 0, ...)`` on a bfloat16
    array therefore died in validation, on the same path fp16 takes without complaint."""
    sdfg = _prep(where_bf16)
    VectorizeGPU(VectorizeConfig(widths=(2,), remainder_strategy=RemainderStrategy.BRANCHED_TAIL)).apply_pass(sdfg, {})
    assert [n for n, node_state in sdfg.all_nodes_recursive() if isinstance(n, TILE_NODE_TYPES)], (
        "the bfloat16 comparison did not reach the tile path at all"
    )
    sdfg.expand_library_nodes()
    cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")
    assert "dace::tileops::tile_load<dace::bfloat16, 2" in cu
    assert "dace::bfloat16" in cu and "dace::float16" not in cu, (
        "the bfloat16 tile must not be lowered through the float16 type"
    )


def test_gpu_half2_emits_tile_ops_in_device_tu():
    """The device (``.cu``) TU calls the ``dace::tileops::tile_*`` contract on
    ``dace::float16`` tiles of width 2, and pulls in the cuda.h header (the
    ``'cuda'`` env key), so the half2 intrinsics are in scope on device."""
    sdfg = _prep(_add16)
    VectorizeGPU(VectorizeConfig(widths=(2,))).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()  # default defers expansion; lower for codegen
    cu = "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")
    assert "dace/tile_ops/cuda.h" in cu, "cuda.h not included in the device TU"
    assert "dace::tileops::tile_binop<dace::float16, 2" in cu
    assert "dace::tileops::tile_load<dace::float16, 2" in cu


@pytest.mark.gpu
@pytest.mark.parametrize("name,prog", [("add16", _add16), ("jacobi2d16", _jacobi2d16), ("heat3d16", _heat3d16)])
def test_gpu_half2_compiles(name, prog):
    """The half2 GPU vectorization of an elementwise / stencil fp16 kernel
    compiles end-to-end with the backend's compiler (deferred expand -> compile)."""
    sdfg = _prep(prog)
    VectorizeGPU(VectorizeConfig(widths=(2,))).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    sdfg.compile()  # raises CompilationError on failure


def test_the_device_file_defines_each_index_helper_once():
    """An allocation inside a kernel is dispatched to the CPU codegen directly; it must still register its
    index helpers against the device file, or a later nest in the same kernel file defines them again
    (``function "s_idx" has already been defined``). Checked on the text, so no GPU compiler is needed."""
    import collections
    import re

    sdfg = _prep(_vsum16)
    VectorizeGPU(VectorizeConfig(widths=(2,), assume_even=True)).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        device = [obj.clean_code for obj in sdfg.generate_code() if "cuda" in obj.target.target_name]
    assert device, "no device file was generated"
    helpers = collections.Counter(re.findall(r"constexpr int64_t (\w+_idx)\(", device[0]))
    assert helpers, "no index helper in the device file, so this asserts nothing"
    assert all(count == 1 for count in helpers.values()), helpers


def test_gpu_reduction_uses_gpu_expansion():
    """A top-level fp16 scalar reduction (``out = sum(A)``) is kept as a
    GPU-scheduled map-exit WCR (``... -[CR:+]-> MapExit -[CR:+]-> out``), NOT lifted
    to a buffer + ``Reduce`` node. Codegen lowers that boundary WCR directly as a GPU
    block-reduce + one atomic per block -- the pure-WCR boundary form (see
    ``VectorizeMultiDim`` reduction handling; the ``Reduce``-node lift is the opt-in
    buffer form, dropped here). The map staying ``GPU_Device`` is what keeps the
    reduction on the device rather than a CPU horizontal fold; it compiles with the backend's compiler."""
    from dace.sdfg.nodes import MapExit

    sdfg = _prep(_vsum16)
    VectorizeGPU(VectorizeConfig(widths=(2,), assume_even=True)).apply_pass(sdfg, {})
    # The reduction stays a map-exit WCR (an in-edge to a MapExit carrying a CR).
    wcr_exits = [
        (st, n)
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.states()
        for n in st.nodes()
        if isinstance(n, MapExit) and any(e.data is not None and e.data.wcr is not None for e in st.in_edges(n))
    ]
    assert wcr_exits, "expected the scalar sum to remain a map-exit WCR reduction"
    for st, mx in wcr_exits:
        assert st.entry_node(mx).map.schedule == ScheduleType.GPU_Device, (
            "reduction map must be GPU-scheduled so codegen emits the GPU block-reduce, not a CPU fold"
        )
    if _gpu_compiler_available():
        sdfg.expand_library_nodes()
        shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
        sdfg.compile()


@pytest.mark.gpu
def test_gpu_half2_lowers_to_native_f16x2(tmp_path):
    """The fp16 add tile lowers to a native packed two-lane fp16 instruction, not the scalar float fallback."""
    asm = _device_assembly(
        tmp_path,
        "__global__ void k(dace::float16* o, const dace::float16* a, const dace::float16* b) {\n"
        "  dace::tileops::tile_binop<dace::float16, 2, '+', false, false, false>(o, a, b, nullptr);\n}\n",
    )
    assert PACKED_F16_ADD[common.get_gpu_backend()] in asm, "fp16 tile add did not lower to packed fp16 SIMD"


@pytest.mark.gpu
def test_gpu_half2_reduce_lowers_to_native_f16x2(tmp_path):
    """The in-map fp16 horizontal reduce (``TileReduce`` -> ``dace::tileops::tile_reduce``) folds via packed
    two-lane fp16 add / max and returns a single ``__half``: neither backend has a "reduce half2 -> half"
    intrinsic, so cuda.h composes one, and the fold must use the packed ops, not the scalar fallback."""
    asm = _device_assembly(
        tmp_path,
        "__global__ void k(dace::float16* o, const dace::float16* a) {\n"
        "  o[0] = dace::tileops::tile_reduce<dace::float16, 8, '+'>(a);\n"
        "  o[1] = dace::tileops::tile_reduce<dace::float16, 8, 'M'>(a);\n}\n",
    )
    backend = common.get_gpu_backend()
    assert PACKED_F16_ADD[backend] in asm, "fp16 tile_reduce(+) did not fold via packed fp16 add"
    assert PACKED_F16_MAX[backend] in asm, "fp16 tile_reduce(max) did not fold via packed fp16 max"


@pytest.mark.gpu
@pytest.mark.parametrize("width", [4, 8])
def test_gpu_vectorize_width_gt2_numeric(width):
    """Widths > 2 (4, 8) tile the GPU map by ``width`` and compute correctly, using the SAME
    half2 fast path as width 2 (two fp16 lanes per instruction, looping ``i += 2`` to emit
    ``width / 2`` consecutive half2 ops) -- still one strided ``0:N:width`` GPU_Device map
    (assume_even), numerically exact for a divisible extent."""
    import cupy
    import numpy as np

    sdfg = _prep(_add16)
    rng = np.random.default_rng(42)
    sdfg.name = f"add16_w{width}"
    VectorizeGPU(VectorizeConfig(widths=(width,), assume_even=True)).apply_pass(sdfg, {})
    maps = _inner_maps(sdfg)
    assert len(maps) == 1, f"assume_even keeps one map; got {len(maps)}"
    n = 8 * width  # a multiple of the width (assume_even)
    A = rng.random(n).astype(np.float16)
    B = rng.random(n).astype(np.float16)
    dA, dB, dC = cupy.asarray(A), cupy.asarray(B), cupy.zeros(n, cupy.float16)
    sdfg(A=dA, B=dB, C=dC, N=n)
    assert np.allclose(dC.get().astype(np.float32), (A + B).astype(np.float32), rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [4, 8])
def test_gpu_half2_wide_emits_width_over_2_f16x2(tmp_path, width):
    """A wide fp16 tile (width 4 / 8) uses the SAME half2 fast path as width 2: the per-tile binop lowers to
    exactly ``width / 2`` packed fp16 adds (the half2 branch loops ``i += 2``, ``#pragma unroll``), not a
    per-lane scalar fallback."""
    asm = _device_assembly(
        tmp_path,
        "__global__ void k(dace::float16* o, const dace::float16* a, const dace::float16* b) {\n"
        f"  dace::tileops::tile_binop<dace::float16, {width}, '+', false, false, false>(o, a, b, nullptr);\n}}\n",
    )
    n = asm.count(PACKED_F16_ADD[common.get_gpu_backend()])
    assert n == width // 2, f"width {width} fp16 add expected {width // 2} packed fp16 adds, got {n}"


@pytest.mark.gpu
def test_gpu_multidim_k2_runs():
    """A K=2 (2D) GPU tile -- ``widths=(2, 2)`` tiles BOTH innermost axes -- compiles and runs
    correctly on the device. K>=2 lowers each tile op to the portable ``pure`` per-lane body
    inside the ``GPU_Device`` kernel (the half2 SIMD intrinsic applies to the single innermost
    contiguous axis, K=1); still one strided map per axis (assume_even) and numerically exact."""
    import cupy
    import numpy as np

    rng = np.random.default_rng(42)
    sdfg = _prep(_add16)
    sdfg.name = "add16_k2"
    VectorizeGPU(VectorizeConfig(widths=(2, 2))).apply_pass(sdfg, {})
    n = 16  # a multiple of 2 on both tiled axes (assume_even)
    A = rng.random((n, n)).astype(np.float16)
    B = rng.random((n, n)).astype(np.float16)
    dA, dB, dC = cupy.asarray(A), cupy.asarray(B), cupy.zeros((n, n), cupy.float16)
    sdfg(A=dA, B=dB, C=dC, N=n)
    assert np.allclose(dC.get().astype(np.float32), (A + B).astype(np.float32), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import tempfile

    test_assume_even_single_strided_gpu_map_no_mask()
    test_deferred_tile_nodes_are_cuda_stamped()
    test_scalar_cast_constant_broadcasts()
    test_gpu_half2_emits_tile_ops_in_device_tu()
    test_gpu_half2_compiles("add16", _add16)
    test_gpu_half2_compiles("jacobi2d16", _jacobi2d16)
    test_gpu_half2_compiles("heat3d16", _heat3d16)
    test_gpu_reduction_uses_gpu_expansion()
    test_gpu_half2_lowers_to_native_f16x2(Path(tempfile.mkdtemp()))
    test_gpu_half2_reduce_lowers_to_native_f16x2(Path(tempfile.mkdtemp()))
    test_gpu_vectorize_width_gt2_numeric(4)
    test_gpu_vectorize_width_gt2_numeric(8)
    test_gpu_half2_wide_emits_width_over_2_f16x2(Path(tempfile.mkdtemp()), 4)
    test_gpu_half2_wide_emits_width_over_2_f16x2(Path(tempfile.mkdtemp()), 8)
    test_gpu_multidim_k2_runs()

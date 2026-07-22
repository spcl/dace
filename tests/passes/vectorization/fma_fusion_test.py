# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fused multiply-add: the ``a*b + c`` -> ``fma`` fusion pass, its ``TileFMA`` lowering, and the
native FMA it emits per backend (scalar ``std::fma`` / AVX ``_mm*_fmadd`` / CUDA ``__hfma2``).

The fusion is OPT-IN (``VectorizeConfig.fuse_multiply_add``): a fused single-rounding differs from
the separate ``*`` then ``+`` (and a NumPy reference) by up to one ULP, so results are compared
with a tolerance, never bit-exact.
"""
import os
import shutil

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.symbolic import fma, pystr_to_symbolic
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.fuse_multiply_add import FuseMultiplyAdd
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
from dace.libraries.tileops import TileFMA, TileBinop

_HAS_NVCC = shutil.which("nvcc") is not None
N = dace.symbol("N")
M = dace.symbol("M")


def _axpy(dt):

    @dace.program
    def axpy(A: dt[M, N], B: dt[M, N], C: dt[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            C[i, j] = A[i, j] * B[i, j] + A[i, j]

    return axpy


def _count(sdfg, node_type):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, node_type))


def test_fma_symbolic_function():
    """``fma`` is a SymPy function: it folds constants (``a*b + c``) and stays symbolic on symbols."""
    assert int(fma(2, 3, 4)) == 10
    e = pystr_to_symbolic("fma(a, b, c)")
    assert str(e) == "fma(a, b, c)"
    assert {str(s) for s in e.free_symbols} == {"a", "b", "c"}
    # 'fma' is a recognised built-in function name, not a per-lane symbol.
    from dace.symbolic import builtin_userfunctions
    assert "fma" in builtin_userfunctions()


def test_fuse_pass_rewrites_mul_add_to_fma():
    """``t = a*b ; d = t + c`` (single-use ``t``) -> one ``fma`` tasklet; no residual ``*`` / ``+``."""
    from dace.sdfg.nodes import Tasklet
    sdfg = _axpy(dace.float64).to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    assert FuseMultiplyAdd().apply_pass(sdfg, {}) == 1
    fmas = [t for t, _ in sdfg.all_nodes_recursive() if isinstance(t, Tasklet) and "fma(" in t.code.as_string]
    assert len(fmas) == 1
    resid = [
        t for t, _ in sdfg.all_nodes_recursive() if isinstance(t, Tasklet) and (
            "*" in t.code.as_string or (" + " in t.code.as_string and "fma" not in t.code.as_string))
    ]
    assert not resid, f"residual mul/add tasklets: {[t.code.as_string for t in resid]}"


def test_fuse_pass_refuses_reused_intermediate():
    """A product read TWICE (not a single-use intermediate) must NOT be fused into an FMA."""
    from dace.sdfg.nodes import Tasklet

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
        for i in dace.map[0:N]:
            t = A[i] * B[i]
            C[i] = t + A[i]
            D[i] = t + B[i]

    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    applied = FuseMultiplyAdd().apply_pass(sdfg, {})
    assert applied is None, "a reused product must not fuse"
    assert not [t for t, _ in sdfg.all_nodes_recursive() if isinstance(t, Tasklet) and "fma(" in t.code.as_string]


def test_tile_fma_registered():
    """``TileFMA`` exposes the pure + cutile + all six ISA expansions."""
    node = TileFMA("fma", widths=[8], kind_a="Tile", kind_b="Tile", kind_c="Tile")
    impls = set(node.implementations)
    assert {"pure", "cutile", "scalar", "avx512", "avx2", "neon", "sve", "cuda"} <= impls
    assert node.default_implementation == "pure"


@pytest.mark.parametrize("isa,dt", [("SCALAR", dace.float32), ("AVX512", dace.float32), ("AVX512", dace.float64)])
def test_cpu_fma_lowers_and_runs(isa, dt):
    """``fuse_multiply_add`` -> the CPU vectorizer emits ``TileFMA`` (no ``TileBinop``) and runs
    within FMA rounding of ``a*b + c``."""
    sdfg = _axpy(dt).to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=isa, fuse_multiply_add=True)).apply_pass(sdfg, {})
    assert _count(sdfg, TileFMA) >= 1
    assert _count(sdfg, TileBinop) == 0
    sdfg.expand_library_nodes()
    sdfg.name = f"fma_cpu_{isa}_{dt.to_string()}"
    A = np.random.rand(16, 64).astype(dt.type)
    B = np.random.rand(16, 64).astype(dt.type)
    C = np.zeros((16, 64), dt.type)
    sdfg(A=A, B=B, C=C, M=16, N=64)
    assert np.allclose(C.astype(np.float64), (A * B + A).astype(np.float64), rtol=1e-4, atol=1e-6)


def test_cpu_fma_off_by_default():
    """Without the flag the pipeline keeps the plain ``*`` / ``+`` (bit-exact, no FMA)."""
    sdfg = _axpy(dace.float32).to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa="AVX512")).apply_pass(sdfg, {})
    assert _count(sdfg, TileFMA) == 0


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; PTX check skipped")
def test_gpu_fma_lowers_to_native_hfma2():
    """A width-8 fp16 ``tile_fma`` lowers to native ``fma.rn.f16x2`` (four packed half2 FMAs),
    NOT separate ``mul.f16x2`` + ``add.f16x2`` -- verified in the PTX."""
    import subprocess
    src = ("#include \"dace/dace.h\"\n#include \"dace/tile_ops/cuda.h\"\n"
           "__global__ void k(dace::float16* o, const dace::float16* a, const dace::float16* b,\n"
           "                  const dace::float16* c) {\n"
           "  dace::tileops::tile_fma<dace::float16, 8, false, false, false, false>(o, a, b, c, nullptr);\n}\n")
    inc = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")
    tmp = os.path.join(os.path.dirname(__file__), "_fma_f16x2_probe.cu")
    ptx = tmp + ".ptx"
    try:
        with open(tmp, "w") as f:
            f.write(src)
        subprocess.run([
            "nvcc", "-I", inc, "--expt-relaxed-constexpr", "-diag-suppress", "128", "-ptx", "-arch=sm_80", tmp, "-o",
            ptx
        ],
                       check=True,
                       capture_output=True)
        text = open(ptx).read()
        assert "fma.rn.f16x2" in text, "fp16 tile_fma did not lower to native hfma2 (fma.rn.f16x2)"
    finally:
        for f in (tmp, ptx):
            if os.path.exists(f):
                os.remove(f)


@pytest.mark.gpu
def test_gpu_fma_runs_fp16():
    """The fp16 GPU vectorizer with ``fuse_multiply_add`` lowers ``A*B + A`` to ``TileFMA`` and
    runs correctly on the device (within fp16 rounding)."""
    import cupy
    sdfg = _axpy(dace.float16).to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    offload_to_gpu(sdfg)
    VectorizeGPU(VectorizeConfig(widths=(8, ), fuse_multiply_add=True)).apply_pass(sdfg, {})
    assert _count(sdfg, TileFMA) >= 1
    sdfg.name = "fma_gpu_run_f16"
    n = 64
    A = np.random.rand(16, n).astype(np.float16)
    B = np.random.rand(16, n).astype(np.float16)
    dA, dB, dC = cupy.asarray(A), cupy.asarray(B), cupy.zeros((16, n), cupy.float16)
    sdfg(A=dA, B=dB, C=dC, M=16, N=n)
    ref = A.astype(np.float32) * B.astype(np.float32) + A.astype(np.float32)
    assert np.allclose(dC.get().astype(np.float32), ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    test_fma_symbolic_function()
    test_fuse_pass_rewrites_mul_add_to_fma()
    test_fuse_pass_refuses_reused_intermediate()
    test_tile_fma_registered()
    test_cpu_fma_off_by_default()
    for _isa, _dt in [("SCALAR", dace.float32), ("AVX512", dace.float32), ("AVX512", dace.float64)]:
        test_cpu_fma_lowers_and_runs(_isa, _dt)
    if _HAS_NVCC:
        test_gpu_fma_lowers_to_native_hfma2()
    print("fma fusion tests ok")

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering tests for the NVIDIA CUDA (half2 / FP16x2) K=1 tile-op backend.

No GPU / nvcc is required: these assert the *lowering selection* and the
*emitted header contract* (the half2 intrinsics live inside
``dace/tile_ops/cuda.h``, pulled in by the ``TileOpsCUDA`` environment). The
GPU path shares ~90% of its pipeline with the CPU one, so the lowering is the
part that needs dedicated coverage.
"""
import os

import pytest

import dace
from dace.libraries.tileops import _dispatch
from dace.libraries.tileops import environments as tile_env
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import _VALID_ISAS

_CUDA_H = os.path.join(os.path.dirname(dace.__file__), "runtime", "include", "dace", "tile_ops", "cuda.h")


def test_cuda_isa_registered():
    assert "CUDA" in _VALID_ISAS
    assert _dispatch._ISA_TO_IMPL["CUDA"] == "cuda"


def test_cuda_environment_pulls_header():
    # The tile ops are emitted inside the GPU kernel, so the header must reach BOTH
    # the device (``'cuda'``, the ``.cu`` TU) and the host (``'frame'``) frames.
    assert tile_env.TileOpsCUDA.headers == {
        "frame": ["dace/tile_ops/cuda.h"],
        "cuda": ["dace/tile_ops/cuda.h"],
    }


def test_cuda_header_has_half2_intrinsics():
    src = open(_CUDA_H).read()
    # fp16x2 arithmetic + min/max + negate/abs (the core SIMD ops).
    for intr in ("__hadd2", "__hsub2", "__hmul2", "__h2div", "__hmin2", "__hmax2", "__hneg2", "__habs2"):
        assert intr in src, f"cuda.h missing fp16x2 intrinsic {intr}"
    # fp16x2 comparisons (set each lane to 1.0/0.0, matching the scalar element form).
    for intr in ("__heq2", "__hne2", "__hlt2", "__hle2", "__hgt2", "__hge2"):
        assert intr in src, f"cuda.h missing fp16x2 comparison intrinsic {intr}"
    # fp16x2 transcendentals / rounding (h2tanh does not exist -> tanh stays scalar).
    for intr in ("h2exp", "h2log", "h2sqrt", "h2sin", "h2cos", "h2floor", "h2ceil"):
        assert intr in src, f"cuda.h missing fp16x2 math intrinsic {intr}"
    # Same tileops contract as the other ISA headers, plus the composed half2
    # horizontal reduce (no native "reduce half2 -> half" intrinsic exists).
    assert "namespace tileops" in src
    for op in ("tile_binop", "tile_unop", "tile_load", "tile_store", "tile_gather", "tile_scatter", "tile_reduce"):
        assert op in src, f"cuda.h missing {op}"
    # tile_reduce folds pairwise via half2 then combines the 2 lanes into one half.
    for intr in ("__hadd", "__hmax", "__hmin", "__low2half", "__high2half"):
        assert intr in src, f"cuda.h missing scalar-half combine intrinsic {intr}"
    # Device-qualified + fp16/fp8 headers.
    assert "cuda_fp16.h" in src and "cuda_fp8.h" in src
    assert "__device__" in src or "DACE_DFI" in src
    # fp8 has no native arithmetic -> computed through float (documented contract).
    assert "float" in src


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_binop_selects_cuda_for_fp16_tile(op):
    from dace.libraries.tileops.nodes.tile_binop import TileBinop
    n = TileBinop("t", op=op, widths=(2, ))
    n.target_isa = "CUDA"
    assert _dispatch.select_tile_implementation(n) == "cuda"


def test_kge2_tile_falls_back_to_pure_under_cuda():
    # K>=2 always lowers to 'pure' regardless of ISA.
    from dace.libraries.tileops.nodes.tile_binop import TileBinop
    n = TileBinop("t", op="+", widths=(2, 2))
    n.target_isa = "CUDA"
    assert _dispatch.select_tile_implementation(n) == "pure"


def test_reduce_selects_cuda_for_full_k1_tile():
    # A full (axis=None), unmasked, K=1 TileReduce lowers to the cuda ``tile_reduce``
    # intrinsic (composed half2 fold for fp16, per-lane fold otherwise).
    from dace.libraries.tileops.nodes.tile_reduce import TileReduce
    n = TileReduce("t", op="+", widths=(2, ))
    n.target_isa = "CUDA"
    assert "cuda" in n.implementations
    assert _dispatch.select_tile_implementation(n) == "cuda"


def test_reduce_kge2_falls_back_to_pure_under_cuda():
    # K>=2 always lowers to 'pure' regardless of ISA (matches the binop rule); the
    # cuda ``tile_reduce`` intrinsic is a K==1 horizontal fold.
    from dace.libraries.tileops.nodes.tile_reduce import TileReduce
    n = TileReduce("t", op="+", widths=(2, 2))
    n.target_isa = "CUDA"
    assert _dispatch.select_tile_implementation(n) == "pure"


def _reduce_sdfg(dtype, W, op="+", axis=None, mask=False):
    """A minimal SDFG holding one ``TileReduce`` (Register tiles), for expansion tests."""
    from dace.libraries.tileops.nodes.tile_reduce import TileReduce
    shape = (W, ) if axis is None else (2, W)
    dst_shape = (1, ) if axis is None else (2, )
    sdfg = dace.SDFG(f"tr_{dtype.to_string()}_{W}")
    sdfg.add_array("src", shape, dtype, storage=dace.StorageType.Register, transient=True)
    sdfg.add_array("dst", dst_shape, dtype, storage=dace.StorageType.Register, transient=True)
    st = sdfg.add_state()
    widths = (W, ) if axis is None else (2, W)
    n = TileReduce("tr", widths=widths, op=op, axis=axis, has_mask=mask)
    s, d = st.add_access("src"), st.add_access("dst")
    st.add_edge(s, None, n, "_src", dace.Memlet.from_array("src", sdfg.arrays["src"]))
    st.add_edge(n, "_dst", d, None,
                dace.Memlet("dst[0]") if axis is None else dace.Memlet.from_array("dst", sdfg.arrays["dst"]))
    if mask:
        sdfg.add_array("msk", shape, dace.bool_, storage=dace.StorageType.Register, transient=True)
        m = st.add_access("msk")
        st.add_edge(m, None, n, "_mask", dace.Memlet.from_array("msk", sdfg.arrays["msk"]))
    return sdfg, st, n


def test_reduce_cuda_emits_intrinsic_for_full_k1():
    # A full (axis=None), unmasked, K=1 fp16 reduce lowers to the tile_reduce intrinsic
    # (the fp16 template picks the composed half2 fold at compile).
    from dace.libraries.tileops.nodes.tile_reduce import ExpandTileReduceCUDA
    sdfg, st, n = _reduce_sdfg(dace.float16, 4, "+")
    code = ExpandTileReduceCUDA.expansion(n, st, sdfg).code.as_string
    assert "dace::tileops::tile_reduce<dace::float16, 4, '+'>(_src)" in code
    # ``max`` maps to the 'M' op char (same legend as tile_binop).
    sdfg, st, n = _reduce_sdfg(dace.float32, 8, "max")
    assert "tile_reduce<float, 8, 'M'>" in ExpandTileReduceCUDA.expansion(n, st, sdfg).code.as_string


def test_reduce_cuda_falls_back_to_pure_for_masked_and_axis():
    # The tile_reduce intrinsic takes no mask and only does a full reduction, so a
    # masked reduce or a single-axis reduce delegates to the pure per-lane expansion.
    from dace.libraries.tileops.nodes.tile_reduce import ExpandTileReduceCUDA
    sdfg, st, n = _reduce_sdfg(dace.float16, 4, "+", mask=True)
    assert "tile_reduce" not in ExpandTileReduceCUDA.expansion(n, st, sdfg).code.as_string
    sdfg, st, n = _reduce_sdfg(dace.float32, 4, "+", axis=1)
    assert "tile_reduce" not in ExpandTileReduceCUDA.expansion(n, st, sdfg).code.as_string


def test_cuda_requires_even_width():
    # half2 packs 2 lanes, so the innermost tile width must be a multiple of 2.
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import _validate_knobs
    _validate_knobs((2, ), "CUDA", "scalar_postamble", "merge", "tile_k1")  # ok
    with pytest.raises(NotImplementedError, match="half2"):
        _validate_knobs((1, ), "CUDA", "scalar_postamble", "merge", "tile_k1")


if __name__ == "__main__":
    test_cuda_isa_registered()
    test_cuda_environment_pulls_header()
    test_cuda_header_has_half2_intrinsics()
    for o in ["+", "-", "*", "m", "M"]:
        test_binop_selects_cuda_for_fp16_tile(o)
    test_kge2_tile_falls_back_to_pure_under_cuda()
    test_reduce_selects_cuda_for_full_k1_tile()
    test_reduce_kge2_falls_back_to_pure_under_cuda()
    test_reduce_cuda_emits_intrinsic_for_full_k1()
    test_reduce_cuda_falls_back_to_pure_for_masked_and_axis()
    test_cuda_requires_even_width()
    print("CUDA tile-lowering tests passed")

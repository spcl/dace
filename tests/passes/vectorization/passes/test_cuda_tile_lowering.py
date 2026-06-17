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
    assert tile_env.TileOpsCUDA.headers == {"frame": ["dace/tile_ops/cuda.h"]}


def test_cuda_header_has_half2_intrinsics():
    src = open(_CUDA_H).read()
    # fp16 SIMD intrinsics (the whole point of the backend).
    for intr in ("__hadd2", "__hsub2", "__hmul2", "__h2div", "__hmin2", "__hmax2", "__hneg2"):
        assert intr in src, f"cuda.h missing fp16x2 intrinsic {intr}"
    # Same tileops contract as the other ISA headers.
    assert "namespace tileops" in src
    for op in ("tile_binop", "tile_unop", "tile_load", "tile_store", "tile_gather", "tile_scatter"):
        assert op in src, f"cuda.h missing {op}"
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


def test_node_without_cuda_backend_falls_back_to_pure():
    # TileReduce does not (yet) define a per-ISA backend -> 'pure' under CUDA.
    from dace.libraries.tileops.nodes.tile_reduce import TileReduce
    n = TileReduce("t", op="+", widths=(2, ))
    n.target_isa = "CUDA"
    assert "cuda" not in n.implementations
    assert _dispatch.select_tile_implementation(n) == "pure"


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
    test_node_without_cuda_backend_falls_back_to_pure()
    test_cuda_requires_even_width()
    print("CUDA tile-lowering tests passed")

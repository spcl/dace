# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The in-kernel thread-block lowerings compile for HIP, on a 64-wide (CDNA) and a 32-wide (RDNA) wavefront.

One node per family -- BLAS, FFT, LAPACK, sparse, the ``gpucub::BlockReduce`` reductions, the lane-strided
element-wise nodes and the block tile ops -- is lowered in a GPU kernel, its device unit generated with the HIP
backend and compiled by hipcc. No AMD GPU is needed; without hipcc the test is skipped.
"""

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

import dace
from dace.libraries.fft.nodes.fft import FFT
from dace.libraries.lapack.block import BLOCK_LAPACK
from dace.libraries.standard.block_reduce import release_kernel_block_size
from dace.libraries.standard.nodes.argminmax import ArgMax
from dace.libraries.tileops.dispatch import TileGroup
from tests.gpu_device_compile import device_compile
from tests.library.blas_level1_in_kernel_test import axpy_per_block
from tests.library.fft_block_gpu_test import batched_kernel as fft_kernel
from tests.library.lapack_block_gpu_test import kernel_getri
from tests.library.sparse_block_gpu_test import spmv_kernel
from tests.library.standard_nodes_in_kernel_test import broadcast_kernel, extreme_index_kernel
from tests.library.tileops.test_tile_nodes_in_gpu_kernels import gpu_sdfg, scanned

BLOCK = "CUDA (block strided)"

#: A wave64 CDNA and a wave32 RDNA target.
ARCHS = ("gfx90a", "gfx1100")


def has_hipcub() -> bool:
    """Whether hipcc finds ``hipcub/hipcub.hpp`` in ROCm's include folder or on ``CPATH``."""
    roots = [Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "include", Path("/usr/include")]
    roots += [Path(p) for p in os.environ.get("CPATH", "").split(os.pathsep) if p]
    return any((root / "hipcub" / "hipcub.hpp").is_file() for root in roots)


def blas() -> dace.SDFG:
    sdfg, state, node = axpy_per_block("hip_block_axpy")
    node.implementation = BLOCK
    return sdfg


def fft() -> dace.SDFG:
    sdfg, state, node = fft_kernel(FFT, dace.complex128)
    node.implementation = BLOCK
    return sdfg


def lapack() -> dace.SDFG:
    node, (sdfg, state) = kernel_getri(dace.float64)
    node.implementation = BLOCK_LAPACK
    release_kernel_block_size(node, state)
    return sdfg


def sparse() -> dace.SDFG:
    sdfg, state, node = spmv_kernel(beta=0.5)
    node.implementation = BLOCK
    return sdfg


def arg_reduction() -> dace.SDFG:
    return extreme_index_kernel(ArgMax, mask=True)[0]


def lane_strided() -> dace.SDFG:
    return broadcast_kernel()[0]


def tile() -> dace.SDFG:
    return gpu_sdfg(scanned.to_sdfg(), TileGroup.BLOCK, "hip_block_tile_scan")


FAMILIES: dict[str, Callable[[], dace.SDFG]] = {
    "blas": blas,
    "fft": fft,
    "lapack": lapack,
    "sparse": sparse,
    "arg_reduction": arg_reduction,
    "lane_strided": lane_strided,
    "tile": tile,
}


def hip_device_unit(sdfg: dace.SDFG) -> str:
    """The generated GPU translation unit of ``sdfg``, under the HIP backend."""
    with dace.config.set_temporary("compiler", "cuda", "backend", value="hip"):
        codes = sdfg.generate_code()
    (unit,) = [c.clean_code for c in codes if c.target.target_name in ("cuda", "experimental_cuda")]
    return unit


@pytest.mark.skipif(shutil.which("hipcc") is None, reason="needs hipcc")
@pytest.mark.parametrize("family", list(FAMILIES))
def test_an_in_kernel_block_lowering_compiles_for_hip(family: str):
    unit = hip_device_unit(FAMILIES[family]())
    assert "hip/hip_runtime.h" in unit
    if "gpucub.cuh" in unit and not has_hipcub():
        pytest.skip("needs hipCUB")
    for arch in ARCHS:
        with (
            tempfile.TemporaryDirectory() as out_dir,
            dace.config.set_temporary("compiler", "cuda", "backend", value="hip"),
            dace.config.set_temporary("compiler", "cuda", "hip_arch", value=arch),
        ):
            build = device_compile(unit, Path(out_dir))
            assert build.result.returncode == 0, f"{family} on {arch}:\n{build.result.stderr}"
            assert build.output.is_file()


if __name__ == "__main__":
    for name in FAMILIES:
        test_an_in_kernel_block_lowering_compiles_for_hip(name)

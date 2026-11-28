# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map index read as a ``double`` converts the whole tile at once, not one lane at a time.

``out[i] = a[i] + (i + off)`` fills an int64 index tile ``base + lane`` and converts it to ``double`` with a cast
``TileUnop``. The cast lowered ``pure`` and the fill was a ``DACE_UNROLL`` loop, which GCC unrolls before it
vectorizes, so even an AVX-512DQ host ran one scalar ``vcvtsi2sd`` per lane. The cast now calls a ``tile_cvt`` tile
function (``vcvtqq2pd`` under AVX-512DQ) and the fill is a ``DACE_LANE_SIMD`` loop. The offsets reach past 2^53, where
the conversion rounds, and the result must still be numpy's bit for bit.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace.libraries.tileops.dispatch import cpu_flags, detect_host_isa
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")


@dace.program
def add_shifted_index(a: dace.float64[N], out: dace.float64[N], off: dace.int64):
    for i in dace.map[0:N]:
        out[i] = a[i] + (i + off)


@pytest.mark.parametrize("isa", [ISA.SCALAR, detect_host_isa()], ids=lambda isa: isa.name)
def test_lane_index_tile_converts_with_simd(isa):
    sdfg = add_shifted_index.to_sdfg(simplify=True)
    sdfg.name = f"lane_index_cast_simd_{isa.name.lower()}"
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(8,), target_isa=isa, remainder_strategy=RemainderStrategy.MASKED_TAIL)
    ).apply_pass(sdfg, {})
    code = sdfg.generate_code()[0].clean_code
    assert "tile_cvt_f64x8_i64_" in code, "the int64 -> double cast of the index tile is no tile function call"
    assert "dace::float64(_a[" not in code, "a cast still lowers to the per-lane loop"
    assert "DACE_LANE_SIMD" in code, "the index tile is not filled by a SIMD lane loop"

    n = 1003  # a masked tail of three lanes
    a = np.random.default_rng(7).random(n)
    for off in (0, -500, (1 << 53) - 400, -(1 << 60) - 3, (1 << 62) + 255):
        out = np.zeros(n)
        sdfg(a=a, out=out, off=off, N=n)
        want = a + (np.arange(n, dtype=np.int64) + np.int64(off)).astype(np.float64)
        np.testing.assert_array_equal(out, want, err_msg=f"off={off}")

    # On an AVX-512DQ host the GCC build converts with the packed ``vcvtqq2pd`` and no scalar conversion is left
    library = os.path.join(sdfg.build_folder, "build", f"lib{sdfg.name}.so")
    objdump = shutil.which("objdump")
    if isa is not ISA.AVX512 or "avx512dq" not in cpu_flags() or objdump is None or not os.path.isfile(library):
        return
    if "clang" in subprocess.run([os.environ.get("CXX", "c++"), "--version"], capture_output=True, text=True).stdout:
        return
    disassembly = subprocess.run([objdump, "-d", library], capture_output=True, text=True, check=True).stdout
    assert "vcvtqq2pd" in disassembly
    assert "vcvtsi2sd" not in disassembly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end checks that K=1 tile ops lower through the per-ISA backend
headers (``dace/tile_ops/<backend>.h``), not just the scalar reference.

For each host-supported ISA the K=1 axpy must (a) compile + run + match numpy,
and (b) actually ``#include`` the chosen backend header — proving the ISA
expansion was selected, rather than silently falling back to ``pure``.
"""
import os

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )


def _host_flags():
    """Return the set of x86 CPU feature flags from ``/proc/cpuinfo`` (empty on
    non-Linux / non-x86)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


_FLAGS = _host_flags()
# (target_isa, required cpuinfo flag, expected backend header).
_ISA_CASES = [
    ("AVX512", "avx512f", "dace/tile_ops/avx512.h"),
    ("AVX2", "avx2", "dace/tile_ops/avx2.h"),
]


def _k1_axpy_sdfg(name):
    """K=1 axpy ``C[i] = A[i] + B[i]`` (unique name per case for parallel runs)."""
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N, ), dace.float64)
    sdfg.add_array("B", (N, ), dace.float64)
    sdfg.add_array("C", (N, ), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "axpy",
        {"i": "0:N"},
        {
            "_a": dace.Memlet("A[i]"),
            "_b": dace.Memlet("B[i]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


@pytest.mark.parametrize("isa,flag,header", _ISA_CASES)
def test_k1_axpy_isa_backend(isa, flag, header):
    """K=1 axpy under ``target_isa=<ISA>`` includes the ISA header and matches numpy."""
    if flag not in _FLAGS:
        pytest.skip(f"host lacks {flag}")
    sdfg = _k1_axpy_sdfg(f"e2e_k1_axpy_{isa.lower()}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa=isa).apply_pass(sdfg, {})
    sdfg.validate()
    csdfg = sdfg.compile()
    cpp = os.path.join(sdfg.build_folder, "src", "cpu", sdfg.name + ".cpp")
    with open(cpp) as f:
        code = f.read()
    assert header in code, f"{isa}: expected {header} include, backend not selected"
    assert "dace::tileops::tile_binop<" in code, f"{isa}: tile_binop call not emitted"

    rng = np.random.default_rng(seed=hash(isa) & 0xFFFF)
    for n in (8, 50, 64, 70):  # aligned, tail, multi-chunk, tail
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        csdfg(A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=0, atol=0)

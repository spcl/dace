# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end checks that K=1 tile ops lower through the per-ISA backend
headers (``dace/tile_ops/<backend>.h``), not just the scalar reference.

For each host-supported ISA the K=1 axpy must (a) compile + run + match numpy,
and (b) actually ``#include`` the chosen backend header — proving the ISA
expansion was selected, rather than silently falling back to ``pure``.
"""

import pytest
# [UNSKIPPED-FOR-ASSESSMENT 2026-06-14] pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import os

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim, )
from tests.corpus import tsvc


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

    rng = np.random.default_rng(seed=tsvc.stable_seed(isa))
    for n in (8, 50, 64, 70):  # aligned, tail, multi-chunk, tail
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        csdfg(A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=0, atol=0)


# (target_isa, required cpuinfo flag (None = always available), backend header).
# SCALAR is always exercised; the SIMD cases skip when the host lacks the flag.
_MASKGEN_CASES = [("SCALAR", None, "dace/tile_ops/scalar.h")] + _ISA_CASES


def _k1_masked_select_sdfg(name):
    """K=1 data-dependent select ``c[i] = a[i]*2 if a[i] > b[i] else b[i]+1``.

    A data-dependent ``if`` lowers to a :class:`TileITE`; over a (symbolically)
    non-divisible map it also gets a masked remainder, hence a
    :class:`TileMaskGen`. Built through the python frontend (the intended
    mask-gen / if test shape).
    """
    N = dace.symbol("N")

    @dace.program
    def masked_select(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
        for i in dace.map[0:N]:
            if a[i] > b[i]:
                c[i] = a[i] * 2.0
            else:
                c[i] = b[i] + 1.0

    sdfg = masked_select.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


@pytest.mark.parametrize("isa,flag,header", _MASKGEN_CASES)
def test_k1_mask_gen_isa_backend(isa, flag, header):
    """K=1 masked-remainder select lowers ``TileMaskGen`` through the per-ISA
    backend header: the iteration mask's ``base + l < ub`` compare becomes the
    ``dace::tileops::tile_mask_gen`` intrinsic, not the scalar pure loop. The
    same kernel's data-dependent ``if`` lowers to ``tile_ite``. Verified against
    numpy over aligned + non-divisible (masked-remainder) sizes."""
    if flag is not None and flag not in _FLAGS:
        pytest.skip(f"host lacks {flag}")
    sdfg = _k1_masked_select_sdfg(f"e2e_k1_maskgen_{isa.lower()}")
    VectorizeCPUMultiDim(widths=(8, ), target_isa=isa, remainder_strategy="masked_tail",
                         branch_mode="merge").apply_pass(sdfg, {})
    sdfg.validate()
    csdfg = sdfg.compile()
    cpp = os.path.join(sdfg.build_folder, "src", "cpu", sdfg.name + ".cpp")
    with open(cpp) as f:
        code = f.read()
    assert header in code, f"{isa}: expected {header} include, backend not selected"
    assert "dace::tileops::tile_mask_gen<" in code, f"{isa}: tile_mask_gen call not emitted"
    rng = np.random.default_rng(seed=tsvc.stable_seed(("maskgen", isa)))
    for n in (8, 22, 64, 70):  # aligned + non-divisible tails (masked remainder fires)
        a = rng.random(n)
        b = rng.random(n)
        c = np.zeros(n)
        csdfg(a=a, b=b, c=c, N=n)
        np.testing.assert_allclose(c, np.where(a > b, a * 2.0, b + 1.0), rtol=0, atol=0)

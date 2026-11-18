# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A served CPF unit carries only its target's scan rescan: the x86 ``simd inscan`` loop or the scalar loop."""

import numpy as np

import dace
from dace.codegen.cpf import X86_GUARD, render
from dace.libraries.tileops.dispatch import ISA
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def running_total(x: dace.float64[N], y: dace.float64[N]):
    y[:] = np.cumsum(x)


def scan_sdfg(name: str) -> dace.SDFG:
    sdfg = running_total.to_sdfg(simplify=True)
    sdfg.name = name
    return sdfg


def test_an_x86_unit_has_only_the_inscan_rescan_and_computes_the_prefix_sum():
    rendering = render(scan_sdfg("target_path_x86"), language="c", isa=ISA.AVX2)
    assert "omp scan" in rendering.code
    assert X86_GUARD not in rendering.code and "#else" not in rendering.code
    for n in (1, 5, 100_003):
        x = np.random.default_rng(n).random(n)
        y = np.zeros(n)
        call_standalone(
            build_standalone(rendering.code, f"target_path_x86_{n}", language="c"),
            rendering.sdfg,
            {"x": x, "y": y, "N": n},
        )
        np.testing.assert_allclose(y, np.cumsum(x), rtol=1e-12)


def test_an_arm_unit_has_only_the_scalar_rescan():
    rendering = render(scan_sdfg("target_path_arm"), language="c", isa=ISA.ARM_NEON, check_compiles=False)
    assert "omp scan" not in rendering.code
    assert X86_GUARD not in rendering.code

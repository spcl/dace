# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the experimental readable CPU generator's ``compiler.cpu.codegen_params.heap_ptr_restrict`` knob. It
defaults to the faster form and only the experimental generator emits the qualified declaration, so legacy output is
unaffected by construction."""

import numpy

import dace
from dace.config import set_temporary

N = dace.symbol("N")


@dace.program
def scaled_twice(A: dace.float64[N], B: dace.float64[N]):
    tmp = numpy.empty(N, dace.float64)  # symbolic-size transient -> heap-allocated pointer
    for i in dace.map[0:N]:
        tmp[i] = A[i] * 2.0
    for i in dace.map[0:N]:
        B[i] = tmp[i] + 1.0


def generate(heap_ptr_restrict="restrict"):
    sdfg = scaled_twice.to_sdfg(simplify=True)
    with (
        set_temporary("compiler", "cpu", "implementation", value="experimental_readable"),
        set_temporary("compiler", "cpu", "codegen_params", "heap_ptr_restrict", value=heap_ptr_restrict),
    ):
        return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")


def test_heap_ptr_restrict_default_emits_restrict():
    code = generate(heap_ptr_restrict="restrict")
    # The fused heap declaration of the transient carries __restrict__.
    assert "double* __restrict__ tmp" in code


def test_heap_ptr_restrict_none_drops_restrict():
    code = generate(heap_ptr_restrict="none")
    assert "__restrict__ tmp" not in code
    assert "double* tmp" in code


def test_index_helpers_are_constexpr():
    code = generate()
    assert "static DACE_HDFI constexpr" in code


def test_legacy_ignores_the_flag():
    """Legacy emits no fused restrict declaration, so its output is identical across both values of this
    experimental-only key."""

    def legacy(hpr):
        sdfg = scaled_twice.to_sdfg(simplify=True)
        with (
            set_temporary("compiler", "cpu", "implementation", value="legacy"),
            set_temporary("compiler", "cpu", "codegen_params", "heap_ptr_restrict", value=hpr),
        ):
            return "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")

    assert legacy("restrict") == legacy("none")


def test_both_configs_compile_and_run():
    for hpr in ("restrict", "none"):
        with (
            set_temporary("compiler", "cpu", "implementation", value="experimental_readable"),
            set_temporary("compiler", "cpu", "codegen_params", "heap_ptr_restrict", value=hpr),
        ):
            A = numpy.random.default_rng(0).random(40)
            B = numpy.zeros(40)
            scaled_twice(A=A, B=B, N=40)
            assert numpy.allclose(B, A * 2.0 + 1.0)


if __name__ == "__main__":
    test_heap_ptr_restrict_default_emits_restrict()
    test_heap_ptr_restrict_none_drops_restrict()
    test_index_helpers_are_constexpr()
    test_legacy_ignores_the_flag()
    test_both_configs_compile_and_run()

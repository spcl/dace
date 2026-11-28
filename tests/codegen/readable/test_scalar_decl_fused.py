# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.scalar_decl = late_fused``.

``eager`` (the default) emits the legacy pair -- ``T x;`` at the scope top, ``x = expr;`` where the write
happens. ``late_fused`` emits ``T x = expr;``: the declaration IS the first write. The knob is
experimental_readable-only, so legacy output is byte-identical for every value.

Folding a declaration into its write means DEFERRING it to that write, and the soundness proof for that is a set
of structural scans (data nodes in other states, interstate edges, region conditions) -- each of which can only
see a use that has a graph representation. ``eager`` needs no such proof, so it is the default.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary
from tests.codegen.readable.conftest import (
    EXPERIMENTAL,
    LEGACY,
    assert_outputs_equivalent,
    generated_code,
    run_isolated,
    use_implementation,
)

N = dace.symbol("N")


@dace.program
def reassigned(A: dace.float64[N], B: dace.float64[N]):
    """``t`` is written twice, so it is exactly the mutable scalar this knob is about. It lives in a map body: the scope a frontend
    actually puts scalars in."""
    for i in dace.map[0:N]:
        t = A[i] * 2.0
        u = t + 1.0
        t = u * u
        B[i] = t + u


def readable_code(sdfg, style):
    with (
        use_implementation(EXPERIMENTAL),
        set_temporary("compiler", "cpu", "codegen_params", "scalar_decl", value=style),
    ):
        return generated_code(sdfg)


def test_fused_folds_declaration_into_first_write(require_experimental):
    """``double t; t = expr;`` becomes ``double t = expr;``; the SECOND write stays a plain
    assignment (a reassigned scalar cannot be const-bound, only defined by its first write)."""
    sdfg = reassigned.to_sdfg(simplify=True)
    split, fused = readable_code(sdfg, "eager"), readable_code(sdfg, "late_fused")

    assert "double t;" in split
    assert "t = (A_index * 2.0);" in split

    assert "double t;" not in fused, fused
    assert "double t = (A_index * 2.0);" in fused, fused
    # The later write is NOT re-declared -- exactly one declaration of `t` survives.
    assert fused.count("double t = ") == 1, fused
    assert "t = t_0;" in fused, fused


def test_default_is_eager(require_experimental):
    """The default readable output equals the explicit ``eager`` output (byte-identical), and is NOT the
    ``late_fused`` one -- so the assertion cannot pass by the two spellings having converged."""
    with use_implementation(EXPERIMENTAL):
        default = generated_code(reassigned.to_sdfg(simplify=True))
    eager = readable_code(reassigned.to_sdfg(simplify=True), "eager")
    assert default == eager
    assert default != readable_code(reassigned.to_sdfg(simplify=True), "late_fused")


def test_legacy_byte_identical_across_style():
    """Legacy ignores the experimental-only key: identical output for eager and late_fused."""

    def legacy(style):
        with (
            use_implementation(LEGACY),
            set_temporary("compiler", "cpu", "codegen_params", "scalar_decl", value=style),
        ):
            return generated_code(reassigned.to_sdfg(simplify=True))

    assert legacy("eager") == legacy("late_fused")


def braced_first_use_sdfg(name):
    """A scalar ``m`` whose FIRST-use tasklet is multi-statement, so the readable generator keeps that
    tasklet's ``{ ... }`` brace and the declaration cannot be folded into it."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    sdfg.add_scalar("m", dace.float64, transient=True, storage=dtypes.StorageType.Register)
    state = sdfg.add_state("main")
    # Two statements -> the readable generator keeps this tasklet's brace.
    t1 = state.add_tasklet("w", {"inp"}, {"o"}, "tmp = inp * 3.0\no = tmp + 1.0")
    state.add_edge(state.add_read("A"), None, t1, "inp", dace.Memlet("A[0]"))
    m = state.add_access("m")
    state.add_edge(t1, "o", m, None, dace.Memlet("m[0]"))
    t2 = state.add_tasklet("rw", {"inp"}, {"o"}, "o = inp + 1.0")
    state.add_edge(m, None, t2, "inp", dace.Memlet("m[0]"))
    m2 = state.add_access("m")
    state.add_edge(t2, "o", m2, None, dace.Memlet("m[0]"))
    t3 = state.add_tasklet("r", {"inp"}, {"o"}, "o = inp")
    state.add_edge(m2, None, t3, "inp", dace.Memlet("m[0]"))
    state.add_edge(t3, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def test_braced_tasklet_falls_back_to_a_plain_declaration(require_experimental):
    """A first-use tasklet that needs its own ``{ ... }`` block cannot carry the declaration -- it would
    be scoped to that block. The candidate must fall back to a plain ``T x;`` ahead of the block, NOT
    silently lose its declaration."""
    fused = readable_code(braced_first_use_sdfg("braced"), "late_fused")
    # However it is spelled, `m` must be declared exactly once and before the brace that writes it.
    assert fused.count("double m") == 1, fused


def test_late_fused_declares_an_unfoldable_scalar_at_first_use(require_experimental):
    """For a candidate it cannot fold (a braced first-use tasklet), ``late_fused`` still emits the ``T x;``
    right before that tasklet rather than at the scope top, where ``eager`` puts it."""
    sdfg = braced_first_use_sdfg("interaction")
    split = readable_code(sdfg, "eager")
    fused = readable_code(sdfg, "late_fused")

    def decl_offset(code):
        """Lines between the scalar's declaration and the brace of the tasklet that first writes it."""
        lines = [l.split("////")[0].rstrip() for l in code.split("\n")]
        decl = next(i for i, l in enumerate(lines) if "double m" in l)
        brace = next(i for i, l in enumerate(lines) if i > decl and l.strip().startswith("{  // w"))
        return brace - decl

    # eager: hoisted to the scope top, so unrelated lines sit between it and the first write.
    assert decl_offset(split) > 1, split
    # late_fused: it could not fold into the braced tasklet, so it lands immediately before it.
    assert decl_offset(fused) == 1, fused
    # Either way the declaration exists exactly once and precedes the write -- never lost to the brace.
    assert split.count("double m") == 1 and fused.count("double m") == 1


@pytest.mark.parametrize("style", ["eager", "late_fused"])
def test_compiles_and_runs_bit_identical(require_experimental, style):
    """Both styles compile and reproduce the legacy result bit-for-bit."""
    A = np.random.default_rng(0).random(16)

    def run(impl, scalar_decl="eager"):

        def build_and_run():
            with (
                use_implementation(impl),
                set_temporary("compiler", "cpu", "codegen_params", "scalar_decl", value=scalar_decl),
            ):
                csdfg = reassigned.to_sdfg(simplify=True).compile()
                B = np.zeros(16)
                csdfg(A=A.copy(), B=B, N=16)
                return {"B": B}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    experimental = run(EXPERIMENTAL, style)
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"scalar_decl={style}")


if __name__ == "__main__":
    test_fused_folds_declaration_into_first_write(None)
    test_default_is_eager(None)
    test_legacy_byte_identical_across_style()
    test_braced_tasklet_falls_back_to_a_plain_declaration(None)
    test_late_fused_declares_an_unfoldable_scalar_at_first_use(None)
    for s in ("eager", "late_fused"):
        test_compiles_and_runs_bit_identical(None, s)
    print("OK")

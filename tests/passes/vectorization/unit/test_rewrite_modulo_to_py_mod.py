# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass-level tests for ``RewriteModuloToPyMod``.

Python / NumPy modulo follows the *divisor's* sign; C's ``%`` follows the
*dividend's*, and ``cppunparse`` (for tasklet bodies / codeblocks) and
``symstr(cpp_mode)`` (for symbolic subsets) both lower a bare ``%`` / sympy
``Mod`` to C's ``%`` -- which miscompiles negative operands (and is ill-formed
for floats). The pass rewrites every ``%`` to ``py_mod`` (which resolves to
``dace::math::py_mod``) so the canonicalized reference, the vectorized body, and
the base codegen all agree, without touching core ``cppunparse``.

``%`` can appear in five places; these tests cover each:

* a **tasklet body** (``c = a % b``) -- also a negative-operand runtime check;
* a **loop-range codeblock** (``range(0, x % 7)`` -> the ``LoopRegion`` bound);
* a **branch condition** (``if a % 2 == 0`` -> a ``ConditionalBlock`` arm);
* a **memlet subset** (``A[i % 7]``);
* an **interstate edge** (a condition and an assignment RHS).
"""
import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod


def _apply(sdfg: dace.SDFG) -> None:
    RewriteModuloToPyMod().apply_pass(sdfg, {})


def test_rewrite_tasklet_body_and_runtime():
    """``c = a % b`` -> ``c = py_mod(a, b)``; negative dividend matches numpy."""
    sdfg = dace.SDFG("mod_body")
    for n in "ABC":
        sdfg.add_array(n, [4], dace.int64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", dict(i="0:4"))
    t = state.add_tasklet("k", {"a", "b"}, {"c"}, "c = a % b", language=dace.Language.Python)
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    state.add_memlet_path(a, me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(b, me, t, dst_conn="b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(t, mx, c, src_conn="c", memlet=dace.Memlet("C[i]"))

    _apply(sdfg)

    assert t.code.as_string.strip() == "c = py_mod(a, b)"
    assert "%" not in t.code.as_string

    A = np.array([-7, -7, 7, 7], dtype=np.int64)
    B = np.array([3, 3, 3, 3], dtype=np.int64)
    C = np.zeros(4, dtype=np.int64)
    sdfg(A=A, B=B, C=C)
    # py_mod (divisor sign), e.g. -7 % 3 == 2 -- C's % would give -1.
    np.testing.assert_array_equal(C, A % B)


def test_rewrite_loop_range_codeblock():
    """A ``range(0, x % 7)`` loop bound (``i < (x % 7)``) is rewritten."""
    sdfg = dace.SDFG("mod_loop")
    sdfg.add_symbol("x", dace.int64)
    loop = LoopRegion("L",
                      condition_expr="i < (x % 7)",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = (i + 1)")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)

    _apply(sdfg)

    cond = loop.loop_condition.as_string
    assert "py_mod(x, 7)" in cond
    assert "%" not in cond


def test_rewrite_branch_condition():
    """An ``if a % 2 == 0`` branch condition is rewritten."""
    sdfg = dace.SDFG("mod_if")
    sdfg.add_symbol("a", dace.int64)
    cblock = ConditionalBlock("C")
    sdfg.add_node(cblock, is_start_block=True)
    branch = ControlFlowRegion("then", sdfg=sdfg)
    cblock.add_branch(CodeBlock("a % 2 == 0"), branch)
    branch.add_state("s", is_start_block=True)

    _apply(sdfg)

    cond = cblock.branches[0][0].as_string
    assert "py_mod(a, 2)" in cond
    assert "%" not in cond


def test_rewrite_memlet_subset():
    """A memlet subset ``A[i % 7]`` is rewritten to ``A[py_mod(i, 7)]`` and codegens."""
    sdfg = dace.SDFG("mod_memlet")
    sdfg.add_array("A", [7], dace.int64)
    sdfg.add_array("B", [7], dace.int64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", dict(i="0:7"))
    t = state.add_tasklet("k", {"a"}, {"b"}, "b = a", language=dace.Language.Python)
    a, b = state.add_access("A"), state.add_access("B")
    state.add_memlet_path(a, me, t, dst_conn="a", memlet=dace.Memlet("A[i % 7]"))
    state.add_memlet_path(t, mx, b, src_conn="b", memlet=dace.Memlet("B[i]"))

    _apply(sdfg)

    gather = next(e.data.subset for e in state.edges()
                  if e.data is not None and e.data.data == "A" and isinstance(e.dst, dace.nodes.Tasklet))
    assert "py_mod(i, 7)" in str(gather)
    assert "%" not in str(gather)
    # The index lowers through ``symstr(cpp_mode)`` to ``dace::math::py_mod``.
    assert "py_mod(i, 7)" in sdfg.generate_code()[0].clean_code


def test_rewrite_interstate_edge():
    """An interstate-edge condition and assignment RHS are both rewritten."""
    sdfg = dace.SDFG("mod_ise")
    sdfg.add_symbol("i", dace.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="i % 3 == 0", assignments={"j": "i % 5"}))

    _apply(sdfg)

    edge = sdfg.edges()[0].data
    assert "py_mod(i, 3)" in edge.condition.as_string
    assert "py_mod(i, 5)" in edge.assignments["j"]
    assert "%" not in edge.condition.as_string
    assert "%" not in edge.assignments["j"]


def test_idempotent_on_existing_py_mod():
    """A body that already calls ``py_mod`` is left unchanged (no double-wrap)."""
    sdfg = dace.SDFG("mod_idem")
    for n in "ABC":
        sdfg.add_array(n, [4], dace.int64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", dict(i="0:4"))
    t = state.add_tasklet("k", {"a", "b"}, {"c"}, "c = py_mod(a, b)", language=dace.Language.Python)
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    state.add_memlet_path(a, me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(b, me, t, dst_conn="b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(t, mx, c, src_conn="c", memlet=dace.Memlet("C[i]"))

    _apply(sdfg)

    assert t.code.as_string.strip() == "c = py_mod(a, b)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

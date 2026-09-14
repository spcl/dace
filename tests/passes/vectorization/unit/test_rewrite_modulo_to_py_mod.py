# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``RewriteModuloToPyMod`` spells every floored modulo as ``py_mod`` and leaves C's ``%`` alone."""
import numpy as np
import pytest

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RewriteModuloToPyMod


def _apply(sdfg: dace.SDFG) -> None:
    RewriteModuloToPyMod().apply_pass(sdfg, {})


def elementwise_sdfg(name: str, body: str) -> tuple[dace.SDFG, dace.nodes.Tasklet]:
    sdfg = dace.SDFG(name)
    for n in "ABC":
        sdfg.add_array(n, [4], dace.int64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", dict(i="0:4"))
    t = state.add_tasklet("k", {"a", "b"}, {"c"}, body, language=dace.Language.Python)
    a, b, c = state.add_access("A"), state.add_access("B"), state.add_access("C")
    state.add_memlet_path(a, me, t, dst_conn="a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(b, me, t, dst_conn="b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(t, mx, c, src_conn="c", memlet=dace.Memlet("C[i]"))
    return sdfg, t


def run_elementwise(sdfg: dace.SDFG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([-7, -7, 7, 7], dtype=np.int64)
    B = np.array([3, -3, 3, -3], dtype=np.int64)
    C = np.zeros(4, dtype=np.int64)
    sdfg(A=A, B=B, C=C)
    return A, B, C


def test_a_floored_modulo_in_a_tasklet_body_becomes_py_mod():
    sdfg, t = elementwise_sdfg("pymod_body", "c = PyMod(a, b)")

    _apply(sdfg)

    assert t.code.as_string.strip() == "c = py_mod(a, b)"
    A, B, C = run_elementwise(sdfg)
    np.testing.assert_array_equal(C, np.mod(A, B))


def test_c_modulo_in_a_tasklet_body_is_left_alone():
    sdfg, t = elementwise_sdfg("cmod_body", "c = a % b")
    before = t.code.as_string

    _apply(sdfg)

    assert t.code.as_string == before and "py_mod" not in before
    A, B, C = run_elementwise(sdfg)
    np.testing.assert_array_equal(C, np.fmod(A, B))


def test_rewrite_loop_range_codeblock():
    sdfg = dace.SDFG("mod_loop")
    sdfg.add_symbol("x", dace.int64)
    loop = LoopRegion("L",
                      condition_expr="i < PyMod(x, 7)",
                      loop_var="i",
                      initialize_expr="i = 0",
                      update_expr="i = (i + 1)")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)

    _apply(sdfg)

    cond = loop.loop_condition.as_string
    assert "py_mod(x, 7)" in cond
    assert "PyMod" not in cond


def test_rewrite_branch_condition():
    sdfg = dace.SDFG("mod_if")
    sdfg.add_symbol("a", dace.int64)
    cblock = ConditionalBlock("C")
    sdfg.add_node(cblock, is_start_block=True)
    branch = ControlFlowRegion("then", sdfg=sdfg)
    cblock.add_branch(CodeBlock("PyMod(a, 2) == 0"), branch)
    branch.add_state("s", is_start_block=True)

    _apply(sdfg)

    cond = cblock.branches[0][0].as_string
    assert "py_mod(a, 2)" in cond
    assert "PyMod" not in cond


def memlet_sdfg(name: str, subset: str) -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [7], dace.int64)
    sdfg.add_array("B", [7], dace.int64)
    state = sdfg.add_state(is_start_block=True)
    me, mx = state.add_map("m", dict(i="0:7"))
    t = state.add_tasklet("k", {"a"}, {"b"}, "b = a", language=dace.Language.Python)
    a, b = state.add_access("A"), state.add_access("B")
    state.add_memlet_path(a, me, t, dst_conn="a", memlet=dace.Memlet(subset))
    state.add_memlet_path(t, mx, b, src_conn="b", memlet=dace.Memlet("B[i]"))
    return sdfg, state


def gather_subset(state: dace.SDFGState) -> dace.subsets.Subset:
    return next(e.data.subset for e in state.edges()
                if e.data is not None and e.data.data == "A" and isinstance(e.dst, dace.nodes.Tasklet))


def test_rewrite_memlet_subset():
    sdfg, state = memlet_sdfg("mod_memlet", "A[PyMod(i, 7)]")

    _apply(sdfg)

    assert "py_mod(i, 7)" in str(gather_subset(state))
    assert "py_mod(i, 7)" in sdfg.generate_code()[0].clean_code


def test_c_modulo_in_a_memlet_subset_is_left_alone():
    sdfg, state = memlet_sdfg("cmod_memlet", "A[i % 7]")
    before = str(gather_subset(state))

    _apply(sdfg)

    assert str(gather_subset(state)) == before
    assert "py_mod" not in before


def test_rewrite_interstate_edge():
    sdfg = dace.SDFG("mod_ise")
    sdfg.add_symbol("i", dace.int64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    s1 = sdfg.add_state("s1")
    sdfg.add_edge(s0, s1, dace.InterstateEdge(condition="PyMod(i, 3) == 0", assignments={"j": "FtnModulo(i, 5)"}))

    _apply(sdfg)

    edge = sdfg.edges()[0].data
    assert "py_mod(i, 3)" in edge.condition.as_string
    assert "py_mod(i, 5)" in edge.assignments["j"]
    assert "PyMod" not in edge.condition.as_string
    assert "FtnModulo" not in edge.assignments["j"]


def test_idempotent_on_existing_py_mod():
    sdfg, t = elementwise_sdfg("mod_idem", "c = py_mod(a, b)")

    _apply(sdfg)

    assert t.code.as_string.strip() == "c = py_mod(a, b)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pass-level tests for ``LowerITEToFpFactor``.

After running, no tasklet body in the SDFG contains an ``ITE(...)`` call;
every ``ITE(c, t, e)`` is rewritten to ``c * t + (1 - c) * e``. The pass
is purely a code-text rewrite -- connectors and edges are unchanged.
"""
import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.vectorization.lower_ite_to_fp_factor import LowerITEToFpFactor
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import (
    SameWriteSetIfElseToITECFG, )


def _build_ite_tasklet_sdfg(ite_code: str):
    """Build a minimal SDFG whose single state contains a tasklet with the
    given ``ITE`` body. Connectors ``_c``, ``_t``, ``_e`` are wired from
    1-element float64 arrays; output ``_o`` goes to ``A``.
    """
    sdfg = dace.SDFG("lower_ite_test")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("T", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("E", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("C", shape=(1, ), dtype=dace.bool_)

    state = sdfg.add_state("only", is_start_block=True)
    rT = state.add_access("T")
    rE = state.add_access("E")
    rC = state.add_access("C")
    wA = state.add_access("A")
    t = state.add_tasklet("ite_A", {"_c", "_t", "_e"}, {"_o"}, ite_code)
    state.add_edge(rC, None, t, "_c", dace.Memlet("C[0]"))
    state.add_edge(rT, None, t, "_t", dace.Memlet("T[0]"))
    state.add_edge(rE, None, t, "_e", dace.Memlet("E[0]"))
    state.add_edge(t, "_o", wA, None, dace.Memlet("A[0]"))
    return sdfg


def test_rewrites_simple_ite_tasklet():
    sdfg = _build_ite_tasklet_sdfg("_o = ITE(_c, _t, _e)")
    rewritten = LowerITEToFpFactor().apply_pass(sdfg, {})
    assert rewritten == 1
    body = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.Tasklet)).code.as_string
    assert "ITE(" not in body
    # The rewrite should preserve the three operands.
    for op in ("_c", "_t", "_e"):
        assert op in body
    # And introduce the (1 - c) complement factor, with the bool cond promoted to
    # the float64 arm dtype (uniform-dtype tile binop; see the pass docstring).
    assert "1 - dace.float64(_c)" in body


def test_no_match_when_no_ite_call():
    sdfg = _build_ite_tasklet_sdfg("_o = _t + _e")
    rewritten = LowerITEToFpFactor().apply_pass(sdfg, {})
    assert rewritten is None


def test_handles_multiple_ites_in_one_tasklet():
    sdfg = _build_ite_tasklet_sdfg("_o = ITE(_c, _t, _e) + ITE(_c, _t, _e)")
    rewritten = LowerITEToFpFactor().apply_pass(sdfg, {})
    assert rewritten == 1
    body = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.Tasklet)).code.as_string
    assert "ITE(" not in body


def test_handles_symbol_cond_in_ite_call():
    """When the upstream branch-normalization passes cannot resolve the cond
    to an array they emit the cond as inline text inside the ITE call, e.g.
    ``ITE(c0 == 1, ...)``. The pass should rewrite this just like the
    in-connector form."""
    sdfg = dace.SDFG("lower_ite_sym_cond")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("T", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("E", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c0", dace.int64)
    state = sdfg.add_state("only", is_start_block=True)
    rT = state.add_access("T")
    rE = state.add_access("E")
    wA = state.add_access("A")
    t = state.add_tasklet("ite_A", {"_t", "_e"}, {"_o"}, "_o = ITE(c0 == 1, _t, _e)")
    state.add_edge(rT, None, t, "_t", dace.Memlet("T[0]"))
    state.add_edge(rE, None, t, "_e", dace.Memlet("E[0]"))
    state.add_edge(t, "_o", wA, None, dace.Memlet("A[0]"))

    rewritten = LowerITEToFpFactor().apply_pass(sdfg, {})
    assert rewritten == 1
    body = next(n for n in sdfg.states()[0].nodes() if isinstance(n, dace.nodes.Tasklet)).code.as_string
    assert "ITE(" not in body
    # The symbol cond appears twice: once in `c0 == 1` and again in `1 - (c0 == 1)`.
    assert body.count("c0") == 2


def test_pass_is_idempotent_after_first_run():
    sdfg = _build_ite_tasklet_sdfg("_o = ITE(_c, _t, _e)")
    first = LowerITEToFpFactor().apply_pass(sdfg, {})
    second = LowerITEToFpFactor().apply_pass(sdfg, {})
    assert first == 1
    assert second is None


def test_numerical_equivalence_against_branch_reference():
    """End-to-end: build a same-write-set if/else, run M3.1b to get canonical
    ITE IR, then LowerITEToFpFactor, then compile and compare against the
    scalar Python reference. The two paths must agree."""
    sdfg = dace.SDFG("end_to_end_lower_ite")
    sdfg.add_array("A", shape=(1, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(1, ), dtype=dace.float64)
    sdfg.add_symbol("c", dace.bool_)
    entry = sdfg.add_state("entry", is_start_block=True)
    exit_state = sdfg.add_state("exit")
    cb = ConditionalBlock("cb")
    sdfg.add_node(cb)
    sdfg.add_edge(entry, cb, dace.InterstateEdge())
    sdfg.add_edge(cb, exit_state, dace.InterstateEdge())

    then_cfr = ControlFlowRegion("then_cfr", sdfg=sdfg)
    ts = then_cfr.add_state("ts", is_start_block=True)
    rB = ts.add_access("B")
    wA = ts.add_access("A")
    tt = ts.add_tasklet("plus", {"_b"}, {"_a"}, "_a = _b + 1.0")
    ts.add_edge(rB, None, tt, "_b", dace.Memlet("B[0]"))
    ts.add_edge(tt, "_a", wA, None, dace.Memlet("A[0]"))
    cb.add_branch(CodeBlock("c"), then_cfr)

    else_cfr = ControlFlowRegion("else_cfr", sdfg=sdfg)
    es = else_cfr.add_state("es", is_start_block=True)
    rB2 = es.add_access("B")
    wA2 = es.add_access("A")
    te = es.add_tasklet("minus", {"_b"}, {"_a"}, "_a = _b - 1.0")
    es.add_edge(rB2, None, te, "_b", dace.Memlet("B[0]"))
    es.add_edge(te, "_a", wA2, None, dace.Memlet("A[0]"))
    cb.add_branch(None, else_cfr)

    SameWriteSetIfElseToITECFG().apply_pass(sdfg, {})
    LowerITEToFpFactor().apply_pass(sdfg, {})

    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.Tasklet):
                assert "ITE(" not in n.code.as_string

    def reference(c: bool, B: np.ndarray) -> np.ndarray:
        return np.array([B[0] + 1.0 if c else B[0] - 1.0], dtype=np.float64)

    csdfg = sdfg.compile()
    for c in (True, False):
        for b in (-2.0, 0.5, 7.0):
            A = np.zeros((1, ), dtype=np.float64)
            B = np.array([b], dtype=np.float64)
            csdfg(A=A, B=B, c=c)
            expected = reference(c, B)
            np.testing.assert_allclose(A, expected, err_msg=f"c={c}, b={b}, got={A}, want={expected}")

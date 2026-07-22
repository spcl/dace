# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Readable-codegen state-scope brace elision (DaCeCodeGenerator.state_needs_brace).

The state machine is lowered to labels + ``goto``. A per-state ``{ }`` C scope confines each state's
declarations so an inter-state goto never crosses an initialization. The readable generator drops that
scope ONLY when the state provably declares nothing at its own scope: structured control flow (no
conditional block out-edge, no unstructured region), ``to_allocate`` empty, and every top-level node a
map scope / access node with no node instrumentation. This proves the elision fires for a pure-map
state and is correctly withheld for the goto / untracked-declaration hazards.
"""
import numpy as np
import pytest

import dace
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap

from tests.codegen.readable.conftest import (LEGACY, EXPERIMENTAL, use_implementation, run_isolated, assert_outputs_equivalent,
                      experimental_available)

N = dace.symbol("N")


def _frame(sdfg):
    # NOTE: state_needs_brace reads the live ``compiler.cpu.implementation`` config, so the caller must
    # hold ``use_implementation(EXPERIMENTAL)`` around BOTH this build and the predicate calls.
    fc = DaCeCodeGenerator(sdfg)
    fc.determine_allocation_lifetime(sdfg)
    return fc


def _pure_map_sdfg():
    sdfg = dace.SDFG("pure_map")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    st = sdfg.add_state("s")
    ra, rb, w = st.add_read("a"), st.add_read("b"), st.add_write("out")
    me, mx = st.add_map("m", dict(i="0:N"))
    t = st.add_tasklet("t", {"x", "y"}, {"o"}, "o = x * y")
    st.add_memlet_path(ra, me, t, dst_conn="x", memlet=dace.Memlet("a[i]"))
    st.add_memlet_path(rb, me, t, dst_conn="y", memlet=dace.Memlet("b[i]"))
    st.add_memlet_path(t, mx, w, src_conn="o", memlet=dace.Memlet("out[i]"))
    return sdfg


def _conditional_edge_sdfg():
    sdfg = dace.SDFG("cond_edge")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    a = sdfg.add_state("a")
    b = sdfg.add_state("b")
    an, ao = b.add_read("a"), b.add_write("out")
    me, mx = b.add_map("m", dict(i="0:N"))
    t = b.add_tasklet("t", {"x"}, {"o"}, "o = x + 1.0")
    b.add_memlet_path(an, me, t, dst_conn="x", memlet=dace.Memlet("a[i]"))
    b.add_memlet_path(t, mx, ao, src_conn="o", memlet=dace.Memlet("out[i]"))
    sdfg.add_edge(a, b, dace.InterstateEdge(condition="N > 4"))  # single CONDITIONAL edge
    return sdfg


def _code_to_code_sdfg():
    sdfg = dace.SDFG("c2c")
    sdfg.add_array("out", [1], dace.float64)
    st = sdfg.add_state("only")
    t1 = st.add_tasklet("t1", {}, {"v"}, "v = 3.0")
    t2 = st.add_tasklet("t2", {"v"}, {"o"}, "o = v * 2.0")
    ao = st.add_access("out")
    st.add_edge(t1, "v", t2, "v", dace.Memlet())  # code->code register at state scope
    st.add_memlet_path(t2, ao, src_conn="o", memlet=dace.Memlet("out[0]"))
    return sdfg


def test_pure_map_state_elides_brace():
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _pure_map_sdfg()
    with use_implementation(EXPERIMENTAL):
        fc = _frame(sdfg)
        assert fc._structured_control_flow(sdfg) is True
        (state, ) = [s for s in sdfg.states() if s.number_of_nodes() > 0]
        assert fc.state_needs_brace(state) is False  # elided


def test_conditional_edge_keeps_brace():
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _conditional_edge_sdfg()
    with use_implementation(EXPERIMENTAL):
        fc = _frame(sdfg)
        assert fc._structured_control_flow(sdfg) is False  # a conditional out-edge emits a crossing goto
        for state in sdfg.states():
            if state.number_of_nodes() > 0:
                assert fc.state_needs_brace(state) is True


def test_code_to_code_keeps_brace():
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _code_to_code_sdfg()
    with use_implementation(EXPERIMENTAL):
        fc = _frame(sdfg)
        (state, ) = list(sdfg.states())
        # a top-level tasklet (code->code register) is not a whitelisted map/access node -> keep the brace
        assert fc.state_needs_brace(state) is True


def test_legacy_always_braces():
    """Legacy generator keeps every non-empty state's scope (byte-identical output)."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _pure_map_sdfg()
    with use_implementation(LEGACY):
        fc = DaCeCodeGenerator(sdfg)
        fc.determine_allocation_lifetime(sdfg)
        (state, ) = [s for s in sdfg.states() if s.number_of_nodes() > 0]
        assert fc.state_needs_brace(state) is True


def test_pure_map_bit_exact():
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")

    def run(impl):

        def build_and_run():
            with use_implementation(impl):
                s = _pure_map_sdfg()
                s.simplify()
                s.apply_transformations_repeated(LoopToMap)
                s.apply_transformations_repeated(MapFusion)
                csdfg = s.compile()
            rng = np.random.default_rng(0)
            a, b, out = rng.random(32), rng.random(32), np.zeros(32)
            csdfg(a=a, b=b, out=out, N=32)
            return {"out": out}

        return run_isolated(build_and_run)

    assert_outputs_equivalent(run(LEGACY), run(EXPERIMENTAL), "cpu", label="pure_map")


if __name__ == "__main__":
    test_pure_map_state_elides_brace()
    test_conditional_edge_keeps_brace()
    test_code_to_code_keeps_brace()
    test_legacy_always_braces()
    test_pure_map_bit_exact()
    print("ok")

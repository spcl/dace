# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The readable generator drops the C scope of a state only when nothing in the state declares at state level and no
jump in its region can cross such a declaration. The legacy generator always keeps it.
"""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.sdfg.state import ControlFlowRegion
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

N = dace.symbol("N")


@contextlib.contextmanager
def recorded_state_scope_decisions() -> Iterator[dict[str, bool]]:
    """State label -> whether the code generator kept the state's scope, filled during code generation."""
    recorded: dict[str, bool] = {}
    original = DaCeCodeGenerator.state_needs_brace

    def record(self, state):
        recorded[state.label] = original(self, state)
        return recorded[state.label]

    DaCeCodeGenerator.state_needs_brace = record
    try:
        yield recorded
    finally:
        DaCeCodeGenerator.state_needs_brace = original


@pytest.fixture
def decisions() -> Iterator[dict[str, bool]]:
    with recorded_state_scope_decisions() as recorded:
        yield recorded


def add_map_state(sdfg, label, region=None):
    state = (sdfg if region is None else region).add_state(label)
    ra, w = state.add_read("a"), state.add_write("out")
    me, mx = state.add_map("m", {"i": "0:N"})
    t = state.add_tasklet("t", {"x"}, {"o"}, "o = x * 2.0")
    state.add_memlet_path(ra, me, t, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(t, mx, w, src_conn="o", memlet=dace.Memlet("out[i]"))
    return state


def base_sdfg(name):
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    return sdfg


def pure_map_sdfg():
    sdfg = base_sdfg("pure_map")
    add_map_state(sdfg, "s")
    return sdfg


def conditional_edge_sdfg():
    sdfg = base_sdfg("cond_edge")
    sdfg.add_edge(add_map_state(sdfg, "first"), add_map_state(sdfg, "second"), dace.InterstateEdge(condition="N > 4"))
    return sdfg


def code_to_code_sdfg():
    sdfg = base_sdfg("c2c")
    state = sdfg.add_state("only")
    t1 = state.add_tasklet("t1", {}, {"v"}, "v = 3.0")
    t2 = state.add_tasklet("t2", {"v"}, {"o"}, "o = v * 2.0")
    state.add_edge(t1, "v", t2, "v", dace.Memlet())
    state.add_memlet_path(t2, state.add_access("out"), src_conn="o", memlet=dace.Memlet("out[0]"))
    return sdfg


def nested_region_sdfg():
    """A fall-through top level followed by a region whose only edge is conditional."""
    sdfg = base_sdfg("regions")
    top = add_map_state(sdfg, "top")
    region = ControlFlowRegion("inner", sdfg=sdfg)
    sdfg.add_node(region)
    sdfg.add_edge(top, region, dace.InterstateEdge())
    inner_first = add_map_state(sdfg, "inner_first", region)
    inner_second = add_map_state(sdfg, "inner_second", region)
    region.add_edge(inner_first, inner_second, dace.InterstateEdge(condition="N > 4"))
    return sdfg


def instrumented_access_sdfg():
    sdfg = pure_map_sdfg()
    (state, ) = sdfg.states()
    next(n for n in state.data_nodes() if n.data == "out").instrument = dtypes.DataInstrumentationType.Save
    return sdfg


def generate(sdfg, implementation):
    # Control flow detection would raise the conditional edges to conditional blocks
    with use_implementation(implementation), set_temporary("optimizer", "detect_control_flow", value=False):
        return "\n".join(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cpp")


SCOPE_CASES = [
    (pure_map_sdfg, {
        "s": False
    }),
    (conditional_edge_sdfg, {
        "first": True,
        "second": True
    }),
    (code_to_code_sdfg, {
        "only": True
    }),
    (instrumented_access_sdfg, {
        "s": True
    }),
    (nested_region_sdfg, {
        "top": False,
        "inner_first": True,
        "inner_second": True
    }),
]


@pytest.mark.parametrize("build, expected", SCOPE_CASES)
def test_state_scope_is_kept_only_where_a_declaration_or_a_jump_could_need_it(build, expected, decisions):
    generate(build(), EXPERIMENTAL)
    kept = {label: kept for label, kept in decisions.items() if label in expected}
    assert kept == expected, kept


def test_legacy_keeps_every_state_scope(decisions):
    generate(pure_map_sdfg(), LEGACY)
    assert decisions == {"s": True}


BALANCED_BUILDS = [pure_map_sdfg, conditional_edge_sdfg, code_to_code_sdfg, nested_region_sdfg]


@pytest.mark.parametrize("build", BALANCED_BUILDS)
def test_generated_scopes_are_balanced(build):
    code = generate(build(), EXPERIMENTAL)
    assert code.count("{") == code.count("}")


def test_pure_map_bit_exact():

    def run(implementation):

        def build_and_run():
            with use_implementation(implementation):
                sdfg = pure_map_sdfg()
                sdfg.simplify()
                sdfg.apply_transformations_repeated(LoopToMap)
                sdfg.apply_transformations_repeated(MapFusion)
                compiled = sdfg.compile()
            a, out = np.random.default_rng(0).random(32), np.zeros(32)
            compiled(a=a, out=out, N=32)
            return {"out": out}

        return run_isolated(build_and_run)

    assert_outputs_equivalent(run(LEGACY), run(EXPERIMENTAL), "cpu", label="pure_map")


if __name__ == "__main__":
    for build, expected in SCOPE_CASES:
        with recorded_state_scope_decisions() as decisions:
            test_state_scope_is_kept_only_where_a_declaration_or_a_jump_could_need_it(build, expected, decisions)
    with recorded_state_scope_decisions() as decisions:
        test_legacy_keeps_every_state_scope(decisions)
    for build in BALANCED_BUILDS:
        test_generated_scopes_are_balanced(build)
    test_pure_map_bit_exact()

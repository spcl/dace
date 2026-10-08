# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import cast

import sympy

import dace
from dace import symbolic
from dace.sdfg.state import LoopRegion, SymbolResolver

ZERO = symbolic.pystr_to_symbolic("0")


def le(lhs: str, rhs: str) -> symbolic.Relation:
    return symbolic.Relation(
        symbolic.RelationKind.LE,
        cast(sympy.Expr, symbolic.pystr_to_symbolic(lhs)),
        cast(sympy.Expr, symbolic.pystr_to_symbolic(rhs)),
    )


def loop_sdfg() -> tuple[dace.SDFG, LoopRegion]:
    sdfg = dace.SDFG("facts_at_loop")
    sdfg.add_symbol("N", dace.int64)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    return sdfg, loop


def map_state(step: str) -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry, dace.nodes.Tasklet]:
    sdfg = dace.SDFG(f"facts_at_map_{step}")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("S", dace.int64)
    sdfg.add_array("A", ["N"], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet, entry, _ = state.add_mapped_tasklet(
        "m", {"j": f"0:N:{step}"}, {}, "a = 1.0", {"a": dace.Memlet("A[j]")}, external_edges=True
    )
    return sdfg, state, entry, tasklet


def test_loop_body_knows_the_iterator_range():
    _, loop = loop_sdfg()
    assert symbolic.ask(le("i", "N - 1"), SymbolResolver().facts_at(loop)) is symbolic.Truth.TRUE


def test_loop_header_does_not_know_the_iterator_range():
    sdfg, _ = loop_sdfg()
    assert symbolic.ask(le("i", "N - 1"), SymbolResolver().facts_at(sdfg)) is symbolic.Truth.UNKNOWN


def test_map_body_knows_the_parameter_range():
    _, state, _, tasklet = map_state("1")
    assert symbolic.ask(le("0", "j"), SymbolResolver().facts_at(state, tasklet)) is symbolic.Truth.TRUE


def test_map_entry_does_not_know_its_own_parameter_range():
    _, state, entry, _ = map_state("1")
    assert symbolic.ask(le("0", "j"), SymbolResolver().facts_at(state, entry)) is symbolic.Truth.UNKNOWN


def test_step_of_unknown_sign_gives_no_range():
    _, state, _, tasklet = map_state("S")
    assert symbolic.ask(le("0", "j"), SymbolResolver().facts_at(state, tasklet)) is symbolic.Truth.UNKNOWN


def test_nested_sdfg_inherits_outer_ranges_through_its_symbol_mapping():
    outer = dace.SDFG("facts_at_outer")
    outer.add_symbol("N", dace.int64)
    state = outer.add_state(is_start_block=True)
    inner = dace.SDFG("facts_at_inner")
    inner.add_symbol("k", dace.int64)
    inner_state = inner.add_state(is_start_block=True)
    entry, exit_node = state.add_map("m", {"j": "0:N"})
    nsdfg = state.add_nested_sdfg(inner, {}, {}, symbol_mapping={"k": "j"})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    state.add_nedge(nsdfg, exit_node, dace.Memlet())
    assert symbolic.ask(le("0", "k"), SymbolResolver().facts_at(inner_state)) is symbolic.Truth.TRUE


def test_unsigned_symbol_is_nonnegative():
    sdfg = dace.SDFG("facts_at_unsigned")
    sdfg.add_symbol("U", dace.uint32)
    assert symbolic.ask(le("0", "U"), SymbolResolver().facts_at(sdfg)) is symbolic.Truth.TRUE


if __name__ == "__main__":
    test_loop_body_knows_the_iterator_range()
    test_loop_header_does_not_know_the_iterator_range()
    test_map_body_knows_the_parameter_range()
    test_map_entry_does_not_know_its_own_parameter_range()
    test_step_of_unknown_sign_gives_no_range()
    test_nested_sdfg_inherits_outer_ranges_through_its_symbol_mapping()
    test_unsigned_symbol_is_nonnegative()

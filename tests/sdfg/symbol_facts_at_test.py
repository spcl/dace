# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import cast

import pytest
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


def nested_with_positive_k(mapped: str) -> dace.SDFG:
    outer = dace.SDFG("assumed_outer")
    outer.add_symbol("N", dace.int64)
    state = outer.add_state(is_start_block=True)
    inner = dace.SDFG("assumed_inner")
    inner.add_symbol("k", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    inner.add_state(is_start_block=True)
    entry, exit_node = state.add_map("m", {"j": "0:N"})
    nsdfg = state.add_nested_sdfg(inner, {}, {}, symbol_mapping={"k": mapped})
    state.add_nedge(entry, nsdfg, dace.Memlet())
    state.add_nedge(nsdfg, exit_node, dace.Memlet())
    return outer


def test_nested_sdfg_assumption_provable_from_the_mapping_is_valid():
    nested_with_positive_k("j + 1").validate()


def test_nested_sdfg_assumption_unprovable_from_the_mapping_is_invalid():
    with pytest.raises(dace.sdfg.InvalidSDFGError, match="not provable"):
        nested_with_positive_k("j").validate()


def test_unsigned_symbol_is_nonnegative():
    sdfg = dace.SDFG("facts_at_unsigned")
    sdfg.add_symbol("U", dace.uint32)
    assert symbolic.ask(le("0", "U"), SymbolResolver().facts_at(sdfg)) is symbolic.Truth.TRUE


def test_inverted_loop_body_does_not_know_the_iterator_range():
    """The body of an inverted loop runs before its condition is checked, so the iterator may leave the range."""
    sdfg = dace.SDFG("facts_at_inverted_loop")
    sdfg.add_symbol("N", dace.int64)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1", inverted=True, update_before_condition=False)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    assert symbolic.ask(le("i", "N - 1"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_nothing_is_assumed_of_a_symbol_an_edge_reassigns():
    """``N`` is positive on entry, but ``N = N - 5`` may make it negative after."""
    sdfg = dace.SDFG("facts_at_reassigned")
    sdfg.add_symbol("N", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    first = sdfg.add_state(is_start_block=True)
    second = sdfg.add_state_after(first, assignments={"N": "N - 5"})
    assert symbolic.ask(le("1", "N"), SymbolResolver().facts_at(second)) is symbolic.Truth.UNKNOWN


def loop_assigning(assignments: dict[str, str]) -> tuple[dace.SDFG, dace.SDFGState]:
    """A loop over ``i < N`` whose body assigns ``assignments`` on an interstate edge; returns the state after it."""
    sdfg, loop = loop_sdfg()
    after = loop.add_state("after")
    loop.add_edge(loop.start_block, after, dace.InterstateEdge(assignments=assignments))
    return sdfg, after


@pytest.mark.parametrize("assignments", [{"i": "i + 100"}, {"N": "0"}])
def test_loop_body_reassigning_the_iterator_or_a_bound_knows_no_range(assignments):
    """After the body reassigns the iterator or a symbol of its bounds, the range no longer bounds the iterator."""
    _, after = loop_assigning(assignments)
    facts = SymbolResolver().facts_at(after)
    assert symbolic.ask(le("i + 1", "N"), facts) is symbolic.Truth.UNKNOWN


def test_loop_body_assigning_another_symbol_keeps_the_range():
    _, after = loop_assigning({"k": "i"})
    facts = SymbolResolver().facts_at(after)
    assert symbolic.ask(le("i + 1", "N"), facts) is symbolic.Truth.TRUE


def test_nested_sdfg_symbol_reassigned_inside_is_not_its_mapped_value():
    """A mapped symbol equals the outer value on entry only: once an interstate edge of the nested SDFG assigns it,
    what held for the outer value says nothing about it."""
    outer = dace.SDFG("facts_at_reassigned_mapping")
    outer.add_symbol("M", dace.int64, predicates=frozenset({symbolic.Predicate.NEGATIVE}))
    inner = dace.SDFG("facts_at_reassigned_mapping_inner")
    inner.add_symbol("N", dace.int64)
    first = inner.add_state("first", is_start_block=True)
    second = inner.add_state("second")
    inner.add_edge(first, second, dace.InterstateEdge(assignments={"N": "10"}))
    outer.add_state(is_start_block=True).add_nested_sdfg(inner, {}, {}, symbol_mapping={"N": "M"})

    facts = SymbolResolver().facts_at(second)
    assert symbolic.ask(le("N + 1", "0"), facts) is symbolic.Truth.UNKNOWN


def test_renamed_extent_moves_its_fact():
    """A dimension's fact is derived from the current descriptors, so a rename leaves nothing of the old name."""
    sdfg = dace.SDFG("shape_fact_renamed")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", ["N"], dace.float64)
    sdfg.add_state(is_start_block=True)
    sdfg.replace_dict({"N": "K"})
    facts = SymbolResolver().facts_at(sdfg)
    assert symbolic.ask(le("0", "K"), facts) is symbolic.Truth.TRUE
    assert symbolic.ask(le("0", "N"), facts) is symbolic.Truth.UNKNOWN

if __name__ == "__main__":
    test_loop_body_knows_the_iterator_range()
    test_loop_header_does_not_know_the_iterator_range()
    test_map_body_knows_the_parameter_range()
    test_map_entry_does_not_know_its_own_parameter_range()
    test_step_of_unknown_sign_gives_no_range()
    test_nested_sdfg_inherits_outer_ranges_through_its_symbol_mapping()
    test_nested_sdfg_assumption_provable_from_the_mapping_is_valid()
    test_nested_sdfg_assumption_unprovable_from_the_mapping_is_invalid()
    test_unsigned_symbol_is_nonnegative()
    test_inverted_loop_body_does_not_know_the_iterator_range()
    test_nothing_is_assumed_of_a_symbol_an_edge_reassigns()
    test_loop_body_reassigning_the_iterator_or_a_bound_knows_no_range({"i": "i + 100"})
    test_loop_body_reassigning_the_iterator_or_a_bound_knows_no_range({"N": "0"})
    test_loop_body_assigning_another_symbol_keeps_the_range()
    test_nested_sdfg_symbol_reassigned_inside_is_not_its_mapped_value()
    test_renamed_extent_moves_its_fact()

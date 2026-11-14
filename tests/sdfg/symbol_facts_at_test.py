# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import cast

import pytest
import sympy

import dace
from dace import symbolic
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SymbolResolver
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

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


def test_step_positive_by_the_sdfg_facts_gives_the_range():
    sdfg, state, _, tasklet = map_state("S")
    sdfg.set_symbol_assumptions("S", frozenset({symbolic.Predicate.POSITIVE}))
    facts = SymbolResolver().facts_at(state, tasklet)
    assert symbolic.ask(le("0", "j"), facts) is symbolic.Truth.TRUE
    assert symbolic.ask(le("j", "N - 1"), facts) is symbolic.Truth.TRUE


def test_step_negative_by_the_sdfg_facts_gives_the_reversed_range():
    sdfg = dace.SDFG("facts_at_map_negative_step")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("S", dace.int64)
    sdfg.set_symbol_assumptions("S", frozenset({symbolic.Predicate.NEGATIVE}))
    sdfg.add_array("A", ["N"], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet, entry, _ = state.add_mapped_tasklet(
        "m", {"j": "0:N"}, {}, "a = 1.0", {"a": dace.Memlet("A[j]")}, external_edges=True
    )
    # Inclusive bounds, first to last: from N - 1 down to 0
    entry.map.range = dace.subsets.Range([tuple(symbolic.pystr_to_symbolic(b) for b in ("N - 1", "0", "S"))])
    facts = SymbolResolver().facts_at(state, tasklet)
    assert symbolic.ask(le("0", "j"), facts) is symbolic.Truth.TRUE
    assert symbolic.ask(le("j", "N - 1"), facts) is symbolic.Truth.TRUE


def test_a_range_empty_by_the_facts_yields_facts_rather_than_a_contradiction():
    """``j in K:N`` under ``N + 1 <= K`` never runs: its bounds contradict the facts, and the body knows less instead
    of making every query at it (memlet propagation, while the map is built) raise."""
    sdfg = dace.SDFG("facts_at_empty_range")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_symbol_relation(le("N + 1", "K"))
    sdfg.add_array("A", ["N + K + 8"], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    tasklet, _, _ = state.add_mapped_tasklet(
        "m", {"j": "K:N"}, {}, "a = 1.0", {"a": dace.Memlet("A[j]")}, external_edges=True
    )
    facts = SymbolResolver().facts_at(state, tasklet)
    assert symbolic.ask(le("N + 1", "K"), facts) is symbolic.Truth.TRUE


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
    what held for the outer value says nothing about it; what the assignment says does."""
    outer = dace.SDFG("facts_at_reassigned_mapping")
    outer.add_symbol("M", dace.int64, predicates=frozenset({symbolic.Predicate.NEGATIVE}))
    inner = dace.SDFG("facts_at_reassigned_mapping_inner")
    inner.add_symbol("N", dace.int64)
    first = inner.add_state("first", is_start_block=True)
    second = inner.add_state("second")
    inner.add_edge(first, second, dace.InterstateEdge(assignments={"N": "10"}))
    outer.add_state(is_start_block=True).add_nested_sdfg(inner, {}, {}, symbol_mapping={"N": "M"})

    facts = SymbolResolver().facts_at(second)
    assert symbolic.ask(le("N + 1", "0"), facts) is symbolic.Truth.FALSE
    assert symbolic.ask(le("10", "N"), facts) is symbolic.Truth.TRUE


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


@pytest.mark.parametrize("dimension, nonnegative", [("2 * N", True), ("N + 1", False), ("M * N", False)])
def test_a_dimension_scaling_one_symbol_makes_it_nonnegative(dimension: str, nonnegative: bool):
    """``2 * N >= 0`` means ``N >= 0``; ``N + 1`` and ``M * N`` say nothing about ``N`` alone."""
    sdfg = dace.SDFG(f"scaled_dimension_{nonnegative}_{len(dimension)}")
    for name in ("M", "N"):
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_array("A", [dimension], dace.float64)
    sdfg.add_state(is_start_block=True)
    truth = symbolic.ask(le("0", "N"), SymbolResolver().facts_at(sdfg))
    assert truth is (symbolic.Truth.TRUE if nonnegative else symbolic.Truth.UNKNOWN)


def loop_over(condition: str, init: str | None, update: str, **symbols) -> tuple[dace.SDFG, LoopRegion]:
    sdfg = dace.SDFG(f"facts_loop_{abs(hash((condition, init, update)))}")
    for name, dtype in symbols.items():
        sdfg.add_symbol(name, dtype)
    loop = LoopRegion("loop", condition, "i", init, update)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    return sdfg, loop


def test_a_range_needs_a_bound_the_body_cannot_write():
    sdfg, loop = loop_over("i < s", "i = 0", "i = i + 1")
    sdfg.add_scalar("s", dace.int64, transient=True)
    body = loop.start_block
    body.add_edge(body.add_tasklet("t", {}, {"o"}, "o = 0"), "o", body.add_write("s"), None, dace.Memlet("s[0]"))
    assert symbolic.ask(le("i", "s - 1"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def csr_loops(write_ptr: bool) -> LoopRegion:
    """``for r < N: for k in range(ptr[r], ptr[r + 1])``, the inner body writing ``ptr`` when asked."""
    sdfg = dace.SDFG(f"facts_csr_{write_ptr}")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("ptr", ["N + 1"], dace.int64)
    rows = LoopRegion("rows", "r < N", "r", "r = 0", "r = r + 1")
    sdfg.add_node(rows, is_start_block=True)
    inner = LoopRegion("inner", "k < ptr[r + 1]", "k", "k = ptr[r]", "k = k + 1")
    rows.add_node(inner, is_start_block=True)
    body = inner.add_state("body", is_start_block=True)
    if write_ptr:
        body.add_edge(
            body.add_tasklet("t", {}, {"o"}, "o = 0"), "o", body.add_write("ptr"), None, dace.Memlet("ptr[r]")
        )
    return inner


def test_a_range_may_read_a_container_the_loop_does_not_write():
    """A CSR row loop ``range(ptr[r], ptr[r + 1])`` bounds its iterator by the row pointers."""
    facts = SymbolResolver().facts_at(csr_loops(write_ptr=False))
    assert symbolic.ask(le("ptr[r]", "k"), facts) is symbolic.Truth.TRUE
    assert symbolic.ask(le("k + 1", "ptr[r + 1]"), facts) is symbolic.Truth.TRUE


def test_a_range_reading_a_container_the_loop_writes_knows_nothing():
    facts = SymbolResolver().facts_at(csr_loops(write_ptr=True))
    assert symbolic.ask(le("ptr[r]", "k"), facts) is symbolic.Truth.UNKNOWN


@pytest.mark.parametrize("predicates, known", [(frozenset({symbolic.Predicate.POSITIVE}), True), (frozenset(), False)])
def test_a_symbolic_loop_step_gives_a_range_when_the_facts_orient_it(predicates, known):
    """``i += S`` under ``i < N`` stays in ``[0, N - 1]`` when ``S`` is positive; of an unknown ``S`` nothing."""
    sdfg, loop = loop_over("i < N", "i = 0", "i = i + S", N=dace.int64)
    sdfg.add_symbol("S", dace.int64, predicates=predicates)
    truth = symbolic.ask(le("i", "N - 1"), SymbolResolver().facts_at(loop))
    assert truth is (symbolic.Truth.TRUE if known else symbolic.Truth.UNKNOWN)


def test_a_range_needs_an_init_that_does_not_read_the_iterator():
    _, loop = loop_over("i < N", "i = 2 * i", "i = i + 1", N=dace.int64)
    assert symbolic.ask(le("i", "0"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_a_range_needs_a_condition_along_the_step():
    _, loop = loop_over("i < N", "i = N - 1", "i = i - 1", N=dace.int64)
    assert symbolic.ask(le("N - 1", "i"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_a_float_limit_gives_no_integer_range():
    _, loop = loop_over("i < x", "i = 0", "i = i + 1", x=dace.float64)
    assert symbolic.ask(le("i + 1", "x"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_a_float_iterator_gives_no_integer_range():
    _, loop = loop_over("i < 1.0", "i = 0.0", "i = i + 0.25")
    assert symbolic.ask(le("i", "0"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_a_loop_header_ends_a_fact_on_entry():
    sdfg, loop = loop_over("i > -3", None, "i = i - 1")
    sdfg.add_symbol("i", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    assert symbolic.ask(le("1", "i"), SymbolResolver().facts_at(loop)) is symbolic.Truth.UNKNOWN


def test_inlining_keeps_the_assumptions_proofs_took():
    inner = dace.SDFG("assumption_inner")
    inner.add_symbol("k", dace.int64)
    inner.add_symbol("N", dace.int64)
    inner.add_state(is_start_block=True)
    relation = symbolic.Relation(symbolic.RelationKind.LT, symbolic.symbol("k"), symbolic.symbol("N"))
    inner.symbol_repo.runtime_assumptions.add(relation)
    outer = dace.SDFG("assumption_outer")
    outer.add_symbol("k", dace.int64)
    outer.add_symbol("N", dace.int64)
    state = outer.add_state(is_start_block=True)
    state.add_nested_sdfg(inner, {}, {}, symbol_mapping={"k": "k", "N": "N"})

    InlineMultistateSDFG.apply_to(outer, nested_sdfg=state.nodes()[0], verify=False, save=False)

    assert relation in outer.symbol_repo.runtime_assumptions


def halving_loop() -> tuple[dace.SDFG, LoopRegion, dace.SDFGState, dace.SDFGState]:
    """``while s > 0: (first) ; s = s // 2 ; (second)``; returns the loop and its two body states."""
    sdfg = dace.SDFG("facts_at_while")
    sdfg.add_symbol("s", dace.int64)
    loop = LoopRegion("halve", "s > 0")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    second = loop.add_state("second")
    loop.add_edge(first, second, dace.InterstateEdge(assignments={"s": "s // 2"}))
    return sdfg, loop, first, second


def test_while_body_start_knows_the_condition():
    """The condition holds where the body starts: ``0 < s`` in ``first``."""
    _, _, first, _ = halving_loop()
    facts = SymbolResolver().facts_at(first)
    assert symbolic.ask(le("1", "s"), facts) is symbolic.Truth.TRUE


def test_a_block_after_the_condition_symbol_is_reassigned_does_not_know_it():
    """``s // 2`` may be 0: after the assignment nothing says ``0 < s``."""
    _, _, _, second = halving_loop()
    facts = SymbolResolver().facts_at(second)
    assert symbolic.ask(le("1", "s"), facts) is symbolic.Truth.UNKNOWN


def test_an_inner_loop_reassigning_the_condition_symbol_does_not_know_it():
    """A loop at the body's start that halves ``s`` itself: its later iterations see ``s`` changed."""
    sdfg, loop, first, _ = halving_loop()
    inner = LoopRegion("inner", "j < 4", "j", "j = 0", "j = j + 1")
    loop.add_node(inner, is_start_block=True)
    loop.add_edge(inner, first, dace.InterstateEdge())
    body = inner.add_state("body", is_start_block=True)
    after = inner.add_state("after")
    inner.add_edge(body, after, dace.InterstateEdge(assignments={"s": "s - 1"}))
    facts = SymbolResolver().facts_at(body)
    assert symbolic.ask(le("1", "s"), facts) is symbolic.Truth.UNKNOWN


def assigned_before_loop(assignments: dict[str, str], body_assignments: dict[str, str] | None = None):
    """``first --assignments--> loop(i < N) { body [--body_assignments--> after] }``, ``N`` positive; returns the
    SDFG, the loop and its body's start state."""
    sdfg = dace.SDFG(f"facts_assigned_{abs(hash((tuple(assignments.items()), str(body_assignments))))}")
    sdfg.add_symbol("N", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    for name in assignments:
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_array("la", [4], dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(first, loop, dace.InterstateEdge(assignments=assignments))
    body = loop.add_state("body", is_start_block=True)
    if body_assignments is not None:
        loop.add_edge(body, loop.add_state("after"), dace.InterstateEdge(assignments=body_assignments))
    return sdfg, loop, body


def test_a_symbol_an_edge_assigns_equals_its_value_after_the_edge():
    """``span = N + 2`` on the edge into a loop: in the body ``span >= 3``, the bound the frontend's ``ceil`` spans and
    ``la_max + lb_max + 1`` extents need."""
    _, loop, body = assigned_before_loop({"span": "N + 2"})
    for block in (loop, body):
        assert symbolic.ask(le("3", "span"), SymbolResolver().facts_at(block)) is symbolic.Truth.TRUE


def test_an_assigned_value_may_read_a_container_nothing_writes_since():
    _, _, body = assigned_before_loop({"m": "la[0] + 1"})
    assert symbolic.ask(le("la[0] + 1", "m"), SymbolResolver().facts_at(body)) is symbolic.Truth.TRUE


def test_an_assigned_value_reading_a_container_written_since_says_nothing():
    _, loop, body = assigned_before_loop({"m": "la[0] + 1"})
    body.add_edge(body.add_tasklet("t", {}, {"o"}, "o = 7"), "o", body.add_write("la"), None, dace.Memlet("la[0]"))
    for block in (loop, body):
        assert symbolic.ask(le("la[0] + 1", "m"), SymbolResolver().facts_at(block)) is symbolic.Truth.UNKNOWN


def test_a_loop_body_reassigning_the_symbol_does_not_know_its_value():
    """Every iteration but the first starts after ``span = span + 1``."""
    _, _, body = assigned_before_loop({"span": "N + 2"}, {"span": "span + 1"})
    assert symbolic.ask(le("3", "span"), SymbolResolver().facts_at(body)) is symbolic.Truth.UNKNOWN


def test_an_increment_is_not_an_equation():
    """``k = k + 1`` reads the old ``k``: as ``k == k + 1`` it would contradict and prove anything."""
    _, _, body = assigned_before_loop({"k": "k + 1"})
    assert symbolic.ask(le("k", "0"), SymbolResolver().facts_at(body)) is symbolic.Truth.UNKNOWN


def test_a_merge_knows_no_assignment_of_either_branch():
    sdfg = dace.SDFG("facts_assigned_merge")
    sdfg.add_symbol("N", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    sdfg.add_symbol("s", dace.int64)
    first = sdfg.add_state("first", is_start_block=True)
    left, right, merge = sdfg.add_state("left"), sdfg.add_state("right"), sdfg.add_state("merge")
    sdfg.add_edge(first, left, dace.InterstateEdge("N > 3", assignments={"s": "N + 2"}))
    sdfg.add_edge(first, right, dace.InterstateEdge("N <= 3", assignments={"s": "0"}))
    sdfg.add_edge(left, merge, dace.InterstateEdge())
    sdfg.add_edge(right, merge, dace.InterstateEdge())
    assert symbolic.ask(le("3", "s"), SymbolResolver().facts_at(left)) is symbolic.Truth.TRUE
    assert symbolic.ask(le("3", "s"), SymbolResolver().facts_at(merge)) is symbolic.Truth.UNKNOWN


def test_a_start_block_entered_again_by_a_back_edge_knows_none_of_its_assignments():
    """Unstructured ``A -> B -> A``: ``A`` also runs first, before ``B -> A`` assigned ``t``; the walk ends."""
    sdfg = dace.SDFG("facts_assigned_back_edge")
    sdfg.add_symbol("N", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    for name in ("s", "t"):
        sdfg.add_symbol(name, dace.int64)
    a = sdfg.add_state("a", is_start_block=True)
    b, c = sdfg.add_state("b"), sdfg.add_state("c")
    sdfg.add_edge(a, b, dace.InterstateEdge(assignments={"s": "N + 2"}))
    sdfg.add_edge(b, a, dace.InterstateEdge("s < 0", assignments={"t": "5"}))
    sdfg.add_edge(b, c, dace.InterstateEdge("s >= 0"))
    assert symbolic.ask(le("5", "t"), SymbolResolver().facts_at(a)) is symbolic.Truth.UNKNOWN
    assert symbolic.ask(le("3", "s"), SymbolResolver().facts_at(c)) is symbolic.Truth.TRUE


def branches(*conditions: str | None, **symbols) -> list[ControlFlowRegion]:
    """``if conditions[0]: ... elif conditions[1]: ... else: ...`` over ``k`` and ``N`` (integers), ``M`` (positive)
    and ``symbols``; returns the branches, each holding one start state."""
    sdfg = dace.SDFG(f"facts_branches_{abs(hash((conditions, tuple(symbols))))}")
    sdfg.add_symbol("M", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE}))
    for name, dtype in {"k": dace.int64, "N": dace.int64, **symbols}.items():
        sdfg.add_symbol(name, dtype)
    sdfg.add_array("la", [4], dace.int64)
    conditional = ConditionalBlock("cond")
    sdfg.add_node(conditional, is_start_block=True)
    regions = []
    for index, condition in enumerate(conditions):
        region = ControlFlowRegion(f"branch_{index}", sdfg=sdfg)
        region.add_state(f"start_{index}", is_start_block=True)
        conditional.add_branch(condition, region)
        regions.append(region)
    return regions


def test_a_branch_knows_its_condition_and_the_failed_earlier_ones():
    """``if k < N`` / ``elif k < 2 * N`` / ``else``: each branch starts knowing its condition and the negation of each
    earlier one."""
    first, second, last = branches("k < N", "k < 2 * N", None)
    facts = [SymbolResolver().facts_at(region.start_block) for region in (first, second, last)]
    assert symbolic.ask(le("k + 1", "N"), facts[0]) is symbolic.Truth.TRUE
    assert symbolic.ask(le("N", "k"), facts[1]) is symbolic.Truth.TRUE
    assert symbolic.ask(le("k + 1", "2 * N"), facts[1]) is symbolic.Truth.TRUE
    assert symbolic.ask(le("2 * N", "k"), facts[2]) is symbolic.Truth.TRUE


def test_the_else_of_a_disjunction_knows_each_negated_part():
    """``if k < 0 or k >= N``: the branch knows neither part, the else both negations, ``0 <= k < N``."""
    then, orelse = branches("k < 0 or k >= N", None)
    assert symbolic.ask(le("0", "k"), SymbolResolver().facts_at(then.start_block)) is symbolic.Truth.UNKNOWN
    facts = SymbolResolver().facts_at(orelse.start_block)
    assert symbolic.ask(le("0", "k"), facts) is symbolic.Truth.TRUE
    assert symbolic.ask(le("k + 1", "N"), facts) is symbolic.Truth.TRUE


def test_a_branch_condition_ends_where_the_branch_reassigns_its_symbol():
    (then,) = branches("k < N")
    later = then.add_state("later")
    then.add_edge(then.start_block, later, dace.InterstateEdge(assignments={"k": "N + 5"}))
    assert symbolic.ask(le("k + 1", "N"), SymbolResolver().facts_at(later)) is symbolic.Truth.FALSE
    assert symbolic.ask(le("k + 1", "N"), SymbolResolver().facts_at(then.start_block)) is symbolic.Truth.TRUE


def test_a_branch_condition_ends_where_the_branch_writes_the_container_it_reads():
    (then,) = branches("la[0] < N")
    start = then.start_block
    later = then.add_state_after(start, "later")
    assert symbolic.ask(le("la[0]", "N"), SymbolResolver().facts_at(later)) is symbolic.Truth.TRUE
    start.add_edge(start.add_tasklet("t", {}, {"o"}, "o = 7"), "o", start.add_write("la"), None, dace.Memlet("la[0]"))
    assert symbolic.ask(le("la[0]", "N"), SymbolResolver().facts_at(later)) is symbolic.Truth.UNKNOWN


def test_a_failed_comparison_of_a_float_is_no_fact():
    """``x < N`` fails for a NaN ``x`` too, so the else knows nothing; the branch itself knows ``x < N``."""
    then, orelse = branches("x < N", None, x=dace.float64)
    assert symbolic.ask(le("x", "N"), SymbolResolver().facts_at(then.start_block)) is symbolic.Truth.TRUE
    assert symbolic.ask(le("N", "x"), SymbolResolver().facts_at(orelse.start_block)) is symbolic.Truth.UNKNOWN


@pytest.mark.parametrize("value, known", [("N + 1", True), ("0.5 * N", False)])
def test_an_undeclared_symbol_negates_only_when_its_assignment_types_it_integer(value, known):
    """``z = value`` on the edge into ``if z < N`` declares ``z`` nowhere else: the else knows ``N <= z`` only when the
    assignment infers an integer, as a float ``z`` may be a NaN."""
    then, orelse = branches("z < N", None)
    sdfg, conditional = then.sdfg, then.parent_graph
    first = sdfg.add_state("first", is_start_block=True)
    sdfg.add_edge(first, conditional, dace.InterstateEdge(assignments={"z": value}))
    truth = symbolic.ask(le("N", "z"), SymbolResolver().facts_at(orelse.start_block))
    assert truth is (symbolic.Truth.TRUE if known else symbolic.Truth.UNKNOWN)


def test_a_failed_comparison_of_a_function_value_is_no_fact():
    """``f(k)`` has no type to resolve: the else of ``f(k) < N`` knows nothing."""
    _, orelse = branches("f(k) < N", None)
    assert symbolic.ask(le("N", "f(k)"), SymbolResolver().facts_at(orelse.start_block)) is symbolic.Truth.UNKNOWN


def test_a_branch_the_facts_prove_dead_knows_less_rather_than_a_contradiction():
    """``M`` is positive, so the else of ``M > 0`` never runs: its ``M <= 0`` is dropped, not a contradiction."""
    _, orelse = branches("M > 0", None)
    assert symbolic.ask(le("1", "M"), SymbolResolver().facts_at(orelse.start_block)) is symbolic.Truth.TRUE


if __name__ == "__main__":
    test_loop_body_knows_the_iterator_range()
    test_loop_header_does_not_know_the_iterator_range()
    test_map_body_knows_the_parameter_range()
    test_map_entry_does_not_know_its_own_parameter_range()
    test_step_of_unknown_sign_gives_no_range()
    test_step_positive_by_the_sdfg_facts_gives_the_range()
    test_step_negative_by_the_sdfg_facts_gives_the_reversed_range()
    test_a_range_empty_by_the_facts_yields_facts_rather_than_a_contradiction()
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
    test_a_dimension_scaling_one_symbol_makes_it_nonnegative("2 * N", True)
    test_a_dimension_scaling_one_symbol_makes_it_nonnegative("N + 1", False)
    test_a_dimension_scaling_one_symbol_makes_it_nonnegative("M * N", False)
    test_a_range_needs_a_bound_the_body_cannot_write()
    test_a_range_may_read_a_container_the_loop_does_not_write()
    test_a_range_reading_a_container_the_loop_writes_knows_nothing()
    test_a_symbolic_loop_step_gives_a_range_when_the_facts_orient_it(frozenset({symbolic.Predicate.POSITIVE}), True)
    test_a_symbolic_loop_step_gives_a_range_when_the_facts_orient_it(frozenset(), False)
    test_a_range_needs_an_init_that_does_not_read_the_iterator()
    test_a_range_needs_a_condition_along_the_step()
    test_a_float_limit_gives_no_integer_range()
    test_a_float_iterator_gives_no_integer_range()
    test_a_loop_header_ends_a_fact_on_entry()
    test_inlining_keeps_the_assumptions_proofs_took()
    test_while_body_start_knows_the_condition()
    test_a_block_after_the_condition_symbol_is_reassigned_does_not_know_it()
    test_an_inner_loop_reassigning_the_condition_symbol_does_not_know_it()
    test_a_symbol_an_edge_assigns_equals_its_value_after_the_edge()
    test_an_assigned_value_may_read_a_container_nothing_writes_since()
    test_an_assigned_value_reading_a_container_written_since_says_nothing()
    test_a_loop_body_reassigning_the_symbol_does_not_know_its_value()
    test_an_increment_is_not_an_equation()
    test_a_merge_knows_no_assignment_of_either_branch()
    test_a_start_block_entered_again_by_a_back_edge_knows_none_of_its_assignments()
    test_a_branch_knows_its_condition_and_the_failed_earlier_ones()
    test_the_else_of_a_disjunction_knows_each_negated_part()
    test_a_branch_condition_ends_where_the_branch_reassigns_its_symbol()
    test_a_branch_condition_ends_where_the_branch_writes_the_container_it_reads()
    test_a_failed_comparison_of_a_float_is_no_fact()
    test_a_branch_the_facts_prove_dead_knows_less_rather_than_a_contradiction()
    test_an_undeclared_symbol_negates_only_when_its_assignment_types_it_integer("N + 1", True)
    test_an_undeclared_symbol_negates_only_when_its_assignment_types_it_integer("0.5 * N", False)
    test_a_failed_comparison_of_a_function_value_is_no_fact()

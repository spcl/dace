# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import json
import pathlib
import re

import pytest
import sympy

import dace
from dace import symbolic
from dace.sdfg.sdfg import NameCollisionError
from dace.sdfg.state import LoopRegion
from dace.sdfg.validation import InvalidSDFGError

POSITIVE = frozenset({symbolic.Predicate.POSITIVE})
NONNEGATIVE = frozenset({symbolic.Predicate.NONNEGATIVE})
N = dace.symbol("N")


@dace.program
def loops_program(A: dace.float64[N]):
    for i in range(N):
        A[i] = A[i] + 1
    for k in range(1, N):
        A[k] += A[k - 1]


def symbols_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_state(is_start_block=True)
    sdfg.add_symbol("N", dace.int64, predicates=POSITIVE)
    sdfg.add_symbol("K", dace.int64)
    return sdfg


def less_than(lhs: str, rhs: str) -> symbolic.Relation:
    return symbolic.Relation(symbolic.RelationKind.LT, symbolic.pystr_to_symbolic(lhs), symbolic.pystr_to_symbolic(rhs))


def test_identical_readd_is_a_no_op():
    sdfg = symbols_sdfg("identical_readd")
    assert sdfg.add_symbol("N", dace.int64, predicates=POSITIVE) == "N"
    assert sdfg.symbol_repo.params.predicates == {"N": POSITIVE}


def test_readd_with_other_predicates_raises():
    sdfg = symbols_sdfg("readd_other_predicates")
    with pytest.raises(symbolic.InconsistentAssumptionsError, match="Inconsistent facts about N"):
        sdfg.add_symbol("N", dace.int64, predicates=NONNEGATIVE)


def test_readd_with_other_type_raises():
    sdfg = symbols_sdfg("readd_other_type")
    with pytest.raises(symbolic.InconsistentAssumptionsError, match="re-added as int"):
        sdfg.add_symbol("K", dace.int32)


def test_add_symbol_takes_dtype_and_declared_assumptions_of_a_symbol():
    by_symbol = dace.SDFG("add_by_symbol")
    by_symbol.add_symbol(dace.symbol("Px", dtype=dace.int32, positive=True))
    by_name = dace.SDFG("add_by_name")
    by_name.add_symbol("Px", dace.int32, predicates=POSITIVE)
    assert by_symbol.symbols == by_name.symbols == {"Px": dace.int32}
    assert by_symbol.symbol_repo.params.predicates == by_name.symbol_repo.params.predicates == {"Px": POSITIVE}
    assert by_name.add_symbol(dace.symbol("Px", dtype=dace.int32, positive=True)) == "Px"


def test_symbol_and_descriptor_names_collide():
    sdfg = symbols_sdfg("name_collision")
    sdfg.add_array("A", [10], dace.float64)
    with pytest.raises(NameCollisionError, match="used by a data descriptor"):
        sdfg.add_symbol("A", dace.int64)
    with pytest.raises(NameCollisionError, match="used by a symbol"):
        sdfg.add_array("K", [10], dace.float64)


def test_contradicting_add_leaves_sdfg_unchanged():
    sdfg = symbols_sdfg("contradicting_add")
    with pytest.raises(symbolic.InconsistentAssumptionsError):
        sdfg.add_symbol(
            "M", dace.int64, predicates=frozenset({symbolic.Predicate.POSITIVE, symbolic.Predicate.NEGATIVE})
        )
    assert "M" not in sdfg.symbols
    assert "M" not in sdfg.symbol_repo.params.predicates


def test_set_symbol_assumptions():
    sdfg = symbols_sdfg("set_assumptions")
    with pytest.raises(KeyError, match="not declared"):
        sdfg.set_symbol_assumptions("M", POSITIVE)
    sdfg.set_symbol_assumptions("K", NONNEGATIVE)
    sdfg.set_symbol_assumptions("N", frozenset())
    assert sdfg.symbol_repo.params.predicates == {"K": NONNEGATIVE}


def test_add_symbol_relation():
    sdfg = symbols_sdfg("add_relation")
    with pytest.raises(KeyError, match='"M" is not declared'):
        sdfg.add_symbol_relation(less_than("K", "M"))
    sdfg.add_symbol_relation(less_than("K", "N"))
    sdfg.add_symbol_relation(less_than("K", "N"))
    assert sdfg.symbol_repo.params.relations == [less_than("K", "N")]
    with pytest.raises(symbolic.InconsistentAssumptionsError):
        sdfg.add_symbol_relation(less_than("N", "K"))
    assert sdfg.symbol_repo.params.relations == [less_than("K", "N")]


def test_remove_symbol_drops_predicates_and_refuses_named_ones():
    sdfg = symbols_sdfg("remove_symbol")
    sdfg.add_symbol("M", dace.int64, predicates=POSITIVE)
    sdfg.remove_symbol("M")
    assert "M" not in sdfg.symbol_repo.params.predicates
    sdfg.add_symbol_relation(less_than("K", "N"))
    with pytest.raises(ValueError, match="still name it"):
        sdfg.remove_symbol("K")


def test_rename_moves_predicates_and_rewrites_relations():
    sdfg = symbols_sdfg("rename_symbols")
    sdfg.add_symbol_relation(less_than("K", "N"))
    sdfg.replace_dict({"N": "M"})
    assert sdfg.symbol_repo.params.predicates == {"M": POSITIVE}
    assert [str(relation) for relation in sdfg.symbol_repo.params.relations] == [str(less_than("K", "M"))]


def test_replacing_by_expression_turns_predicates_into_relations():
    sdfg = symbols_sdfg("replace_by_expression")
    sdfg.replace_dict({"N": "K + 1"})
    assert "N" not in sdfg.symbol_repo.params.predicates
    assert sdfg.symbol_repo.params.relations == [
        symbolic.Relation(symbolic.RelationKind.LT, sympy.Integer(0), symbolic.pystr_to_symbolic("K + 1"))
    ]


def test_facts_take_integers_from_dtypes():
    sdfg = symbols_sdfg("facts_integers")
    sdfg.add_symbol("alpha", dace.float64)
    sdfg.add_symbol_relation(less_than("K", "N"))
    facts = sdfg.facts()
    assert facts.integers == frozenset({"N", "K"})
    query = symbolic.Relation(
        symbolic.RelationKind.LE, symbolic.pystr_to_symbolic("K"), symbolic.pystr_to_symbolic("N - 1")
    )
    assert symbolic.ask(query, facts) is symbolic.Truth.TRUE


def test_json_round_trip_keeps_facts():
    sdfg = symbols_sdfg("json_round_trip")
    sdfg.add_symbol_relation(less_than("K", "N"))
    loaded = dace.SDFG.from_json(json.loads(json.dumps(sdfg.to_json())))
    assert loaded.symbol_repo.params.predicates == {"N": POSITIVE}
    # Compared by name: a symbol's dtype is still part of its identity until symbols become names only
    assert [str(relation) for relation in loaded.symbol_repo.params.relations] == [str(less_than("K", "N"))]


def test_sdfg_without_facts_serializes_no_fact_keys():
    sdfg = dace.SDFG("no_facts")
    sdfg.add_state(is_start_block=True)
    sdfg.add_symbol("N", dace.int64)
    assert sdfg.to_json()["attributes"]["symbol_repo"] == {"params": {"types": {"N": "int64"}}}


def test_validation_rejects_facts_on_undeclared_symbols():
    sdfg = symbols_sdfg("undeclared_facts")
    sdfg.symbol_repo.params.predicates["M"] = POSITIVE
    with pytest.raises(InvalidSDFGError, match="undeclared symbols"):
        sdfg.validate()


def test_loop_variable_cannot_be_added_as_symbol():
    sdfg = dace.SDFG("loop_variable_symbol")
    loop = LoopRegion("loop", "i < 10", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    with pytest.raises(NameCollisionError, match="a loop or map scope binds it"):
        sdfg.add_symbol("i", dace.int64)


def test_frontend_loop_iterators_are_not_symbols():
    sdfg = loops_program.to_sdfg(simplify=False)
    sdfg.validate()
    assert set(sdfg.symbols) == {"N"}


def test_validation_rejects_symbol_bound_by_a_map():
    sdfg = dace.SDFG("map_parameter_symbol")
    sdfg.add_array("A", [10], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet("m", {"j": "0:10"}, {}, "a = 1.0", {"a": dace.Memlet("A[j]")}, external_edges=True)
    sdfg.symbol_repo.add("j", dace.int64)
    with pytest.raises(InvalidSDFGError, match="bound by a loop or map scope"):
        sdfg.validate()


def test_descriptor_sized_by_a_map_parameter_does_not_register_it():
    sdfg = dace.SDFG("scoped_shape")
    sdfg.add_array("A", [10], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet("m", {"j": "0:10"}, {}, "a = 1.0", {"a": dace.Memlet("A[j]")}, external_edges=True)
    sdfg.add_transient("T", [dace.symbol("j") + 1], dace.float64)
    assert "j" not in sdfg.symbols


def test_only_the_frontend_boundary_reads_symbol_declarations():
    """A symbol object's declaration is read where a user's symbol enters; everything else asks the SDFG."""
    root = pathlib.Path(dace.__file__).parent
    boundary = {"symbolic.py", "sdfg/sdfg.py", "sdfg/type_inference.py", "data/core.py", "data/creation.py"}
    readers = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if re.search(r"\.declaration\b", path.read_text(encoding="utf-8"))
    }
    assert {r for r in readers if not r.startswith("frontend/python/")} <= boundary


if __name__ == "__main__":
    test_loop_variable_cannot_be_added_as_symbol()
    test_frontend_loop_iterators_are_not_symbols()
    test_validation_rejects_symbol_bound_by_a_map()
    test_descriptor_sized_by_a_map_parameter_does_not_register_it()
    test_identical_readd_is_a_no_op()
    test_readd_with_other_predicates_raises()
    test_readd_with_other_type_raises()
    test_add_symbol_takes_dtype_and_declared_assumptions_of_a_symbol()
    test_symbol_and_descriptor_names_collide()
    test_contradicting_add_leaves_sdfg_unchanged()
    test_set_symbol_assumptions()
    test_add_symbol_relation()
    test_remove_symbol_drops_predicates_and_refuses_named_ones()
    test_rename_moves_predicates_and_rewrites_relations()
    test_replacing_by_expression_turns_predicates_into_relations()
    test_facts_take_integers_from_dtypes()
    test_json_round_trip_keeps_facts()
    test_sdfg_without_facts_serializes_no_fact_keys()
    test_validation_rejects_facts_on_undeclared_symbols()
    test_only_the_frontend_boundary_reads_symbol_declarations()

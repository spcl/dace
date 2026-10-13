# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""SMALL symbols: much smaller than every default symbol (``<<``, which implies ``<``), with each order a proof takes
from the scales checked when the program is called."""

import numpy as np
import pytest
import sympy

import dace
from dace import symbolic
from dace.symbolic import Relation, RelationKind, Scale, Truth
from dace.transformation.interstate import LoopToMap

n1, n2, N = (symbolic.symbol(name) for name in ("n1", "n2", "N"))


def scaled_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("n1", dace.int64, scale=Scale.SMALL)
    sdfg.add_symbol("n2", dace.int64, scale=Scale.SMALL)
    sdfg.add_symbol("N", dace.int64)
    return sdfg


def test_small_symbols_are_much_less_than_default_ones():
    sdfg = scaled_sdfg("scale_orders")
    facts = sdfg.facts()
    assert symbolic.ask(Relation(RelationKind.LT, n1 + n2, N), facts) is Truth.TRUE
    assert symbolic.ask(Relation(RelationKind.MUCH_LT, n1, N - 1), facts) is Truth.TRUE
    assert symbolic.ask(Relation(RelationKind.LE, N, n1), facts) is Truth.FALSE
    # ``10 - N`` shrinks as N grows, so nothing follows
    assert symbolic.ask(Relation(RelationKind.LT, n1, 10 - N), facts) is Truth.UNKNOWN
    assert list(sdfg.symbol_repo.runtime_assumptions) == [
        Relation(RelationKind.LT, n1 + n2, N),
        Relation(RelationKind.LT, n1, N - 1),
        Relation(RelationKind.LT, n1, N),
    ]


def test_scales_prove_nothing_without_a_place_to_record():
    """Facts that are not an SDFG's carry no scales, so no order is taken from them unchecked."""
    assert symbolic.ask(Relation(RelationKind.LT, n1 + n2, N), symbolic.Facts.none()) is Truth.UNKNOWN


def test_scales_and_assumptions_survive_serialization():
    sdfg = scaled_sdfg("scale_json")
    symbolic.ask(Relation(RelationKind.LT, n1 + n2, N), sdfg.facts())
    loaded = dace.SDFG.from_json(sdfg.to_json())
    assert list(loaded.symbol_repo.small) == ["n1", "n2"]
    assert list(loaded.symbol_repo.runtime_assumptions) == [Relation(RelationKind.LT, n1 + n2, N)]


def test_a_symbol_declared_small_in_the_program():
    k = dace.symbol("k", dace.int64, small=True)

    @dace.program
    def small_declared(A: dace.float64[k]):
        A[:] = 1.0

    assert list(small_declared.to_sdfg().symbol_repo.small) == ["k"]


def loop_over(extent: str) -> tuple[dace.SDFG, dace.sdfg.state.LoopRegion]:
    sdfg = scaled_sdfg(f"scale_loop_{extent.replace(' ', '').replace('+', '_')}")
    sdfg.add_array("A", ["N"], dace.float64)
    loop = dace.sdfg.state.LoopRegion("loop", f"i < {extent}", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    tasklet = body.add_tasklet("one", {}, {"o"}, "o = 1.0")
    body.add_edge(tasklet, "o", body.add_write("A"), None, dace.Memlet("A[i]"))
    return sdfg, loop


def test_loop_over_small_symbols_stays_sequential():
    sdfg, loop = loop_over("n1 + n2")
    assert not LoopToMap.can_be_applied_to(sdfg, loop=loop)
    assert LoopToMap.can_be_applied_to(sdfg, options={"keep_small_loops_sequential": False}, loop=loop)


def test_loop_over_a_default_symbol_becomes_a_map():
    sdfg, loop = loop_over("N")
    assert LoopToMap.can_be_applied_to(sdfg, loop=loop)


def checked_program(name: str) -> dace.SDFG:
    """Copies ``A[0:n1 + n2]`` to ``B``, with the order ``n1 + n2 < N`` taken from the scales, as a proof would."""
    sdfg = scaled_sdfg(name)
    sdfg.add_array("A", ["N"], dace.float64)
    sdfg.add_array("B", ["N"], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "copy",
        {"i": "0:n1 + n2"},
        {"a": dace.Memlet("A[i]")},
        "b = a",
        {"b": dace.Memlet("B[i]")},
        external_edges=True,
    )
    assert symbolic.ask(Relation(RelationKind.LT, n1 + n2, N), sdfg.facts()) is Truth.TRUE
    return sdfg


def test_call_checks_the_assumed_orders():
    csdfg = checked_program("scale_checked_call").compile()
    A, B = np.arange(20, dtype=np.float64), np.zeros(20)
    csdfg(A=A, B=B, n1=3, n2=4, N=20)
    assert np.array_equal(B[:7], A[:7])
    # n1 + n2 = 16 is not below N = 10
    A, B = np.arange(10, dtype=np.float64), np.zeros(10)
    with pytest.raises(ValueError, match=r"n1 \+ n2 < N"):
        csdfg(A=A, B=B, n1=8, n2=8, N=10)


def test_init_refuses_values_that_break_an_assumed_order():
    """A caller outside Python, such as a Fortran binding, gets a null handle from ``__dace_init``."""
    import ctypes

    sdfg = checked_program("scale_checked_init")
    csdfg = sdfg.compile()
    init = csdfg._lib.get_symbol(f"__dace_init_{sdfg.name}", restype=ctypes.c_void_p)
    # The init parameters are the free symbols, sorted by name
    assert init(ctypes.c_longlong(10), ctypes.c_longlong(8), ctypes.c_longlong(8)) is None


def test_a_cached_proof_records_its_assumption_in_every_sdfg():
    """Equal facts share cached proofs, but each SDFG still records the order its result relies on."""
    first, second = scaled_sdfg("scale_cache_first"), scaled_sdfg("scale_cache_second")
    for sdfg in (first, second):
        assert symbolic.simplify(sympy.Max(n1, N), sdfg.facts()) == N
        assert Relation(RelationKind.LT, n1, N) in sdfg.symbol_repo.runtime_assumptions
    facts = first.facts()
    unrecorded = symbolic.Facts(facts.relations, facts.integers, facts.small, facts.parameters)
    assert symbolic.simplify(sympy.Max(n1, N), unrecorded) == sympy.Max(n1, N)


if __name__ == "__main__":
    test_small_symbols_are_much_less_than_default_ones()
    test_scales_prove_nothing_without_a_place_to_record()
    test_scales_and_assumptions_survive_serialization()
    test_a_symbol_declared_small_in_the_program()
    test_loop_over_small_symbols_stays_sequential()
    test_loop_over_a_default_symbol_becomes_a_map()
    test_call_checks_the_assumed_orders()
    test_init_refuses_values_that_break_an_assumed_order()
    test_a_cached_proof_records_its_assumption_in_every_sdfg()

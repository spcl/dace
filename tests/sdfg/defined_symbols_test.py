# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFGState.defined_symbols`` types interstate assignments from one environment kept per call, and
``add_nested_sdfg`` takes that environment as an argument instead of recomputing it per nested SDFG."""

from unittest import mock

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion, SDFGState


def assigning_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("assigning")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", [dace.symbol("N")], dace.float32)
    sdfg.add_scalar("k", dace.float64, transient=True)
    first = sdfg.add_state("first", is_start_block=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    loop.add_state("body", is_start_block=True)
    sdfg.add_node(loop)
    last = sdfg.add_state("last")
    # ``m`` is typed from ``N``, ``w`` from the array element, ``k`` shadows the scalar descriptor and
    # ``z`` is typed from ``k`` -- which must resolve to the DESCRIPTOR's dtype, not the symbol's.
    sdfg.add_edge(first, loop, InterstateEdge(assignments={"m": "N + 1", "w": "A[0]"}))
    sdfg.add_edge(loop, last, InterstateEdge(assignments={"k": "m * 2", "z": "k + 1"}))
    return sdfg


def reference_defined_symbols(state: dace.SDFGState) -> dict:
    """The per-edge environment the method used to rebuild, kept verbatim as the oracle."""
    sdfg = state.sdfg
    defined = dict(sdfg.symbols)
    for desc in sdfg.arrays.values():
        for sym in desc.free_symbols:
            if sym.dtype is not None:
                defined.setdefault(str(sym), sym.dtype)
    for edge in sdfg.all_interstate_edges():
        defined.update({k: v for k, v in edge.data.new_symbols(sdfg, defined).items() if v is not None})
        if isinstance(edge.dst, LoopRegion):
            defined.update({k: v for k, v in edge.dst.new_symbols(defined).items() if v is not None})
    return defined


def test_defined_symbols_types_every_assignment_as_before():
    sdfg = assigning_sdfg()
    state = next(s for s in sdfg.states() if s.label == "last")
    assert state.defined_symbols() == reference_defined_symbols(state)
    assert state.defined_symbols()["z"] == dace.float64
    # The declared ``N: int64`` wins over the default-typed ``N`` of the array extent.
    assert state.defined_symbols()["N"] == dace.int64


def test_defined_symbols_builds_no_type_environment_per_edge():
    """``new_symbols(sdfg, ...)`` copies the symbols and every array's dtype per assigning edge:
    2.4 s per call on warpx_field_gather, once per LoopToMap lift that nests a body."""
    sdfg = assigning_sdfg()
    state = next(s for s in sdfg.states() if s.label == "last")
    with mock.patch.object(
        InterstateEdge, "new_symbols", autospec=True, side_effect=InterstateEdge.new_symbols
    ) as new_symbols:
        state.defined_symbols()
    assert new_symbols.called
    assert [c for c in new_symbols.call_args_list if c.args[1] is not None] == []


def nesting_parts() -> tuple[dace.SDFGState, dace.SDFG]:
    """The state of ``nesting`` — which declares the symbol ``N`` — and ``nested``, whose tasklet uses
    the free symbol ``n``; ``add_nested_sdfg`` maps ``n`` to ``N`` and types the nested SDFG's ``n``
    from the symbol environment that call uses."""
    nested = dace.SDFG("nested")
    nested.add_array("a", [1], dace.float64)
    nested_state = nested.add_state()
    tasklet = nested_state.add_tasklet("w", {}, {"o"}, "o = n")
    nested_state.add_edge(tasklet, "o", nested_state.add_write("a"), None, dace.Memlet("a[0]"))

    nesting = dace.SDFG("nesting")
    nesting.add_symbol("N", dace.int16)
    return nesting.add_state(), nested


def test_supplied_defined_symbols_type_the_nested_sdfg_without_recomputing():
    """A cached environment types the nested SDFG's free symbol like the computed one, without a second
    ``defined_symbols`` scan of every data descriptor (gt4py issue 2971: a split-fused program that
    nests many SDFGs into one state)."""
    state, nested = nesting_parts()
    cached = state.defined_symbols()

    with mock.patch.object(
        SDFGState, "defined_symbols", autospec=True, side_effect=SDFGState.defined_symbols
    ) as defined_symbols:
        state.add_nested_sdfg(nested, {}, {"a"}, symbol_mapping={"n": "N"}, defined_symbols=cached)

    assert not defined_symbols.called
    assert nested.symbols["n"] == dace.int16


def test_empty_supplied_defined_symbols_fall_back_to_the_symbol_default():
    """A supplied empty environment is the detached-state default (``self.sdfg is None``): the free
    symbol keeps its own default type instead of the parent's, and no ``defined_symbols`` scan happens."""
    state, nested = nesting_parts()

    with mock.patch.object(
        SDFGState, "defined_symbols", autospec=True, side_effect=SDFGState.defined_symbols
    ) as defined_symbols:
        state.add_nested_sdfg(nested, {}, {"a"}, symbol_mapping={"n": "N"}, defined_symbols={})

    assert not defined_symbols.called
    assert nested.symbols["n"] != dace.int16, "typed from the parent, not the default"


def test_defined_symbols_are_recomputed_when_not_supplied():
    """Without the argument, ``add_nested_sdfg`` computes the environment itself and types the nested
    SDFG's free symbol from it."""
    state, nested = nesting_parts()

    with mock.patch.object(
        SDFGState, "defined_symbols", autospec=True, side_effect=SDFGState.defined_symbols
    ) as defined_symbols:
        state.add_nested_sdfg(nested, {}, {"a"}, symbol_mapping={"n": "N"})

    assert defined_symbols.called
    assert nested.symbols["n"] == dace.int16


if __name__ == "__main__":
    test_defined_symbols_types_every_assignment_as_before()
    test_defined_symbols_builds_no_type_environment_per_edge()
    test_supplied_defined_symbols_type_the_nested_sdfg_without_recomputing()
    test_empty_supplied_defined_symbols_fall_back_to_the_symbol_default()
    test_defined_symbols_are_recomputed_when_not_supplied()

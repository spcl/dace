# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Implementation selection for CPU and cuTile nodes."""

import pytest

import dace
from dace.libraries.tileops import TileBinop, TileMaskGen
from dace.libraries.tileops import dispatch
from dace.libraries.tileops.dispatch import ISA, ISA_TO_IMPL, detect_host_isa, select_tile_implementation


def wired_binop(widths: tuple[int, ...], isa: ISA) -> tuple[TileBinop, dace.SDFGState]:
    sdfg = dace.SDFG("tile_dispatch")
    state = sdfg.add_state()
    for name in ("A", "B", "C"):
        sdfg.add_array(name, widths, dace.float64, transient=True, storage=dace.StorageType.Register)
    node = TileBinop("binop", widths, op="+")
    node.target_isa = isa
    state.add_node(node)
    for name, connector in (("A", "_a"), ("B", "_b")):
        state.add_edge(state.add_read(name), None, node, connector, dace.Memlet.from_array(name, sdfg.arrays[name]))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return node, state


@pytest.mark.parametrize("isa", list(ISA))
@pytest.mark.parametrize("widths", [(8, 4), (4, 4, 2)])
def test_multidimensional_selection(widths: tuple[int, ...], isa: ISA):
    node, state = wired_binop(widths, isa)
    assert select_tile_implementation(node, state) == ("cutile" if isa is ISA.CUTILE else "pure")


@pytest.mark.parametrize("isa, expected", [*ISA_TO_IMPL.items(), (ISA.CUTILE, "cutile")])
def test_single_dimensional_selection(isa: ISA, expected: str, monkeypatch):
    # Selection only: no instructions from a foreign architecture are executed.
    monkeypatch.setattr(dispatch, "host_supported_isas", lambda: frozenset(ISA))
    node, state = wired_binop((8,), isa)
    assert select_tile_implementation(node, state) == expected


def test_unknown_isa_is_rejected():
    node, _ = wired_binop((8,), ISA.SCALAR)
    with pytest.raises(TypeError):
        node.target_isa = "FOO"


def test_default_is_scalar():
    node, state = wired_binop((8,), ISA.SCALAR)
    assert TileBinop("default", (8,)).target_isa is ISA.SCALAR
    assert select_tile_implementation(node, state) == "scalar"


def test_auto_uses_host_isa():
    node, state = wired_binop((8,), ISA.AUTO)
    assert select_tile_implementation(node, state) == ISA_TO_IMPL[detect_host_isa()]


def test_cutile_mask_selection():
    sdfg = dace.SDFG("mask_dispatch")
    state = sdfg.add_state()
    node = TileMaskGen("mask", (8,), iter_vars=("i",), global_ubs=("N",))
    node.target_isa = ISA.CUTILE
    state.add_node(node)
    assert select_tile_implementation(node, state) == "cutile"

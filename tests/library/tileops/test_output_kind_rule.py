# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the design section 6.2 output-kind rule on tile-op lib nodes.

Per user direction 2026-06-09:

* Any Tile input -> the output ``_c`` / ``_o`` must be tile-shape.
* All inputs Scalar / Symbol -> the output may be Scalar / length-1 (preferred,
  for compositional chains) OR tile-shape (allowed).

Both ``validate()`` enforcement and the pure expansion's Scalar-output branch
are exercised here.
"""
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileITE, TileUnop
from dace.memlet import Memlet


def _wire_binop(kind_a, kind_b, out_shape, src_dtype=dace.float64):
    sdfg = dace.SDFG("binop_fixture")
    sdfg.add_array("A", (8, ), src_dtype, transient=True)
    sdfg.add_array("B", (8, ), src_dtype, transient=True)
    if out_shape == "scalar":
        sdfg.add_scalar("C", dace.float64, transient=True)
        c_memlet = Memlet("C")
    elif out_shape == "len1":
        sdfg.add_array("C", (1, ), dace.float64, transient=True)
        c_memlet = Memlet("C[0]")
    else:
        sdfg.add_array("C", (8, ), dace.float64, transient=True)
        c_memlet = Memlet("C[0:8]")
    state = sdfg.add_state("s")
    a_an = state.add_access("A")
    b_an = state.add_access("B")
    c_an = state.add_access("C")
    node = TileBinop("bn", widths=(8, ), op="+", kind_a=kind_a, kind_b=kind_b)
    state.add_node(node)
    state.add_edge(a_an, None, node, "_a", Memlet("A[0:8]"))
    state.add_edge(b_an, None, node, "_b", Memlet("B[0:8]"))
    state.add_edge(node, "_c", c_an, None, c_memlet)
    return sdfg, state, node


def test_binop_validate_refuses_scalar_output_when_any_input_is_tile():
    """Any Tile input + Scalar/length-1 output -> validate raises NotImplementedError."""
    sdfg, state, node = _wire_binop("Tile", "Tile", out_shape="scalar")
    with pytest.raises(NotImplementedError, match=r"output-kind rule violated"):
        node.validate(sdfg, state)
    sdfg, state, node = _wire_binop("Tile", "Scalar", out_shape="len1")
    sdfg.arrays["B"] = dace.data.Scalar(dtype=dace.float64, transient=True)
    state.in_edges(node)[1].data = Memlet("B")
    with pytest.raises(NotImplementedError, match=r"output-kind rule violated"):
        node.validate(sdfg, state)


def test_binop_validate_accepts_scalar_output_when_all_inputs_non_tile():
    """All Scalar/Symbol inputs + Scalar output -> validate passes."""
    sdfg = dace.SDFG("binop_scalar_out")
    sdfg.add_scalar("A", dace.float64, transient=True)
    sdfg.add_scalar("B", dace.float64, transient=True)
    sdfg.add_scalar("C", dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileBinop("bn", widths=(8, ), op="+", kind_a="Scalar", kind_b="Scalar")
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", Memlet("A"))
    state.add_edge(state.add_access("B"), None, node, "_b", Memlet("B"))
    state.add_edge(node, "_c", state.add_access("C"), None, Memlet("C"))
    node.validate(sdfg, state)


def test_binop_pure_expansion_emits_single_assignment_for_scalar_output():
    """For Scalar+Scalar inputs + Scalar output the pure expansion is a single
    assignment, no K-fold lane loop."""
    sdfg = dace.SDFG("binop_pure_scalar")
    sdfg.add_scalar("A", dace.float64, transient=True)
    sdfg.add_scalar("B", dace.float64, transient=True)
    sdfg.add_scalar("C", dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileBinop("bn", widths=(8, ), op="+", kind_a="Scalar", kind_b="Scalar")
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", Memlet("A"))
    state.add_edge(state.add_access("B"), None, node, "_b", Memlet("B"))
    state.add_edge(node, "_c", state.add_access("C"), None, Memlet("C"))
    sdfg.expand_library_nodes()
    state = next(s for s in sdfg.states())
    tasklet = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    code = tasklet.code.as_string
    assert "for" not in code, f"expected no tile loop in Scalar-output body, got:\n{code}"
    assert code.strip().startswith("_c"), f"expected `_c = ...` assignment, got:\n{code}"


def test_binop_pure_expansion_keeps_tile_loop_when_any_input_is_tile():
    """Tile + Scalar inputs + Tile output -> body has the K-fold lane loop."""
    sdfg = dace.SDFG("binop_pure_tile")
    sdfg.add_array("A", (8, ), dace.float64, transient=True)
    sdfg.add_scalar("B", dace.float64, transient=True)
    sdfg.add_array("C", (8, ), dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileBinop("bn", widths=(8, ), op="+", kind_a="Tile", kind_b="Scalar")
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", Memlet("A[0:8]"))
    state.add_edge(state.add_access("B"), None, node, "_b", Memlet("B"))
    state.add_edge(node, "_c", state.add_access("C"), None, Memlet("C[0:8]"))
    sdfg.expand_library_nodes()
    state = next(s for s in sdfg.states())
    tasklet = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    code = tasklet.code.as_string
    assert "for" in code, f"expected a K-fold lane loop in Tile-output body, got:\n{code}"


def test_unop_validate_refuses_scalar_output_when_input_is_tile():
    """Tile input + Scalar output -> validate raises NotImplementedError."""
    sdfg = dace.SDFG("unop_bad")
    sdfg.add_array("A", (8, ), dace.float64, transient=True)
    sdfg.add_scalar("C", dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileUnop("un", widths=(8, ), op="abs", kind_a="Tile")
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", Memlet("A[0:8]"))
    state.add_edge(node, "_c", state.add_access("C"), None, Memlet("C"))
    with pytest.raises(NotImplementedError, match=r"output-kind rule violated"):
        node.validate(sdfg, state)


def test_unop_pure_expansion_emits_single_assignment_for_scalar_output():
    """Scalar input + Scalar output -> single assignment, no lane loop."""
    sdfg = dace.SDFG("unop_pure_scalar")
    sdfg.add_scalar("A", dace.float64, transient=True)
    sdfg.add_scalar("C", dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileUnop("un", widths=(8, ), op="abs", kind_a="Scalar")
    state.add_node(node)
    state.add_edge(state.add_access("A"), None, node, "_a", Memlet("A"))
    state.add_edge(node, "_c", state.add_access("C"), None, Memlet("C"))
    sdfg.expand_library_nodes()
    state = next(s for s in sdfg.states())
    tasklet = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet))
    code = tasklet.code.as_string
    assert "for" not in code
    assert code.strip().startswith("_c")


def test_ite_validate_refuses_scalar_output():
    """TileITE inputs are implicitly Tile; Scalar output -> validate raises."""
    sdfg = dace.SDFG("ite_bad")
    sdfg.add_array("Cond", (8, ), dace.bool_, transient=True)
    sdfg.add_array("T", (8, ), dace.float64, transient=True)
    sdfg.add_array("E", (8, ), dace.float64, transient=True)
    sdfg.add_scalar("O", dace.float64, transient=True)
    state = sdfg.add_state("s")
    node = TileITE("ite", widths=(8, ))
    state.add_node(node)
    state.add_edge(state.add_access("Cond"), None, node, "_cond", Memlet("Cond[0:8]"))
    state.add_edge(state.add_access("T"), None, node, "_t", Memlet("T[0:8]"))
    state.add_edge(state.add_access("E"), None, node, "_e", Memlet("E[0:8]"))
    state.add_edge(node, "_o", state.add_access("O"), None, Memlet("O"))
    with pytest.raises(NotImplementedError, match=r"output-kind rule violated"):
        node.validate(sdfg, state)

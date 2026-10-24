# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Inlining a nested SDFG reuses a constant the parent already holds, and renames only a genuine clash."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import StateSubgraphView
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.interstate.sdfg_nesting import InlineSDFG

VALUES = np.arange(4, dtype=np.float64)


def sdfg_with_a_constant(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_constant("C", VALUES.copy(), dace.data.Array(dace.float64, [4], transient=True))
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", {}, {"o"}, "o = 1.0")
    state.add_edge(tasklet, "o", state.add_write("A"), None, dace.Memlet("A[0]"))
    return sdfg


def inline(xform_type: type, sdfg: dace.SDFG, state: dace.SDFGState, node: dace.nodes.NestedSDFG) -> None:
    xform = xform_type()
    xform.setup_match(sdfg, sdfg.cfg_id, sdfg.node_id(state), {xform_type.nested_sdfg: state.node_id(node)}, 0)
    xform.apply(state, sdfg)


@pytest.mark.parametrize("xform_type", [InlineMultistateSDFG, InlineSDFG])
def test_nesting_and_inlining_again_and_again_keeps_one_constant(xform_type: type) -> None:
    """Every round copied the parent's constants into the nested SDFG and inlining renamed each copy apart: the
    constants doubled per round (cp2k_density_matrix_trs4 never finished canonicalizing)."""
    sdfg = sdfg_with_a_constant(f"constant_rounds_{xform_type.__name__}")
    for _ in range(8):
        state = next(s for s in sdfg.states() if any(isinstance(n, dace.nodes.Tasklet) for n in s.nodes()))
        node = nest_state_subgraph(sdfg, state, StateSubgraphView(state, state.nodes()))
        inline(xform_type, sdfg, state, node)
    assert list(sdfg.constants_prop) == ["C"], list(sdfg.constants_prop)
    assert np.array_equal(sdfg.constants["C"], VALUES)
    sdfg.validate()


@pytest.mark.parametrize("xform_type", [InlineMultistateSDFG, InlineSDFG])
def test_a_nested_constant_of_another_value_is_renamed_apart(xform_type: type) -> None:
    sdfg = sdfg_with_a_constant(f"constant_clash_{xform_type.__name__}")
    state = next(iter(sdfg.states()))
    node = nest_state_subgraph(sdfg, state, StateSubgraphView(state, state.nodes()))
    node.sdfg.constants_prop["C"] = (dace.data.Array(dace.float64, [4], transient=True), VALUES + 1.0)
    inline(xform_type, sdfg, state, node)
    assert len(sdfg.constants_prop) == 2
    assert np.array_equal(sdfg.constants_prop["C"][1], VALUES)
    (renamed,) = [name for name in sdfg.constants_prop if name != "C"]
    assert np.array_equal(sdfg.constants_prop[renamed][1], VALUES + 1.0)


if __name__ == "__main__":
    test_nesting_and_inlining_again_and_again_keeps_one_constant(InlineMultistateSDFG)
    test_nesting_and_inlining_again_and_again_keeps_one_constant(InlineSDFG)
    test_a_nested_constant_of_another_value_is_renamed_apart(InlineMultistateSDFG)
    test_a_nested_constant_of_another_value_is_renamed_apart(InlineSDFG)

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FoldConstantTables on its own: which fills become constants, and that the program still computes the same."""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import FoldConstantTables

TABLE = (2.0, 3.0, 4.0, -99.0)


def table_then_use(skip: int = -1, looped: bool = False, rewrite: bool = False) -> dace.SDFG:
    """``table[k] = TABLE[k]`` by one tasklet per element, then ``out[i] = table[i] * 2``.

    ``skip`` leaves one element unfilled, ``looped`` puts the fill in a loop, ``rewrite`` writes ``table[0]`` again
    in the second state.
    """
    sdfg = dace.SDFG(f"table_then_use_{skip + 1}_{looped}_{rewrite}")
    sdfg.add_array("out", [len(TABLE)], dace.float64)
    sdfg.add_array("table", [len(TABLE)], dace.float64, transient=True)
    if looped:
        region = LoopRegion("twice", "r < 2", "r", "r = 0", "r = r + 1")
        sdfg.add_node(region, is_start_block=True)
        fill = region.add_state("fill", is_start_block=True)
    else:
        region = fill = sdfg.add_state("fill", is_start_block=True)
    table = fill.add_write("table")
    for index, value in enumerate(TABLE):
        if index != skip:
            tasklet = fill.add_tasklet(f"fill_{index}", {}, {"o"}, f"o = {value}")
            fill.add_edge(tasklet, "o", table, None, dace.Memlet(f"table[{index}]"))
    use = sdfg.add_state("use")
    sdfg.add_edge(region, use, dace.InterstateEdge())
    if rewrite:
        tasklet = use.add_tasklet("rewrite", {}, {"o"}, "o = 7.0")
        use.add_edge(tasklet, "o", use.add_write("table"), None, dace.Memlet("table[0]"))
    use.add_mapped_tasklet(
        "double",
        {"i": f"0:{len(TABLE)}"},
        {"t": dace.Memlet("table[i]")},
        "o = t * 2.0",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_table_filled_once_with_literals_becomes_a_constant():
    sdfg = table_then_use()
    folded = FoldConstantTables().apply_pass(sdfg, {})
    sdfg.validate()
    assert list(folded) == ["table"]
    assert list(sdfg.constants["table"]) == list(TABLE)
    fill = next(state for state in sdfg.states() if state.label == "fill")
    assert fill.number_of_nodes() == 0, fill.nodes()
    out = np.zeros(len(TABLE))
    sdfg(out=out)
    np.testing.assert_array_equal(out, np.array(TABLE) * 2.0)


@pytest.mark.parametrize("options", [{"skip": 1}, {"looped": True}, {"rewrite": True}])
def test_a_table_not_filled_once_with_literals_is_left_alone(options):
    """A partial fill, a fill that runs more than once, and a table written again are not compile-time constants."""
    sdfg = table_then_use(**options)
    assert FoldConstantTables().apply_pass(sdfg, {}) is None
    assert "table" not in sdfg.constants

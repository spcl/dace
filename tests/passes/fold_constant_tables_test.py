# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A table filled once with literals becomes an SDFG constant; any other write keeps it a container."""

import numpy as np
import pytest

import dace
from dace.sdfg.graph import SubgraphView
from dace.transformation import helpers as xfh
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.fold_constant_tables import FoldConstantTables

N = 8
TABLE = (1, 2, 1, 2, 0)


def lookup(sdfg: dace.SDFG, state: dace.SDFGState) -> None:
    """``out[i] = iphase[i % 5]`` in ``state``."""
    state.add_mapped_tasklet(
        "lookup",
        {"i": f"0:{N}"},
        {"t": dace.Memlet("iphase[i % 5]")},
        "o = t",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )


def table_then_lookup(overwrite: bool = False, nested: bool = False) -> dace.SDFG:
    """``iphase[0:5] = 1, 2, 1, 2, 0`` by input-less tasklets, then ``out[i] = iphase[i % 5]``, in a nested SDFG
    that receives the table when ``nested``; ``overwrite`` writes ``iphase[0]`` again in the second state."""
    sdfg = dace.SDFG("table_then_lookup" + ("_overwritten" if overwrite else "") + ("_nested" if nested else ""))
    sdfg.add_array("iphase", [5], dace.int32, transient=True)
    sdfg.add_array("out", [N], dace.int32)
    fill = sdfg.add_state("fill", is_start_block=True)
    table = fill.add_write("iphase")
    for index, value in enumerate(TABLE):
        tasklet = fill.add_tasklet(f"set_{index}", {}, {"o"}, f"o = {value}")
        fill.add_edge(tasklet, "o", table, None, dace.Memlet(f"iphase[{index}]"))
    use = sdfg.add_state_after(fill, "use")
    if overwrite:
        tasklet = use.add_tasklet("reset", {}, {"o"}, "o = 7")
        use.add_edge(tasklet, "o", use.add_write("iphase"), None, dace.Memlet("iphase[0]"))
        use = sdfg.add_state_after(use, "lookup")
    if nested:
        body = dace.SDFG("body")
        body.add_array("iphase", [5], dace.int32)
        body.add_array("out", [N], dace.int32)
        lookup(body, body.add_state())
        node = use.add_nested_sdfg(body, {"iphase"}, {"out"})
        use.add_edge(use.add_read("iphase"), None, node, "iphase", dace.Memlet("iphase[0:5]"))
        use.add_edge(node, "out", use.add_write("out"), None, dace.Memlet(f"out[0:{N}]"))
    else:
        lookup(sdfg, use)
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("nested", [False, True])
def test_a_literal_table_becomes_a_constant_and_computes_the_same(nested: bool):
    sdfg = table_then_lookup(nested=nested)

    assert Pipeline([FoldConstantTables()]).apply_pass(sdfg, {})

    assert "iphase" in sdfg.constants
    assert np.array_equal(sdfg.constants["iphase"], np.array(TABLE, dtype=np.int32))
    assert not any(n.label.startswith("set_") for n, _ in sdfg.all_nodes_recursive())
    out = np.zeros(N, dtype=np.int32)
    sdfg(out=out)
    assert np.array_equal(out, np.array([TABLE[i % 5] for i in range(N)], dtype=np.int32))


def test_an_outlined_reader_of_the_constant_still_compiles_and_computes_the_same():
    """Outlining the lookup into a nested SDFG reads the constant there, not through a connector of its name, while
    a later state still reads it outside."""
    sdfg = table_then_lookup()
    outlined = list(sdfg.states())
    sdfg.add_array("first", [1], dace.int32)
    after = sdfg.add_state_after(outlined[-1], "after")
    tasklet = after.add_tasklet("first", {"t"}, {"o"}, "o = t")
    after.add_edge(after.add_read("iphase"), None, tasklet, "t", dace.Memlet("iphase[1]"))
    after.add_edge(tasklet, "o", after.add_write("first"), None, dace.Memlet("first[0]"))
    assert Pipeline([FoldConstantTables()]).apply_pass(sdfg, {})

    nested = xfh.nest_sdfg_subgraph(sdfg, SubgraphView(sdfg, outlined))

    node = next(n for n in nested.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert "iphase" not in node.in_connectors
    out, first = np.zeros(N, dtype=np.int32), np.zeros(1, dtype=np.int32)
    sdfg(out=out, first=first)
    assert np.array_equal(out, np.array([TABLE[i % 5] for i in range(N)], dtype=np.int32))
    assert first[0] == TABLE[1]


def test_a_table_written_again_stays_a_container():
    sdfg = table_then_lookup(overwrite=True)

    assert Pipeline([FoldConstantTables()]).apply_pass(sdfg, {}) is None

    assert "iphase" not in sdfg.constants


if __name__ == "__main__":
    test_a_literal_table_becomes_a_constant_and_computes_the_same(False)
    test_a_literal_table_becomes_a_constant_and_computes_the_same(True)
    test_an_outlined_reader_of_the_constant_still_compiles_and_computes_the_same()
    test_a_table_written_again_stays_a_container()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A table filled once with literals becomes an SDFG constant; any other write keeps it a container."""

import numpy as np

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.fold_constant_tables import FoldConstantTables

N = 8
TABLE = (1, 2, 1, 2, 0)


def table_then_lookup(overwrite: bool = False) -> dace.SDFG:
    """``iphase[0:5] = 1, 2, 1, 2, 0`` by input-less tasklets, then ``out[i] = iphase[i % 5]``; ``overwrite`` writes
    ``iphase[0]`` again in the second state."""
    sdfg = dace.SDFG("table_then_lookup" + ("_overwritten" if overwrite else ""))
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
    use.add_mapped_tasklet(
        "lookup",
        {"i": f"0:{N}"},
        {"t": dace.Memlet("iphase[i % 5]")},
        "o = t",
        {"o": dace.Memlet("out[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def test_a_literal_table_becomes_a_constant_and_computes_the_same():
    sdfg = table_then_lookup()

    assert Pipeline([FoldConstantTables()]).apply_pass(sdfg, {})

    assert "iphase" in sdfg.constants
    assert np.array_equal(sdfg.constants["iphase"], np.array(TABLE, dtype=np.int32))
    assert not any(n.label.startswith("set_") for n, _ in sdfg.all_nodes_recursive())
    out = np.zeros(N, dtype=np.int32)
    sdfg(out=out)
    assert np.array_equal(out, np.array([TABLE[i % 5] for i in range(N)], dtype=np.int32))


def test_a_table_written_again_stays_a_container():
    sdfg = table_then_lookup(overwrite=True)

    assert Pipeline([FoldConstantTables()]).apply_pass(sdfg, {}) is None

    assert "iphase" not in sdfg.constants


if __name__ == "__main__":
    test_a_literal_table_becomes_a_constant_and_computes_the_same()
    test_a_table_written_again_stays_a_container()

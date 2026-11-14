# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RedundantArray must not let a producer overtake a read that an ordering edge protects.

``tmp`` is produced by ``double`` and copied into ``B``. A reader of the old ``B`` must run before that copy, which an
ordering edge from the reader into ``tmp`` records. ``double`` already runs before the reader. Removing ``tmp`` would
make ``double`` write ``B`` directly, before the reader, which then reads the new value.
"""

import numpy as np

import dace
from dace.transformation.dataflow import RedundantArray


def ordered_read_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.AccessNode, dace.nodes.AccessNode]:
    sdfg = dace.SDFG("redundant_array_ordered_read")
    sdfg.add_array("X", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_transient("tmp", [1], dace.float64)
    state = sdfg.add_state()
    double = state.add_tasklet("double", {"x"}, {"y"}, "y = 2 * x")
    tmp, written = state.add_access("tmp"), state.add_write("B")
    state.add_edge(state.add_read("X"), None, double, "x", dace.Memlet("X[0]"))
    state.add_edge(double, "y", tmp, None, dace.Memlet("tmp[0]"))
    state.add_nedge(tmp, written, dace.Memlet("tmp[0] -> [0]"))
    reader = state.add_tasklet("reader", {"b"}, {"c"}, "c = b + 1")
    state.add_edge(state.add_read("B"), None, reader, "b", dace.Memlet("B[0]"))
    state.add_edge(reader, "c", state.add_write("C"), None, dace.Memlet("C[0]"))
    state.add_nedge(double, reader, dace.Memlet())  # the producer already runs before the reader
    state.add_nedge(reader, tmp, dace.Memlet())  # the reader sees B before the copy overwrites it
    sdfg.validate()
    return sdfg, state, tmp, written


def test_a_producer_ahead_of_an_ordered_read_keeps_its_transient():
    sdfg, state, tmp, written = ordered_read_sdfg()
    assert not RedundantArray.can_be_applied_to(sdfg, in_array=tmp, out_array=written)


def test_the_ordered_read_sees_the_old_value():
    sdfg, _, _, _ = ordered_read_sdfg()
    sdfg.apply_transformations_repeated(RedundantArray)
    x, b, c = np.array([3.0]), np.array([5.0]), np.zeros(1)
    sdfg(X=x, B=b, C=c)
    assert c[0] == 6.0 and b[0] == 6.0


if __name__ == "__main__":
    test_a_producer_ahead_of_an_ordered_read_keeps_its_transient()
    test_the_ordered_read_sees_the_old_value()

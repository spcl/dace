# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SplitConstantIndexedBuffers``: a small buffer accessed at constant elements becomes one scalar per element."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.split_constant_indexed_buffers import SplitConstantIndexedBuffers


def buffer_copied_out(name: str) -> dace.SDFG:
    """``B[0, 0] = a[0] * 2; B[1, 0] = a[1] + 1; out[2:4] <- B[0:2, 0]`` -- a fused map's per-iteration buffer."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [2], dace.float64)
    sdfg.add_array("out", [5], dace.float64)
    sdfg.add_array("B", [2, 1], dace.float64, transient=True)
    state = sdfg.add_state()
    buffer = state.add_access("B")
    for index, code in enumerate(("y = x * 2.0", "y = x + 1.0")):
        tasklet = state.add_tasklet(f"t{index}", {"x"}, {"y"}, code)
        state.add_edge(state.add_read("a"), None, tasklet, "x", dace.Memlet(f"a[{index}]"))
        state.add_edge(tasklet, "y", buffer, None, dace.Memlet(f"B[{index}, 0]"))
    state.add_edge(buffer, None, state.add_write("out"), None, dace.Memlet("B[0:2, 0] -> [2:4]"))
    sdfg.validate()
    return sdfg


def buffers(sdfg: dace.SDFG) -> set[str]:
    return {n.data for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode)} - {"a", "out"}


def test_a_buffer_copied_out_by_range_becomes_one_scalar_per_element():
    sdfg = buffer_copied_out("split_buffer_copied_out")
    assert SplitConstantIndexedBuffers().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "B" not in sdfg.arrays
    split = buffers(sdfg)
    assert len(split) == 2 and all(isinstance(sdfg.arrays[name], dace.data.Scalar) for name in split)

    a = np.array([3.0, 5.0])
    out = np.zeros(5)
    sdfg(a=a, out=out)
    np.testing.assert_array_equal(out, [0.0, 0.0, 6.0, 6.0, 0.0])


def test_a_buffer_read_at_a_symbolic_index_is_left_alone():
    sdfg = buffer_copied_out("split_buffer_symbolic_read")
    sdfg.add_symbol("k", dace.int64)
    state = sdfg.add_state_after(sdfg.start_block)
    tasklet = state.add_tasklet("read", {"x"}, {"y"}, "y = x")
    state.add_edge(state.add_read("B"), None, tasklet, "x", dace.Memlet("B[k, 0]"))
    state.add_edge(tasklet, "y", state.add_write("out"), None, dace.Memlet("out[0]"))
    assert SplitConstantIndexedBuffers().apply_pass(sdfg, {}) is None
    assert "B" in sdfg.arrays


if __name__ == "__main__":
    test_a_buffer_copied_out_by_range_becomes_one_scalar_per_element()
    test_a_buffer_read_at_a_symbolic_index_is_left_alone()

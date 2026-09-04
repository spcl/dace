# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An empty memlet inside a map scope is a happens-before edge, not data.

The two chains below are independent in dataflow -- each stages one array through its own transient
and writes the other -- so nothing but the ordering edge decides which of them reads first. Each
iteration touches only its own index, so the map stays data-parallel, which is what a Map means: a
cross-iteration dependence belongs in a sequential loop, and asserting one here would only be
asserting that the generator declines to vectorize.
"""

import pytest
import dace
import numpy as np


def _build(reverse: bool) -> dace.SDFG:
    """``A[i] = B[i]`` and ``B[i] = A[i]``, ordered by one empty memlet.

    ``reverse`` picks which chain is held back: the delayed chain stages a value the other has
    already overwritten, so both arrays end up holding the same one.
    """
    sdfg = dace.SDFG("mapped_dependency_edge")
    state = sdfg.add_state()

    sdfg.add_array("A", shape=[2], dtype=dace.int32)
    sdfg.add_array("B", shape=[2], dtype=dace.int32)
    sdfg.add_transient("tmp_A", shape=[1], dtype=dace.int32)
    sdfg.add_transient("tmp_B", shape=[1], dtype=dace.int32)

    map_entry, map_exit = state.add_map("map", {"i": "0:2"}, schedule=dace.dtypes.ScheduleType.Sequential)
    for conn in ("IN_A", "IN_B"):
        map_entry.add_in_connector(conn)
    for conn in ("OUT_A", "OUT_B"):
        map_entry.add_out_connector(conn)
    for conn in ("IN_A", "IN_B"):
        map_exit.add_in_connector(conn)
    for conn in ("OUT_A", "OUT_B"):
        map_exit.add_out_connector(conn)

    A_in, B_in = state.add_read("A"), state.add_read("B")
    A_written, B_written = state.add_write("A"), state.add_write("B")
    A_out, B_out = state.add_write("A"), state.add_write("B")
    tmp_A, tmp_B = state.add_write("tmp_A"), state.add_write("tmp_B")

    state.add_edge(A_in, None, map_entry, "IN_A", dace.Memlet("A[0:2]"))
    state.add_edge(B_in, None, map_entry, "IN_B", dace.Memlet("B[0:2]"))

    # Chain 1 stages A[i] and writes it to B[i]; chain 2 stages B[i] and writes it to A[i].
    state.add_edge(map_entry, "OUT_A", tmp_A, None, dace.Memlet("A[i]"))
    state.add_edge(tmp_A, None, B_written, None, dace.Memlet("tmp_A[0] -> [i]"))
    state.add_edge(B_written, None, map_exit, "IN_B", dace.Memlet("B[0:2]"))

    state.add_edge(map_entry, "OUT_B", tmp_B, None, dace.Memlet("B[i]"))
    state.add_edge(tmp_B, None, A_written, None, dace.Memlet("tmp_B[0] -> [i]"))
    state.add_edge(A_written, None, map_exit, "IN_A", dace.Memlet("A[0:2]"))

    # The dependency edge: the staged read it points at must follow the write it comes from.
    if reverse:
        state.add_edge(A_written, None, tmp_A, None, dace.Memlet())
    else:
        state.add_edge(B_written, None, tmp_B, None, dace.Memlet())

    state.add_edge(map_exit, "OUT_A", A_out, None, dace.Memlet("A[0:2]"))
    state.add_edge(map_exit, "OUT_B", B_out, None, dace.Memlet("B[0:2]"))

    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("reverse", [True, False])
def test_mapped_dependency_edge(reverse):
    """ Tests dependency edges in a map scope """
    sdfg = _build(reverse)

    a = np.random.randint(0, 100, 2).astype(np.int32)
    b = np.random.randint(0, 100, 2).astype(np.int32)
    a_before, b_before = a.copy(), b.copy()
    sdfg(A=a, B=b)

    if reverse:
        # A[i] = B[i] lands first, so the held-back chain stages the new A and writes it back to B.
        expected = b_before
    else:
        expected = a_before
    # Ignoring the edge leaves the two chains staging the original values, i.e. a plain swap.
    assert np.array_equal(a, expected) and np.array_equal(b, expected), \
        f"reverse={reverse}: expected both arrays to hold {expected}, got A={a}, B={b}"


if __name__ == "__main__":
    test_mapped_dependency_edge(False)
    test_mapped_dependency_edge(True)

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A pointer-typed tasklet input over a read-only scalar keeps the scalar's constness."""

import numpy as np

import dace


def test_pointer_connector_over_read_only_nested_scalar():
    """A nested SDFG receives its scalar input as ``const float&``; a pointer connector over it must be declared
    ``const float*`` (taking ``&`` of a const reference into a ``float*`` does not compile)."""
    nsdfg = dace.SDFG("pointer_over_const_scalar_nested")
    nsdfg.add_scalar("s", dace.float32)
    nsdfg.add_array("out", [1], dace.float32)
    nstate = nsdfg.add_state()
    tasklet = nstate.add_tasklet(
        "read_through_pointer",
        {"inp": dace.pointer(dace.float32)},
        {"o": dace.float32},
        "o = *inp + 1;",
        language=dace.Language.CPP,
    )
    nstate.add_edge(nstate.add_read("s"), None, tasklet, "inp", dace.Memlet("s"))
    nstate.add_edge(tasklet, "o", nstate.add_write("out"), None, dace.Memlet("out[0]"))

    sdfg = dace.SDFG("pointer_over_const_scalar")
    sdfg.add_scalar("s_outer", dace.float32)
    sdfg.add_array("out_outer", [1], dace.float32)
    state = sdfg.add_state()
    node = state.add_nested_sdfg(nsdfg, {"s"}, {"out"})
    state.add_edge(state.add_read("s_outer"), None, node, "s", dace.Memlet("s_outer"))
    state.add_edge(node, "out", state.add_write("out_outer"), None, dace.Memlet("out_outer[0]"))

    out = np.zeros(1, dtype=np.float32)
    sdfg(s_outer=np.float32(2), out_outer=out)
    assert out[0] == 3


if __name__ == "__main__":
    test_pointer_connector_over_read_only_nested_scalar()

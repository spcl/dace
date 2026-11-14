# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StageGlobalArrayThroughScalars`` leaves a bridge whose write is the masked write ``IT`` in place."""

import numpy

import dace
from dace import data as dt
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars,
)


def masked_write_read_in_the_body() -> dace.SDFG:
    """``A[i] = 3 v[i]``, ``A[i] = IT(A[i] < 1, 0)``, ``out[i] = 2 A[i]``, each hop through its own ``A`` node:
    CloudSC's ``zlcond1`` once ``BranchNormalization`` made ``if zlcond1 < rlmin: zlcond1 = 0`` a masked write."""
    sdfg = dace.SDFG("stage_masked_write")
    for name in ("A", "v", "out"):
        sdfg.add_array(name, [8], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("m", {"i": "0:8"})
    scale = state.add_tasklet("scale", {"_v"}, {"_o"}, "_o = 3.0 * _v")
    clip = state.add_tasklet("clip", {"_a"}, {"_o"}, "_o = IT(_a < 1.0, 0.0)")
    read = state.add_tasklet("read", {"x"}, {"y"}, "y = 2.0 * x")
    scaled, clipped = state.add_access("A"), state.add_access("A")
    state.add_memlet_path(state.add_read("v"), entry, scale, dst_conn="_v", memlet=dace.Memlet("v[i]"))
    state.add_edge(scale, "_o", scaled, None, dace.Memlet("A[i]"))
    state.add_edge(scaled, None, clip, "_a", dace.Memlet("A[i]"))
    written = dace.Memlet("A[i]")
    written.dynamic = True
    state.add_edge(clip, "_o", clipped, None, written)
    state.add_edge(clipped, None, read, "x", dace.Memlet("A[i]"))
    state.add_memlet_path(read, exit_node, state.add_write("out"), src_conn="y", memlet=dace.Memlet("out[i]"))
    state.add_memlet_path(clipped, exit_node, state.add_write("A"), memlet=dace.Memlet("A[i]"))
    return sdfg


def test_a_masked_write_bridge_is_not_staged():
    """Staged, the masked write lands in a fresh scalar with no old value for the elements it skips, under a plain
    memlet the readable generator inlines (``stage_A = IT(...)``, which does not compile), and the ``3 v`` hop it
    follows counts as overwritten, so that value never reaches ``A`` where the guard fails."""
    sdfg = masked_write_read_in_the_body()
    assert not StageGlobalArrayThroughScalars().apply_pass(sdfg, {})
    sdfg.validate()

    state = sdfg.start_state
    staged = [n for n in state.data_nodes() if isinstance(n.desc(sdfg), dt.Scalar)]
    assert not staged, f"the masked write was staged through {[n.data for n in staged]}"
    clip = next(n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == "clip")
    (edge,) = state.out_edges(clip)
    assert edge.dst.data == "A" and edge.data.dynamic

    v = numpy.linspace(0.0, 0.7, 8)
    expected = numpy.where(3.0 * v < 1.0, 0.0, 3.0 * v)
    A, out = numpy.full(8, -1.0), numpy.zeros(8)
    sdfg(A=A, v=v, out=out)
    assert numpy.array_equal(A, expected)
    assert numpy.array_equal(out, 2.0 * expected)


if __name__ == "__main__":
    test_a_masked_write_bridge_is_not_staged()

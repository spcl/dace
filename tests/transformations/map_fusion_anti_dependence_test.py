# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MapFusionVertical keeps a read of the first Map before a write of the same container by the second Map: apart, the
Map boundary orders them; fused, the body needs a path (spcl/NestForge#50)."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusionVertical

N = dace.symbol("N", dtype=dace.int64)


def mapped(state: dace.SDFGState, label: str, param: str, tasklets: list[tuple[str, str, str, str]], inputs, outputs):
    """A Map over ``0:N`` whose body holds one tasklet ``(label, read, code, written)`` per entry, each reading and
    writing one element; ``inputs``/``outputs`` are the access nodes outside the Map, by data name."""
    entry, exit_ = state.add_map(label, {param: "0:N"})
    for name, read, code, written in tasklets:
        tasklet = state.add_tasklet(name, {"a"}, {"o"}, code)
        state.add_memlet_path(inputs[read], entry, tasklet, dst_conn="a", memlet=dace.Memlet(f"{read}[{param}]"))
        state.add_memlet_path(tasklet, exit_, outputs[written], src_conn="o", memlet=dace.Memlet(f"{written}[{param}]"))


def read_then_overwrite() -> dace.SDFG:
    """The first Map reads x (``first``: t = x + 1) and, in another tasklet, computes f = 2 * y; the second Map writes
    out = t and x = f (``write_x``). The write of x depends only on f, never on the read of x."""
    sdfg = dace.SDFG("map_fusion_anti_dependence")
    for name in ("x", "y", "out"):
        sdfg.add_array(name, [N], dace.float64)
    for name in ("t", "f"):
        sdfg.add_array(name, [N], dace.float64, transient=True)
    state = sdfg.add_state()
    middle = {"t": state.add_access("t"), "f": state.add_access("f")}
    reads = {"x": state.add_read("x"), "y": state.add_read("y")}
    mapped(
        state, "producer", "i", [("first", "x", "o = a + 1.0", "t"), ("double", "y", "o = 2.0 * a", "f")], reads, middle
    )
    writes = {"out": state.add_write("out"), "x": state.add_write("x")}
    mapped(state, "consumer", "j", [("copy_t", "t", "o = a", "out"), ("write_x", "f", "o = a", "x")], middle, writes)
    return sdfg


def fused() -> dace.SDFG:
    sdfg = read_then_overwrite()
    sdfg.validate()
    assert sdfg.apply_transformations_repeated(MapFusionVertical) == 1
    return sdfg


def test_the_fused_body_reads_x_before_it_writes_x():
    sdfg = fused()
    state = sdfg.start_state
    reader = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == "first")
    writer = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == "write_x")

    assert len([n for n in state.nodes() if isinstance(n, nodes.MapEntry)]) == 1
    assert reader in dace.sdfg.utils.find_upstream_nodes(writer, state)


def test_the_fused_program_reads_the_old_x():
    sdfg = fused()
    x, y, out = np.arange(8.0), np.full(8, 3.0), np.zeros(8)

    sdfg(x=x, y=y, out=out, N=8)

    assert np.array_equal(out, np.arange(8.0) + 1.0)
    assert np.array_equal(x, np.full(8, 6.0))


if __name__ == "__main__":
    test_the_fused_body_reads_x_before_it_writes_x()
    test_the_fused_program_reads_the_old_x()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineSDFG keeps the order a nested SDFG enforced as a unit: a later write to a container the nest read must still
come after that read once the nest is spliced in."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG


def single_state(name: str, tasklets: list[tuple[str, dict[str, str], str, dict[str, str]]]) -> dace.SDFG:
    """One state of tasklets; each reads and writes one-element arrays named by its connector maps."""
    sdfg = dace.SDFG(name)
    for _, reads, _, writes in tasklets:
        for data in (*reads.values(), *writes.values()):
            if data not in sdfg.arrays:
                sdfg.add_array(data, [1], dace.float64)
    state = sdfg.add_state()
    for label, reads, code, writes in tasklets:
        tasklet = state.add_tasklet(label, dict.fromkeys(reads), dict.fromkeys(writes), code)
        for conn, data in reads.items():
            state.add_edge(state.add_read(data), None, tasklet, conn, dace.Memlet(f"{data}[0]"))
        for conn, data in writes.items():
            state.add_edge(tasklet, conn, state.add_write(data), None, dace.Memlet(f"{data}[0]"))
    return sdfg


def read_then_write() -> dace.SDFG:
    """``first`` reads G (t = 2 * G) and computes f = 3 * x; its sibling ``second`` writes q = t and G = f. Only the
    nesting orders ``first``'s read of G before ``second``'s write of it: the write depends on f, not on the read."""
    sdfg = dace.SDFG("inline_keeps_nest_order")
    for name in ("G", "q", "x"):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_transient("T", [1], dace.float64)
    sdfg.add_transient("F", [1], dace.float64)
    state = sdfg.add_state()
    first = state.add_nested_sdfg(
        single_state(
            "first",
            [
                ("read_g", {"a": "g_in"}, "b = 2.0 * a", {"b": "t_out"}),
                ("scale", {"a": "x_in"}, "b = 3.0 * a", {"b": "f_out"}),
            ],
        ),
        dict.fromkeys(["g_in", "x_in"]),
        dict.fromkeys(["t_out", "f_out"]),
    )
    second = state.add_nested_sdfg(
        single_state(
            "second",
            [("write_q", {"a": "t_in"}, "b = a", {"b": "q_out"}), ("write_g", {"a": "f_in"}, "b = a", {"b": "g_out"})],
        ),
        dict.fromkeys(["t_in", "f_in"]),
        dict.fromkeys(["g_out", "q_out"]),
    )
    state.add_edge(state.add_read("G"), None, first, "g_in", dace.Memlet("G[0]"))
    state.add_edge(state.add_read("x"), None, first, "x_in", dace.Memlet("x[0]"))
    t, f = state.add_access("T"), state.add_access("F")
    state.add_edge(first, "t_out", t, None, dace.Memlet("T[0]"))
    state.add_edge(first, "f_out", f, None, dace.Memlet("F[0]"))
    state.add_edge(t, None, second, "t_in", dace.Memlet("T[0]"))
    state.add_edge(f, None, second, "f_in", dace.Memlet("F[0]"))
    state.add_edge(second, "g_out", state.add_write("G"), None, dace.Memlet("G[0]"))
    state.add_edge(second, "q_out", state.add_write("q"), None, dace.Memlet("q[0]"))
    sdfg.validate()
    return sdfg


def has_path(state: dace.SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    stack, seen = [src], {src}
    while stack:
        for edge in state.out_edges(stack.pop()):
            if edge.dst is dst:
                return True
            if edge.dst not in seen:
                seen.add(edge.dst)
                stack.append(edge.dst)
    return False


def tasklet(state: dace.SDFGState, label: str) -> nodes.Tasklet:
    return next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == label)


def test_the_read_of_a_container_stays_before_a_later_write_of_it():
    sdfg = read_then_write()

    assert sdfg.apply_transformations_repeated(InlineSDFG, validate=False) == 2

    state = sdfg.start_state
    assert has_path(state, tasklet(state, "read_g"), tasklet(state, "write_g"))
    sdfg.validate()


def test_the_inlined_program_reads_the_old_value():
    sdfg = read_then_write()
    sdfg.apply_transformations_repeated(InlineSDFG)
    G, q, x = np.array([5.0]), np.array([0.0]), np.array([1.0])

    sdfg(G=G, q=q, x=x)

    assert (q[0], G[0]) == (10.0, 3.0)

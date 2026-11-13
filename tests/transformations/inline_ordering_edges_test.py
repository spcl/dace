# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineSDFG inlines a nested SDFG whose connector access nodes carry empty-memlet ordering edges, as an earlier
inline leaves them, and keeps every such order on an empty tasklet."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.interstate.sdfg_nesting import InlineSDFG

N = 8


def ordered_inputs(name: str, copy_input: bool) -> dace.SDFG:
    """``map i: r = 0; r = r + 2 s; c = f + 1`` where empty memlets order the reads of ``s`` and ``f`` after the
    first write of ``r`` (``r -> s``, ``r -> f -> f``). With ``copy_input`` ``f`` is instead copied into ``s``, which
    is then written and so an output as well."""
    sdfg = dace.SDFG(name)
    for data in "SFRC":
        sdfg.add_array(data, [N], dace.float64)
    inner = dace.SDFG(f"{name}_inner")
    inner.add_symbol("i", dace.int64)
    for data in "sfrc":
        inner.add_array(data, [N], dace.float64)
    nstate = inner.add_state()
    init = nstate.add_tasklet("init", {}, {"y"}, "y = 0")
    acc = nstate.add_tasklet("acc", {"x", "z"}, {"y"}, "y = x + 2 * z")
    inc = nstate.add_tasklet("inc", {"x"}, {"y"}, "y = x + 1")
    r1, r2, s = nstate.add_access("r"), nstate.add_write("r"), nstate.add_access("s")
    f1, f2 = nstate.add_access("f"), nstate.add_access("f")
    nstate.add_edge(init, "y", r1, None, dace.Memlet("r[i]"))
    nstate.add_nedge(r1, s, dace.Memlet())
    if copy_input:
        nstate.add_edge(f1, None, s, None, dace.Memlet(data="f", subset="i", other_subset="i"))
    else:
        nstate.add_nedge(r1, f1, dace.Memlet())
    nstate.add_nedge(f1, f2, dace.Memlet())
    nstate.add_edge(r1, None, acc, "x", dace.Memlet("r[i]"))
    nstate.add_edge(s, None, acc, "z", dace.Memlet("s[i]"))
    nstate.add_edge(acc, "y", r2, None, dace.Memlet("r[i]"))
    nstate.add_edge(f2, None, inc, "x", dace.Memlet("f[i]"))
    nstate.add_edge(inc, "y", nstate.add_write("c"), None, dace.Memlet("c[i]"))

    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": f"0:{N}"})
    outputs = "rcs" if copy_input else "rc"
    node = state.add_nested_sdfg(inner, {"s": None, "f": None}, dict.fromkeys(outputs), symbol_mapping={"i": "i"})
    for conn in "sf":
        state.add_memlet_path(
            state.add_read(conn.upper()), entry, node, dst_conn=conn, memlet=dace.Memlet(f"{conn.upper()}[i]")
        )
    for conn in outputs:
        state.add_memlet_path(
            node, exit_, state.add_write(conn.upper()), src_conn=conn, memlet=dace.Memlet(f"{conn.upper()}[i]")
        )
    sdfg.validate()
    return sdfg


def reaches(state: dace.SDFGState, src: nodes.Node, dst: nodes.Node) -> bool:
    stack, seen = [src], {src}
    while stack:
        for edge in state.out_edges(stack.pop()):
            if edge.dst is dst:
                return True
            if edge.dst not in seen:
                seen.add(edge.dst)
                stack.append(edge.dst)
    return False


def test_inline_keeps_ordering_edges_of_boundary_nodes_on_a_tasklet():
    sdfg = ordered_inputs("inline_ordering_edges", copy_input=False)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    InlineSDFG.apply_to(sdfg, nested_sdfg=node)
    sdfg.validate()
    assert not any(isinstance(n, nodes.NestedSDFG) for n, _ in sdfg.all_nodes_recursive())
    state = sdfg.start_block
    tasklets = {n.label: n for n in state.nodes() if isinstance(n, nodes.Tasklet)}
    assert reaches(state, tasklets["init"], tasklets["acc"])
    assert reaches(state, tasklets["init"], tasklets["inc"]), "the ordering of f after r was lost"
    s, f = np.random.rand(N), np.random.rand(N)
    r, c = np.zeros(N), np.zeros(N)
    sdfg(S=s, F=f, R=r, C=c)
    assert np.allclose(r, 2 * s) and np.allclose(c, f + 1)


def test_inline_refuses_an_input_copied_into_another_behind_ordering_edges():
    sdfg = ordered_inputs("inline_ordering_edges_copy", copy_input=True)
    node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
    assert not InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=node)


if __name__ == "__main__":
    test_inline_keeps_ordering_edges_of_boundary_nodes_on_a_tasklet()
    test_inline_refuses_an_input_copied_into_another_behind_ordering_edges()

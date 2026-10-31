# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The border memlet propagated out of a nested SDFG addresses the connector's own container, also where the access is
a copy whose memlet names the other end."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.propagation import propagate_memlets_sdfg

N = 8


def gather_through_an_inner_index() -> dace.SDFG:
    """``for i in map[0:N]: B[i] = A[N - 1 - i]``, the nest staging ``a[k]`` into a length-1 ``t`` by a copy that names
    ``t``, at an index ``k`` the nest assigns itself."""
    sdfg = dace.SDFG("gather_through_an_inner_index")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("B", [N], dace.float64)
    state = sdfg.add_state()
    inner = dace.SDFG("gather_body")
    inner.add_array("a", [N], dace.float64)
    inner.add_array("b", [N], dace.float64)
    inner.add_array("t", [1], dace.float64, transient=True)
    inner.add_symbol("i", dace.int64)
    inner.add_symbol("k", dace.int64)
    head = inner.add_state(is_start_block=True)
    body = inner.add_state()
    inner.add_edge(head, body, dace.InterstateEdge(assignments={"k": f"{N - 1} - i"}))
    t = body.add_access("t")
    body.add_edge(body.add_read("a"), None, t, None, dace.Memlet(data="t", subset="0", other_subset="k"))
    body.add_edge(t, None, body.add_write("b"), None, dace.Memlet(data="b", subset="i", other_subset="0"))
    nsdfg = state.add_nested_sdfg(inner, {"a"}, {"b"}, {"i": "i"})
    entry, exit_ = state.add_map("rows", {"i": f"0:{N}"})
    state.add_memlet_path(state.add_read("A"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet(f"A[0:{N}]"))
    state.add_memlet_path(nsdfg, exit_, state.add_write("B"), src_conn="b", memlet=dace.Memlet(f"B[0:{N}]"))
    return sdfg


def test_the_border_memlet_of_a_copy_naming_the_other_end_addresses_the_connector():
    sdfg = gather_through_an_inner_index()
    propagate_memlets_sdfg(sdfg)
    state = sdfg.start_state
    nsdfg = next(n for n in state.nodes() if isinstance(n, nodes.NestedSDFG))
    (into,) = state.in_edges(nsdfg)
    # ``k`` is unknown outside the nest, so the read widens to all of ``a``, not to ``t``'s one element.
    assert (into.data.data, str(into.data.subset)) == ("A", f"0:{N}"), into.data
    a = np.arange(N, dtype=np.float64)
    b = np.zeros(N)
    sdfg(A=a, B=b)
    assert np.array_equal(b, a[::-1])


if __name__ == "__main__":
    test_the_border_memlet_of_a_copy_naming_the_other_end_addresses_the_connector()

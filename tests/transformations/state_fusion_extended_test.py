# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import networkx as nx
import numpy as np

from dace import SDFG, InterstateEdge, Memlet, dtypes, nodes
from dace.transformation.interstate import StateFusionExtended


def test_extended_fusion():
    """
    Test the extended state fusion transformation.
    It should fuse the two states into one and add a dependency between the two uses of tmp.
    """
    sdfg = SDFG("extended_state_fusion_test")
    sdfg.add_array("A", [20, 20], dtypes.float64)
    sdfg.add_array("B", [20, 20], dtypes.float64)
    sdfg.add_array("C", [20, 20], dtypes.float64)
    sdfg.add_array("D", [20, 20], dtypes.float64)
    sdfg.add_array("E", [20, 20], dtypes.float64)
    sdfg.add_array("F", [20, 20], dtypes.float64)

    sdfg.add_scalar("tmp", dtypes.float64)

    strt = sdfg.add_state("start")
    mid = sdfg.add_state("middle")

    sdfg.add_edge(strt, mid, InterstateEdge())

    acc_a = strt.add_read("A")
    acc_b = strt.add_read("B")
    acc_c = strt.add_write("C")
    acc_tmp = strt.add_access("tmp")

    acc2_d = mid.add_read("D")
    acc2_e = mid.add_read("E")
    acc2_f = mid.add_write("F")
    acc2_tmp = mid.add_access("tmp")

    t1 = strt.add_tasklet(
        "t1",
        {"a", "b"},
        {
            "c",
        },
        "c = a + b",
    )
    t2 = strt.add_tasklet(
        "t2",
        {},
        {
            "tmpa",
        },
        "tmpa=4",
    )

    t3 = mid.add_tasklet(
        "t3",
        {"d", "e"},
        {
            "f",
        },
        "f = e + d",
    )
    t4 = mid.add_tasklet(
        "t4",
        {},
        {
            "tmpa",
        },
        "tmpa=7",
    )

    strt.add_edge(acc_a, None, t1, "a", Memlet.simple("A", "1,1"))
    strt.add_edge(acc_b, None, t1, "b", Memlet.simple("B", "1,1"))
    strt.add_edge(t1, "c", acc_c, None, Memlet.simple("C", "1,1"))
    strt.add_edge(t2, "tmpa", acc_tmp, None, Memlet.simple("tmp", "0"))

    mid.add_edge(acc2_d, None, t3, "d", Memlet.simple("D", "1,1"))
    mid.add_edge(acc2_e, None, t3, "e", Memlet.simple("E", "1,1"))
    mid.add_edge(t3, "f", acc2_f, None, Memlet.simple("F", "1,1"))
    mid.add_edge(t4, "tmpa", acc2_tmp, None, Memlet.simple("tmp", "0"))
    sdfg.simplify()
    sdfg.apply_transformations_repeated(StateFusionExtended)
    assert sdfg.number_of_nodes() == 1


def test_extended_fusion_orders_unsafe_write_after_read():
    """A read state followed by an in-place write of the same array fuses WITH a
    happens-before ordering, not by refusing.

    ``s1`` reads ``A[k]`` in two tasklets; the later ``s2`` does ``A[k] = A[k] + 1``.
    Fusing the two states drops the interstate edge that kept the write after the
    reads, so ``StateFusionExtended`` records the write-after-read anti-dependency and
    adds an ordering edge from every first-state ``A`` reader to the second-state write.
    The result is a single fused state that still computes the reference: ``B`` and ``C``
    see the ORIGINAL ``A[k]`` and ``A[k]`` is incremented after. (An earlier version
    refused this fusion outright; ordering it is sound and strictly more capable.)
    """
    sdfg = SDFG("state_fusion_war_ordering")
    sdfg.add_array("A", [8], dtypes.float64)
    sdfg.add_array("B", [8], dtypes.float64)
    sdfg.add_array("C", [8], dtypes.float64)
    sdfg.add_symbol("k", dtypes.int64)

    s1 = sdfg.add_state("read_A", is_start_block=True)
    s2 = sdfg.add_state("increment_A")
    sdfg.add_edge(s1, s2, InterstateEdge())

    ar_b = s1.add_read("A")
    bw = s1.add_write("B")
    tb = s1.add_tasklet("rb", {"_in"}, {"_out"}, "_out = _in")
    s1.add_edge(ar_b, None, tb, "_in", Memlet("A[k]"))
    s1.add_edge(tb, "_out", bw, None, Memlet("B[k]"))

    ar_c = s1.add_read("A")
    cw = s1.add_write("C")
    tc = s1.add_tasklet("rc", {"_in"}, {"_out"}, "_out = _in")
    s1.add_edge(ar_c, None, tc, "_in", Memlet("A[k]"))
    s1.add_edge(tc, "_out", cw, None, Memlet("C[k]"))

    ar2 = s2.add_read("A")
    aw2 = s2.add_write("A")
    ti = s2.add_tasklet("inc", {"_in"}, {"_out"}, "_out = _in + 1.0")
    s2.add_edge(ar2, None, ti, "_in", Memlet("A[k]"))
    s2.add_edge(ti, "_out", aw2, None, Memlet("A[k]"))
    sdfg.validate()

    applied = sdfg.apply_transformations_repeated(StateFusionExtended)
    assert applied >= 1, "write-after-read fusion should apply with an ordering edge"
    assert sdfg.number_of_nodes() == 1, "the two states must fuse into one"

    # The ordering edge keeps the write after the reads: B, C get the original A[k]; A is incremented.
    A = np.arange(8, dtype=np.float64) + 10.0
    B = np.zeros(8)
    C = np.zeros(8)
    A0 = A.copy()
    sdfg(A=A, B=B, C=C, k=3)
    assert A[3] == A0[3] + 1.0
    assert B[3] == A0[3]
    assert C[3] == A0[3]


def test_extended_fusion_orders_a_reused_transient():
    """Both states write then read ``tmp``; the second write must wait for the first state's read."""
    sdfg = SDFG("state_fusion_reused_transient")
    sdfg.add_array("A", [4], dtypes.float64)
    sdfg.add_array("B", [4], dtypes.float64)
    sdfg.add_scalar("tmp", dtypes.float64, transient=True)

    s1 = sdfg.add_state("c0", is_start_block=True)
    s2 = sdfg.add_state_after(s1, "c1")
    for st, idx in ((s1, 0), (s2, 1)):
        tmp = st.add_access("tmp")
        load = st.add_tasklet(f"load{idx}", {"_i"}, {"_o"}, "_o = _i * 2.0")
        store = st.add_tasklet(f"store{idx}", {"_i"}, {"_o"}, "_o = _i")
        st.add_edge(st.add_read("A"), None, load, "_i", Memlet(f"A[{idx}]"))
        st.add_edge(load, "_o", tmp, None, Memlet("tmp[0]"))
        st.add_edge(tmp, None, store, "_i", Memlet("tmp[0]"))
        st.add_edge(store, "_o", st.add_write("B"), None, Memlet(f"B[{idx}]"))

    assert sdfg.apply_transformations_repeated(StateFusionExtended) == 1
    A = np.array([10.0, 20.0, 30.0, 40.0])
    B = np.zeros(4)
    sdfg(A=A, B=B)
    assert B[0] == 20.0 and B[1] == 40.0


def test_extended_fusion_compares_a_copy_at_its_destination():
    # ``A[5:10] -> B`` writes ``B[0:5]``, which the second state writes again: the writes must be ordered
    sdfg = SDFG("extended_copy_other_subset")
    sdfg.add_array("A", [10], dtypes.float64)
    sdfg.add_array("B", [5], dtypes.float64)
    s1 = sdfg.add_state("s1", is_start_block=True)
    s1.add_nedge(s1.add_read("A"), s1.add_write("B"), Memlet("A[5:10]"))
    s2 = sdfg.add_state_after(s1, "s2")
    s2.add_mapped_tasklet("w", {"i": "0:5"}, {}, "b = 1.0", {"b": Memlet("B[i]")}, external_edges=True)
    sdfg.validate()

    sdfg.apply_transformations_repeated(StateFusionExtended)
    for state in sdfg.states():
        writers = [n for n in state.data_nodes() if n.data == "B" and state.in_degree(n) > 0]
        assert all(nx.has_path(state.nx, a, b) or nx.has_path(state.nx, b, a) for a in writers for b in writers)


def test_extended_fusion_orders_a_writer_its_ordered_sibling_does_not_cover():
    # ``B[0]`` reaches the second state through ``C``; its sibling ``B[1]`` writer does not, and the second state
    # writes ``B[1]``
    sdfg = SDFG("extended_fusion_orders_a_writer_its")
    sdfg.add_array("A", [1], dtypes.float64)
    sdfg.add_array("B", [2], dtypes.float64)
    sdfg.add_array("C", [1], dtypes.float64)
    first = sdfg.add_state("first", is_start_block=True)
    a = first.add_read("A")
    t0 = first.add_tasklet("w0", {"i"}, {"o"}, "o = i")
    first.add_edge(a, None, t0, "i", Memlet("A[0]"))
    b0 = first.add_access("B")
    first.add_edge(t0, "o", b0, None, Memlet("B[0]"))
    copy = first.add_tasklet("cp", {"i"}, {"o"}, "o = i")
    first.add_edge(b0, None, copy, "i", Memlet("B[0]"))
    first.add_edge(copy, "o", first.add_write("C"), None, Memlet("C[0]"))
    t1 = first.add_tasklet("w1", {"i"}, {"o"}, "o = i + 1")
    first.add_edge(a, None, t1, "i", Memlet("A[0]"))
    first.add_edge(t1, "o", first.add_write("B"), None, Memlet("B[1]"))
    second = sdfg.add_state_after(first, "second")
    t2 = second.add_tasklet("w2", {"i"}, {"o"}, "o = i")
    second.add_edge(second.add_read("C"), None, t2, "i", Memlet("C[0]"))
    second.add_edge(t2, "o", second.add_write("B"), None, Memlet("B[1]"))
    sdfg.validate()

    sdfg.apply_transformations_repeated(StateFusionExtended)
    for state in sdfg.states():
        writers = [
            n for n in state.data_nodes() if n.data == "B" and any(str(e.data.subset) == "1" for e in state.in_edges(n))
        ]
        assert all(
            nx.has_path(state.nx, a, b) or nx.has_path(state.nx, b, a) for a in writers for b in writers if a is not b
        )


def test_extended_fusion_a_read_behind_an_ordering_edge_is_a_read():
    # ``B`` is only read in the first state; its one in-edge orders, it writes nothing
    sdfg = SDFG("extended_fusion_a_read_behind_an_orderin")
    for name in "ABC":
        sdfg.add_array(name, [1], dtypes.float64)
    first = sdfg.add_state("first", is_start_block=True)
    b = first.add_access("B")
    first.add_nedge(first.add_access("A"), b, Memlet())
    read = first.add_tasklet("rd", {"i"}, {"o"}, "o = i")
    first.add_edge(b, None, read, "i", Memlet("B[0]"))
    first.add_edge(read, "o", first.add_write("C"), None, Memlet("C[0]"))
    second = sdfg.add_state_after(first, "second")
    write = second.add_tasklet("w", {}, {"o"}, "o = 7.0")
    second.add_edge(write, "o", second.add_write("B"), None, Memlet("B[0]"))
    sdfg.validate()

    sdfg.apply_transformations_repeated(StateFusionExtended)
    for state in sdfg.states():
        tasklets = {n.label: n for n in state.nodes() if isinstance(n, nodes.Tasklet)}
        if tasklets.keys() == {"rd", "w"}:
            assert nx.has_path(state.nx, tasklets["rd"], tasklets["w"])


def test_extended_fusion_orders_a_transient_with_two_producers():
    """The first state writes ``t[0]`` and ``t[1]`` through two access nodes and reads both; the second state writes
    ``t[0]`` again, which must wait for that read."""
    sdfg = SDFG("state_fusion_two_producers")
    sdfg.add_array("arr", [8], dtypes.float64)
    sdfg.add_array("out1", [1], dtypes.float64)
    sdfg.add_array("out2", [1], dtypes.float64)
    sdfg.add_transient("t", [2], dtypes.float64)

    s1 = sdfg.add_state("s1", is_start_block=True)
    arr = s1.add_read("arr")
    reader = s1.add_tasklet("rd", {"i0", "i1"}, {"o"}, "o = i0 + i1")
    for k in range(2):
        write = s1.add_tasklet(f"w{k}", {"i"}, {"o"}, "o = i")
        t = s1.add_access("t")
        s1.add_edge(arr, None, write, "i", Memlet(f"arr[{k}]"))
        s1.add_edge(write, "o", t, None, Memlet(f"t[{k}]"))
        s1.add_edge(t, None, reader, f"i{k}", Memlet(f"t[{k}]"))
    s1.add_edge(reader, "o", s1.add_write("out1"), None, Memlet("out1[0]"))

    s2 = sdfg.add_state_after(s1, "s2")
    t = s2.add_access("t")
    write = s2.add_tasklet("w2", {"i"}, {"o"}, "o = i")
    copy = s2.add_tasklet("r2", {"i"}, {"o"}, "o = i")
    s2.add_edge(s2.add_read("arr"), None, write, "i", Memlet("arr[2]"))
    s2.add_edge(write, "o", t, None, Memlet("t[0]"))
    s2.add_edge(t, None, copy, "i", Memlet("t[0]"))
    s2.add_edge(copy, "o", s2.add_write("out2"), None, Memlet("out2[0]"))

    sdfg.apply_transformations_repeated(StateFusionExtended)
    out1, out2 = np.zeros(1), np.zeros(1)
    sdfg(arr=np.arange(8, dtype=np.float64), out1=out1, out2=out2)
    assert out1[0] == 1.0 and out2[0] == 2.0


if __name__ == "__main__":
    test_extended_fusion()
    test_extended_fusion_orders_unsafe_write_after_read()
    test_extended_fusion_orders_a_reused_transient()
    test_extended_fusion_compares_a_copy_at_its_destination()
    test_extended_fusion_orders_a_writer_its_ordered_sibling_does_not_cover()
    test_extended_fusion_a_read_behind_an_ordering_edge_is_a_read()
    test_extended_fusion_orders_a_transient_with_two_producers()

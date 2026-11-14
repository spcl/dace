# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A slice update spelled as an assignment, ``A[S] = A[S] + e``, reaches the SDFG as ``A[S] -> map -> t -> A[S]``;
``EliminateCopyBack`` has the map write ``A[S]`` in place when each iteration reads only the element it writes."""

import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.dealias import windowed_connectors
from dace.transformation.passes.canonicalize.eliminate_copy_back import EliminateCopyBack

N, M, K = dace.symbol("N"), dace.symbol("M"), dace.symbol("K")


@dace.program
def deposit(A: dace.float64[M + 2, 2], w: dace.float64[M], c: dace.float64[3]):
    for p in range(M):
        A[p : p + 3, 1] = A[p : p + 3, 1] + w[p] * c


@dace.program
def shifted(A: dace.float64[N], c: dace.float64[3]):
    A[K : K + 3] = A[K + 1 : K + 4] + c


def transients_copied_back(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for st in sdfg.states()
        for n in st.data_nodes()
        if sdfg.arrays[n.data].transient
        and st.out_degree(n) == 1
        and isinstance(st.out_edges(n)[0].dst, nodes.AccessNode)
        and st.out_edges(n)[0].dst.data == "A"
    )


def test_a_slice_update_writes_its_target_in_place():
    sdfg = deposit.to_sdfg(simplify=True)
    assert transients_copied_back(sdfg) == 1
    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    assert transients_copied_back(sdfg) == 0
    sdfg.validate()
    m = 7
    rng = np.random.default_rng(0)
    w, c = rng.random(m), rng.random(3)
    A = rng.random((m + 2, 2))
    expected = A.copy()
    for p in range(m):
        expected[p : p + 3, 1] += w[p] * c
    sdfg(A=A, w=w, c=c, M=m)
    assert np.allclose(A, expected)


def test_a_read_of_the_neighbouring_element_keeps_the_copy_back():
    """Iteration ``i`` reads ``A[K + i + 1]``, which iteration ``i + 1`` writes: in place it would read the new value."""
    sdfg = shifted.to_sdfg(simplify=True)
    before = copy.deepcopy(sdfg)
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None
    assert transients_copied_back(sdfg) == transients_copied_back(before)


def test_a_transient_only_ordered_after_a_map_is_left_alone():
    """``map -> (empty) -> t -> A``: the map orders ``t``'s copy and writes nothing into it (nbody's pipeline died on
    the connector-less edge)."""
    sdfg = dace.SDFG("ordered_copy_back")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("B", [4], dace.float64)
    sdfg.add_transient("t", [4], dace.float64)
    state = sdfg.add_state()
    _, _, exit_node = state.add_mapped_tasklet(
        "fill", {"i": "0:4"}, {}, "o = 1.0", {"o": dace.Memlet("B[i]")}, external_edges=True
    )
    t = state.add_access("t")
    state.add_nedge(exit_node, t, dace.Memlet())
    state.add_nedge(t, state.add_write("A"), dace.Memlet("t[0:4] -> [0:4]"))
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None


def row_update(update: str, read: str, transient_shape: list, name: str) -> dace.SDFG:
    """``t[i] = <read>`` over ``i`` in ``0:4``, then the copy ``t[...] -> A[update]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [3, 6], dace.float64)
    sdfg.add_transient("t", transient_shape, dace.float64)
    state = sdfg.add_state()
    a = state.add_read("A")
    t = state.add_access("t")
    state.add_mapped_tasklet(
        "update",
        {"i": "0:4"},
        {"x": dace.Memlet(f"A[{read}]")},
        "o = x * 2.0",
        {"o": dace.Memlet("t[i + 1]" if transient_shape == [6] else "t[i]")},
        input_nodes={"A": a},
        output_nodes={"t": t},
        external_edges=True,
    )
    whole = "1:5" if transient_shape == [6] else "0:4"
    state.add_nedge(t, state.add_write("A"), dace.Memlet(f"t[{whole}] -> [{update}]"))
    sdfg.validate()
    return sdfg


def test_a_read_outside_the_updated_slice_writes_in_place():
    sdfg = row_update("0, 0:4", "1, i", [4], "copy_back_other_row")
    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    A = np.arange(18.0).reshape(3, 6)
    expected = A.copy()
    expected[0, 0:4] = 2.0 * A[1, 0:4]
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_a_transient_used_in_part_folds_at_its_origin():
    sdfg = row_update("0, 1:5", "0, i + 1", [6], "copy_back_part_used")
    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    assert "t" not in sdfg.arrays
    A = np.arange(18.0).reshape(3, 6)
    expected = A.copy()
    expected[0, 1:5] = 2.0 * A[0, 1:5]
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_a_read_inside_the_slice_at_another_element_keeps_the_copy_back():
    sdfg = row_update("0, 0:4", "0, 3 - i", [4], "copy_back_reversed")
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None


def test_a_sibling_read_of_the_written_element_keeps_the_copy_back():
    """``t[i] = A[i] + 1; y[i] = A[i] * 2`` in one map: in place, ``y`` could read the new ``A[i]``."""
    sdfg = dace.SDFG("copy_back_sibling_read")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("y", [4], dace.float64)
    sdfg.add_transient("t", [4], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("both", {"i": "0:4"})
    a, t = state.add_read("A"), state.add_access("t")
    update = state.add_tasklet("update", {"x"}, {"o"}, "o = x + 1.0")
    sibling = state.add_tasklet("sibling", {"x"}, {"o"}, "o = x * 2.0")
    state.add_memlet_path(a, entry, update, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(a, entry, sibling, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(update, exit_node, t, src_conn="o", memlet=dace.Memlet("t[i]"))
    state.add_memlet_path(sibling, exit_node, state.add_write("y"), src_conn="o", memlet=dace.Memlet("y[i]"))
    state.add_nedge(t, state.add_write("A"), dace.Memlet("t[0:4] -> [0:4]"))
    sdfg.validate()
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None


def test_a_stepped_copy_back_stays():
    sdfg = dace.SDFG("copy_back_stepped")
    sdfg.add_array("A", [6], dace.float64)
    sdfg.add_transient("t", [3], dace.float64)
    state = sdfg.add_state()
    t = state.add_access("t")
    state.add_mapped_tasklet(
        "m",
        {"j": "0:3"},
        {"q": dace.Memlet("A[j]")},
        "o = q + 1",
        {"o": dace.Memlet("t[j]")},
        external_edges=True,
        output_nodes={"t": t},
    )
    state.add_edge(t, None, state.add_write("A"), None, dace.Memlet(data="A", subset="0:6:2", other_subset="0:3"))
    assert EliminateCopyBack().apply_pass(sdfg, {}) is None


def test_folding_a_nested_writer_onto_a_wider_array_restates_the_connector():
    """The body wrote the transient, so its connector describes that transient. Folding the copy onto a row of ``A``
    wires the connector to ``A``; it has to describe ``A``, not the transient (viterbi ``__rdo0`` / ``back``)."""
    sdfg = dace.SDFG("nested_copy_back")
    sdfg.add_array("A", [3, 4], dace.int64)
    sdfg.add_transient("t", [4], dace.int64)
    body = dace.SDFG("loop_body")
    body.add_symbol("i", dace.int64)
    body.add_array("t", [4], dace.int64)
    inner = body.add_state()
    task = inner.add_tasklet("set", {}, {"o"}, "o = 3")
    inner.add_edge(task, "o", inner.add_write("t"), None, dace.Memlet("t[i]"))
    state = sdfg.add_state()
    entry, exit_node = state.add_map("update", {"i": "0:4"})
    nested = state.add_nested_sdfg(body, set(), {"t"}, {"i": "i"})
    transient = state.add_access("t")
    state.add_nedge(entry, nested, dace.Memlet())
    state.add_memlet_path(nested, exit_node, transient, src_conn="t", memlet=dace.Memlet("t[i]"))
    state.add_nedge(transient, state.add_write("A"), dace.Memlet("t[0:4] -> [1, 0:4]"))
    sdfg.validate()
    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    leaked = [
        conn
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.NestedSDFG) and node.sdfg is not None
        for conn in windowed_connectors(node.sdfg)
    ]
    assert leaked == []
    got = np.zeros((3, 4), dtype=np.int64)
    sdfg(A=got)
    expected = np.zeros((3, 4), dtype=np.int64)
    expected[1, :] = 3
    assert np.array_equal(got, expected)


def test_ordering_edges_of_the_transient_move_to_the_map_and_the_target():
    """``D``, ordered ahead of the transient, stays ahead of the map writing ``A``; ``later``, ordered behind it,
    waits for ``A``."""
    sdfg = dace.SDFG("copy_back_behind_ordering")
    sdfg.add_array("A", [4], dace.float64)
    sdfg.add_array("C", [4], dace.float64)
    sdfg.add_array("D", [1], dace.float64)
    sdfg.add_transient("T", [4], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("m", {"i": "0:4"})
    tasklet = state.add_tasklet("dbl", {"a"}, {"o"}, "o = 2 * a")
    state.add_memlet_path(state.add_read("A"), entry, tasklet, dst_conn="a", memlet=dace.Memlet("A[i]"))
    tmp = state.add_access("T")
    state.add_memlet_path(tasklet, exit_, tmp, src_conn="o", memlet=dace.Memlet("T[i]"))
    target = state.add_write("A")
    state.add_nedge(tmp, target, dace.Memlet("T[0:4] -> [0:4]"))
    first = state.add_tasklet("first", {"c"}, {"o"}, "o = c")
    state.add_edge(state.add_read("C"), None, first, "c", dace.Memlet("C[0]"))
    d = state.add_write("D")
    state.add_edge(first, "o", d, None, dace.Memlet("D[0]"))
    state.add_nedge(d, tmp, dace.Memlet())
    later = state.add_tasklet("later", {}, {"o"}, "o = 7.0")
    state.add_nedge(tmp, later, dace.Memlet())
    state.add_edge(later, "o", state.add_write("C"), None, dace.Memlet("C[1]"))

    assert EliminateCopyBack().apply_pass(sdfg, {}) == 1
    assert "T" not in sdfg.arrays
    assert state.edges_between(d, entry) and state.edges_between(target, later)

    a, c, d_val = np.arange(4.0), np.full(4, 3.0), np.zeros(1)
    sdfg(A=a, C=c, D=d_val)
    assert np.array_equal(a, 2 * np.arange(4.0)) and d_val[0] == 3.0 and c[1] == 7.0


if __name__ == "__main__":
    test_a_slice_update_writes_its_target_in_place()
    test_a_read_of_the_neighbouring_element_keeps_the_copy_back()
    test_a_transient_only_ordered_after_a_map_is_left_alone()
    test_a_read_outside_the_updated_slice_writes_in_place()
    test_a_transient_used_in_part_folds_at_its_origin()
    test_a_read_inside_the_slice_at_another_element_keeps_the_copy_back()
    test_a_sibling_read_of_the_written_element_keeps_the_copy_back()
    test_a_stepped_copy_back_stays()
    test_folding_a_nested_writer_onto_a_wider_array_restates_the_connector()
    test_ordering_edges_of_the_transient_move_to_the_map_and_the_target()

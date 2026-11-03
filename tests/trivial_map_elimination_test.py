# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import unittest

import networkx

import dace
from dace.transformation.dataflow import TrivialMapElimination


def trivial_map_sdfg():
    sdfg = dace.SDFG("trivial_map")
    sdfg.add_array("A", [5], dace.float64)
    sdfg.add_array("B", [5], dace.float64)
    state = sdfg.add_state()

    # Nodes
    read = state.add_read("A")
    map_entry, map_exit = state.add_map("map", dict(i="0:1"))
    tasklet = state.add_tasklet("tasklet", {"a"}, {"b"}, "b = a")
    write = state.add_write("B")

    # Edges
    state.add_memlet_path(read, map_entry, tasklet, memlet=dace.Memlet.simple("A", "0"), dst_conn="a")
    state.add_memlet_path(tasklet, map_exit, write, memlet=dace.Memlet.simple("B", "i"), src_conn="b")

    sdfg.validate()
    return sdfg


def trivial_map_init_sdfg():
    sdfg = dace.SDFG("trivial_map_range_expanded")
    sdfg.add_array("B", [5, 1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    map_entry_outer, map_exit_outer = state.add_map("map_outer", dict(j="0:5"))
    map_entry_inner, map_exit_inner = state.add_map("map_inner", dict(i="0:1"))

    tasklet = state.add_tasklet("tasklet", {}, {"b"}, "b = 1")
    write = state.add_write("B")

    # Edges
    state.add_memlet_path(map_entry_outer, map_entry_inner, memlet=dace.Memlet())
    state.add_memlet_path(map_entry_inner, tasklet, memlet=dace.Memlet())

    state.add_memlet_path(
        tasklet, map_exit_inner, memlet=dace.Memlet.simple("B", "j, i"), src_conn="b", dst_conn="IN_B"
    )
    state.add_memlet_path(
        map_exit_inner, map_exit_outer, memlet=dace.Memlet.simple("B", "j, 0"), src_conn="OUT_B", dst_conn="IN_B"
    )
    state.add_memlet_path(map_exit_outer, write, memlet=dace.Memlet.simple("B", "0:5, 0"), src_conn="OUT_B")

    sdfg.validate()
    return sdfg


def trivial_map_with_dynamic_map_range_sdfg():
    sdfg = dace.SDFG("trivial_map_with_dynamic_map_range")
    state = sdfg.add_state("state1", is_start_block=True)

    for name in "ABC":
        sdfg.add_scalar(name, dtype=dace.float32, transient=False)
    A, B, C = (state.add_access(name) for name in "ABC")

    _, me, _ = state.add_mapped_tasklet(
        name="MAP",
        map_ranges=[("__i", "0:1"), ("__j", "10:11")],
        inputs={"__in": dace.Memlet("A[0]")},
        input_nodes={"A": A},
        code="__out = __in + 1",
        outputs={"__out": dace.Memlet("B[0]")},
        output_nodes={"B": B},
        external_edges=True,
    )
    state.add_edge(
        C,
        None,
        me,
        "dynamic_variable",
        dace.Memlet("C[0]"),
    )
    me.add_in_connector("dynamic_variable")
    sdfg.validate()

    return sdfg


def trivial_map_pseudo_init_sdfg():
    sdfg = dace.SDFG("trivial_map_range_expanded")
    sdfg.add_array("A", [5, 1], dace.float64)
    sdfg.add_array("B", [5, 1], dace.float64)
    state = sdfg.add_state()

    # Nodes
    map_entry_outer, map_exit_outer = state.add_map("map_outer", dict(j="0:5"))
    map_entry_inner, map_exit_inner = state.add_map("map_inner", dict(i="0:1"))

    read = state.add_read("A")
    tasklet = state.add_tasklet("tasklet", {"a"}, {"b"}, "b = a")
    write = state.add_write("B")

    # Edges
    state.add_memlet_path(map_entry_outer, map_entry_inner, memlet=dace.Memlet())
    state.add_memlet_path(
        read, map_entry_outer, map_entry_inner, memlet=dace.Memlet.simple("A", "0:5, 0"), dst_conn="IN_A"
    )
    state.add_memlet_path(map_entry_inner, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(
        map_entry_inner, tasklet, memlet=dace.Memlet.simple("A", "j, 0"), src_conn="OUT_A", dst_conn="a"
    )

    state.add_memlet_path(
        tasklet, map_exit_inner, memlet=dace.Memlet.simple("B", "j, i"), src_conn="b", dst_conn="IN_B"
    )
    state.add_memlet_path(
        map_exit_inner, map_exit_outer, memlet=dace.Memlet.simple("B", "j, 0"), src_conn="OUT_B", dst_conn="IN_B"
    )
    state.add_memlet_path(map_exit_outer, write, memlet=dace.Memlet.simple("B", "0:5, 0"), src_conn="OUT_B")

    sdfg.validate()
    return sdfg


def ordered_trivial_map_sdfg():
    """``A[0] = 1`` (access node A1), then a one-iteration map reading ``A[0]`` through a second access node A2 and
    writing ``B[0]``, then ``A[0] = 2`` through A3; the two hazards on A are held only by empty ordering edges
    ``A1 -> map entry`` and ``map exit -> writer of A3``."""
    sdfg = dace.SDFG("trivial_map_ordering")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    first = state.add_tasklet("first", {}, {"o"}, "o = 1.0")
    a1 = state.add_write("A")
    state.add_edge(first, "o", a1, None, dace.Memlet("A[0]"))
    entry, exit_ = state.add_map("map", dict(i="0:1"))
    read = state.add_tasklet("read", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(state.add_read("A"), entry, read, dst_conn="x", memlet=dace.Memlet("A[0]"))
    state.add_memlet_path(read, exit_, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[0]"))
    state.add_nedge(a1, entry, dace.Memlet())
    last = state.add_tasklet("last", {}, {"o"}, "o = 2.0")
    state.add_edge(last, "o", state.add_write("A"), None, dace.Memlet("A[0]"))
    state.add_nedge(exit_, last, dace.Memlet())
    sdfg.validate()
    return sdfg, state, first, read, last


def test_removing_a_trivial_map_keeps_the_order_its_empty_edges_held():
    sdfg, state, first, read, last = ordered_trivial_map_sdfg()
    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    assert not any(isinstance(n, dace.nodes.MapEntry) for n in state.nodes())
    sdfg.validate()
    assert networkx.has_path(state._nx, first, read), "the write of A[0] no longer precedes its read"
    assert networkx.has_path(state._nx, read, last), "the read of A[0] no longer precedes its overwrite"


def data_ordered_trivial_map_sdfg():
    """``t = 2 * u`` (``first``), then a one-iteration map whose ``use`` reads ``t`` and whose ``bump`` updates ``u``
    in place. Only the map entry orders ``bump`` after ``first``'s read of ``u``: ``bump`` takes no input from ``t``
    (warpx_boris_push on GPU, whose ``uz_p`` read the updated ``uz``)."""
    sdfg = dace.SDFG("trivial_map_data_ordering")
    for name in ("u", "t", "b"):
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state()
    u_in = state.add_read("u")
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = 2 * x")
    t = state.add_access("t")
    state.add_edge(u_in, None, first, "x", dace.Memlet("u[0]"))
    state.add_edge(first, "y", t, None, dace.Memlet("t[0]"))
    entry, exit_ = state.add_map("map", dict(i="0:1"))
    use = state.add_tasklet("use", {"x"}, {"y"}, "y = x + 1")
    state.add_memlet_path(t, entry, use, dst_conn="x", memlet=dace.Memlet("t[0]"))
    state.add_memlet_path(use, exit_, state.add_write("b"), src_conn="y", memlet=dace.Memlet("b[0]"))
    bump = state.add_tasklet("bump", {"x"}, {"y"}, "y = x + 1")
    state.add_memlet_path(u_in, entry, bump, dst_conn="x", memlet=dace.Memlet("u[0]"))
    state.add_memlet_path(bump, exit_, state.add_write("u"), src_conn="y", memlet=dace.Memlet("u[0]"))
    sdfg.validate()
    return sdfg, state, first, bump


def test_removing_a_trivial_map_keeps_the_order_its_data_inputs_held():
    sdfg, state, first, bump = data_ordered_trivial_map_sdfg()
    assert networkx.has_path(state._nx, first, bump)
    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert networkx.has_path(state._nx, first, bump), "the read of u no longer precedes its in-place update"


def test_removing_a_trivial_map_orders_a_nested_scope_by_its_entry():
    """The in-place update of ``u`` sits in a nested map; its order after ``first`` must attach to the nested entry."""
    sdfg = dace.SDFG("trivial_map_nested_data_ordering")
    for name, size in (("u", 2), ("t", 1), ("b", 1)):
        sdfg.add_array(name, [size], dace.float64)
    state = sdfg.add_state()
    u_in = state.add_read("u")
    first = state.add_tasklet("first", {"x"}, {"y"}, "y = 2 * x")
    t = state.add_access("t")
    state.add_edge(u_in, None, first, "x", dace.Memlet("u[0]"))
    state.add_edge(first, "y", t, None, dace.Memlet("t[0]"))
    entry, exit_ = state.add_map("map", dict(i="0:1"))
    use = state.add_tasklet("use", {"x"}, {"y"}, "y = x + 1")
    state.add_memlet_path(t, entry, use, dst_conn="x", memlet=dace.Memlet("t[0]"))
    state.add_memlet_path(use, exit_, state.add_write("b"), src_conn="y", memlet=dace.Memlet("b[0]"))
    bump = state.add_tasklet("bump", {"x"}, {"y"}, "y = x + 1")
    inner_entry, inner_exit = state.add_map("inner", dict(j="0:2"))
    state.add_memlet_path(u_in, entry, inner_entry, bump, dst_conn="x", memlet=dace.Memlet("u[j]"))
    state.add_memlet_path(bump, inner_exit, exit_, state.add_write("u"), src_conn="y", memlet=dace.Memlet("u[j]"))
    sdfg.validate()
    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert state.scope_dict()[bump] is inner_entry
    assert networkx.has_path(state._nx, first, inner_entry), "the read of u no longer precedes its in-place update"


class TrivialMapEliminationTest(unittest.TestCase):
    """
    Tests the case where the map has an empty input edge
    """

    def test_can_be_applied(self):
        graph = trivial_map_sdfg()

        count = graph.apply_transformations(TrivialMapElimination)

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 0)

    def test_raplaces_map_params_in_scope(self):
        # Tests if the 'i' in the range of the memlet to B gets replaced
        # with the value 'i' obtains in the map, namely '0'.

        graph = trivial_map_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        B = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.AccessNode) and n.data == "B"][0]
        out_memlet = state.in_edges(B)[0]
        self.assertEqual(out_memlet.data.subset, dace.subsets.Range([(0, 0, 1)]))


class TrivialMapInitEliminationTest(unittest.TestCase):
    def test_can_be_applied(self):
        graph = trivial_map_init_sdfg()

        count = graph.apply_transformations(TrivialMapElimination, validate=False, validate_all=False)
        graph.validate()

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_init_sdfg()

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 2)

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)

    def test_reconnects_edges(self):
        graph = trivial_map_init_sdfg()

        graph.apply_transformations(TrivialMapElimination)
        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        # Check that there is an outgoing edge from the map entry
        self.assertEqual(len(state.out_edges(map_entries[0])), 1)


class TrivialMapPseudoInitEliminationTest(unittest.TestCase):
    """
    Test cases where the map has an empty input and a non empty input
    """

    def test_can_be_applied(self):
        graph = trivial_map_pseudo_init_sdfg()

        count = graph.apply_transformations(TrivialMapElimination, validate=False, validate_all=False)
        graph.validate()

        self.assertGreater(count, 0)

    def test_removes_map(self):
        graph = trivial_map_pseudo_init_sdfg()

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 2)

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)

    def test_reconnects_edges(self):
        graph = trivial_map_pseudo_init_sdfg()

        graph.apply_transformations(TrivialMapElimination)
        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        # Check that there is an outgoing edge from the map entry
        self.assertEqual(len(state.out_edges(map_entries[0])), 1)


class TrivialMapEliminationWithDynamicMapRangesTest(unittest.TestCase):
    """
    Tests the case where the map has trivial ranges and dynamic map ranges.
    """

    def test_can_be_applied(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        count = graph.apply_transformations(TrivialMapElimination)
        graph.validate()

        self.assertEqual(count, 1)

    def test_removes_map(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        graph.apply_transformations(TrivialMapElimination)

        state = graph.nodes()[0]
        map_entries = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry)]
        self.assertEqual(len(map_entries), 1)
        self.assertEqual(state.in_degree(map_entries[0]), 2)
        self.assertTrue(any(e.dst_conn.startswith("IN_") for e in state.in_edges(map_entries[0])))
        self.assertTrue(any(not e.dst_conn.startswith("IN_") for e in state.in_edges(map_entries[0])))

    def test_not_remove_dynamic_map_range(self):
        graph = trivial_map_with_dynamic_map_range_sdfg()

        count1 = graph.apply_transformations(TrivialMapElimination)
        self.assertEqual(count1, 1)

        count2 = graph.apply_transformations(TrivialMapElimination)
        self.assertEqual(count2, 0)


def test_an_ordering_edge_into_the_removed_map_still_orders_its_body():
    """``t0 -> A -(empty)-> map[i=0] { t1 -> A }``: the empty edge is all that orders the two writes of ``A``."""
    sdfg = dace.SDFG("ordering_into_entry")
    sdfg.add_array("A", [1], dace.float64)
    state = sdfg.add_state()
    first = state.add_access("A")
    t0 = state.add_tasklet("t0", {}, {"o"}, "o = 1")
    state.add_edge(t0, "o", first, None, dace.Memlet("A[0]"))
    me, mx = state.add_map("trivial", {"i": "0:1"})
    t1 = state.add_tasklet("t1", {}, {"o"}, "o = 2")
    state.add_edge(first, None, me, None, dace.Memlet())
    state.add_edge(me, None, t1, None, dace.Memlet())
    state.add_memlet_path(t1, mx, state.add_access("A"), src_conn="o", memlet=dace.Memlet("A[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert networkx.has_path(state.nx, first, t1)


def test_a_body_node_the_entry_reaches_only_by_an_ordering_edge_stays_after_the_inputs():
    """``P -> C; C -(empty)-> B -> map[i=0] { B -> t1; (empty) -> t2 -> C }``: ``t2`` overwrites ``C`` and only the
    map ordered it after ``P``."""
    sdfg = dace.SDFG("ordering_out_of_entry")
    for name in "BCD":
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state()
    c_first = state.add_access("C")
    p = state.add_tasklet("p", {}, {"o"}, "o = 1")
    state.add_edge(p, "o", c_first, None, dace.Memlet("C[0]"))
    b = state.add_access("B")
    state.add_edge(c_first, None, b, None, dace.Memlet())
    me, mx = state.add_map("trivial", {"i": "0:1"})
    t1 = state.add_tasklet("t1", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(b, me, t1, dst_conn="x", memlet=dace.Memlet("B[0]"))
    state.add_memlet_path(t1, mx, state.add_access("D"), src_conn="y", memlet=dace.Memlet("D[0]"))
    t2 = state.add_tasklet("t2", {}, {"o"}, "o = 2")
    state.add_edge(me, None, t2, None, dace.Memlet())
    state.add_memlet_path(t2, mx, state.add_access("C"), src_conn="o", memlet=dace.Memlet("C[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert networkx.has_path(state.nx, c_first, t2)


def test_an_ordering_edge_out_of_the_removed_map_still_orders_its_body():
    """``map[i=0] { A -> t1 -> D } -(empty)-> t3 -> A``: ``t3`` overwrites what ``t1`` reads, ordered by the exit."""
    sdfg = dace.SDFG("ordering_out_of_exit")
    for name in "AD":
        sdfg.add_array(name, [1], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("trivial", {"i": "0:1"})
    t1 = state.add_tasklet("t1", {"x"}, {"y"}, "y = x")
    state.add_memlet_path(state.add_access("A"), me, t1, dst_conn="x", memlet=dace.Memlet("A[0]"))
    state.add_memlet_path(t1, mx, state.add_access("D"), src_conn="y", memlet=dace.Memlet("D[0]"))
    t3 = state.add_tasklet("t3", {}, {"o"}, "o = 3")
    state.add_edge(mx, None, t3, None, dace.Memlet())
    state.add_edge(t3, "o", state.add_access("A"), None, dace.Memlet("A[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert networkx.has_path(state.nx, t1, t3)


def test_the_order_reaches_a_nested_scope_at_its_entry():
    """The removed map's body is a nested map: its order attaches to the nested entry, never inside the scope."""
    sdfg = dace.SDFG("ordering_nested_scope")
    sdfg.add_array("A", [4], dace.float64)
    state = sdfg.add_state()
    first = state.add_access("A")
    t0 = state.add_tasklet("t0", {}, {"o"}, "o = 1")
    state.add_edge(t0, "o", first, None, dace.Memlet("A[0]"))
    me, mx = state.add_map("trivial", {"i": "0:1"})
    ie, ix = state.add_map("inner", {"j": "0:4"})
    t1 = state.add_tasklet("t1", {}, {"o"}, "o = 2")
    state.add_edge(first, None, me, None, dace.Memlet())
    state.add_edge(me, None, ie, None, dace.Memlet())
    state.add_edge(ie, None, t1, None, dace.Memlet())
    state.add_memlet_path(t1, ix, mx, state.add_access("A"), src_conn="o", memlet=dace.Memlet("A[j]"))
    sdfg.validate()

    assert sdfg.apply_transformations(TrivialMapElimination) == 1
    sdfg.validate()
    assert networkx.has_path(state.nx, first, t1)
    assert state.scope_dict()[t1] is ie
    assert all(e.src is ie for e in state.in_edges(t1))


if __name__ == "__main__":
    test_removing_a_trivial_map_keeps_the_order_its_empty_edges_held()
    test_removing_a_trivial_map_keeps_the_order_its_data_inputs_held()
    test_removing_a_trivial_map_orders_a_nested_scope_by_its_entry()
    test_an_ordering_edge_into_the_removed_map_still_orders_its_body()
    test_a_body_node_the_entry_reaches_only_by_an_ordering_edge_stays_after_the_inputs()
    test_an_ordering_edge_out_of_the_removed_map_still_orders_its_body()
    test_the_order_reaches_a_nested_scope_at_its_entry()
    unittest.main()

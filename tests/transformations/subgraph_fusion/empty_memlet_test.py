# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SubgraphFusion`` with empty memlets (ordering edges, no connectors) on the maps it fuses."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion

N = dace.symbol("N")


class Chain:
    """``B[i] = A[i] + 1`` then ``C[i] = 2 * B[i]`` in one state, ``B`` the intermediate; the cases add empty
    memlets around the two maps."""

    def __init__(self, name: str) -> None:
        self.sdfg = dace.SDFG(name)
        for array in "ACDEH":
            self.sdfg.add_array(array, [N], dace.float64)
        self.sdfg.add_array("F", [1], dace.float64)
        self.sdfg.add_transient("B", [N], dace.float64)
        self.state = self.sdfg.add_state("chain")
        self.a, self.b = self.state.add_read("A"), self.state.add_access("B")
        self.first = self.mapped("first", "a", "A", "b = a + 1", "b", "B", self.b)
        self.second = self.mapped("second", "b", "B", "c = 2 * b", "c", "C", None, self.b)
        self.fused: list[nodes.Node] = [*self.scope(self.first), self.b, *self.scope(self.second)]

    def mapped(self, label, src_conn, src, code, dst_conn, dst, out_node=None, in_node=None):
        """A map ``dst[i] = code(src[i])``; returns ``(tasklet, entry, exit)``."""
        return self.state.add_mapped_tasklet(
            label,
            {"i": "0:N"},
            {src_conn: dace.Memlet(f"{src}[i]")},
            code,
            {dst_conn: dace.Memlet(f"{dst}[i]")},
            external_edges=True,
            input_nodes={src: in_node} if in_node is not None else ({src: self.a} if src == "A" else None),
            output_nodes={dst: out_node} if out_node is not None else None,
        )

    @staticmethod
    def scope(mapped) -> list[nodes.Node]:
        tasklet, entry, exit_ = mapped
        return [entry, tasklet, exit_]

    def flag(self) -> nodes.AccessNode:
        """``F[0] = 1`` by a tasklet outside the maps."""
        flag = self.state.add_access("F")
        self.state.add_edge(self.state.add_tasklet("flag", {}, {"f"}, "f = 1"), "f", flag, None, dace.Memlet("F[0]"))
        return flag

    def fuse(self) -> SubgraphFusion:
        self.sdfg.validate()
        fusion = SubgraphFusion()
        fusion.setup_match(SubgraphView(self.state, self.fused), self.sdfg.cfg_id, self.sdfg.node_id(self.state))
        return fusion

    def fused_scope(self) -> tuple[nodes.MapEntry, nodes.MapExit]:
        """The one top-level map that is not ``after`` (the reader the out-of-exit case adds)."""
        entries = [
            n
            for n in self.state.nodes()
            if isinstance(n, nodes.MapEntry)
            and self.state.entry_node(n) is None
            and not n.map.label.startswith("after")
        ]
        assert len(entries) == 1
        return entries[0], self.state.exit_node(entries[0])


def run(sdfg: dace.SDFG, size: int = 17) -> dict[str, np.ndarray]:
    arrays = {name: np.zeros(size) for name in "CDEH"}
    arrays["A"] = np.random.default_rng(0).random(size)
    arrays["F"] = np.zeros(1)
    sdfg(**arrays, N=size)
    return arrays


def test_an_ordering_edge_into_both_maps_orders_the_fused_map_once():
    """``F`` orders both maps; after fusion one empty memlet orders the fused map after ``F``."""
    chain = Chain("empty_memlet_into_entries")
    flag = chain.flag()
    for mapped in (chain.first, chain.second):
        chain.state.add_nedge(flag, mapped[1], dace.Memlet())
    fusion = chain.fuse()
    assert fusion.can_be_applied(chain.sdfg, fusion.subgraph_view(chain.sdfg))
    fusion.apply(chain.sdfg)
    chain.sdfg.validate()

    entry, exit_ = chain.fused_scope()
    assert [e.dst for e in chain.state.out_edges(flag)] == [entry]
    arrays = run(chain.sdfg)
    np.testing.assert_allclose(arrays["C"], 2 * (arrays["A"] + 1))
    assert arrays["F"][0] == 1


def test_an_ordering_edge_out_of_a_map_follows_the_fused_map():
    """The first map's exit orders ``E``, which a later map reads; after fusion the fused map's exit orders it."""
    chain = Chain("empty_memlet_out_of_exit")
    e = chain.state.add_access("E")
    chain.state.add_nedge(chain.first[2], e, dace.Memlet())
    chain.mapped("after", "e", "E", "h = e", "h", "H", in_node=e)
    fusion = chain.fuse()
    assert fusion.can_be_applied(chain.sdfg, fusion.subgraph_view(chain.sdfg))
    fusion.apply(chain.sdfg)
    chain.sdfg.validate()

    entry, exit_ = chain.fused_scope()
    assert [edge.src for edge in chain.state.in_edges(e) if edge.data.is_empty()] == [exit_]
    arrays = run(chain.sdfg)
    np.testing.assert_allclose(arrays["C"], 2 * (arrays["A"] + 1))


def test_an_ordering_edge_into_the_intermediate_is_not_fused():
    """The first map's exit orders ``E``, which the second map reads: ``E`` is an intermediate tied to the whole
    first map, and the fusion declines (it used to raise ``IndexError``)."""
    chain = Chain("empty_memlet_into_intermediate")
    e = chain.state.add_access("E")
    chain.state.add_nedge(chain.first[2], e, dace.Memlet())
    chain.second[0].add_in_connector("e")
    chain.state.add_memlet_path(e, chain.second[1], chain.second[0], dst_conn="e", memlet=dace.Memlet("E[i]"))
    chain.fused.append(e)
    fusion = chain.fuse()
    assert not fusion.can_be_applied(chain.sdfg, fusion.subgraph_view(chain.sdfg))


def test_tasklets_without_inputs_fuse():
    """Inner empty memlets (an entry into a tasklet that reads nothing) keep their place inside the fused map."""
    sdfg = dace.SDFG("empty_memlet_inner")
    sdfg.add_array("C", [N], dace.float64)
    sdfg.add_transient("B", [N], dace.float64)
    state = sdfg.add_state("chain")
    b = state.add_access("B")
    first = state.add_mapped_tasklet(
        "first", {"i": "0:N"}, {}, "b = i", {"b": dace.Memlet("B[i]")}, external_edges=True, output_nodes={"B": b}
    )
    second = state.add_mapped_tasklet(
        "second",
        {"i": "0:N"},
        {"b": dace.Memlet("B[i]")},
        "c = 2 * b",
        {"c": dace.Memlet("C[i]")},
        external_edges=True,
        input_nodes={"B": b},
    )
    sdfg.validate()
    fusion = SubgraphFusion()
    fusion.setup_match(SubgraphView(state, [*Chain.scope(first), b, *Chain.scope(second)]), sdfg.cfg_id, 0)
    assert fusion.can_be_applied(sdfg, fusion.subgraph_view(sdfg))
    fusion.apply(sdfg)
    sdfg.validate()

    c = np.zeros(17)
    sdfg(C=c, N=17)
    np.testing.assert_allclose(c, 2 * np.arange(17))


def dynamic_range_chain(case: str) -> tuple[dace.SDFG, dace.SDFGState, list[nodes.Node]]:
    """``B[i] = A[i] + 1`` then ``C[i] = 2 * B[i]`` for ``i`` in ``0:lim``, both maps reading ``lim`` through a
    dynamic map range connector.

    :param case: ``"one_node"``: both read one access node of ``LIM``. ``"two_nodes"``: each reads its own.
                 ``"written"``: like ``two_nodes``, but a tasklet writes ``LIM`` into the second map's node.
                 ``"other_element"``: the second map reads ``LIM[1]`` instead of ``LIM[0]``.
    :return: The SDFG, its state, and the nodes to fuse.
    """
    sdfg = dace.SDFG(f"dynamic_range_{case}")
    for name in "AC":
        sdfg.add_array(name, [N], dace.float64)
    sdfg.add_array("LIM", [2], dace.int64)
    sdfg.add_transient("B", [N], dace.float64)
    state = sdfg.add_state("chain")
    first_lim = state.add_read("LIM")
    second_lim = first_lim if case == "one_node" else state.add_access("LIM")
    if case == "written":
        writer = state.add_tasklet("set_lim", {}, {"l"}, "l = 10")
        state.add_edge(writer, "l", second_lim, None, dace.Memlet("LIM[0]"))
    b = state.add_access("B")
    fused = [b]
    for label, src, code, dst, lim, element in [
        ("first", state.add_read("A"), "o = a + 1", b, first_lim, 0),
        ("second", b, "o = 2 * a", state.add_write("C"), second_lim, 1 if case == "other_element" else 0),
    ]:
        entry, exit_ = state.add_map(label, {"i": "0:lim"})
        entry.add_in_connector("lim")
        state.add_edge(lim, None, entry, "lim", dace.Memlet(f"LIM[{element}]"))
        tasklet = state.add_tasklet(label, {"a"}, {"o"}, code)
        state.add_memlet_path(src, entry, tasklet, dst_conn="a", memlet=dace.Memlet(f"{src.data}[i]"))
        state.add_memlet_path(tasklet, exit_, dst, src_conn="o", memlet=dace.Memlet(f"{dst.data}[i]"))
        fused += [entry, tasklet, exit_]
    sdfg.validate()
    return sdfg, state, fused


def fuse_dynamic_range_chain(case: str) -> tuple[dace.SDFG, dace.SDFGState, bool]:
    sdfg, state, fused = dynamic_range_chain(case)
    fusion = SubgraphFusion()
    fusion.setup_match(SubgraphView(state, fused), sdfg.cfg_id, sdfg.node_id(state))
    applicable = fusion.can_be_applied(sdfg, fusion.subgraph_view(sdfg))
    if applicable:
        fusion.apply(sdfg)
        sdfg.validate()
    return sdfg, state, applicable


@pytest.mark.parametrize("case", ["one_node", "two_nodes"])
def test_maps_reading_the_same_dynamic_range_input_fuse(case):
    """Both maps read ``LIM[0]``, which nothing in the state writes: the fused map reads it once."""
    sdfg, state, applicable = fuse_dynamic_range_chain(case)
    assert applicable
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None)
    assert [e.dst_conn for e in state.in_edges(entry) if e.dst_conn == "lim"] == ["lim"]

    size, lim = 17, 10
    a = np.random.default_rng(0).random(size)
    c = np.zeros(size)
    sdfg(A=a, C=c, LIM=np.array([lim, 0], dtype=np.int64), N=size)
    np.testing.assert_allclose(c[:lim], 2 * (a[:lim] + 1))
    assert not c[lim:].any()


@pytest.mark.parametrize("case", ["written", "other_element"])
def test_maps_that_may_read_different_dynamic_range_values_are_not_fused(case):
    """A write into ``LIM`` in the state, or a different element, may give the maps different ranges."""
    sdfg, state, applicable = fuse_dynamic_range_chain(case)
    assert not applicable


def test_an_ordering_edge_on_a_scope_connector_still_orders_the_body():
    """``F -> IN_x -> OUT_x -> tasklet`` with empty memlets orders the second map's body after ``F``; fusion dropped
    both edges."""
    chain = Chain("empty_memlet_on_scope_connector")
    flag = chain.flag()
    tasklet, entry, _ = chain.second
    entry.add_in_connector("IN_order")
    entry.add_out_connector("OUT_order")
    chain.state.add_edge(flag, None, entry, "IN_order", dace.Memlet())
    chain.state.add_edge(entry, "OUT_order", tasklet, None, dace.Memlet())
    fusion = chain.fuse()
    assert fusion.can_be_applied(chain.sdfg, fusion.subgraph_view(chain.sdfg))
    fusion.apply(chain.sdfg)
    chain.sdfg.validate()

    assert tasklet in set(chain.state.bfs_nodes(flag))
    arrays = run(chain.sdfg)
    np.testing.assert_allclose(arrays["C"], 2 * (arrays["A"] + 1))


def test_fuse_leaves_a_dynamic_map_range_input_unfused():
    """``fuse`` called without ``can_be_applied`` leaves the maps as they are rather than drop the range input."""
    chain = Chain("empty_memlet_fuse_dynamic_range")
    chain.sdfg.add_scalar("lim", dace.int64)
    entry = chain.second[1]
    entry.add_in_connector("lim")
    chain.state.add_edge(chain.state.add_read("lim"), None, entry, "lim", dace.Memlet("lim[0]"))

    SubgraphFusion().fuse(chain.sdfg, chain.state, [chain.first[1], entry])

    chain.sdfg.validate()
    assert {chain.first[1], entry} <= set(chain.state.nodes())


def test_fuse_leaves_a_map_ordered_after_another_unfused():
    """An empty memlet from the first map's exit into the second map's entry orders the maps whole; ``fuse`` called
    without ``can_be_applied`` leaves them as they are."""
    chain = Chain("empty_memlet_fuse_exit_to_entry")
    chain.state.add_nedge(chain.first[2], chain.second[1], dace.Memlet())

    SubgraphFusion().fuse(chain.sdfg, chain.state, [chain.first[1], chain.second[1]])

    chain.sdfg.validate()
    assert {chain.first[1], chain.second[1]} <= set(chain.state.nodes())


def test_an_ordering_edge_beside_a_data_edge_adds_no_empty_memlet():
    """``B`` feeds the second map and also orders its entry: after fusion only the data edge connects ``B`` and the
    body."""
    chain = Chain("empty_memlet_beside_data_edge")
    chain.state.add_nedge(chain.b, chain.second[1], dace.Memlet())
    fusion = chain.fuse()
    assert fusion.can_be_applied(chain.sdfg, fusion.subgraph_view(chain.sdfg))
    fusion.apply(chain.sdfg)
    chain.sdfg.validate()

    assert [e.data.is_empty() for e in chain.state.out_edges(chain.b)] == [False]
    arrays = run(chain.sdfg)
    np.testing.assert_allclose(arrays["C"], 2 * (arrays["A"] + 1))


if __name__ == "__main__":
    test_an_ordering_edge_into_both_maps_orders_the_fused_map_once()
    test_an_ordering_edge_out_of_a_map_follows_the_fused_map()
    test_an_ordering_edge_into_the_intermediate_is_not_fused()
    test_tasklets_without_inputs_fuse()
    for case in ["one_node", "two_nodes"]:
        test_maps_reading_the_same_dynamic_range_input_fuse(case)
    for case in ["written", "other_element"]:
        test_maps_that_may_read_different_dynamic_range_values_are_not_fused(case)
    test_an_ordering_edge_on_a_scope_connector_still_orders_the_body()
    test_fuse_leaves_a_dynamic_map_range_input_unfused()
    test_fuse_leaves_a_map_ordered_after_another_unfused()
    test_an_ordering_edge_beside_a_data_edge_adds_no_empty_memlet()

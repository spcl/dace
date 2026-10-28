# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SubgraphFusion`` with empty memlets (ordering edges, no connectors) on the maps it fuses."""

import numpy as np

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


if __name__ == "__main__":
    test_an_ordering_edge_into_both_maps_orders_the_fused_map_once()
    test_an_ordering_edge_out_of_a_map_follows_the_fused_map()
    test_an_ordering_edge_into_the_intermediate_is_not_fused()

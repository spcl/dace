# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Split a map whose body is a nested SDFG into two maps at a chosen block of that body."""

from typing import Optional

from dace import properties, sdfg as sd
from dace.ordered import OrderedSet
from dace.sdfg import graph as gr, nodes, utils as sdutil
from dace.sdfg import state as cf
from dace.transformation import helpers, transformation
from dace.transformation.dataflow import map_fission
from dace.transformation.interstate import multistate_inline


def drop_uninitialized_inputs(body: sd.SDFG) -> None:
    """Remove every nested-SDFG input of ``body`` that reads a transient no earlier state wrote.

    ``nest_sdfg_subgraph`` makes an input of every container a nest reads, including one it writes first; read from
    outside, such a transient is uninitialized.
    """
    written = OrderedSet()
    for state in sdutil.dfs_topological_sort(body):
        for nest in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
            for edge in state.in_edges(nest):
                if body.arrays[edge.data.data].transient and edge.data.data not in written:
                    state.remove_edge(edge)
                    nest.remove_in_connector(edge.dst_conn)
                    if state.degree(edge.src) == 0:
                        state.remove_node(edge.src)
            written.update(edge.data.data for edge in state.out_edges(nest))


@properties.make_properties
@transformation.explicit_cf_compatible
class SubgraphFission(transformation.SingleStateTransformation):
    """``map i: {A; B}`` -> ``map i: A; map i: B``, cutting the map's nested-SDFG body after the block ``cut``.

    ``MapFission`` splits a nested-SDFG body at every top-level block; this splits it at one. The blocks up to
    ``cut`` and the blocks after it are nested into one nested SDFG each, then ``MapFission`` makes one map per
    half and widens every transient the halves share by the map's range.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    cut = properties.Property(dtype=str, default='', desc='Label of the last top-level body block of the first map')

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def cut_block(self) -> Optional[cf.ControlFlowBlock]:
        return next((b for b in self.nested_sdfg.sdfg.nodes() if b.label == self.cut), None)

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        body = self.nested_sdfg.sdfg
        cut = self.cut_block()
        if cut is None:
            return False
        # a straight line of blocks, cut before its last, joined by a plain edge the halves cannot lose
        outs = body.out_edges(cut)
        if len(outs) != 1 or any(body.in_degree(b) > 1 or body.out_degree(b) > 1 for b in body.nodes()):
            return False
        if not outs[0].data.is_unconditional() or outs[0].data.assignments:
            return False
        # MapFission judges the unsplit body at least as strictly as the split one: nesting the halves hides their
        # interstate assignments from the map parameters it refuses
        return map_fission.MapFission.can_be_applied_to(sdfg,
                                                        expr_index=1,
                                                        map_entry=self.map_entry,
                                                        nested_sdfg=self.nested_sdfg)

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        entry, nsdfg, cut = self.map_entry, self.nested_sdfg, self.cut_block()
        body = nsdfg.sdfg
        blocks = list(sdutil.dfs_topological_sort(body))
        index = blocks.index(cut)
        for group in (blocks[:index + 1], blocks[index + 1:]):
            if len(group) == 1 and isinstance(group[0], sd.SDFGState):
                # nest_sdfg_subgraph leaves a lone state in place, and MapFission would split it per component
                group = [group[0], body.add_state_after(group[0])]
            helpers.nest_sdfg_subgraph(body, gr.SubgraphView(body, group))
        drop_uninitialized_inputs(body)
        # MapFission moves the two maps inside the nested SDFG; inlining it leaves them in two states
        map_fission.MapFission.apply_to(sdfg, expr_index=1, map_entry=entry, nested_sdfg=nsdfg)
        multistate_inline.InlineMultistateSDFG.apply_to(sdfg, nested_sdfg=nsdfg)
        root = sdfg.root_sdfg
        sdutil.set_nested_sdfg_parent_references(root)
        root.reset_cfg_list()

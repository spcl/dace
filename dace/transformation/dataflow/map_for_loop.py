# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement a map->for loop transformation.
"""

import dace
from dace import properties, symbolic
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import LoopRegion
from dace.transformation import transformation
from typing import Tuple, Optional


@properties.make_properties
class MapToForLoop(transformation.SingleStateTransformation):
    """Implements the Map to for-loop transformation.

    Takes a map and enforces a sequential schedule by transforming it
    into a LoopRegion. The historical implementation wrapped the map's
    body in a NestedSDFG and put the LoopRegion inside it; with
    ``inline_after=True`` (default) the wrapping NSDFG is flattened
    via :class:`~dace.transformation.interstate.expand_nested_sdfg_inputs.ExpandNestedSDFGInputs`
    + :class:`~dace.transformation.interstate.multistate_inline.InlineMultistateSDFG`
    so the LoopRegion lands directly at the parent CFR. Set
    ``inline_after=False`` to keep the legacy wrapped form.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    loop_region: Optional[LoopRegion] = None

    inline_after = properties.Property(dtype=bool,
                                       default=True,
                                       desc='Flatten the wrapping NestedSDFG via '
                                       'ExpandNestedSDFGInputs + InlineMultistateSDFG so the resulting LoopRegion '
                                       'lives at the parent CFR. On by default so canonicalization sees a clean '
                                       'CFR nest without spurious NSDFG boundaries. Set to False to keep the '
                                       'legacy wrapped form (e.g., when a downstream test asserts on the NSDFG).')

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # Only uni-dimensional maps are accepted.
        if len(self.map_entry.map.params) > 1:
            return False

        # Refuse a map that still has a WCR (reduction) output. By this point
        # WCRToAugAssign has rewritten every conflict-free (injective) WCR into an
        # explicit RMW, so a surviving WCR output is a genuine parallel reduction.
        # Lowering it to a sequential loop serializes the reduction AND severs an
        # in-state-consumed accumulator; keep it a parallel map so it codegens to an
        # OpenMP reduction and the producer->consumer edge is preserved.
        map_exit = graph.exit_node(self.map_entry)
        for e in graph.out_edges(map_exit):
            if e.data is not None and e.data.wcr is not None:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> Tuple[nodes.NestedSDFG, SDFGState]:
        """ Applies the transformation and returns a tuple with the new nested
            SDFG node and the main state in the for-loop. """

        # Avoid import loop
        from dace.transformation.helpers import nest_state_subgraph

        # Retrieve map entry and exit nodes.
        map_entry = self.map_entry
        map_exit = graph.exit_node(map_entry)

        loop_idx = map_entry.map.params[0]
        loop_from, loop_to, loop_step = map_entry.map.range[0]

        # Turn the map scope into a nested SDFG
        node = nest_state_subgraph(sdfg, graph, graph.scope_subgraph(map_entry))

        nsdfg: SDFG = node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]

        # If map range is dynamic, replace loop expressions with memlets
        param_to_edge = {}
        for edge in nstate.in_edges(map_entry):
            if edge.dst_conn and not edge.dst_conn.startswith('IN_'):
                param = '__DACE_P%d' % len(param_to_edge)
                repldict = {symbolic.pystr_to_symbolic(edge.dst_conn): param}
                param_to_edge[param] = edge
                loop_from = loop_from.subs(repldict)
                loop_to = loop_to.subs(repldict)
                loop_step = loop_step.subs(repldict)

        # Avoiding import loop
        from dace.codegen.targets.cpp import cpp_array_expr

        def replace_param(param):
            param = symbolic.symstr(param, cpp_mode=False)
            for p, pval in param_to_edge.items():
                # TODO: Correct w.r.t. connector type
                param = param.replace(p, cpp_array_expr(nsdfg, pval.data))
            return param

        # End of dynamic input range

        # Create a loop inside the nested SDFG
        loop_region = LoopRegion('loop_' + map_entry.map.label, '%s < %s' % (loop_idx, replace_param(loop_to + 1)),
                                 loop_idx, '%s = %s' % (loop_idx, replace_param(loop_from)),
                                 '%s = %s + %s' % (loop_idx, loop_idx, replace_param(loop_step)))
        nsdfg.add_node(loop_region, is_start_block=True)
        nsdfg.remove_node(nstate)
        loop_region.add_node(nstate, is_start_block=True)
        # store as object field for external access
        self.loop_region = loop_region
        # Skip map in input edges
        for edge in nstate.out_edges(map_entry):
            src_node = nstate.memlet_path(edge)[0].src
            nstate.add_edge(src_node, None, edge.dst, edge.dst_conn, edge.data)
            nstate.remove_edge(edge)

        # Skip map in output edges
        for edge in nstate.in_edges(map_exit):
            dst_node = nstate.memlet_path(edge)[-1].dst
            nstate.add_edge(edge.src, edge.src_conn, dst_node, None, edge.data)
            nstate.remove_edge(edge)

        # Remove nodes from dynamic map range
        nstate.remove_nodes_from([e.src for e in dace.sdfg.dynamic_map_inputs(nstate, map_entry)])
        # Remove scope nodes
        nstate.remove_nodes_from([map_entry, map_exit])

        # create object field for external nsdfg access
        self.nsdfg = nsdfg

        sdfg.reset_cfg_list()
        # Ensure the SDFG is marked as containing CFG regions
        sdfg.root_sdfg.using_explicit_control_flow = True

        if self.inline_after:
            # Flatten the wrapping NSDFG so the resulting LoopRegion
            # ends up directly at the parent CFR. The widening step is
            # the InlineMultistateSDFG prerequisite (its can_be_applied
            # refuses narrowed in/out subsets); both run on the single
            # NSDFG we just created (no SDFG-wide sweep).
            #
            # External-reference preservation: ``InlineMultistateSDFG``
            # calls ``isolate_nested_sdfg`` which splits / renames /
            # removes the state holding the NSDFG. Any external Python
            # references to ``graph`` (the start_block of the parent
            # CFR, predecessor->graph interstate edges, etc.) would
            # become stale. We avoid that by proactively migrating
            # ``graph``'s dataflow content into a fresh successor state
            # ``target_state`` and running expand+inline against THAT
            # state. ``graph`` stays in place as an empty placeholder;
            # the parent CFR sees ``graph -> target_state`` and any
            # external interstate references to ``graph`` remain valid.
            from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
            from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

            parent_cfr = graph.parent_graph
            target_state = graph
            if parent_cfr is not None:
                was_start = parent_cfr.start_block is graph
                target_state = parent_cfr.add_state(label=f"{graph.label}_for_inline")
                # Move ``graph``'s INTRA-state dataflow to ``target_state``.
                nodes_to_move = list(graph.nodes())
                edges_to_move = [(e.src, e.src_conn, e.dst, e.dst_conn, e.data) for e in graph.edges()]
                for n in nodes_to_move:
                    graph.remove_node(n)
                for n in nodes_to_move:
                    target_state.add_node(n)
                for src, sc, dst, dc, data in edges_to_move:
                    target_state.add_edge(src, sc, dst, dc, data)
                # Reparent every existing ``graph -> *`` INTERSTATE edge to
                # ``target_state -> *``. Otherwise the placeholder ``graph``
                # would end up with both the new ``graph -> target_state``
                # edge AND the original successor edges, giving it multiple
                # unconditional out-edges that later trip
                # ``control_flow_raising`` (else-not-last) and
                # ``DeadStateElimination`` validation.
                successor_edges = list(parent_cfr.out_edges(graph))
                for e in successor_edges:
                    parent_cfr.remove_edge(e)
                    parent_cfr.add_edge(target_state, e.dst, e.data)
                parent_cfr.add_edge(graph, target_state, dace.InterstateEdge())
                if was_start:
                    parent_cfr.start_block = parent_cfr.node_id(graph)

            expand = ExpandNestedSDFGInputs()
            expand.nested_sdfg = node
            expand.expr_index = 0
            if expand.can_be_applied(target_state, 0, sdfg, permissive=False):
                expand.apply(target_state, sdfg)
            inline = InlineMultistateSDFG()
            inline.nested_sdfg = node
            inline.expr_index = 0
            if inline.can_be_applied(target_state, 0, sdfg, permissive=False):
                inline.apply(target_state, sdfg)
                # After inline, the NSDFG node is gone and the LoopRegion
                # has been hoisted into the parent CFR. Clear the stale
                # nsdfg reference; ``self.loop_region`` is still valid
                # (it was reparented, not destroyed).
                self.nsdfg = None
            sdfg.reset_cfg_list()

        return node, nstate

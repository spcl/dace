# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of nested ``GPU_Device`` maps into a single kernel guarded by bound checks."""
from typing import Any, Dict, List, Optional, Tuple

import dace
from dace import SDFG, properties, subsets, symbolic
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState, StateSubgraphView
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from ordered_set import OrderedSet

GPU_DEVICE = dace.dtypes.ScheduleType.GPU_Device


def enclosing_gpu_device_maps(state: SDFGState, node: nodes.Node) -> List[nodes.MapEntry]:
    """``GPU_Device`` map scopes enclosing ``node`` within ``state``, outermost first.

    Scoped to ``state`` alone -- it does NOT ascend into parent SDFGs, which is what distinguishes a
    map nested in this state from one merely reached through a NestedSDFG boundary.

    :param state: State holding ``node``.
    :param node: Node whose enclosing scopes are collected.
    :returns: The enclosing ``GPU_Device`` ``MapEntry`` nodes, outermost first.
    """
    chain: List[nodes.MapEntry] = []
    scope = state.entry_node(node)
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule == GPU_DEVICE:
            chain.append(scope)
        scope = state.entry_node(scope)
    chain.reverse()
    return chain


def bound_check(map_entry: nodes.MapEntry) -> str:
    """Condition selecting exactly the iterations ``map_entry``'s range owns.

    The step is part of the range: absorbing ``0:N:2`` into a unit-step parent would otherwise let
    the odd iterations, which the map never owned, into its body.

    :param map_entry: Map whose range the condition reproduces.
    :returns: A Python condition over the map's parameters.
    """
    terms = []
    for param, (begin, end, step) in zip(map_entry.map.params, map_entry.map.range):
        terms.append(f'({param} >= {begin} and {param} <= {end})')
        if step != 1:
            terms.append(f'(({param} - {begin}) % {step} == 0)')
    return ' and '.join(terms) if terms else 'True'


@properties.make_properties
@transformation.explicit_cf_compatible
class NestedGPUDeviceMapLowering(ppl.Pass):
    """Lower nested ``GPU_Device`` maps into one kernel whose body is bound-checked.

    A ``GPU_Device`` map whose body holds further ``GPU_Device`` maps has no direct hardware
    meaning. The outer map absorbs the inner maps' parameters -- their ranges merged into one
    bounding box -- and each inner body becomes a nested SDFG guarded by the condition selecting
    the iterations that body actually owns.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def move_map_to_if(self, state: SDFGState, map_entry: nodes.MapEntry) -> None:
        """Replace a map scope with a bound-checked nested SDFG holding its body.

        :param state: State holding the map.
        :param map_entry: Map whose scope is dissolved.
        """
        map_exit = state.exit_node(map_entry)
        body = list(state.all_nodes_between(map_entry, map_exit))
        nsdfg_node = helpers.nest_state_subgraph(state.sdfg,
                                                 state,
                                                 StateSubgraphView(state, body),
                                                 name=f'if_of_nested_{map_entry.label}',
                                                 full_data=True)
        inner = nsdfg_node.sdfg

        # ``nest_state_subgraph`` reads the map's params off the scope symbol table, which goes
        # stale against the in-place ``map.params`` / ``map.range`` mutation below. Thread them
        # explicitly, else validation fires ``Missing symbols on nested SDFG``.
        for param in map_entry.map.params:
            if param not in inner.symbols:
                inner.add_symbol(param, symbolic.DEFAULT_SYMBOL_TYPE)
            if param not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[param] = param

        body_state = inner.nodes()[0]
        guard = ConditionalBlock(f'bound_check_{map_entry.label}', sdfg=inner, parent=inner)
        branch = ControlFlowRegion(f'body_{map_entry.label}', sdfg=inner, parent=guard)
        inner.remove_node(body_state)
        branch.add_node(body_state, is_start_block=True)
        guard.add_branch(condition=CodeBlock(bound_check(map_entry)), branch=branch)
        inner.add_node(guard, is_start_block=True)

        self.dissolve_map_scope(state, map_entry, map_exit)
        sdutil.set_nested_sdfg_parent_references(state.sdfg)
        state.sdfg.reset_cfg_list()

    def dissolve_map_scope(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> None:
        """Remove a map scope, reconnecting its contents straight to the scope's outer neighbours.

        Each edge is rerouted through its memlet path rather than by connector name, so a connector
        the nesting renamed still finds its outer counterpart.

        :param state: State holding the map.
        :param map_entry: Entry of the scope to remove.
        :param map_exit: Matching exit.
        """
        for edge in state.out_edges(map_entry):
            if edge.data.is_empty():
                continue
            path = state.memlet_path(edge)
            outer = path[path.index(edge) - 1]
            state.add_edge(outer.src, outer.src_conn, edge.dst, edge.dst_conn, edge.data)
        for edge in state.in_edges(map_exit):
            path = state.memlet_path(edge)
            index = path.index(edge)
            if len(path) > index + 1:
                state.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)
        # ``remove_nodes_from`` drops the incident edges with the nodes.
        state.remove_nodes_from([map_entry, map_exit])

    def next_level_maps(self, state: SDFGState,
                        gpu_dev_map: nodes.MapEntry) -> OrderedSet[Tuple[SDFGState, nodes.MapEntry]]:
        """``GPU_Device`` maps one nesting level below ``gpu_dev_map``.

        They sit either directly in its scope or, when it holds none, in the nearest NestedSDFGs
        below it.

        :param state: State holding ``gpu_dev_map``.
        :param gpu_dev_map: Kernel map to search under.
        :returns: ``(state, map entry)`` pairs, in state node order.
        """
        scope = list(state.all_nodes_between(gpu_dev_map, state.exit_node(gpu_dev_map)))
        direct = OrderedSet((state, n) for n in scope if isinstance(n, nodes.MapEntry) and n.map.schedule == GPU_DEVICE)
        if direct:
            return OrderedSet((s, m) for s, m in direct if len(enclosing_gpu_device_maps(s, m)) == 1)

        frontier = OrderedSet(n for n in scope if isinstance(n, nodes.NestedSDFG))
        while frontier:
            found: OrderedSet[Tuple[SDFGState, nodes.MapEntry]] = OrderedSet()
            deeper: OrderedSet[nodes.NestedSDFG] = OrderedSet()
            for nsdfg_node in frontier:
                for nested_state in nsdfg_node.sdfg.all_states():
                    for node in nested_state.nodes():
                        if isinstance(node, nodes.MapEntry) and node.map.schedule == GPU_DEVICE:
                            found.add((nested_state, node))
                        elif isinstance(node, nodes.NestedSDFG):
                            deeper.add(node)
            if found:
                return OrderedSet((s, m) for s, m in found if not enclosing_gpu_device_maps(s, m))
            frontier = deeper
        return OrderedSet()

    def top_level_kernels(self, state: SDFGState) -> OrderedSet[nodes.MapEntry]:
        """``GPU_Device`` maps in ``state`` that no other scope encloses.

        :param state: State to search.
        :returns: The outermost kernel maps, in state node order.
        """
        return OrderedSet(
            node for node in state.nodes()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == GPU_DEVICE and state.entry_node(node) is None)

    def absorb(self, state: SDFGState, gpu_dev_map: nodes.MapEntry) -> int:
        """Absorb one kernel's next level of nested ``GPU_Device`` maps into it.

        :param state: State holding ``gpu_dev_map``.
        :param gpu_dev_map: Kernel map that grows by the inner maps' parameters.
        :returns: How many inner maps were lowered.
        :raises NotImplementedError: An inner map itself contains further ``GPU_Device`` maps.
        """
        inner_maps = self.next_level_maps(state, gpu_dev_map)
        if not inner_maps:
            return 0

        ranges: Dict[str, subsets.Range] = {}
        for map_state, inner_map in inner_maps:
            if self.next_level_maps(map_state, inner_map):
                raise NotImplementedError('Multiple levels of nestedness in GPU Device Maps are not supported')
            for param, rng in zip(inner_map.map.params, inner_map.map.range):
                # Bounding box over the siblings sharing a parameter; each body's own bound check
                # discards the iterations it does not own. Seeded with the first range, never with
                # zero, which would widen the box down to an origin no map asked for.
                one = subsets.Range([rng])
                ranges[param] = one if param not in ranges else subsets.union(ranges[param], one)

        new_ranges = list(gpu_dev_map.map.range)
        for param, merged in ranges.items():
            gpu_dev_map.map.params.append(param)
            new_ranges.append(merged[0])
        gpu_dev_map.map.range = subsets.Range(new_ranges)

        # The absorbed parameters are resolved at the kernel level, so every NestedSDFG between it
        # and an inner body has to carry them down.
        for map_state, _ in inner_maps:
            cur_sdfg = map_state.sdfg
            while cur_sdfg.parent_nsdfg_node is not None and cur_sdfg is not state.sdfg:
                nsdfg_node = cur_sdfg.parent_nsdfg_node
                for param in ranges:
                    if param not in cur_sdfg.symbols:
                        cur_sdfg.add_symbol(param, symbolic.DEFAULT_SYMBOL_TYPE)
                    if param not in nsdfg_node.symbol_mapping:
                        nsdfg_node.symbol_mapping[param] = param
                cur_sdfg = cur_sdfg.parent_sdfg

        for map_state, inner_map in inner_maps:
            self.move_map_to_if(map_state, inner_map)
        return len(inner_maps)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Lower every nested ``GPU_Device`` map in the hierarchy.

        :param sdfg: SDFG to lower, modified in place.
        :param pipeline_results: Unused.
        :returns: How many maps were lowered, or ``None`` if none were.
        :raises ValueError: A nested ``GPU_Device`` map survived the lowering.
        """
        lowered = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                # Absorbing one level can expose the next, so each kernel is drained before moving on.
                for kernel in self.top_level_kernels(state):
                    while True:
                        applied = self.absorb(state, kernel)
                        if applied == 0:
                            break
                        lowered += applied

        sdfg.validate()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for kernel in self.top_level_kernels(state):
                    if self.next_level_maps(state, kernel):
                        raise ValueError(f'Nested GPU_Device maps remain under {kernel} after lowering')
        return lowered or None

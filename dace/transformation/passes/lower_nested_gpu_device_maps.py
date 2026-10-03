# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of nested ``GPU_Device`` maps into a single kernel guarded by bound checks."""
import copy
from collections.abc import Iterator
from typing import TypeGuard

import dace
from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState, StateSubgraphView
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from ordered_set import OrderedSet

InnerMap = tuple[SDFGState, nodes.MapEntry]


def is_gpu_device_map(node: nodes.Node) -> TypeGuard[nodes.MapEntry]:
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device


def gpu_maps_below(state: SDFGState, scope: nodes.EntryNode | None) -> Iterator[InnerMap]:
    """Outermost ``GPU_Device`` maps inside ``scope`` (``None``: the whole state), through scopes and NestedSDFGs."""
    for node in state.scope_children()[scope]:
        if is_gpu_device_map(node):
            yield state, node
        elif isinstance(node, nodes.EntryNode):
            yield from gpu_maps_below(state, node)
        elif isinstance(node, nodes.NestedSDFG):
            for nested_state in node.sdfg.all_states():
                yield from gpu_maps_below(nested_state, None)


def nested_sdfg_chain(inner: SDFG, outer: SDFG) -> list[SDFG]:
    """SDFGs from ``inner`` up to, but excluding, its ancestor ``outer``, innermost first."""
    chain = []
    while inner is not outer:
        chain.append(inner)
        inner = inner.parent_sdfg
    return chain


def bound_check(map_entry: nodes.MapEntry) -> str:
    """Condition selecting the iterations of ``map_entry``'s range; maps never have a negative step."""
    terms = []
    for param, (begin, end, step) in zip(map_entry.map.params, map_entry.map.range, strict=True):
        terms.append(f'({param} >= {begin} and {param} <= {end})')
        if step != 1:
            terms.append(f'(({param} - {begin}) % {step} == 0)')
    return ' and '.join(terms)


@properties.make_properties
@transformation.explicit_cf_compatible
class NestedGPUDeviceMapLowering(ppl.Pass):
    """Lower nested ``GPU_Device`` maps, directly or behind NestedSDFGs, into one bound-checked kernel.

    The outermost map absorbs the parameters of the maps below it, merging sibling ranges into a bounding box.
    Each absorbed body becomes a nested SDFG guarded by the condition selecting the iterations it owns.
    Bounds must be evaluable where the kernel is launched.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def move_map_to_if(self, state: SDFGState, map_entry: nodes.MapEntry) -> None:
        """Replace a map scope with a bound-checked nested SDFG holding its body."""
        map_exit = state.exit_node(map_entry)
        # The map's own params are defined by the map itself.
        defined = state.symbols_defined_at(map_entry)
        defined.update(map_entry.new_symbols(state.sdfg, state, defined))
        nsdfg_node = helpers.nest_state_subgraph(state.sdfg,
                                                 state,
                                                 StateSubgraphView(state,
                                                                   list(state.all_nodes_between(map_entry, map_exit))),
                                                 name=f'if_of_nested_{map_entry.label}',
                                                 full_data=True)
        inner = nsdfg_node.sdfg
        for sym, sym_type in defined.items():
            if sym not in inner.symbols:
                inner.add_symbol(sym, sym_type)
            nsdfg_node.symbol_mapping.setdefault(sym, sym)

        body_state = inner.nodes()[0]
        guard = ConditionalBlock(f'bound_check_{map_entry.label}', sdfg=inner, parent=inner)
        branch = ControlFlowRegion(f'body_{map_entry.label}', sdfg=inner, parent=guard)
        inner.remove_node(body_state)
        branch.add_node(body_state, is_start_block=True)
        guard.add_branch(condition=CodeBlock(bound_check(map_entry)), branch=branch)
        inner.add_node(guard, is_start_block=True)

        self.dissolve_map_scope(state, map_entry, map_exit)
        sdutil.set_nested_sdfg_parent_references(state.sdfg)

    def dissolve_map_scope(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> None:
        """Remove a map scope, reconnecting its contents to the scope's outer neighbors."""
        enclosing = state.entry_node(map_entry)
        for edge in state.out_edges(map_entry):
            if edge.data.is_empty():
                # An ordering edge has no memlet path.
                if enclosing is not None:
                    state.add_edge(enclosing, None, edge.dst, None, dace.Memlet())
                continue
            path = state.memlet_path(edge)
            outer = path[path.index(edge) - 1]
            state.add_edge(outer.src, outer.src_conn, edge.dst, edge.dst_conn, edge.data)
        for edge in state.in_edges(map_exit):
            path = state.memlet_path(edge)
            index = path.index(edge)
            if len(path) > index + 1:
                state.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)
        state.remove_nodes_from([map_entry, map_exit])

    def hoisted_ranges(self, state: SDFGState, kernel: nodes.MapEntry,
                       inner_maps: list[InnerMap]) -> list[subsets.Range]:
        """Each inner map's range in the kernel SDFG's symbols; refuses a bound the host cannot evaluate."""
        host_symbols = OrderedSet(state.symbols_defined_at_state()).union(state.sdfg.constants)
        hoisted = []
        for map_state, inner_map in inner_maps:
            rng = copy.deepcopy(inner_map.map.range)
            for sdfg in nested_sdfg_chain(map_state.sdfg, state.sdfg):
                symbolic.safe_replace(sdfg.parent_nsdfg_node.symbol_mapping, rng.replace)
            unavailable = sorted(str(s) for s in rng.free_symbols if str(s) not in host_symbols)
            if unavailable:
                raise NotImplementedError(f'Cannot absorb {inner_map.map.label} into {kernel.map.label}: its '
                                          f'range {rng} names {unavailable}, undefined where the grid is sized')
            hoisted.append(rng)
        return hoisted

    def fresh_param_names(self, state: SDFGState, kernel: nodes.MapEntry, inner_maps: list[InnerMap]) -> dict[str, str]:
        """New names for inner params clashing with a kernel param or a symbol on the way down; siblings share."""
        scope_sdfgs = OrderedSet(sdfg for s, _ in inner_maps for sdfg in nested_sdfg_chain(s.sdfg, state.sdfg))
        scope_sdfgs.add(state.sdfg)
        inner_params = OrderedSet(p for _, m in inner_maps for p in m.map.params)
        clashing = OrderedSet(kernel.map.params).union(*(sdfg.symbols for sdfg in scope_sdfgs))
        taken = clashing.union(inner_params, *(sdfg.arrays for sdfg in scope_sdfgs))
        fresh: dict[str, str] = {}
        for param in inner_params & clashing:
            fresh[param] = dace.utils.find_new_name(param, taken)
            taken.add(fresh[param])
        return fresh

    def rename_params(self, inner_maps: list[InnerMap], fresh: dict[str, str]) -> None:
        for map_state, inner_map in inner_maps:
            repl = {p: fresh[p] for p in inner_map.map.params if p in fresh}
            if not repl:
                continue
            # The range is evaluated outside the scope, where the old names keep their meaning.
            outer_range = copy.deepcopy(inner_map.map.range)
            symbolic.safe_replace(repl, map_state.scope_subgraph(inner_map).replace_dict)
            inner_map.map.params = [fresh.get(p, p) for p in inner_map.map.params]
            inner_map.map.range = outer_range

    def absorb(self, state: SDFGState, kernel: nodes.MapEntry) -> int:
        """Absorb the outermost ``GPU_Device`` maps below ``kernel``; returns how many."""
        inner_maps = list(gpu_maps_below(state, kernel))
        if not inner_maps:
            return 0
        hoisted = self.hoisted_ranges(state, kernel, inner_maps)
        self.rename_params(inner_maps, self.fresh_param_names(state, kernel, inner_maps))

        # Siblings sharing a param get a bounding box; each guard drops what its body does not own.
        ranges: dict[str, subsets.Range] = {}
        param_types: dict[str, dtypes.typeclass] = {}
        for (map_state, inner_map), rng in zip(inner_maps, hoisted, strict=True):
            param_types.update(inner_map.new_symbols(map_state.sdfg, map_state, {}))
            for dim, param in enumerate(inner_map.map.params):
                one = subsets.Range([rng[dim]])
                ranges[param] = one if param not in ranges else subsets.union(ranges[param], one)
                if ranges[param] is None:
                    raise NotImplementedError(f'Cannot bound the union of the ranges of {param}')

        kernel.map.params.extend(ranges)
        kernel.map.range = subsets.Range(list(kernel.map.range) + [merged[0] for merged in ranges.values()])

        # Every NestedSDFG on the way down binds the absorbed params.
        for map_state, _ in inner_maps:
            for sdfg in nested_sdfg_chain(map_state.sdfg, state.sdfg):
                for param in ranges:
                    if param not in sdfg.symbols:
                        sdfg.add_symbol(param, param_types[param])
                    sdfg.parent_nsdfg_node.symbol_mapping.setdefault(param, param)

        for map_state, inner_map in inner_maps:
            self.move_map_to_if(map_state, inner_map)
        return len(inner_maps)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Lower every nested ``GPU_Device`` map; returns how many, or ``None`` if there were none.

        :raises NotImplementedError: A bound cannot be evaluated at the kernel launch.
        """
        lowered = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for kernel in [n for n in state.scope_children()[None] if is_gpu_device_map(n)]:
                    # Each layer's bodies become NestedSDFGs holding the next layer.
                    while absorbed := self.absorb(state, kernel):
                        lowered += absorbed
        sdfg.validate()
        return lowered or None

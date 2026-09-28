# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of nested ``GPU_Device`` maps into a single kernel guarded by bound checks."""
import copy
from typing import Any, Dict, List, Optional, Tuple

import dace
from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState, StateSubgraphView
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from ordered_set import OrderedSet

GPU_DEVICE = dace.dtypes.ScheduleType.GPU_Device

InnerMap = Tuple[SDFGState, nodes.MapEntry]


def is_gpu_device_map(node: nodes.Node) -> bool:
    return isinstance(node, nodes.MapEntry) and node.map.schedule == GPU_DEVICE


def gpu_device_depth(state: SDFGState, node: nodes.Node) -> int:
    """Number of ``GPU_Device`` scopes enclosing ``node`` within ``state``, not across NestedSDFGs."""
    depth = 0
    scope = state.entry_node(node)
    while scope is not None:
        depth += is_gpu_device_map(scope)
        scope = state.entry_node(scope)
    return depth


def scan_nested_level(
        frontier: OrderedSet[nodes.NestedSDFG]) -> Tuple[OrderedSet[InnerMap], OrderedSet[nodes.NestedSDFG]]:
    """Outermost ``GPU_Device`` maps in the states of ``frontier``, and the NestedSDFGs one level deeper."""
    found: OrderedSet[InnerMap] = OrderedSet()
    deeper: OrderedSet[nodes.NestedSDFG] = OrderedSet()
    for nested_state in (st for nsdfg_node in frontier for st in nsdfg_node.sdfg.all_states()):
        for node in nested_state.nodes():
            if is_gpu_device_map(node) and gpu_device_depth(nested_state, node) == 0:
                found.add((nested_state, node))
            elif isinstance(node, nodes.NestedSDFG):
                deeper.add(node)
    return found, deeper


def bound_check(map_entry: nodes.MapEntry) -> str:
    """Condition selecting exactly the iterations ``map_entry``'s range owns, step included."""
    terms = []
    for param, (begin, end, step) in zip(map_entry.map.params, map_entry.map.range):
        terms.append(f'({param} >= {begin} and {param} <= {end})')
        if step != 1:
            terms.append(f'(({param} - {begin}) % {step} == 0)')
    return ' and '.join(terms) if terms else 'True'


def nested_sdfg_chain(inner: SDFG, outer: SDFG) -> List[SDFG]:
    """SDFGs from ``inner`` up to, but excluding, its ancestor ``outer``, innermost first."""
    chain = []
    while inner is not outer:
        chain.append(inner)
        inner = inner.parent_sdfg
    return chain


def hoist_range(rng: subsets.Range, chain: List[SDFG]) -> subsets.Range:
    """``rng`` rewritten outward through every ``symbol_mapping`` on ``chain``, each applied simultaneously."""
    hoisted = copy.deepcopy(rng)
    for sdfg in chain:
        symbolic.safe_replace(sdfg.parent_nsdfg_node.symbol_mapping, hoisted.replace)
    return hoisted


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
        """Replace a map scope with a bound-checked nested SDFG holding its body."""
        map_exit = state.exit_node(map_entry)
        # The map's own params are defined by the map, so the scope symbol table cannot list them.
        defined = state.symbols_defined_at(map_entry)
        defined.update(map_entry.new_symbols(state.sdfg, state, defined))
        body = list(state.all_nodes_between(map_entry, map_exit))
        nsdfg_node = helpers.nest_state_subgraph(state.sdfg,
                                                 state,
                                                 StateSubgraphView(state, body),
                                                 name=f'if_of_nested_{map_entry.label}',
                                                 full_data=True)
        inner = nsdfg_node.sdfg
        for sym, sym_type in defined.items():
            if sym not in inner.symbols:
                inner.add_symbol(sym, sym_type)
            if sym not in nsdfg_node.symbol_mapping:
                nsdfg_node.symbol_mapping[sym] = sym

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
        """Remove a map scope, reconnecting its contents to the scope's outer neighbors along memlet paths.

        :param state: State holding the map.
        :param map_entry: Entry of the scope to remove.
        :param map_exit: Matching exit.
        """
        enclosing = state.entry_node(map_entry)
        for edge in state.out_edges(map_entry):
            if edge.data.is_empty():
                # An ordering edge has no memlet path; re-anchor it on the enclosing scope.
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

    def next_level_maps(self, state: SDFGState, gpu_dev_map: nodes.MapEntry) -> OrderedSet[InnerMap]:
        """``GPU_Device`` maps directly in ``gpu_dev_map``'s scope, else in the nearest NestedSDFGs below it."""
        scope = list(state.all_nodes_between(gpu_dev_map, state.exit_node(gpu_dev_map)))
        direct = OrderedSet((state, n) for n in scope if is_gpu_device_map(n) and gpu_device_depth(state, n) == 1)
        if direct:
            return direct
        frontier = OrderedSet(n for n in scope if isinstance(n, nodes.NestedSDFG))
        while frontier:
            found, frontier = scan_nested_level(frontier)
            if found:
                return found
        return OrderedSet()

    def top_level_kernels(self, state: SDFGState) -> List[nodes.MapEntry]:
        return [node for node in state.nodes() if is_gpu_device_map(node) and state.entry_node(node) is None]

    def hoisted_ranges(self, state: SDFGState, kernel: nodes.MapEntry,
                       inner_maps: OrderedSet[InnerMap]) -> List[subsets.Range]:
        """Each inner map's range in the kernel SDFG's symbols; refuses a bound the host cannot evaluate."""
        host_symbols = OrderedSet(state.symbols_defined_at_state()) | OrderedSet(state.sdfg.constants)
        hoisted = []
        for map_state, inner_map in inner_maps:
            rng = hoist_range(inner_map.map.range, nested_sdfg_chain(map_state.sdfg, state.sdfg))
            unavailable = sorted(str(s) for s in rng.free_symbols if str(s) not in host_symbols)
            if unavailable:
                raise NotImplementedError(f'Cannot absorb {inner_map.map.label} into {kernel.map.label}: its '
                                          f'range {rng} names {unavailable}, undefined where the grid is sized')
            hoisted.append(rng)
        return hoisted

    def fresh_param_names(self, state: SDFGState, kernel: nodes.MapEntry,
                          inner_maps: OrderedSet[InnerMap]) -> Dict[str, str]:
        """New names for inner params clashing with a kernel param or a symbol on the way down; siblings share."""
        scope_sdfgs = OrderedSet(sdfg for s, _ in inner_maps for sdfg in nested_sdfg_chain(s.sdfg, state.sdfg))
        scope_sdfgs.add(state.sdfg)
        inner_params = OrderedSet(p for _, m in inner_maps for p in m.map.params)
        clashing = OrderedSet(kernel.map.params).union(*(sdfg.symbols for sdfg in scope_sdfgs))
        taken = clashing.union(inner_params, *(sdfg.arrays for sdfg in scope_sdfgs))
        fresh: Dict[str, str] = {}
        for param in inner_params & clashing:
            fresh[param] = dace.utils.find_new_name(param, taken)
            taken.add(fresh[param])
        return fresh

    def rename_params(self, inner_maps: OrderedSet[InnerMap], fresh: Dict[str, str]) -> None:
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
        """Absorb one kernel's next level of nested ``GPU_Device`` maps into it."""
        inner_maps = self.next_level_maps(state, kernel)
        if not inner_maps:
            return 0
        if any(self.next_level_maps(s, m) for s, m in inner_maps):
            raise NotImplementedError('Multiple levels of nestedness in GPU Device Maps are not supported')
        hoisted = self.hoisted_ranges(state, kernel, inner_maps)
        self.rename_params(inner_maps, self.fresh_param_names(state, kernel, inner_maps))

        # Bounding box over the siblings sharing a param; each body's guard drops what it does not own.
        ranges: Dict[str, subsets.Range] = {}
        param_types: Dict[str, dtypes.typeclass] = {}
        for (map_state, inner_map), rng in zip(inner_maps, hoisted):
            param_types.update(inner_map.new_symbols(map_state.sdfg, map_state, {}))
            for dim, param in enumerate(inner_map.map.params):
                one = subsets.Range([rng[dim]])
                ranges[param] = one if param not in ranges else subsets.union(ranges[param], one)
                if ranges[param] is None:
                    raise NotImplementedError(f'Cannot bound the union of the ranges of {param}')

        kernel.map.params.extend(ranges)
        kernel.map.range = subsets.Range(list(kernel.map.range) + [merged[0] for merged in ranges.values()])

        # The absorbed params are resolved at the kernel, so every NestedSDFG on the way down binds them.
        for map_state, _ in inner_maps:
            for sdfg in nested_sdfg_chain(map_state.sdfg, state.sdfg):
                for param in ranges:
                    if param not in sdfg.symbols:
                        sdfg.add_symbol(param, param_types[param])
                    sdfg.parent_nsdfg_node.symbol_mapping.setdefault(param, param)

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
                    applied = self.absorb(state, kernel)
                    while applied:
                        lowered += applied
                        applied = self.absorb(state, kernel)

        sdfg.validate()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for kernel in self.top_level_kernels(state):
                    if self.next_level_maps(state, kernel):
                        raise ValueError(f'Nested GPU_Device maps remain under {kernel} after lowering')
        return lowered or None

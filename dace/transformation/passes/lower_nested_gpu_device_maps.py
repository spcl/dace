# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of nested ``GPU_Device`` maps into a single kernel guarded by bound checks."""

import copy
import typing
import warnings
from collections.abc import Iterator
from typing import TypeGuard

from ordered_set import OrderedSet

import dace
from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.sdfg import nodes, propagation
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, SDFGState, StateSubgraphView
from dace.transformation import helpers, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.map_fission import MapFission

InnerMap = tuple[SDFGState, nodes.MapEntry]

#: ``absorb`` split the kernel instead: its state holds new kernels
SPLIT = -1


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
            for nested_state in node.sdfg.states():
                yield from gpu_maps_below(nested_state, None)


def nested_sdfg_chain(inner: SDFG, outer: SDFG) -> list[SDFG]:
    """SDFGs from ``inner`` up to, but excluding, its ancestor ``outer``, innermost first."""
    chain = []
    while inner is not outer:
        chain.append(inner)
        inner = inner.parent_sdfg
    return chain


def has_map(nodes_: typing.Iterable[nodes.Node], schedules: tuple[dtypes.ScheduleType, ...]) -> bool:
    """Whether a map of one of ``schedules`` is among ``nodes_``, also through NestedSDFGs."""
    for node in nodes_:
        if isinstance(node, nodes.MapEntry) and node.map.schedule in schedules:
            return True
        if isinstance(node, nodes.NestedSDFG) and any(
            has_map(state.nodes(), schedules) for state in node.sdfg.states()
        ):
            return True
    return False


def write_scope(sdfg: SDFG, state: SDFGState, component: list[nodes.Node]) -> str:
    """Who runs a map-free part of a persistent kernel: one thread of the ``grid`` if it writes global memory, one
    thread of each ``block`` if it writes shared memory, else every ``thread`` (e.g., one that updates a register)."""
    written = [n for n in component if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0]
    storages = {sdfg.arrays[n.data].storage for n in written}
    if dtypes.StorageType.Register in storages:
        return "thread"
    if dtypes.StorageType.GPU_Shared in storages:
        return "block"
    if storages & {dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned}:
        return "grid"
    return "thread"


def add_sync_after(region: ControlFlowRegion, block: ControlFlowBlock, scope: SyncScope) -> SDFGState:
    """A state of ``region`` that holds a barrier of ``scope`` and runs right after ``block``."""
    sync = region.add_state(f"{block.label}_sync")
    sync.add_node(Sync(sync.label, scope=scope))
    for edge in region.out_edges(block):
        region.add_edge(sync, edge.dst, edge.data)
        region.remove_edge(edge)
    region.add_edge(block, sync, dace.InterstateEdge())
    return sync


def inner_accesses(state: SDFGState, entry: nodes.MapEntry) -> dict[str, list[tuple[str, bool, object]]]:
    """The innermost accesses of the map ``entry`` to each container it reads or writes: subset, write, WCR."""
    accesses: dict[str, list[tuple[str, bool, object]]] = {}
    for outer, write in [(e, False) for e in state.in_edges(entry)] + [
        (e, True) for e in state.out_edges(state.exit_node(entry))
    ]:
        if outer.data.is_empty():
            continue
        for leaf in state.memlet_tree(outer).leaves():
            accesses.setdefault(outer.data.data, []).append((str(leaf.data.subset), write, leaf.data.wcr))
    return accesses


def same_iteration(inner_maps: list[InnerMap]) -> bool:
    """Whether the maps only share containers element by element: every access of a container that one of them
    writes has the same subset, so that each iteration of the merged kernel touches only its own elements (or all
    of them only accumulate). Maps in different SDFGs that share a container cannot be compared."""
    accesses = [(map_state.sdfg, inner_accesses(map_state, entry)) for map_state, entry in inner_maps]
    for index, (sdfg, mine) in enumerate(accesses):
        for other_sdfg, theirs in accesses[index + 1 :]:
            for name in mine.keys() & theirs.keys():
                both = mine[name] + theirs[name]
                if not any(write for _, write, _ in both):
                    continue
                if other_sdfg is not sdfg:
                    return False
                if all(write and wcr is not None for _, write, wcr in both):
                    continue
                if len({subset for subset, _, _ in both}) != 1 or any(wcr is not None for _, _, wcr in both):
                    return False
    return True


def separate_device_maps(sdfg: SDFG) -> bool:
    """Moves each top-level ``GPU_Device`` map of a state of ``sdfg``, with what it needs, into a state of its own
    ahead of the rest; returns whether any state was split."""
    split = False
    for state in list(sdfg.states()):
        while True:
            entries = [
                n for n in sdutil.dfs_topological_sort(state) if is_gpu_device_map(n) and state.entry_node(n) is None
            ]
            if len(entries) < 2:
                break
            first = entries[0]
            exit_ = state.exit_node(first)
            part = OrderedSet(state.scope_subgraph(first).nodes())
            part.update(n for n in sdutil.find_upstream_nodes(first, state) if not isinstance(n, nodes.ExitNode))
            part.update(e.dst for e in state.out_edges(exit_) if isinstance(e.dst, nodes.AccessNode))
            helpers.state_fission(SubgraphView(state, list(part)))
            split = True
    return split


def bound_check(map_entry: nodes.MapEntry) -> str:
    """Condition selecting the iterations of ``map_entry``'s range; maps never have a negative step."""
    terms = []
    for param, (begin, end, step) in zip(map_entry.map.params, map_entry.map.range, strict=True):
        terms.append(f"({param} >= {begin} and {param} <= {end})")
        if step != 1:
            terms.append(f"(({param} - {begin}) % {step} == 0)")
    return " and ".join(terms)


@properties.make_properties
@transformation.explicit_cf_compatible
class NestedGPUDeviceMapLowering(ppl.Pass):
    """Lower nested ``GPU_Device`` maps, directly or behind NestedSDFGs, into one bound-checked kernel.

    The outermost map absorbs the parameters of the maps below it, merging sibling ranges into a bounding box.
    Each absorbed body becomes a nested SDFG guarded by the condition selecting the iterations it owns.
    Bounds must be evaluable where the kernel is launched.

    In a ``GPU_Persistent`` kernel, whose blocks run its whole body, every one-dimensional ``GPU_Device`` map becomes a
    sequential grid-stride loop: over the threads of the grid, or over its blocks if the map has thread-block maps of
    its own. The parts without maps run where codegen would run them (one thread of the grid if they write global
    memory, one thread of each block if they write shared memory, else every thread), and a grid ``Sync`` follows
    every state of the body.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def move_map_to_if(self, state: SDFGState, map_entry: nodes.MapEntry) -> None:
        """Replace a map scope with a bound-checked nested SDFG holding its body."""
        map_exit = typing.cast(nodes.MapExit, state.exit_node(map_entry))
        # The map's own params are defined by the map itself.
        defined = state.symbols_defined_at(map_entry)
        defined.update(map_entry.new_symbols(state.sdfg, state, defined))
        nsdfg_node = helpers.nest_state_subgraph(
            state.sdfg,
            state,
            StateSubgraphView(state, list(state.all_nodes_between(map_entry, map_exit))),
            name=f"if_of_nested_{map_entry.map.label}",
        )
        inner = nsdfg_node.sdfg
        body_state = inner.start_block
        guard = ConditionalBlock(f"bound_check_{map_entry.map.label}", sdfg=inner, parent=inner)
        branch = ControlFlowRegion(f"body_{map_entry.map.label}", sdfg=inner, parent=guard)
        inner.remove_node(body_state)
        branch.add_node(body_state, is_start_block=True)
        guard.add_branch(condition=properties.CodeBlock(bound_check(map_entry)), branch=branch)
        inner.add_node(guard, is_start_block=True)

        for sym in sorted(inner.free_symbols & defined.keys()):
            if sym not in inner.symbols:
                inner.add_symbol(sym, defined[sym])
            nsdfg_node.symbol_mapping.setdefault(sym, sym)

        self.dissolve_map_scope(state, map_entry, map_exit)

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

    def hoisted_ranges(
        self, state: SDFGState, kernel: nodes.MapEntry, inner_maps: list[InnerMap]
    ) -> list[subsets.Range]:
        """Each inner map's range in the kernel SDFG's symbols; refuses a bound the host cannot evaluate."""
        host_symbols = OrderedSet(state.symbols_defined_at_state()).union(state.sdfg.constants.keys())
        hoisted = []
        for map_state, inner_map in inner_maps:
            rng = copy.deepcopy(inner_map.map.range)
            for sdfg in nested_sdfg_chain(map_state.sdfg, state.sdfg):
                symbolic.safe_replace(sdfg.parent_nsdfg_node.symbol_mapping, rng.replace)
            unavailable = sorted(s for s in rng.free_symbols if s not in host_symbols)
            if unavailable:
                raise NotImplementedError(
                    f"Cannot absorb {inner_map.map.label} into {kernel.map.label}: its "
                    f"range {rng} names {unavailable}, undefined where the grid is sized"
                )
            hoisted.append(rng)
        return hoisted

    def fresh_param_names(self, state: SDFGState, kernel: nodes.MapEntry, inner_maps: list[InnerMap]) -> dict[str, str]:
        """New names for inner params clashing with a kernel param or a symbol on the way down; siblings share."""
        scope_sdfgs = OrderedSet(sdfg for s, _ in inner_maps for sdfg in nested_sdfg_chain(s.sdfg, state.sdfg))
        scope_sdfgs.add(state.sdfg)
        inner_params = OrderedSet(p for _, m in inner_maps for p in m.map.params)
        clashing = OrderedSet(kernel.map.params).union(*(sdfg.symbols.keys() for sdfg in scope_sdfgs))
        taken = clashing.union(inner_params, *(sdfg.arrays.keys() for sdfg in scope_sdfgs))
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
        if not same_iteration(inner_maps):
            # The merged kernel would need a grid barrier between the maps: split it into one kernel per part
            children = [n for n in state.scope_children()[kernel] if n is not state.exit_node(kernel)]
            matched = {"map_entry": kernel}
            if len(children) == 1 and isinstance(children[0], nodes.NestedSDFG):
                # Fission splits a nested SDFG at its states
                matched["nested_sdfg"] = children[0]
                separate_device_maps(children[0].sdfg)
            try:
                MapFission.apply_to(state.sdfg, expr_index=len(matched) - 1, verify=True, **matched)
            except ValueError:
                warnings.warn(
                    f"The device maps nested in {kernel.map.label} depend on each other across iterations and the "
                    "kernel cannot be split; they stay nested"
                )
                return 0
            return SPLIT
        hoisted = self.hoisted_ranges(state, kernel, inner_maps)
        self.rename_params(inner_maps, self.fresh_param_names(state, kernel, inner_maps))

        # Siblings sharing a param get a bounding box; each guard drops what its body does not own.
        ranges: dict[str, subsets.Range] = {}
        param_types: dict[str, dtypes.typeclass] = {}
        for (map_state, inner_map), rng in zip(inner_maps, hoisted, strict=True):
            param_types.update(inner_map.new_symbols(map_state.sdfg, map_state, {}))
            for dim, param in enumerate(inner_map.map.params):
                one = subsets.Range([rng[dim]])
                merged = one if param not in ranges else subsets.union(ranges[param], one)
                if not isinstance(merged, subsets.Range):
                    raise NotImplementedError(f"Cannot bound the union of the ranges of {param}")
                ranges[param] = merged

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

    def lower_persistent(self, state: SDFGState, kernel: nodes.MapEntry) -> int:
        """Lower the ``GPU_Device`` maps and the grid barriers of the persistent kernel ``kernel``; returns how many
        maps."""
        if kernel.map.gpu_grid_syncs_lowered:
            return 0
        kernel_exit = state.exit_node(kernel)
        inside = list(state.all_nodes_between(kernel, kernel_exit))
        if len(inside) == 1 and isinstance(inside[0], nodes.NestedSDFG):
            body_node = inside[0]
        else:
            body_node = helpers.nest_state_subgraph(
                state.sdfg, state, StateSubgraphView(state, inside), name=f"{kernel.map.label}_body"
            )
        body = body_node.sdfg

        # The grid position of the thread, read once
        taken = OrderedSet(body.symbols) | OrderedSet(body.arrays) | OrderedSet(body.free_symbols)
        registers = {"blk": "blockIdx.x", "nblk": "gridDim.x", "tid": "threadIdx.x", "nthr": "blockDim.x"}
        ids = {}
        for key in registers:
            ids[key] = dace.utils.find_new_name(f"__dace_{key}", taken)
            taken.add(ids[key])
        start = body.start_block
        id_state = body.add_state("persistent_ids", is_start_block=True)
        body.add_edge(id_state, start, dace.InterstateEdge(assignments={ids[k]: f"{ids[k]}_value" for k in ids}))
        tasklet = id_state.add_tasklet(
            "persistent_ids",
            {},
            {f"__{k}" for k in ids},
            "\n".join(f"__{k} = {registers[k]};" for k in ids),
            language=dtypes.Language.CPP,
        )
        for key, name in ids.items():
            body.add_scalar(f"{name}_value", dtypes.int32, storage=dtypes.StorageType.Register, transient=True)
            id_state.add_edge(
                tasklet, f"__{key}", id_state.add_write(f"{name}_value"), None, dace.Memlet(f"{name}_value[0]")
            )
            body.add_symbol(name, dtypes.int32)
        blk, nblk, tid, nthr = (symbolic.pystr_to_symbolic(ids[k]) for k in ("blk", "nblk", "tid", "nthr"))

        lowered = 0
        for region_state in [s for s in body.states() if s is not id_state]:
            # Codegen runs a thread-block map with a dynamic range sequentially in each thread
            for node in region_state.nodes():
                if (
                    isinstance(node, nodes.MapEntry)
                    and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
                    and sdutil.has_dynamic_map_inputs(region_state, node)
                ):
                    node.map.schedule = dtypes.ScheduleType.Sequential
            for entry in [n for n in region_state.scope_children()[None] if is_gpu_device_map(n)]:
                if len(entry.map.params) != 1:
                    raise NotImplementedError(f"The device map {entry.map.label} of a persistent kernel is not 1D")
                begin, end, step = entry.map.range[0]
                if has_map(region_state.scope_subgraph(entry).nodes(), (dtypes.ScheduleType.GPU_ThreadBlock,)):
                    # The thread-block maps of the map spread each iteration over the threads of a block
                    first, stride = begin + blk * step, nblk * step
                else:
                    first, stride = begin + (blk * nthr + tid) * step, nblk * nthr * step
                entry.map.range = subsets.Range([(first, end, stride)])
                entry.map.schedule = dtypes.ScheduleType.Sequential
                lowered += 1
            propagation.propagate_memlets_state(body, region_state)
        if has_map([n for s in body.states() for n in s.nodes()], (dtypes.ScheduleType.GPU_Device,)):
            raise NotImplementedError(f"The persistent kernel {kernel.map.label} has device maps nested in others")

        self.synchronize(body, id_state, ids)
        kernel.map.gpu_grid_syncs_lowered = True
        return lowered

    def synchronize(self, sdfg: SDFG, id_state: SDFGState | None, ids: dict[str, str]) -> None:
        """Runs the map-free parts of the states of ``sdfg`` where codegen would and puts a grid barrier after every
        state, also in the NestedSDFGs that hold maps."""
        conditions = {"grid": f"{ids['blk']} == 0 and {ids['tid']} == 0", "block": f"{ids['tid']} == 0"}
        for state in [s for s in sdfg.states() if s is not id_state]:
            region = state.parent_graph
            # The block of ``region`` the barrier follows
            last = state
            for component in dace.sdfg.concurrent_subgraphs(state):
                component_nodes = list(component.nodes())
                if has_map(component_nodes, (dtypes.ScheduleType.Sequential, dtypes.ScheduleType.GPU_ThreadBlock)):
                    for node in component_nodes:
                        if isinstance(node, nodes.NestedSDFG) and state.entry_node(node) is None:
                            for name in ids.values():
                                if name not in node.sdfg.symbols:
                                    node.sdfg.add_symbol(name, dtypes.int32)
                                node.symbol_mapping.setdefault(name, name)
                            self.synchronize(node.sdfg, None, ids)
                    continue
                scope = write_scope(sdfg, state, component_nodes)
                if scope == "thread":
                    continue
                # Guarded ahead of the rest of the state, which it does not depend on
                whole = len(component_nodes) == state.number_of_nodes()
                part = state if whole else helpers.state_fission(SubgraphView(state, component_nodes))
                guard = ConditionalBlock(f"{part.label}_{scope}_guard", sdfg=region.sdfg, parent=region)
                region.add_node(guard, is_start_block=region.start_block is part)
                for edge in region.in_edges(part):
                    region.add_edge(edge.src, guard, edge.data)
                for edge in region.out_edges(part):
                    region.add_edge(guard, edge.dst, edge.data)
                region.remove_node(part)
                branch = ControlFlowRegion(f"{part.label}_{scope}_only", sdfg=region.sdfg, parent=guard)
                branch.add_node(part, is_start_block=True)
                guard.add_branch(properties.CodeBlock(conditions[scope]), branch)
                if whole:
                    last = guard
                    break
            add_sync_after(region, last, SyncScope.GRID)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Lower every nested ``GPU_Device`` map; returns how many, or ``None`` if there were none.

        :raises NotImplementedError: A bound cannot be evaluated at the kernel launch.
        """
        lowered = 0
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for state in nsdfg.states():
                kernels = [n for n in state.scope_children()[None] if is_gpu_device_map(n)]
                while kernels:
                    kernel = kernels.pop(0)
                    # Each layer's bodies become NestedSDFGs holding the next layer.
                    while absorbed := self.absorb(state, kernel):
                        if absorbed == SPLIT:
                            # Lower the kernels the split left in the state
                            kernels = [n for n in state.scope_children()[None] if is_gpu_device_map(n)]
                            break
                        lowered += absorbed
                for kernel in [
                    n
                    for n in state.scope_children()[None]
                    if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Persistent
                ]:
                    lowered += self.lower_persistent(state, kernel)
        return lowered or None

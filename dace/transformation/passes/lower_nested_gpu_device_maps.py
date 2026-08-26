# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import enclosing_map_chain
from ordered_set import OrderedSet


@properties.make_properties
@transformation.explicit_cf_compatible
class NestedGPUDeviceMapLowering(ppl.Pass):
    """
    Lowers nested ``GPU_Device`` maps (a ``GPU_Device`` map whose body contains further
    ``GPU_Device`` maps): the outer map is range-expanded to absorb the inner maps' params
    and each inner body is wrapped in an if-bound-check NSDFG, realizing the nesting through
    the codegen's special support for nested GPU Device maps.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Nodes)

    def _rm_map(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        """Delete a map scope and its contents. ``remove_node`` drops the incident edges."""
        map_exit = state.exit_node(map_entry)
        state.remove_nodes_from([*state.all_nodes_between(map_entry, map_exit), map_entry, map_exit])

    def _move_map_to_if(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        map_exit = state.exit_node(map_entry)
        map_inner_nodes = {n for n in state.all_nodes_between(map_entry, map_exit)}
        map_inner_edges = state.all_edges(*map_inner_nodes)
        map_in_edges = state.in_edges(map_entry)
        map_out_edges = state.out_edges(map_exit)
        inputs = {ie.data.data for ie in state.in_edges(map_entry) if ie.data.data is not None}
        outputs = {oe.data.data for oe in state.out_edges(state.exit_node(map_entry)) if oe.data.data is not None}

        inner_sdfg = SDFG(name=f"if_of_nested_{map_entry.label}")

        if_bound_check = ConditionalBlock(label=f"bound_check_{map_entry.label}", sdfg=inner_sdfg, parent=inner_sdfg)
        inner_sdfg.add_node(if_bound_check)

        if_body = ControlFlowRegion(label=f"body_{map_entry.label}", sdfg=inner_sdfg, parent=if_bound_check)

        bound_check = " and ".join(
            [f"({p} >= {b} and {p} <= {e})" for p, (b, e, s) in zip(map_entry.map.params, map_entry.map.range)])
        if_bound_check.add_branch(
            condition=CodeBlock(bound_check),
            branch=if_body,
        )

        if_body_state = if_body.add_state(f"state_{map_entry.label}", is_start_block=True)

        # inout nodes can be written inside kernels: key off ``n.data`` (a string), not the AccessNode.
        for n in map_inner_nodes:
            if isinstance(n, dace.nodes.AccessNode) and state.sdfg.arrays[n.data].transient is False:
                if n.data not in inputs and state.out_degree(n) > 0:
                    inputs.add(n.data)
                if n.data not in outputs and state.in_degree(n) > 0:
                    outputs.add(n.data)

        nsdfg = state.add_nested_sdfg(
            sdfg=inner_sdfg,
            inputs=inputs,
            outputs=outputs,
        )

        for ie in map_in_edges:
            if ie.data.data is not None:
                state.add_edge(ie.src, ie.src_conn, nsdfg, ie.data.data,
                               dace.memlet.Memlet.from_array(ie.data.data, state.sdfg.arrays[ie.data.data]))
            else:
                state.add_edge(ie.src, None, nsdfg, None, dace.memlet.Memlet(None))
        for oe in map_out_edges:
            if oe.data.data is not None:
                state.add_edge(nsdfg, oe.data.data, oe.dst, oe.dst_conn,
                               dace.memlet.Memlet.from_array(oe.data.data, state.sdfg.arrays[oe.data.data]))
            else:
                state.add_edge(nsdfg, None, oe.dst, None, dace.memlet.Memlet(None))

        for data_name in inputs.union(outputs):
            if data_name not in inner_sdfg.arrays:
                copydesc = copy.deepcopy(state.sdfg.arrays[data_name])
                copydesc.transient = False
                inner_sdfg.add_datadesc(data_name, copydesc)

        for sym, symtype in state.symbols_defined_at(map_entry).items():
            if sym not in inner_sdfg.symbols:
                inner_sdfg.add_symbol(sym, symtype)
            if sym not in nsdfg.symbol_mapping:
                nsdfg.symbol_mapping[sym] = sym

        # ``symbols_defined_at`` may miss the inner map's params: its scope cache goes stale after
        # the in-place ``map.params``/``map.range`` mutation. Thread them explicitly instead.
        for sym in map_entry.map.params:
            if sym not in inner_sdfg.symbols:
                inner_sdfg.add_symbol(sym, symbolic.DEFAULT_SYMBOL_TYPE)
            if sym not in nsdfg.symbol_mapping:
                nsdfg.symbol_mapping[sym] = sym

        node_map = {n: copy.deepcopy(n) for n in map_inner_nodes}
        for v in node_map.values():
            if_body_state.add_node(v)
        for e in map_inner_edges:
            if e.src in node_map and e.dst in node_map:
                if_body_state.add_edge(node_map[e.src], e.src_conn, node_map[e.dst], e.dst_conn, copy.deepcopy(e.data))
            elif e.src in node_map and e.dst not in node_map:
                if e.data.data is not None:
                    if_body_state.add_edge(node_map[e.src], e.src_conn, if_body_state.add_access(e.data.data), None,
                                           copy.deepcopy(e.data))
            elif e.dst in node_map and e.src not in node_map:
                if e.data.data is not None:
                    if_body_state.add_edge(if_body_state.add_access(e.data.data), None, node_map[e.dst], e.dst_conn,
                                           copy.deepcopy(e.data))
            else:
                raise ValueError(f'Edge {e} has neither endpoint inside the map scope')

        self._rm_map(state, map_entry)

        sdutil.set_nested_sdfg_parent_references(state.sdfg)
        state.sdfg.reset_cfg_list()

    def _move_dev_maps_in_sdfg_to_ifs(self, sdfg: SDFG):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                    self._move_map_to_if(state, node)

    def _get_next_level_maps(self, state: SDFGState, gpu_dev_map: dace.nodes.MapEntry):
        gpu_maps_between = {
            (state, n)
            for n in state.all_nodes_between(gpu_dev_map, state.exit_node(gpu_dev_map))
            if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        }

        if len(gpu_maps_between) == 0:
            all_nsdfgs = {
                n
                for n in state.all_nodes_between(gpu_dev_map, state.exit_node(gpu_dev_map))
                if isinstance(n, dace.nodes.NestedSDFG)
            }

            def collect_map_candidates_and_new_nsdfg(all_nsdfgs):
                new_all_nsdfgs = OrderedSet()
                next_level_map_candidates = OrderedSet()
                for nsdfg in all_nsdfgs:
                    for state in nsdfg.sdfg.all_states():
                        for node in state.nodes():
                            if (isinstance(node, dace.nodes.MapEntry)
                                    and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device):
                                next_level_map_candidates.add((state, node))
                        new_all_nsdfgs = new_all_nsdfgs.union(
                            {n
                             for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)})
                return new_all_nsdfgs, next_level_map_candidates

            while True:
                all_nsdfgs, next_level_map_candidates = collect_map_candidates_and_new_nsdfg(all_nsdfgs)
                if next_level_map_candidates or not all_nsdfgs:
                    break

            next_level_maps = {(state, m)
                               for (state, m) in next_level_map_candidates
                               if len(enclosing_map_chain(state, m, dace.dtypes.ScheduleType.GPU_Device)) == 0}
            return next_level_maps
        else:
            next_level_maps = {(state, m)
                               for (state, m) in gpu_maps_between
                               if len(enclosing_map_chain(state, m, dace.dtypes.ScheduleType.GPU_Device)) == 1}
            return next_level_maps

    def _apply(self, sdfg: SDFG) -> int:
        num_applied = 0
        for state in sdfg.all_states():
            parentless_device_maps: OrderedSet[dace.nodes.MapEntry] = OrderedSet()
            for node in state.nodes():
                if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                        and state.scope_dict()[node] is None):
                    parentless_device_maps.add(node)

            for gpu_dev_map in parentless_device_maps:
                next_level_maps = self._get_next_level_maps(state, gpu_dev_map)

                nested_map_params_and_ranges = dict()
                for map_state, nested_gpu_map in next_level_maps:
                    if not self._no_further_nested_gpu_dev_maps(map_state, nested_gpu_map):
                        raise NotImplementedError(
                            "Multiple levels of nestedness in GPU Device Maps are not supported by the pass")

                    for p, range in zip(nested_gpu_map.map.params, nested_gpu_map.map.range):
                        if p not in nested_map_params_and_ranges:
                            nested_map_params_and_ranges[p] = list()
                        nested_map_params_and_ranges[p].append(range)

                # Bounding-box union: the per-map bound check filters iterations an inner map does not own.
                new_ranges_to_add = {}
                for p, ranges in nested_map_params_and_ranges.items():
                    merged = dace.subsets.Range([(0, 0, 1)])
                    for map_range in ranges:
                        merged = dace.subsets.union(merged, dace.subsets.Range([map_range]))
                    new_ranges_to_add[p] = merged

                new_range_list = list(gpu_dev_map.map.range)
                for param, merged in new_ranges_to_add.items():
                    gpu_dev_map.map.params.append(param)
                    new_range_list.append(merged[0])
                gpu_dev_map.map.range = dace.subsets.Range(new_range_list)

                # Thread the new outer-map params through every NSDFG down to each inner kernel state,
                # else validation fires ``Missing symbols on nested SDFG: ['__i', '__j']``.
                new_symbol_names = list(new_ranges_to_add.keys())
                for map_state, _inner_gpu_map in next_level_maps:
                    cur_sdfg = map_state.sdfg
                    while cur_sdfg.parent_nsdfg_node is not None and cur_sdfg is not sdfg:
                        nsdfg_node = cur_sdfg.parent_nsdfg_node
                        for sym in new_symbol_names:
                            if sym not in cur_sdfg.symbols:
                                cur_sdfg.add_symbol(sym, symbolic.DEFAULT_SYMBOL_TYPE)
                            if sym not in nsdfg_node.symbol_mapping:
                                nsdfg_node.symbol_mapping[sym] = sym
                        cur_sdfg = cur_sdfg.parent_sdfg

                for map_state, inner_gpu_map in next_level_maps:
                    self._move_map_to_if(map_state, inner_gpu_map)
                    num_applied += 1

        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    num_applied += self._apply(node.sdfg)

        return num_applied

    def _no_further_nested_gpu_dev_maps(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        nodes = set(state.all_nodes_between(map_entry, state.exit_node(map_entry)))
        for node in nodes:
            if isinstance(node, dace.nodes.NestedSDFG):
                nodes = nodes.union({n for n, g in node.sdfg.all_nodes_recursive()})
        return not any({
            n
            for n in nodes
            if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        })

    def _assert_no_nested_gpu_device_maps(self, sdfg: SDFG):
        for state in sdfg.all_states():
            parentless_device_maps = OrderedSet()
            for node in state.nodes():
                if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                        and state.scope_dict()[node] is None):
                    parentless_device_maps.add(node)

            for gpu_dev_map in parentless_device_maps:
                if not self._no_further_nested_gpu_dev_maps(state, gpu_dev_map):
                    raise ValueError(f'Nested GPU_Device maps remain under {gpu_dev_map} after lowering')

    def apply_pass(
        self,
        sdfg: SDFG,
        _,
    ) -> None:
        num_applied = self._apply(sdfg)
        while num_applied > 0:
            num_applied = self._apply(sdfg)
        sdfg.validate()
        self._assert_no_nested_gpu_device_maps(sdfg)

        return None

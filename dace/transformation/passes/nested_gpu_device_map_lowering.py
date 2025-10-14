# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Dict, Iterator, Optional, Sequence, Set, Tuple

import copy
import sympy

import dace
from dace import SDFG, data, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class NestedGPUDeviceMapLowering(ppl.Pass):
    """
    for i in dace.map[0:128] @ GPU_Device:
        for j in dace.map[0:32]  @ GPU_Device:
            with dace.tasklet:
                out >> A[i, j]
                out = 5

        for j in dace.map[33:60]  @ GPU_Device:
            with dace.tasklet:
                inp << A[i, j]
                out >> A[i, j]
                out = 6 + inp

    Is implement through special codegen features for nested GPU Device maps
    This should become and IfCheck though
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Adding new nested maps means adding new nodes
        return modified & (ppl.Modifies.Nodes)

    def _rm_map(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        map_exit = state.exit_node(map_entry)
        map_inner_nodes = {n for n in state.all_nodes_between(map_entry, map_exit)}
        map_inner_edges = state.all_edges(*map_inner_nodes)
        # Rm all edges
        for e in map_inner_edges:
            state.remove_edge(e)
        # Rm body nodes
        for n in map_inner_nodes:
            state.remove_node(n)

        for e in state.in_edges(map_entry):
            state.remove_edge(e)
        for e in state.out_edges(map_exit):
            state.remove_edge(e)

        state.remove_node(map_entry)
        state.remove_node(map_exit)

    def _move_map_to_if(self, state: SDFGState, map_entry: dace.nodes.MapEntry):
        map_exit = state.exit_node(map_entry)
        map_inner_nodes = {n for n in state.all_nodes_between(map_entry, map_exit)}
        map_inner_edges = state.all_edges(*map_inner_nodes)
        map_in_edges = state.in_edges(map_entry)
        map_out_edges = state.out_edges(map_exit)
        inputs = {ie.data.data for ie in state.in_edges(map_entry) if ie.data.data is not None}
        outputs = {oe.data.data for oe in state.out_edges(state.exit_node(map_entry)) if oe.data.data is not None}
        #assert all({isinstance(ie.src, dace.nodes.AccessNode) for ie in map_in_edges if ie.data is not None}), f"{[ie.src for ie in map_in_edges if ie.data is not None]}"
        #assert all({isinstance(oe.dst, dace.nodes.AccessNode) for oe in map_out_edges if oe.data is not None}), f"{[oe.dst for oe in map_out_edges if ot.data is not None]}"

        # Inputs -> Map (range)[MapNodes] -> Outputs need to become
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
        assert if_bound_check in inner_sdfg.nodes()
        assert if_body in if_bound_check.nodes()

        if_body_state = if_body.add_state(f"state_{map_entry.label}", is_start_block=True)
        assert if_body_state in if_body.nodes()
        assert if_body_state in inner_sdfg.all_states()

        # inout nodes can be written inside kernels (not inside nsdfg)
        for n in map_inner_nodes:
            if isinstance(n, dace.nodes.AccessNode) and state.sdfg.arrays[n.data].transient is False:
                if n not in inputs and state.out_degree(n) > 0:
                    inputs.add(n)
                if n not in outputs and state.in_degree(n) > 0:
                    outputs.add(n)

        nsdfg = state.add_nested_sdfg(
            sdfg=inner_sdfg,
            inputs=inputs,
            outputs=outputs,
        )

        # Connect nsdfg
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

        # Copy over map inputs
        for data_name in inputs.union(outputs):
            if data_name not in inner_sdfg.arrays:
                copydesc = copy.deepcopy(state.sdfg.arrays[data_name])
                copydesc.transient = False
                inner_sdfg.add_datadesc(data_name, copydesc)

        # Copy over symbols
        for sym, symtype in state.symbols_defined_at(next(iter(map_inner_nodes))).items():
            assert inner_sdfg == nsdfg.sdfg
            if sym not in inner_sdfg.symbols:
                print(f"Add {sym} to {inner_sdfg.label}")
                inner_sdfg.add_symbol(sym, symtype)
                assert sym not in nsdfg.symbol_mapping
                print(f"Add {sym} to symbol mapping of {nsdfg}")
                nsdfg.symbol_mapping[sym] = sym

        # Copy over symbol mappins
        for n in map_inner_nodes:
            if isinstance(n, dace.nodes.NestedSDFG):
                for sym, val in n.symbol_mapping.items():
                    assert sym == str(val), f"Non identity smybol mappings are not supported, {sym} != {val}"
                    symtype = n.sdfg.symbols[sym]
                    if sym not in inner_sdfg.symbols:
                        inner_sdfg.add_symbol(sym, symtype)
                        assert sym not in nsdfg.symbol_mapping
                        nsdfg.symbol_mapping[sym] = sym

        # Copy over nodes (and generate accesses when needed)
        node_map = {n: copy.deepcopy(n) for n in map_inner_nodes}
        for v in node_map.values():
            if_body_state.add_node(v)
        for e in map_inner_edges:
            if e.src in node_map and e.dst in node_map:
                if_body_state.add_edge(node_map[e.src], e.src_conn, node_map[e.dst], e.dst_conn, copy.deepcopy(e.data))
            elif e.src in node_map and e.dst not in node_map:
                # Src was the map entry
                if e.data.data is not None:
                    if_body_state.add_edge(node_map[e.src], e.src_conn, if_body_state.add_access(e.data.data), None,
                                           copy.deepcopy(e.data))
            elif e.dst in node_map and e.src not in node_map:
                # Dst was the map exit
                if e.data.data is not None:
                    if_body_state.add_edge(if_body_state.add_access(e.data.data), None, node_map[e.dst], e.dst_conn,
                                           copy.deepcopy(e.data))
            else:
                assert False

        # Rm map from the state
        self._rm_map(state, map_entry)

        sdutil.set_nested_sdfg_parent_references(state.sdfg)
        state.sdfg.reset_cfg_list()

    def _move_dev_maps_in_sdfg_to_ifs(self, sdfg: SDFG):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                    # Nested GPU Device map move the map to an If
                    # Move the body of the map to an If
                    self._move_map_to_if(state, node)

    def _get_device_map_parents(self, state: SDFGState, cur_map: dace.nodes.MapEntry, gpu_dev_map: dace.nodes.MapEntry):
        sdict = state.scope_dict()
        cur_parent = sdict[cur_map]
        parents = [cur_parent]
        while cur_parent is not None:
            cur_parent = sdict[cur_parent]
            if cur_parent is not None:
                parents.append(cur_parent)

        num_dev_maps = len({
            k
            for k in parents
            if isinstance(k, dace.nodes.MapEntry) and k.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        })
        if gpu_dev_map is not None:
            assert gpu_dev_map in parents, f"{gpu_dev_map} not in {parents}"

        return num_dev_maps

    def _get_next_level_maps(self, state: SDFGState, gpu_dev_map: dace.nodes.MapEntry):
        # Gets all the maps of the next depth
        # If inside same nsdfg, then it means no parent
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

            # While candidates are zero increase depth
            # Iterate the graphs one by one if we have a lot of GPU device maps in the NestedSDFG
            # Then we first process top-level gpu device maps one by one
            # And as long as we have nestedSDFGs or GPU device maps we iterate one level further
            def collect_map_candidates_and_new_nsdfg(all_nsdfgs):
                new_all_nsdfgs = set()
                next_level_map_candidates = set()
                for nsdfg in all_nsdfgs:
                    for state in nsdfg.sdfg.all_states():
                        for node in state.nodes():
                            if isinstance(node, dace.nodes.MapEntry):
                                next_level_map_candidates.add((state, node))
                        new_all_nsdfgs = new_all_nsdfgs.union(
                            {n
                             for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)})
                return new_all_nsdfgs, next_level_map_candidates

            all_nsdfgs, next_level_map_candidates = collect_map_candidates_and_new_nsdfg(all_nsdfgs)

            while len(next_level_map_candidates) == 0:
                all_nsdfgs, next_level_map_candidates = collect_map_candidates_and_new_nsdfg(all_nsdfgs)

                # If we exchaust all nsdfgs it is time to stop
                if len(all_nsdfgs) == 0:
                    break

            next_level_maps = {(state, m)
                               for (state, m) in next_level_map_candidates
                               if self._get_device_map_parents(state, m, None) == 0}
            return next_level_maps
        else:
            next_level_maps = {(state, m)
                               for (state, m) in gpu_maps_between
                               if self._get_device_map_parents(state, m, gpu_dev_map) == 1}
            return next_level_maps

    def _apply(self, sdfg: SDFG) -> int:
        num_applied = 0
        for state in sdfg.all_states():
            parentless_device_maps: Set[dace.nodes.MapEntry] = set()
            for node in state.nodes():
                if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                        and state.scope_dict()[node] is None):
                    parentless_device_maps.add(node)

            for gpu_dev_map in parentless_device_maps:
                next_level_maps = self._get_next_level_maps(state, gpu_dev_map)

                nested_map_params_and_ranges = dict()
                # Collect all the ranges to build the union of the ranges later
                for map_state, nested_gpu_map in next_level_maps:
                    #if not self._no_further_nested_gpu_dev_maps(map_state, nested_gpu_map):
                    #    raise NotImplementedError("Multiple levels of nestedness in GPU Device Maps are not supported by the pass")

                    for p, range in zip(nested_gpu_map.map.params, nested_gpu_map.map.range):
                        if p not in nested_map_params_and_ranges:
                            nested_map_params_and_ranges[p] = list()
                        nested_map_params_and_ranges[p].append(range)

                # Build the union of the collected ranges
                new_ranges_to_add = {p: dace.subsets.Range([(0, 0, 1)]) for p in nested_map_params_and_ranges}
                for p, ranges in nested_map_params_and_ranges.items():
                    for map_range in ranges:
                        old_b, old_e, old_s = map_range
                        assert isinstance(new_ranges_to_add[p], dace.subsets.Range)
                        assert len(new_ranges_to_add[p]) == 1
                        cur_b, cur_e, cur_s = new_ranges_to_add[p][0]
                        new_ranges_to_add[p] = dace.subsets.Range([
                            (sympy.Min(old_b, cur_b).simplify(), sympy.Max(old_e, cur_e).simplify(), 1),
                        ])

                # Append the new dimensions
                new_range_list = []
                for (b, e, s) in gpu_dev_map.map.range:
                    new_range_list.append((b, e, s))
                for k, v in new_ranges_to_add.items():
                    gpu_dev_map.map.params.append(k)
                    assert len(v) == 1
                    (b, e, s) = v[0]
                    new_range_list.append((b, e, s))
                gpu_dev_map.map.range = dace.subsets.Range(new_range_list)

                for map_state, inner_gpu_map in next_level_maps:
                    self._move_map_to_if(map_state, inner_gpu_map)
                    num_applied += 1

                for p in new_ranges_to_add:
                    oes = state.out_edges(gpu_dev_map)
                    first_dst = oes[0].dst
                    symtype = state.symbols_defined_at(first_dst)[p]
                    for n in state.all_nodes_between(gpu_dev_map, state.exit_node(gpu_dev_map)):
                        if isinstance(n, dace.nodes.NestedSDFG):
                            if p not in n.sdfg.symbols:
                                n.sdfg.add_symbol(p, symtype)
                            n.symbol_mapping[p] = p
                            for n2, g in n.sdfg.all_nodes_recursive():
                                if isinstance(n2, dace.nodes.NestedSDFG):
                                    if p not in n2.sdfg.symbols:
                                        n2.sdfg.add_symbol(p, symtype)
                                    n2.symbol_mapping[p] = p

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
            parentless_device_maps = set()
            for node in state.nodes():
                if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                        and state.scope_dict()[node] is None):
                    parentless_device_maps.add(node)

            for gpu_dev_map in parentless_device_maps:
                assert self._no_further_nested_gpu_dev_maps(
                    state, gpu_dev_map
                ), f"There are nested GPU Device maps left after applying the pass (implementation error of the pass)"

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

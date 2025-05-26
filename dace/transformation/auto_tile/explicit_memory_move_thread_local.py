# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from ast import Tuple
import copy
from typing import Union
from itertools import product
import dace
from dace import subsets
from dace.data import Array
from dace.properties import DictProperty, ListProperty, Property, SubsetProperty, make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.symbolic import SymExpr
from dace.transformation import transformation
from dace import dtypes
from functools import reduce
from typing import Any, List, Dict
from dace import symbol
from sympy import nextprime
from dace.codegen.targets import cpp

from typing import NamedTuple


@make_properties
class ExplicitMemoryMoveThreadLocal(transformation.SingleStateTransformation):
    """
    For all memlets that connect the outer map to inner map,
    The src location is inferred and the memory is moved explicitly using a library not the given memory location
    """

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_group_map_entry = transformation.PatternNode(nodes.MapEntry)
    map_entry = transformation.PatternNode(nodes.MapEntry)
    tiles_evenly = Property(dtype=bool, default=False, desc="No remainder loop")
    dst_memory_location = Property(
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.Register,
        desc="Destination memory location",
    )
    src_memory_location = Property(
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.GPU_Global,
        desc="Source memory location",
    )
    intermediate_memory_location = Property(
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.GPU_Shared,
        desc="Intermediate memory location",
    )
    location_prefixes = DictProperty(key_type=dace.dtypes.StorageType, value_type=str,
                                     default={}, desc="Name mapping")
    level = Property(dtype=int, default=0, desc="Level of the map")
    pad_contig_dim = Property(dtype=bool, default=False, allow_none=False, desc="Pad contiguous dimension to a prime number bigger than the contig dimension")
    prepend_purpose_to_name = Property(dtype=bool, default=False, allow_none=False, desc="Prepend the purpose to the name of the array")
    max_levels = Property(dtype=int, default=2, desc="Maximum number of levels")
    level_list_reversed = Property(dtype=bool, default=False, desc="Reverse the level list iteraiton")

    locations_with_purpose = DictProperty(
        key_type=str, value_type=dtypes.StorageType, default=dict(), desc="Locations with purpose"
    )
    exclude_from_explicit_memory = ListProperty(element_type=str, default=[], desc="List of arrays to exclude from explicit memory movement")
    thread_local = Property(dtype=bool, default=False, desc="Memory owned by single thread")

    def __init__(self):
        super().__init__()

    def remove_prefix(self, src_arr_name: str):
        level_prefixes = []
        for i in range(1,self.max_levels+1):
            level_prefixes += [f"A{i}", f"B{i}", f"C{i}"]
        all_combinations = level_prefixes
        #all_combinations = [f"{a}_{b}" for a, b in product(level_prefixes, self.location_prefixes.values())]
        #print(all_combinations + list(self.location_prefixes.values()))
        for prefix in all_combinations + list(self.location_prefixes.values()):
            if src_arr_name.startswith(prefix):
                return src_arr_name[len(prefix)+1:]
        return src_arr_name

    def find_next_prime(self, number):
        return nextprime(number)


    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(
                cls.device_map_entry, cls.thread_group_map_entry, cls.map_entry
            )
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def _infer_source(
        self, state: SDFGState, sdfg: SDFG, edge: MultiConnectorEdge[Memlet]
    ):
        u, uc, v, vc, memlet = edge
        return (memlet.data, sdfg.arrays[memlet.data].storage)

    def _location_to_prefix(self, storage: dtypes.StorageType):
        return self.location_prefixes.get(storage, storage.name)

    def filter_terms(self, expr : SymExpr, vars):
        if isinstance(expr, int):
            return expr

        # Base case: if the expression is a single term
        #simplified = dace.symbolic.simplify(expr)
        #print(simplified, type(simplified))
        # TODO: right now
        #print(expr, type(expr))
        def try_simplify(expr):
            try:
                return dace.symbolic.simplify(expr)
            except Exception as e:
                return expr
        simplified = try_simplify(expr)
        if isinstance(simplified, dace.symbolic.SymExpr):
            _e = simplified.expr
            #print(_e, type(_e))
            simplified = simplified.expr
        #print(simplified, type(simplified), simplified.is_constant())

        if simplified.is_Atom:
            for var in vars:
                if simplified.has(symbol(var)):
                    return simplified
            if simplified.is_number:
                return simplified
            else:
                return 0

        # Recursive case: apply the function to each argument of the expression
        filtered_expr = 0
        for term in expr.args:
            filtered_term = self.filter_terms(term, vars)
            if filtered_term != 0:
                filtered_expr += filtered_term

        return filtered_expr

    def apply(self, state: SDFGState, sdfg: SDFG):
        offsets = dict()
        loc1_to_loc2_map = dict()
        num_loads = 0
        tiles_evenly = self.tiles_evenly
        for out_edge in state.out_edges(self.map_entry):
            u, uc, v, vc, memlet = out_edge
            if memlet is not None and memlet.data is not None:
                src_arr_name, src_arrstorage_type = self._infer_source(state, sdfg, out_edge)
                if src_arrstorage_type == self.src_memory_location:
                    if src_arr_name not in self.exclude_from_explicit_memory:
                        num_loads += 1

        current_load = 0
        for out_edge in state.out_edges(self.map_entry):
            u, uc, v, vc, memlet = out_edge
            if memlet is None or memlet.data is None:
                continue

            src_arr_name, src_arrstorage_type = self._infer_source(state, sdfg, out_edge)

            if src_arrstorage_type != self.src_memory_location or isinstance(
                sdfg.arrays[src_arr_name], dace.data.Scalar
            ):
                continue
            if src_arr_name in self.exclude_from_explicit_memory:
                continue

            pruned_src_arr_name = self.remove_prefix(src_arr_name)
            purpose_dict = self.device_map_entry.purpose_dict if self.prepend_purpose_to_name and hasattr(self.device_map_entry, "purpose_dict") else dict()
            if src_arr_name in purpose_dict:
                if (str(self.dst_memory_location) + "@" + purpose_dict[pruned_src_arr_name]) in self.locations_with_purpose:
                    self.dst_memory_location = self.locations_with_purpose[str(self.dst_memory_location) + "@" + purpose_dict[pruned_src_arr_name]]

            parsedstorage_type = src_arrstorage_type.name

            parsed_memory_location = self.dst_memory_location.name

            # Map the subset accessed by a thread to the subset accessed by the threadblock
            subset_to_pass = []
            shape = []
            to_replace = []
            if self.level == 0:
                for i, (beg, end, step) in enumerate(memlet.subset):
                    for sym in set.union(beg.free_symbols, end.free_symbols):
                        for param in self.thread_group_map_entry.map.params:
                            if str(sym) == param:
                                to_replace.append(i)
                                break
                for in_edge in state.in_edges(self.thread_group_map_entry):
                    _, _, _, _, _memlet = in_edge
                    if _memlet.data == memlet.data:
                        for j in range(len(memlet.subset)):
                            if j in to_replace:
                                subset_to_pass.append(_memlet.subset[j])
                                b, e, s = _memlet.subset[j]
                                assert s == 1
                                shape.append(e + 1 - b)
                            else:
                                subset_to_pass.append(memlet.subset[j])
                                b, e, s = memlet.subset[j]
                                assert s == 1
                                shape.append(e + 1 - b)
            else:
                for i, (beg, end, step) in enumerate(memlet.subset):
                    for sym in set.union(beg.free_symbols, end.free_symbols):
                        for param in self.thread_group_map_entry.map.params:
                            if str(sym) == param:
                                to_replace.append(i)
                                break
                # Prune tblock prefix from memlet
                # Prune work map prefix from memlet
                smys_to_rm = set()
                for p in self.thread_group_map_entry.map.params:
                    smys_to_rm.add(dace.symbol(p))
                for p in self.map_entry.map.params:
                    smys_to_rm.add(dace.symbol(p))

                subset_to_pass = []
                for i, (beg, end, step) in enumerate(memlet.subset):
                    subs_dict = {sym: 0 for sym in smys_to_rm}
                    _beg = beg.subs(subs_dict)
                    _end = end.subs(subs_dict)
                    _step = step.subs(subs_dict)
                    subset_to_pass.append((_beg, _end, _step))

                shape = [(end + 1 - beg)//step for beg, end, step in subset_to_pass]
            # End Mapping
            #raise Exception(subset_to_pass)

            mem_loc_a = parsedstorage_type
            mem_loc_b = parsed_memory_location
            lib_node_name = f"move_{memlet.data}_from_{mem_loc_a}_to_{mem_loc_b}"
            dst_arr_shape = shape
            #print(f"dst_arr_shape: {dst_arr_shape}")
            num_threads = [
                int((e + 1 - b) / s)
                for b, e, s in self.thread_group_map_entry.map.range
            ]
            dst_arr_strides = None
            if self.pad_contig_dim:
                dst_arr_stride_0 = 1

                if len(shape) == 1:
                    dst_arr_strides = [dst_arr_stride_0]
                elif len(shape) == 2:
                    dst_arr_stride_1 = self.find_next_prime(dst_arr_shape[-1])
                    dst_arr_strides = [dst_arr_stride_1, dst_arr_stride_0]
                elif len(shape) > 2:
                    dst_arr_stride_1 = self.find_next_prime(dst_arr_shape[-1])
                    dst_arr_strides = [dst_arr_stride_1, dst_arr_stride_0]
                    for sh in reversed(dst_arr_shape[1:-1]):
                        dst_arr_stride_1 *= sh
                        dst_arr_strides.insert(0, dst_arr_stride_1)
            else:
                dst_arr_strides = None


            dst_arr_name = (
                self._location_to_prefix(self.dst_memory_location) + "_" + pruned_src_arr_name
            )
            intermediate_arr_name = (
                self._location_to_prefix(self.intermediate_memory_location) + "_" + pruned_src_arr_name
            )
            c = 0
            while dst_arr_name in sdfg.arrays:
                if not (dst_arr_name + str(c) in sdfg.arrays):
                    dst_arr_name = dst_arr_name + str(c)
                else:
                    c += 1
            c = 0
            while intermediate_arr_name in sdfg.arrays:
                if not (intermediate_arr_name + str(c) in sdfg.arrays):
                    intermediate_arr_name = intermediate_arr_name + str(c)
                else:
                    c += 1
            (dst_arr_name, dst_arr) = sdfg.add_array(
                name=dst_arr_name,
                dtype=sdfg.arrays[src_arr_name].dtype,
                shape=dst_arr_shape,
                strides=dst_arr_strides,
                transient=True,
                storage=self.dst_memory_location,
            )
            (intermediate_arr_name, intermediate_arr) = sdfg.add_array(
                name=intermediate_arr_name,
                dtype=sdfg.arrays[src_arr_name].dtype,
                shape=dst_arr_shape,
                strides=dst_arr_strides,
                transient=True,
                storage=self.intermediate_memory_location,
            )

            # raise Exception(type(subset_to_pass), ", ", type(subset_to_pass[0]))

            state.remove_edge(out_edge)

            # Add offsets for the rest of the accesses
            offsets[src_arr_name] = [beg for (beg, end, step) in memlet.subset]
            # Loc1 -> Loc2 mapping list
            loc1_to_loc2_map[src_arr_name] = (
                dst_arr_name,
                [beg for (beg, end, step) in memlet.subset],
            )
            # Map thread offsets (tbock + thread offsets) to tblock offsets
            for n, (n2, offset) in loc1_to_loc2_map.items():
                noffsets = []
                for _expr in offset:
                    expr = _expr
                    syms = expr.free_symbols
                    for sym in syms:
                        expr = expr.subs(sym, dace.symbolic.symbol(str(sym).replace("d_", "b_")))
                    noffsets.append(expr)
                loc1_to_loc2_map[n] = (n2, noffsets)


            dst_access_node = nodes.AccessNode(data=dst_arr_name)
            intermediate_access_node = nodes.AccessNode(data=intermediate_arr_name)
            state.add_node(dst_access_node)
            state.add_node(intermediate_access_node )
            # Compute thread block offset
            old_subset = memlet.subset



            # This removes any parameter that depends on the grid loop
            new_subset = []
            thread_group_offset = []
            if self.level == 0:
                for beg, end, step in old_subset:
                    _beg = self.filter_terms(
                        SymExpr(beg), self.thread_group_map_entry.map.params
                    )
                    _end = self.filter_terms(
                        SymExpr(end), self.thread_group_map_entry.map.params
                    )
                    _step = self.filter_terms(
                        SymExpr(step), self.thread_group_map_entry.map.params
                    )
                    if isinstance(_beg, SymExpr) or isinstance(_beg, symbol):
                        thread_group_offset.append(
                            any(
                                [
                                    _beg.has(symbol(v))
                                    for v in self.thread_group_map_entry.map.params
                                ]
                            )
                        )
                    else:
                        thread_group_offset.append(False)

                    if _beg == 0:
                        new_subset.append((_beg, _end, _step))
                    else:
                        new_subset.append((_beg - beg, _end - beg, _step))
            else:
                for i, (beg, end, step) in enumerate(memlet.subset):
                    subs_dict = {sym: 0 for sym in smys_to_rm}
                    _beg = beg.subs(subs_dict)
                    _end = end.subs(subs_dict)
                    _step = step.subs(subs_dict)
                    new_subset.append((_beg, _end, _step))
                    if isinstance(_beg, SymExpr) or isinstance(_beg, symbol):
                        thread_group_offset.append(
                            any(
                                [
                                    _beg.has(symbol(v))
                                    for v in self.thread_group_map_entry.map.params
                                ]
                            )
                        )
                    else:
                        thread_group_offset.append(False)

            new_range_list = new_subset
            to_dst_memlet = Memlet(
                subset=subsets.Range(new_range_list), data=dst_arr_name
            )
            to_map_memlet = Memlet(
                subset=subsets.Range(new_range_list), data=dst_arr_name
            )
            intermediate_memlet = Memlet(
                subset=subsets.Range(new_range_list), data=intermediate_arr_name
            )

            state.add_edge(u, uc, intermediate_access_node, None, memlet)
            state.add_edge(intermediate_access_node, None, dst_access_node, None, intermediate_memlet)
            state.add_edge(dst_access_node, None, v, vc, to_map_memlet)

            # Update any memlet that accesses any of the mapped arrays
            edges_to_check = set()

            nodeset = set([v])
            while nodeset:
                n = nodeset.pop()
                if not isinstance(n, nodes.MapExit):
                    edges_to_check = edges_to_check.union(set([
                        e for e in state.out_edges(n) if not isinstance(e.dst, dace.nodes.MapExit)])
                        )
                    nodeset = nodeset.union(
                        set([v for u, uc, v, vc, m in state.out_edges(n) if not isinstance(v, dace.nodes.MapExit)])
                    )

            for edge in edges_to_check:
                u, uc, v, vc, memlet = edge
                if memlet.data in loc1_to_loc2_map.keys():
                    dst_name, offset = loc1_to_loc2_map[memlet.data]
                    new_subset_list = [
                        (beg - o, end - o, step)
                        for (beg, end, step), o in zip(memlet.subset, offset)
                    ]

                    new_memlet = Memlet(
                        subset=subsets.Range(new_subset_list), data=dst_name
                    )
                    state.remove_edge(edge)
                    state.add_edge(u, uc, v, vc, new_memlet)


    @staticmethod
    def annotates_memlets():
        return True

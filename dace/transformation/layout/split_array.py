# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from typing import Dict, List, Set, Optional
from dace import SDFG, InterstateEdge, Tuple, properties
from dace.sdfg.state import ControlFlowRegion, LoopRegion, ReturnBlock
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.sdfg.graph import MultiConnectorEdge
import itertools
import copy
import sympy as sp

@properties.make_properties
@transformation.explicit_cf_compatible
class SplitArray(ppl.Pass):
    CATEGORY: str = 'Layout'

    def __init__(self,
                 symbol_map: Dict[str, int],
                 name_map: Dict[str, List[str]]):
        super().__init__()
        self._symbol_map = symbol_map
        self._name_map = name_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _unroll_loops_that_depend_only_on_split_dimensions(
            self, sdfg: dace.SDFG,
        ):
        maps_to_unroll = set()
        loops_to_unroll = set()

        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                for p, r in zip(n.map.params, n.map.range):
                    extent: dace.symbolic.SymExpr = ((r[1]+1) - r[1] // r[2])
                    freesyms = {str(s) for s in extent.free_symbols}
                    if freesyms.issubset(self._symbol_map.keys()):
                        maps_to_unroll.add((n, g))
            elif isinstance(n, LoopRegion):
                beg = loop_analysis.get_init_assignment(n)
                end = loop_analysis.get_loop_end(n)
                step = loop_analysis.get_loop_stride(n)
                extent = ((end+1) - beg) // step
                freesyms = {str(s) for s in extent.free_symbols}
                if freesyms.issubset(self._symbol_map.keys()):
                    loops_to_unroll.add((n, g))
        
        # Replace the symbols with integers
        sdfg.replace_dict(self._symbol_map)

        for n, g in maps_to_unroll:
            if n in g.nodes():
                MapUnroll().apply_to(sdfg=n.sdfg,
                                     options={},
                                     map_entry=n)

        for n, g in loops_to_unroll:
            if n in g.nodes():
                LoopUnroll().apply_to(sdfg=n.sdfg,
                                      options={
                                          "inline_iterations": True,
                                      },
                                      loop=n)

    def _collect_arrays_to_split(self, sdfg: dace.SDFG) -> Dict[str, List[Optional[str]]]:
        """
        Collect which dimensions will be split before we change the symbols
        """
        split_map = {}
        for arrname, desc in sdfg.arrays.items():
            split_list = [None] * len(desc.shape)
            for i, d_expr in enumerate(desc.shape):
                if str(d_expr) in self._symbol_map:
                    split_list[i] = str(d_expr)

            # Do not add arrays that we will not split at all
            if not all(s is None for s in split_list):
                split_map[arrname] = split_list

        return split_map

    def _get_corresponding_array(self,
                                 sdfg: dace.SDFG,
                                 edge: MultiConnectorEdge[dace.Memlet],
                                 split_map: Dict[str, List[Optional[str]]]):
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]

        # Ensure access is a single element
        new_name_expr = []
        new_subset_expr = []
        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                assert ((e+1)-b)//s == 1
                try:
                    access_offset = int(b)
                except Exception as e:
                    print()
                    raise Exception(f"Expression {b} is not an integer, can't get the corresponding array"
                                    f" for {edge.data.data}, access subset: {edge.data.subset},"
                                    f" {edge.src} -> {edge.dst}. Exception: {e}.")
                new_name_expr.append(self._name_map[splitd][access_offset])
            else:
                new_subset_expr.append((b,e,s))

        print(f"{edge.data.subset} -> {new_subset_expr}")
        return (f"{edge.data.data}_{'_'.join(map(str, new_name_expr))}",
                dace.subsets.Range(new_subset_expr))


    def _split_data_descriptors(self, sdfg: dace.SDFG,
                                split_map: Dict[str, List[Optional[str]]]):
        new_descs = dict()
        print(split_map)

        for arr, desc in sdfg.arrays.items():
            if arr not in split_map:
                continue
            else:
                assert isinstance(desc, dace.data.Array)

            split_config = split_map[arr]

            assert desc.is_packed_c_strides() or desc.is_packed_fortran_strides()

            # Remaining shape/strides for non-split dimensions
            filtered_shape = [dim for dim, split in zip(desc.shape, split_config)
                            if split is None]
            if desc.is_packed_c_strides():
                filtered_strides = [1] * len(filtered_shape)
                for i in range(len(filtered_shape) - 2, -1, -1):
                    filtered_strides[i] = filtered_strides[i + 1] * filtered_shape[i + 1]
            else:
                filtered_strides = [1] * len(filtered_shape)
                for i in range(1, len(filtered_shape)):
                    filtered_strides[i] = filtered_strides[i - 1] * filtered_shape[i - 1]

            # Collect (dim_name, extent) pairs for each split dimension
            # e.g. split_config = ["a", None, "b"], shape = [4, 5, 3]
            #   -> split_dims = [("a", 4), ("b", 3)]
            split_dims = [(dname, dim) for dname, dim in zip(split_config, desc.shape)
                        if dname is not None]

            # Cartesian product of all split index ranges
            # e.g. [("a", 4), ("b", 3)] -> product(range(4), range(3)) = 12 combos
            ranges = [range(extent) for _, extent in split_dims]

            # Get the dimensions names 
            array_name_map = {}
            i = 0
            for d in split_config:
                if d is not None:
                    array_name_map[i] = d
                    i += 1

            # The ordering is the same as the array dimensions, so we can just zip them together
            for indices in itertools.product(*ranges):
                suffix = "_".join(str(self._name_map[array_name_map[i]][int(idx)]) for i, idx in enumerate(indices))
                new_name = f"{arr}_{suffix}"

                new_descs[new_name] = dace.data.Array(
                    shape=filtered_shape,
                    strides=filtered_strides,
                    dtype=desc.dtype,
                    storage=desc.storage,
                    transient=desc.transient,
                    lifetime=desc.lifetime,
                )

            # Not the complete subset might be used, but if we are
            # passing the original array, then we should pass all new arrays
            # to the nsdfg as well, so need to create all of them
        for name, desc in new_descs.items():
            print(f"Add {name}")
            sdfg.add_datadesc(name, desc)

    def _replace_memlets(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        for state in sdfg.all_states():
            for edge in state.edges():
                mapped_data, new_subset = self._get_corresponding_array(sdfg, edge, split_map)
                if  mapped_data != edge.data.data:
                    edge.data = dace.memlet.Memlet(
                        data=mapped_data, subset=new_subset
                    )

    def _replace_iedges(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        def _rewrite_expr(expr):
            """Recursively rewrite sympy expr, replacing array-as-function calls
            whose name appears in split_map.
            e.g. zsolqa(4, j, 2) with split_config=["a", None, "b"]
                -> zsolqa_<name_a_4>_<name_b_2>(j)
            """
            if not expr.args:
                return expr

            # Recurse into children first — preserve order via tuple
            new_args = tuple(_rewrite_expr(a) for a in expr.args)

            if not expr.is_Function:
                return expr.func(*new_args)

            fname = type(expr).__name__
            if fname not in split_map:
                # Rebuild with same non-commutative type to preserve arg order
                ncfunc = sp.Function(fname, commutative=False)
                return ncfunc(*new_args)

            dim_filter = split_map[fname]

            assert len(dim_filter) == len(new_args), (
                f"{fname}: split_config has {len(dim_filter)} dims "
                f"but function call has {len(new_args)} args: {expr}"
            )

            name_parts = []
            kept_args = []
            for splitd, arg in zip(dim_filter, new_args):
                if splitd is not None:
                    name_parts.append(str(self._name_map[splitd][int(arg)]))
                else:
                    kept_args.append(arg)

            suffix = "_".join(name_parts)
            new_name = f"{fname}_{suffix}"

            if kept_args:
                ncfunc = sp.Function(new_name, commutative=False)
                return ncfunc(*kept_args)
            else:
                return sp.Symbol(new_name)


        for iedge in sdfg.all_interstate_edges():
            # Rewrite assignments
            changed = False
            new_assignments = {}
            for k, v in iedge.data.assignments.items():
                v_sym = dace.symbolic.pystr_to_symbolic(v)
                v_new = _rewrite_expr(v_sym)
                new_assignments[k] = dace.symbolic.symstr(v_new, arrayexprs=frozenset(sdfg.arrays.keys()))
                if v_new is not v_sym:
                    changed = True
            if changed:
                iedge.data.assignments = new_assignments

    def _remove_split_arrays(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        for arr in split_map.keys():
            sdfg.remove_data(arr)

    def _replace_access_nodes(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        for state in sdfg.all_states():
            for dnode in state.data_nodes():
                if dnode.data in split_map:
                    in_edges = state.in_edges(dnode)
                    out_edges = state.out_edges(dnode)

                    in_names = {ie.data.data for ie in in_edges if ie.data.data is not None}
                    out_names = {oe.data.data for oe in out_edges if oe.data.data is not None}

                    unique_names = in_names.union(out_names)

                    # If input name and output name are different we can split
                    assert len(unique_names) == 1 or (len(in_names) == 1 and len(out_names) == 1), f"Expected only one unique data name for the node (or 1 unique input, 1 unique output) got: In: {in_names}, out: {out_names} in state {state}."
                    
                    if len(in_names) == 1 and len(out_names) == 1:
                        dup_dnode = state.add_access(dnode.data)
                        for oe in out_edges:
                            state.remove_edge(oe)
                            state.add_edge(
                                dup_dnode, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data)
                            )

                        dnode.data = in_names.pop()
                        dup_dnode.data = out_names.pop()

                        # Since they are unique arrays we dont need a dependency edge
                        # state.add_edge(dnode, None, dup_dnode, None, dace.Memlet())
                    else:
                        # Having 1 name means that all edges are consistent with each other, so we can just replace them all with the same new name
                        original_name = unique_names.pop()
                        dnode.data = original_name

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        # Assume we have an array:
        # `iphase[0:nclv]` where where we want to split the array into 
        # `iphase_0, ..., iphase_$(nclv-1)` if the `nclv` is known
        #
        # This means all accesses to the array is replaced with the access 
        # to the corresponding split array. For example, `iphase[i]` is replaced with `iphase_i`.
        # if known.
        # For cases where this is not know:
        # e.g. `iphase[idx[i]]`then we need to generate branching
        # if `idx[i] == 0` ... etc.


        # If the access happens inside a loop:
        # jm = 1..(nclv+1) then we need to to unroll for performance.
        # (Since the dimension we split is known it should be possible)

        # This needs to be followed by cleaning the code (cleaning the branches)

        # Collect dimensions we will split

        # 1. Unroll all maps and loops where the only symbol necessary is the 
        # provided
        # 1.1 It replaces the symbol with a constant so we need to collect dimensions that split
        split_map = self._collect_arrays_to_split(sdfg)
        self._unroll_loops_that_depend_only_on_split_dimensions(sdfg)
        sdfg.validate()
        sdfg.save("after_unrolling_0.sdfg")

        # 2. Create the new arrays that will be used after splitting
        self._split_data_descriptors(sdfg, split_map)
        sdfg.save("after_unrolling.sdfg")


        # 3. Split arrays for accesses
        # There should be no access over the split dimension that has
        # a volume > 1 on the split dimension
        self._replace_memlets(sdfg, split_map)
        self._replace_access_nodes(sdfg, split_map)
        # Update interstate edge accesses
        self._replace_iedges(sdfg, split_map)

        # TODO: Passing to nested SDFGs
        # self._pass_to_nsdfgs(sdfg)
        sdfg.save("after_replacing.sdfg")

        # 3. Pass the arrays to the Nested SDFGs
        # (Don't forget they can go through maps)
        self._remove_split_arrays(sdfg, split_map)

        pass
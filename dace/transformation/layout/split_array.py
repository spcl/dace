# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from typing import Dict, List, Set, Optional
from dace import SDFG, InterstateEdge, Tuple, properties
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.sdfg.graph import MultiConnectorEdge
import itertools
import copy
import sympy as sp
from collections import deque
import sympy


def reverse_bfs_assignments(cfg: ControlFlowRegion, start_node: dace.nodes.Node):
    # Find symbol assignments in a reverse BFS manner starting from the given node, and return the last assignment for each symbol
    result = {}
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        node = queue.popleft()
        for edge in cfg.in_edges(node):
            for key, val in edge.data.assignments.items():
                if key not in result:
                    result[key] = val
            if edge.src not in visited:
                visited.add(edge.src)
                queue.append(edge.src)

    return result


def resolve_aliases(data_dependent_dims: List[Optional[dace.symbolic.SymExpr]],
                    iedge_assignments: Dict[str, str]) -> List[Optional[dace.symbolic.SymExpr]]:
    """Replace aliased symbols with a single canonical representative."""
    # Group symbols by their assigned value string
    val_to_canonical = {}
    alias_map = {}  # aliased_sym -> canonical_sym
    for k, v in iedge_assignments.items():
        v_str = str(v)
        if v_str not in val_to_canonical:
            val_to_canonical[v_str] = k
        else:
            alias_map[k] = val_to_canonical[v_str]

    resolved = []
    for expr in data_dependent_dims:
        if expr is None:
            resolved.append(None)
            continue
        for old, new in alias_map.items():
            expr = expr.subs(sympy.Symbol(old), sympy.Symbol(new))
        resolved.append(expr)
    return resolved


def copy_state_contents(old_state: dace.SDFGState, new_state: dace.SDFGState) -> Dict[dace.nodes.Node, dace.nodes.Node]:
    """
    Deep-copies all nodes and edges from one SDFG state into another.

    Args:
        old_state: The source SDFG state to copy from.
        new_state: The destination SDFG state to copy into.

    Returns:
        A mapping from original nodes in `old_state` to their deep-copied
        counterparts in `new_state`.

    Notes:
        - Node objects are deep-copied.
        - Edge data are also deep-copied.
        - Connections between the newly created nodes are preserved.
    """
    node_map = dict()

    # Copy all nodes
    for n in old_state.nodes():
        c_n = copy.deepcopy(n)
        node_map[n] = c_n
        new_state.add_node(c_n)

    # Copy all edges, reconnecting them to their new node counterparts
    for e in old_state.edges():
        c_src = node_map[e.src]
        c_dst = node_map[e.dst]
        new_state.add_edge(c_src, e.src_conn, c_dst, e.dst_conn, copy.deepcopy(e.data))

    return node_map


@properties.make_properties
@transformation.explicit_cf_compatible
class SplitArray(ppl.Pass):
    CATEGORY: str = 'Layout'

    def __init__(self, symbol_map: Dict[str, int], name_map: Dict[str, List[str]]):
        super().__init__()
        self._symbol_map = symbol_map
        self._name_map = name_map

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _unroll_loops_that_depend_only_on_split_dimensions(self, sdfg: dace.SDFG):
        # Phase 1: Identify candidates by label BEFORE replacing symbols
        unrollable_map_labels = set()
        unrollable_loop_labels = set()

        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                all_valid = True
                for p, r in zip(n.map.params, n.map.range):
                    extent = ((r[1] + 1) - r[0]) // r[2]
                    freesyms = {str(s) for s in extent.free_symbols}
                    if not freesyms.issubset(self._symbol_map.keys()):
                        all_valid = False
                        break
                if all_valid:
                    unrollable_map_labels.add(n.map.label)
            elif isinstance(n, LoopRegion):
                beg = loop_analysis.get_init_assignment(n)
                end = loop_analysis.get_loop_end(n)
                step = loop_analysis.get_loop_stride(n)
                if beg is None or end is None or step is None:
                    continue
                extent = ((end + 1) - beg) // step
                freesyms = {str(s) for s in extent.free_symbols}
                if freesyms.issubset(self._symbol_map.keys()):
                    unrollable_loop_labels.add(n.label)

        # Phase 2: Replace symbols with integers
        sdfg.replace_dict(self._symbol_map)

        # Phase 3: Unroll one at a time, rescanning after each mutation
        while True:
            target = self._find_map_by_label(sdfg, unrollable_map_labels)
            if target is None:
                break
            n, g = target
            MapUnroll().apply_to(sdfg=g.sdfg, options={}, map_entry=n)

        while True:
            target = self._find_loop_by_label(sdfg, unrollable_loop_labels)
            if target is None:
                break
            n, g = target
            LoopUnroll().apply_to(sdfg=n.sdfg, options={"inline_iterations": True}, loop=n)

    def _find_map_by_label(self, sdfg, labels):
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry) and n.map.label in labels:
                return (n, g)
        return None

    def _find_loop_by_label(self, sdfg, labels):
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, LoopRegion) and n.label in labels:
                return (n, g)
        return None


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
                                 edge: MultiConnectorEdge[dace.Memlet],
                                 split_map: Dict[str, List[Optional[str]]],
                                 access_mapping: Optional[Dict[str, int]] = None):
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]

        # Ensure access is a single element
        new_name_expr = []
        new_subset_expr = []
        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                assert ((e + 1) - b) // s == 1
                try:
                    access_offset = int(b)
                    new_name_expr.append(self._name_map[splitd][access_offset])
                except Exception as e:
                    if access_mapping is None:
                        raise Exception(f"Expression {b} is not an integer, can't get the corresponding array"
                                        f" for {edge.data.data}, access subset: {edge.data.subset},"
                                        f" {edge.src} -> {edge.dst}. Exception: {e}.")
                    else:
                        if str(b) in access_mapping:
                            new_name_expr.append(self._name_map[splitd][access_mapping[str(b)]])
                        else:
                            raise Exception(
                                f"(Internal Error) Access mapping provided ({access_mapping}) but the access offset ({b}) is not preset in it."
                            )
            else:
                new_subset_expr.append((b, e, s))

        print(f"{edge.data.subset} -> {new_subset_expr}")
        return (f"{edge.data.data}_{'_'.join(map(str, new_name_expr))}", dace.subsets.Range(new_subset_expr))

    def _get_data_dependent_dims(
        self, state: dace.SDFGState, edge: MultiConnectorEdge[dace.Memlet], split_map: Dict[str, List[Optional[str]]]
    ) -> Tuple[List[Optional[dace.symbolic.SymExpr]], List[Optional[dace.symbolic.SymExpr]]]:
        if edge.data.data not in split_map:
            return edge.data.data, edge.data.subset

        dim_filter = split_map[edge.data.data]

        # Ensure access is a single element
        data_dependent_dims = []
        for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
            b, e, s = access_expr
            if splitd is not None:
                assert ((e + 1) - b) // s == 1
                try:
                    access_offset = int(b)
                except Exception as e:
                    # This can happen if we have an access like `iphase[idx[i]]` where `idx[i]` is not known at compile time, so we can't determine which split array
                    # So we need to generated branches for each possible value of `idx[i]` (e.g. `if idx[i] == 0` then access `iphase_0` etc.)

                    # Need to resolve aliasing
                    # Dace can have symbols such has imelt_index_0 = imelt[0], imelt_index_1 = imelt[0]
                    data_dependent_dims.append(b)
            else:
                data_dependent_dims.append(None)

        iedge_assignments = reverse_bfs_assignments(state.parent_graph, state)
        all_exprs = copy.deepcopy(data_dependent_dims)
        data_dependent_dims = resolve_aliases(data_dependent_dims, iedge_assignments)

        return data_dependent_dims, all_exprs

    def _split_data_descriptors(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
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
            filtered_shape = [dim for dim, split in zip(desc.shape, split_config) if split is None]
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
            split_dims = [(dname, dim) for dname, dim in zip(split_config, desc.shape) if dname is not None]

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

    def _has_non_integer_access(self, sdfg: dace.SDFG, state: dace.SDFGState,
                                split_map: Dict[str, List[Optional[str]]]) -> Tuple[Optional[str], Optional[List[str]]]:
        all_data_depdent_dims = set()
        all_access_exprs = set()
        for edge in state.edges():
            if edge.data.data is not None and edge.data.data in split_map:
                data_dependent_dims, access_exprs = self._get_data_dependent_dims(state, edge, split_map)
                for d in data_dependent_dims:
                    if d is not None:
                        all_data_depdent_dims.add(str(d))
                        all_access_exprs = all_access_exprs.union(set(map(str, access_exprs)))

        if len(all_data_depdent_dims) > 0:
            if len(all_data_depdent_dims) > 1:
                sdfg.save("failing.sdfgz", compress=True)
            assert len(
                all_data_depdent_dims
            ) == 1, f"Multiple data dependent dims not supported yet. Found: {all_data_depdent_dims} in state {state} of sdfg {sdfg}."
            return all_data_depdent_dims.pop(), all_access_exprs
        else:
            return None, None

    def _get_non_int_access_dims(self, state: dace.SDFGState, split_map: Dict[str, List[Optional[str]]],
                                 exprs: Set[str], canonical_access: str) -> Set[str]:
        dims = set()

        for edge in state.edges():
            if edge.data.data not in split_map:
                continue
            dim_filter = split_map[edge.data.data]

            for i, (splitd, access_expr) in enumerate(zip(dim_filter, edge.data.subset)):
                b, e, s = access_expr
                if splitd is not None:
                    assert ((e + 1) - b) // s == 1
                    try:
                        access_offset = int(b)
                    except Exception as e:
                        # This is the data dependent access that we need to generate branches for
                        if not (any(str(expr) in str(b) for expr in exprs)):
                            raise Exception(
                                f"Expected any of the non-integer access expression {exprs} to be part of the access subset {b}."
                            )
                        else:
                            if splitd != "None":
                                dims.add(splitd)

        return dims

    c = 0

    def _replace_memlets(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        for state in sdfg.all_states():
            # See if we have a non-integer read of the array which can't be resolves statically (compile-time)
            # Then we need to generated branches
            canonical_access, all_access_exprs = self._has_non_integer_access(sdfg, state, split_map)

            if canonical_access is not None:
                # We need to put in branches, this is the number of branches we need
                # Need to find the dimensions these access is used in
                dims = self._get_non_int_access_dims(state, split_map, all_access_exprs, canonical_access)
                assert len(dims) == 1
                dim = dims.pop()
                extent = self._symbol_map[dim]

                # We need to duplicate the state in extent many branches and generate the corresponding
                # Add a CFG before this state, add a single state to this CFG
                # Copy contents of this state to that state
                cb = ConditionalBlock(label=f"extent_check_{SplitArray.c}", sdfg=sdfg, parent=state.parent_graph)
                g = state.parent_graph

                # Remove old state, put in cb instead
                g.add_node(cb, g.start_block == state)
                for ie in g.in_edges(state):
                    g.add_edge(ie.src, cb, dace.InterstateEdge())
                for oe in g.out_edges(state):
                    g.add_edge(cb, oe.dst, dace.InterstateEdge())
                g.remove_node(state)

                for i in range(extent):
                    cfg = ControlFlowRegion(
                        label=f"extent_check_{SplitArray.c}_branch_{i}",
                        sdfg=cb.sdfg,
                        parent=cb,
                    )
                    ns = cfg.add_state(f"extent_check_{SplitArray.c}_body_{i}", True)
                    node_map = copy_state_contents(state, ns)
                    cb.add_branch(condition=properties.CodeBlock(f"{canonical_access} == {i}"), branch=cfg)
                    access_mapping = dict()
                    for k in all_access_exprs:
                        access_mapping[k] = i
                    for edge in ns.edges():
                        mapped_data, new_subset = self._get_corresponding_array(edge, split_map, access_mapping)
                        if mapped_data != edge.data.data:
                            edge.data = dace.memlet.Memlet(data=mapped_data, subset=new_subset)
                SplitArray.c += 1
                #sdfg.validate()
                #sdfg.save("hmm.sdfg")
                #raise Exception(extent, canonical_access, all_access_exprs, dims)
            else:
                for edge in state.edges():
                    mapped_data, new_subset = self._get_corresponding_array(edge, split_map)
                    if mapped_data != edge.data.data:
                        edge.data = dace.memlet.Memlet(data=mapped_data, subset=new_subset)

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

            assert len(dim_filter) == len(new_args), (f"{fname}: split_config has {len(dim_filter)} dims "
                                                      f"but function call has {len(new_args)} args: {expr}")

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
                    assert len(unique_names) == 1 or (
                        len(in_names) == 1 and len(out_names) == 1
                    ), f"Expected only one unique data name for the node (or 1 unique input, 1 unique output) got: In: {in_names}, out: {out_names} in state {state}."

                    if len(in_names) == 1 and len(out_names) == 1:
                        dup_dnode = state.add_access(dnode.data)
                        for oe in out_edges:
                            state.remove_edge(oe)
                            state.add_edge(dup_dnode, None, oe.dst, oe.dst_conn, copy.deepcopy(oe.data))

                        dnode.data = in_names.pop()
                        dup_dnode.data = out_names.pop()

                        # Since they are unique arrays we dont need a dependency edge
                        # state.add_edge(dnode, None, dup_dnode, None, dace.Memlet())
                    else:
                        # Having 1 name means that all edges are consistent with each other, so we can just replace them all with the same new name
                        original_name = unique_names.pop()
                        dnode.data = original_name

    def _pass_to_nsdfgs(self, sdfg: dace.SDFG, split_map: Dict[str, List[Optional[str]]]):
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    in_names = set(n.in_connectors.keys())
                    out_names = set(n.out_connectors.keys())
                    unique_names = in_names.union(out_names)
                    if any(name in split_map for name in unique_names):
                        raise Exception(
                            f"TODO: Arrays split being passed to nested SDFGs is not supported yet. Found in {n} in state {state} of sdfg {sdfg}. Split arrays: {split_map.keys()}"
                        )

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
        sdfg.validate()
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
        sdfg.validate()

        # 3. Pass the arrays to the Nested SDFGs
        # (Don't forget they can go through maps)
        self._pass_to_nsdfgs(sdfg, split_map)

        # 4. Cleanup
        self._remove_split_arrays(sdfg, split_map)

        pass

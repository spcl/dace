import dace

from typing import Dict, List, Any
from dace.transformation import pass_pipeline as ppl
from dataclasses import dataclass


@dataclass
class PermuteDimensions(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        # This is an analysis pass, so it does not modify anything
        return (ppl.Modifies.States & ppl.Modifies.AccessNodes & ppl.Modifies.Edges & ppl.Modifies.Descriptors
                & ppl.Modifies.NestedSDFGs & ppl.Modifies.Memlets)

    def __init__(self, permute_map: Dict[str, List[int]],
                 add_permute_maps: bool,
                 use_permute_libnodes: bool = False,
                 column_major: bool = False):
        self._permute_map = permute_map
        self._use_permute_libnodes = use_permute_libnodes
        self._add_permute_maps = add_permute_maps
        self._column_major = column_major

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Once permutaiton is done, no re-application is needed, ever
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        self._permute_index(sdfg, sdfg, self._permute_map, self._add_permute_maps)
        return 0

    def _add_permute_map(self, sdfg: dace.SDFG, state: dace.SDFGState, old_shape: List[int], new_shape: List[int],
                            permute_indices: List[int], old_name: str, new_name: str):
        """
        Adds a transpose that copies data from the original layout to the permuted layout
        using the TensorTranspose library node.

        For GPU arrays, the cuTENSOR implementation is used (cutensorPermute).
        For CPU arrays, the pure (map-based) implementation is used.
        """
        if self._use_permute_libnodes:
            from dace.libraries.standard import TensorTranspose

            old_access = state.add_access(old_name)
            new_access = state.add_access(new_name)

            assert len(old_shape) == len(new_shape), \
                f"Old shape {old_shape} and new shape {new_shape} must have the same length"

            impl = "pure"

            tnode = TensorTranspose(f"permute_{old_name}_to_{new_name}", axes=permute_indices)
            tnode.implementation = impl
            state.add_node(tnode)

            state.add_edge(old_access, None, tnode, "_inp_tensor",
                        dace.Memlet.from_array(old_name, sdfg.arrays[old_name]))
            state.add_edge(tnode, "_out_tensor", new_access, None,
                        dace.Memlet.from_array(new_name, sdfg.arrays[new_name]))
        else:
            # Map iterates over the OLD shape
            map_params = [f"__i{d}" for d in range(len(old_shape))]
            map_ranges = {p: f"0:{s}" for p, s in zip(list(reversed(map_params)),
                                                      list(reversed(old_shape)))}

            # Read indices: i0, i1, i2, ...
            read_indices = ", ".join(map_params)

            # Write indices: permuted, e.g. [1,0,2] → __i1, __i0, __i2
            write_indices = ", ".join(map_params[permute_indices[d]] for d in range(len(permute_indices)))

            state.add_mapped_tasklet(
                name=f"permute_{old_name}_to_{new_name}",
                map_ranges=map_ranges,
                inputs={"__inp": dace.Memlet.simple(old_name, read_indices)},
                code="__out = __inp",
                outputs={"__out": dace.Memlet.simple(new_name, write_indices)},
                external_edges=True,
            )

    def _inverse_permute_indices(self, permute_indices: List[int]) -> List[int]:
        # implicit([0, 1, 2, 3]) -> [0, 3, 1, 2]
        # 1. get as a dictionary {0:0, 1:3, 2:1, 3:2}
        # 2. invert keys and values to get {0:0, 3:1, 1:2, 2:3}
        # 3. sort by keys to get {0:0, 1:3, 2:1, 3:2} -> [0, 3, 1, 2]
        # 1: Create mapping dictionary
        perm_map = {i: p for i, p in enumerate(permute_indices)}
        # 2: Invert the dictionary
        inverse_map = {v: k for k, v in perm_map.items()}
        # 3: Sort by key and extract values
        inverse_perm = [inverse_map[i] for i in sorted(inverse_map)]
        return inverse_perm

    def _permute_index(self, root: dace.SDFG, sdfg: dace.SDFG, permute_map: Dict[str, List[int]],
                       add_permute_maps: bool):
        # If top-level SDFG, namely the root is equal to the sdfg, we might need to add a transpose state and maps to
        # permute the arrays, otherwise we just replace the arrays with the permuted shape
        name_map = dict()
        permute_states_to_skip = set()

        for arr_name, arr in list(sdfg.arrays.items()):
            if arr_name in permute_map:
                permute_indices = permute_map[arr_name]

                arr_shape = arr.shape

                # Generate new shape
                permuted_shape = []
                assert len(permute_indices) == len(
                    arr_shape
                ), f"Permute indices {permute_indices} and array shape {arr_shape} must have the same length {arr_name}"
                for i in permute_indices:
                    permuted_shape.append(arr_shape[i])

                # Permuted array is packed (contiguous one-dimensional memory)
                strides = None
                if self._column_major:
                    strides = [1]
                    for i in range(len(permuted_shape) - 1):
                        strides.append(strides[-1] * permuted_shape[i])

                permuted_arr = dace.data.Array(
                    dtype=arr.dtype,
                    shape=permuted_shape,
                    strides=strides,
                    transient=True if (add_permute_maps and root == sdfg) else arr.transient,
                    allow_conflicts=arr.allow_conflicts,
                    storage=arr.storage,
                    alignment=arr.alignment,
                    lifetime=arr.lifetime,
                )

                # Change the in connector name
                # If before it was A -> (A)(NestedSDFG)
                # it will be per_A -> (A)(NestedSDFG)
                # If before it was A -> (nA)(NestedSDFG)
                # it will be per_A -> (nA)(NestedSDFG)
                # The nested SDFG needs to have identity as the name map
                if add_permute_maps and root == sdfg:
                    sdfg.add_datadesc(name="permuted_" + arr_name, datadesc=permuted_arr, find_new_name=False)
                else:
                    sdfg.remove_data(name=arr_name, validate=False)
                    sdfg.add_datadesc(name=arr_name,
                                      datadesc=permuted_arr)  # Need to transpose memlets before validation

                name_map[arr_name] = "permuted_" + arr_name if (add_permute_maps and root == sdfg) else arr_name

        if root == sdfg:
            if add_permute_maps:
                permute_state = sdfg.add_state_before(sdfg.start_state, "permute_in")
                permute_states_to_skip.add(permute_state)
                final_block = [v for v in sdfg.nodes() if sdfg.out_degree(v) == 0][0]
                permute_out_state = sdfg.add_state_after(final_block, "permute_out")
                permute_states_to_skip.add(permute_out_state)

                # Add maps to permute the input arrays to their permuted shape
                for old_name, new_name in name_map.items():
                    old_shape = sdfg.arrays[old_name].shape
                    new_shape = sdfg.arrays[new_name].shape
                    permute_indices = permute_map[old_name]

                    # Only non-transient glb arrays are input arrays
                    self._add_permute_map(sdfg=sdfg,
                                          state=permute_state,
                                          old_shape=old_shape,
                                          new_shape=new_shape,
                                          permute_indices=permute_indices,
                                          old_name=old_name,
                                          new_name=new_name)

                # Add maps to permute the arrays back to their original shape
                for old_name, new_name in name_map.items():
                    old_shape = sdfg.arrays[old_name].shape
                    new_shape = sdfg.arrays[new_name].shape
                    # Permute map is of form map[old] = new, we need to invert it
                    inverse_permute_indices = self._inverse_permute_indices(permute_map[old_name])

                    self._add_permute_map(sdfg=sdfg,
                                            state=permute_out_state,
                                            old_shape=new_shape,
                                            new_shape=old_shape,
                                            permute_indices=inverse_permute_indices,
                                            old_name=new_name,
                                            new_name=old_name)

        # The transformation has added the permuted shapes and maps to permute them if the user requested it.
        # The transformation has yet permuted the memlets as we want to access the previous defined arrays
        # The arrays passed to the NestedSDFG nodes need to be permuted as well, recursively go deeper
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    new_permute_map = dict()
                    # Change the in connector name
                    # If before it was A -> (A)(NestedSDFG)
                    # it will be per_A -> (A)(NestedSDFG)
                    # If before it was A -> (nA)(NestedSDFG)
                    # it will be per_A -> (nA)(NestedSDFG)

                    # The nested SDFG needs to have identity as the name map
                    # Update the names for the nested SDFG
                    # But only if the full array is passed, for example A[i] (array) -> tmp_X (scalar) does not require replacement

                    # TODO: views, do they need any changes?
                    # Open issue, what to do when we get all subsets, vs. only some of the subsets
                    # If we permute [a, b, c] -> [b, a, c], if we get subset [c] nothing changes,
                    # but if we get subset [a, b] we need to change it to [b, a]
                    # For now: the length of shape should be either 1 (no change needed), or the full length of the array (full change needed), otherwise we raise an exception as we do not know how to permute it
                    for ie in state.in_edges(node):
                        src_name = ie.data.data
                        dst_name = ie.dst_conn
                        dst_dimensionality = len(node.sdfg.arrays[dst_name].shape)
                        src_dimensionality = len(sdfg.arrays[src_name].shape)
                        if src_name in permute_map and src_dimensionality == dst_dimensionality:
                            new_permute_map[dst_name] = permute_map[src_name]
                    for oe in state.out_edges(node):
                        dst_name = oe.src_conn
                        src_name = oe.data.data
                        dst_dimensionality = len(node.sdfg.arrays[dst_name].shape)
                        src_dimensionality = len(sdfg.arrays[src_name].shape)
                        if src_name in permute_map and src_dimensionality == dst_dimensionality:
                            new_permute_map[dst_name] = permute_map[src_name]

                    self._permute_index(root=root, sdfg=node.sdfg, permute_map=new_permute_map, add_permute_maps=False)

        for state in sdfg.all_states():
            if sdfg == root and (state in permute_states_to_skip):
                continue
            for node in state.nodes():
                # Replace array access with the new Name (can be identity if we have not added permute maps)
                if isinstance(node, dace.nodes.AccessNode):
                    if node.data in name_map:
                        node.data = name_map[node.data]

            # Go through all memlets
            for edge in state.edges():
                if edge.data is not None and edge.data.data is not None and edge.data.data in name_map:
                    # Replace map connectors to reference to correct permuted array (e.g. IN_A -> IN_per_A)
                    # Do not change nested SDFG connectors
                    if edge.dst_conn == "IN_" + edge.data.data:
                        edge.dst_conn = "IN_" + name_map[edge.data.data]
                        edge.dst.remove_in_connector("IN_" + edge.data.data)
                        edge.dst.add_in_connector("IN_" + name_map[edge.data.data])
                    if edge.src_conn == "OUT_" + edge.data.data:
                        edge.src_conn = "OUT_" + name_map[edge.data.data]
                        edge.src.remove_out_connector("OUT_" + edge.data.data)
                        edge.src.add_out_connector("OUT_" + name_map[edge.data.data])

                    # Change data of the memlet
                    old_name = edge.data.data
                    edge.data.data = name_map[old_name]

                    # Permute the memlet subset
                    new_subset = []
                    permute_indices = permute_map[old_name]
                    for i in range(len(permute_indices)):
                        new_subset.append(edge.data.subset[permute_indices[i]])
                    edge.data.subset = dace.subsets.Range(new_subset)

        # Go through all interstate edges
        for edge in sdfg.all_interstate_edges():
            new_assignments = dict()
            for k, v in edge.data.assignments.items():
                # Replace array names if present, according to the permute conditions
                if any(name in v or name in k for name in permute_map.keys()):
                    # Time to replace
                    new_v = _parse_interstate_edge(v, permute_map, sdfg)
                    new_assignments[k] = new_v
                else:
                    new_assignments[k] = v
            edge.data.assignments = new_assignments


def permute_args(expr, permute_map: dict[str, list[int]]):
    """Recursively permute function call arguments in a SymPy/DaCe expression.

    permute_map: {func_name: perm} where perm[new_pos] = old_pos.
    Returns the original expression object if nothing changed.
    """
    if not expr.args:
        return expr
    args = tuple(permute_args(a, permute_map) for a in expr.args)
    name = str(expr.func)
    if name in permute_map:
        perm = permute_map[name]
        args = tuple(args[perm[i]] for i in range(len(args)))
    if args == expr.args:
        return expr
    return expr.func(*args)


def _parse_interstate_edge(edge_data: str, permute_map: dict[str, list[int]], sdfg: dace.SDFG = None):
    symbolic_expr: dace.symbolic.SymExpr = dace.symbolic.pystr_to_symbolic(edge_data)
    permuted_symbolic_expr: dace.symbolic.SymExpr = permute_args(symbolic_expr, permute_map)
    permuted_str_expr: str = dace.symbolic.symstr(sym=permuted_symbolic_expr, arrayexprs=frozenset(sdfg.arrays.keys()))
    return permuted_str_expr

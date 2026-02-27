import dace
from typing import Dict, List, Any
from dace.transformation import pass_pipeline as ppl
from dataclasses import dataclass


@dataclass
class PermuteArrayDimensions(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        # This is an analysis pass, so it does not modify anything
        return (ppl.Modifies.States & ppl.Modifies.AccessNodes & ppl.Modifies.Edges & ppl.Modifies.Descriptors
                & ppl.Modifies.NestedSDFGs & ppl.Modifies.Memlets)

    def __init__(self, permute_map: Dict[str, List[int]], add_permute_maps: bool):
        self._permute_map = permute_map
        self._add_permute_maps = add_permute_maps

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        self._permute_index(sdfg, sdfg, self._permute_map, self._add_permute_maps)
        return 0

    def _add_permute_map(self, sdfg: dace.SDFG, state: dace.SDFGState, old_shape: List[int], new_shape: List[int],
                         permute_indices: List[int], old_name: str, new_name: str):
        old_access = state.add_access(old_name)
        new_access = state.add_access(new_name)
        range_dict = dict()
        assert len(old_shape) == len(
            new_shape), f"Old shape {old_shape} and new shape {new_shape} must have the same length"
        for i in range(len(old_shape)):
            range_dict[f"i{i}"] = f"0:{new_shape[i]}"  # Could use old shape too

        # Add map that computes B[permute_indices[i], ..., permute_indices[k]] = A[i, j, ..., k]
        map_entry, map_exit = state.add_map("permute_impl", range_dict)

        src_access = ", ".join(f"i{i}" for i in range(len(permute_indices)))
        dst_access = ", ".join(f"i{permute_indices[i]}" for i in range(len(permute_indices)))
        map_entry.add_in_connector("IN_" + old_name)
        map_entry.add_out_connector("OUT_" + old_name)
        map_exit.add_in_connector("IN_" + new_name)
        map_exit.add_out_connector("OUT_" + new_name)
        state.add_edge(old_access, None, map_entry, "IN_" + old_name,
                       dace.Memlet.from_array(old_name, sdfg.arrays[old_name]))
        state.add_edge(map_exit, "OUT_" + new_name, new_access, None,
                       dace.Memlet.from_array(new_name, sdfg.arrays[new_name]))
        assign_tasklet = state.add_tasklet("assign", {"_in1"}, {"_out1"}, f"_out1 = _in1")
        state.add_edge(map_entry, "OUT_" + old_name, assign_tasklet, "_in1",
                       dace.Memlet(expr=f"{old_name}[{src_access}]"))
        state.add_edge(assign_tasklet, "_out1", map_exit, "IN_" + new_name,
                       dace.Memlet(expr=f"{new_name}[{dst_access}]"))

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
                permuted_arr = dace.data.Array(
                    dtype=arr.dtype,
                    shape=permuted_shape,
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
                    if sdfg.arrays[old_name].transient is False:
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
                    # Only non-transient glb arrays are output arrays
                    if sdfg.arrays[old_name].transient is False:
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
                    for ie in state.in_edges(node):
                        src_name = ie.data.data
                        dst_name = ie.dst_conn
                        if src_name in permute_map and sdfg.arrays[src_name].shape == node.sdfg.arrays[dst_name].shape:
                            new_permute_map[dst_name] = permute_map[src_name]
                    for oe in state.out_edges(node):
                        src_name = oe.src_conn
                        dst_name = oe.data.data
                        if dst_name in permute_map and sdfg.arrays[dst_name].shape == node.sdfg.arrays[src_name].shape:
                            new_permute_map[src_name] = permute_map[dst_name]
                    self._permute_index(root=root, sdfg=node.sdfg, permute_map=new_permute_map, add_permute_maps=False)

        for state in sdfg.all_states():
            if sdfg == root and (state in permute_states_to_skip):
                continue
            for node in state.nodes():
                # Replace array access with the new Name
                if isinstance(node, dace.nodes.AccessNode):
                    if node.data in name_map:
                        node.data = name_map[node.data]
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

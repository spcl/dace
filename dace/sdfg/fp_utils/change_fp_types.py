import dace
import copy
from typing import Union, Set, Dict, List

def _repl_recursive_with_connectors(sdfg: dace.SDFG, repl_dict):
    sdfg.replace_dict(repl_dict)

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                in_conns = copy.deepcopy(node.in_connectors)
                out_conns = copy.deepcopy(node.out_connectors)
                for in_conn in in_conns:
                    if in_conn in repl_dict:
                        node.remove_in_connector(in_conn)
                        node.add_in_connector(repl_dict[in_conn], force=True)
                for out_conn in out_conns:
                    if out_conn in repl_dict:
                        node.remove_out_connector(out_conn)
                        node.add_out_connector(repl_dict[out_conn], force=True)

                inner_sdfg = node.sdfg
                for name, arr in inner_sdfg.arrays.items():
                    if name in repl_dict:
                        inner_sdfg.remove_data(name, validate=False)
                        new_arr = copy.deepcopy(arr)
                        inner_sdfg.add_datadesc(name, new_arr)

                _repl_recursive_with_connectors(node.sdfg, repl_dict)

def _get_array_view_paths(sdfg: dace.SDFG, arrays_to_replace: Set[str]) -> Dict[str, Set[str]]:
    # TODO: Inefficient implementation, improve it later
    view_sets = {name: set() for name in arrays_to_replace}
    for state in sdfg.all_states():
        for node in state.nodes():
            for node2 in state.nodes():
                if node == node2:
                    continue
                if (isinstance(node, dace.nodes.AccessNode) and 
                    isinstance(node2, dace.nodes.AccessNode) and 
                    isinstance(sdfg.arrays[node.data], dace.data.Data) and
                    isinstance(sdfg.arrays[node2.data], dace.data.View) and
                    node.data in arrays_to_replace
                    ):
                    paths_iterator = state.all_simple_paths(node, node2)
                    if any(True for _ in paths_iterator):
                        view_sets[node.data].add(node2.data)
    return view_sets

def _change_fp_type_recursive(sdfg: dace.SDFG,
                              src_fptype: dace.dtypes.typeclass,
                              dst_fptype: dace.dtypes.typeclass,
                              arrays_to_replace: Set[str]):
    # Need to collect all views that depend on the arrays and add them to the set
    view_sets = _get_array_view_paths(sdfg, arrays_to_replace)
    print(view_sets)

    for k, view_set in view_sets.items():
        arrays_to_replace = arrays_to_replace.union(view_set)

    # Change FP-types of the arrays (and views that depend on them)
    named_array_set = set()
    for name in arrays_to_replace:
        arr = sdfg.arrays[name]
        named_array_set.add((name, arr))
        arr.dtype = dst_fptype

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                # Continue changing FP-types depending on the connectors
                new_replacements = set()
                for ie in state.in_edges(node):
                    if ie.data is not None and ie.data.data in arrays_to_replace:
                        new_replacements.add(ie.dst_conn)
                for oe in state.out_edges(node):
                    if oe.data is not None and oe.data.data in arrays_to_replace:
                        new_replacements.add(oe.dsst_conn)
                _change_fp_type_recursive(sdfg, src_fptype, dst_fptype, new_replacements)

def change_fptype(sdfg: dace.SDFG,
                  src_fptype: dace.dtypes.typeclass,
                  dst_fptype: dace.dtypes.typeclass,
                  cast_in_and_out_data: bool = False,
                  arrays_to_replace: Union[Set[str], None] = None):
    # If arrays_to_replace is None, all arrays are replaced
    if arrays_to_replace is None:
        arrays_to_replace = {name for name, arr in sdfg.arrays.items() if arr.dtype == src_fptype}
    else:
        # Check types are matchig
        for name in arrays_to_replace:
            arr = sdfg.arrays[name]
            if arr.dtype != src_fptype:
                raise ValueError(f"Array {name} from the passed inputs has fptype {arr.dtype} but function was provided src fp type ({src_fptype})")

    _change_fp_type_recursive(sdfg, src_fptype, dst_fptype, arrays_to_replace)

    # Replace all occurences of the arrays in the SDFG with their replaced counterparts
    if cast_in_and_out_data is True:
        # If we cast in data, the interface stays the same, we add new arrays
        # e.g. A -> A_(dst_fptype.to_string)
        array_name_mapping = dict()
        for name in arrays_to_replace:
            array_name_mapping[name] = f"{name}_{dst_fptype.to_string()}"

        _repl_recursive_with_connectors(sdfg, array_name_mapping)

        # Copy-in and copy-out extensions
        # For all arrays that we made transient, we need to ensure we add the appropriate copy-in and copy-out nodes
        copy_in_state = sdfg.add_state_before(state=sdfg.start_block, label="copy_in")
        last_blocks = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0] # Only last block has no successors
        assert len(last_blocks) == 1
        last_block = last_blocks[0]
        copy_out_state = sdfg.add_state_after(state=last_block, label="copy_out")

        # Add a new array descriptor for each transient array, and add copy-in or copy-out
        src_dst_paris = set()
        for original_arr_name, casted_arr_name in array_name_mapping.items():
            arr = sdfg.arrays[casted_arr_name]
            original_arr_desc = copy.deepcopy(arr) # Transientness does not change for this array
            arr.transient = True # Always transient due to copy-in
            original_arr_desc.dtype = src_fptype # Change back to the original fp type
            sdfg.add_datadesc(name=original_arr_name, datadesc=original_arr_desc, find_new_name=False)
            src_dst_paris.add(((casted_arr_name, arr), (original_arr_name, original_arr_desc)))

        def _add_copy_map(state: dace.SDFGState, src_arr_name:str, src_arr:dace.data.Data, dst_arr_name:str, dst_arr:dace.data.Data):
            """
            Add a copy map to the given state in the SDFG.
            """
            assert src_arr.shape == dst_arr.shape, "Source and destination arrays must have the same shape."
            # Add a tasklet that perfmorms the type cast
            tasklet = state.add_tasklet(
                name=f"copy_{src_arr_name}_to_{dst_arr_name}",
                inputs={"in"},
                outputs={"out"},
                code=f"out = static_cast<{dst_arr.dtype.ctype}>(in);",
                language=dace.Language.CPP)

            if isinstance(src_arr, dace.data.Array):
                assert isinstance(dst_arr, dace.data.Array)
                # Create a new map node
                map_ranges = dict()
                for dim, size in enumerate(src_arr.shape):
                    map_ranges[f"i{dim}"] = f"0:{size}"

                map_entry, map_exit = state.add_map(name=f"copy_map_{src_arr_name}_to_{dst_arr_name}", ndrange=map_ranges)

                # Add access nodes for source and destination arrays
                src_access = state.add_access(src_arr_name)
                dst_access = state.add_access(dst_arr_name)

                # Add edges from the map to the access nodes, care about the connector
                state.add_edge(src_access, None, map_entry, f"IN_{src_arr_name}", dace.memlet.Memlet.from_array(src_arr_name, src_arr))
                state.add_edge(map_exit, f"OUT_{dst_arr_name}", dst_access, None, dace.memlet.Memlet.from_array(dst_arr_name, dst_arr))
                map_entry.add_in_connector(f"IN_{src_arr_name}")
                map_entry.add_out_connector(f"OUT_{src_arr_name}")
                map_exit.add_in_connector(f"IN_{dst_arr_name}")
                map_exit.add_out_connector(f"OUT_{dst_arr_name}")
                access_str = f", ".join([str(s) for s in map_ranges.keys()])
                state.add_edge(map_entry, f"OUT_{src_arr_name}", tasklet, "in", dace.Memlet(expr=f"{src_arr_name}[{access_str}]"))
                state.add_edge(tasklet, "out", map_exit, f"IN_{dst_arr_name}", dace.Memlet(expr=f"{dst_arr_name}[{access_str}]"))
            else:
                assert isinstance(src_arr, dace.data.Scalar)
                assert isinstance(dst_arr, dace.data.Scalar)
                src_access = state.add_access(src_arr_name)
                dst_access = state.add_access(dst_arr_name)
                state.add_edge(src_access, None, tasklet, "in", dace.Memlet(expr=f"{src_arr_name}[0]"))
                state.add_edge(tasklet, "out", dst_access, None, dace.Memlet(expr=f"{dst_arr_name}[0]"))

        for (transient_arr_name, transient_arr), (nontransient_arr_name, nontransient_arr) in src_dst_paris:
            # No need to copy from transient to transient
            if transient_arr.transient and nontransient_arr.transient:
                continue
            _add_copy_map(state=copy_in_state,
                            src_arr_name=nontransient_arr_name,
                            src_arr=nontransient_arr,
                            dst_arr_name=transient_arr_name,
                            dst_arr=transient_arr)
            _add_copy_map(state=copy_out_state,
                            src_arr_name=transient_arr_name,
                            src_arr=transient_arr,
                            dst_arr_name=nontransient_arr_name,
                            dst_arr=nontransient_arr)
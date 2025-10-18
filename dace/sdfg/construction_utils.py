import re
from typing import Dict, Set, Union
import dace
import copy

from dace.sdfg.propagation import propagate_memlets_state
import copy
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion


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


def insert_non_transient_data_through_parent_scopes(
    non_transient_data: Set[str],
    nsdfg_node: 'dace.nodes.NestedSDFG',
    parent_graph: 'dace.SDFGState',
    parent_sdfg: 'dace.SDFG',
    add_to_output_too: bool = False,
    add_with_exact_subset: bool = False,
    exact_subset: Union[None, dace.subsets.Range] = None,
):
    """
    Inserts non-transient data containers into all relevant parent scopes (through all map scopes).

    This function connect data from top-level data
    into nested SDFGs (and vice versa) by connecting AccessNodes, MapEntries,
    and NestedSDFG connectors appropriately.

    Args:
        non_transient_data: Set of data container names to propagate.
        nsdfg_node: The nested SDFG node where the data should be connected.
        parent_graph: The parent SDFG state that contains the NestedSDFG node.
        parent_sdfg: The parent SDFG corresponding to `parent_graph.sdfg`.
        add_to_output_too: If True, also connect the data as an output from the nested SDFG.
        add_with_exact_subset: If True, use an explicitly provided subset for the memlet.
        exact_subset: The explicit subset (if any) to use when `add_with_exact_subset` is True.

    Behavior:
        - Adds data descriptors for any missing non-transient arrays to both
          the parent SDFG and the nested SDFG.
        - Connects data through all enclosing parent scopes (e.g., nested maps).
        - Optionally adds symmetric output connections.
        - Propagates memlets if exact subsets are used.
        - Adds any newly required symbols (from shapes or strides) to the nested SDFG.
    """

    descs = [None] * len(non_transient_data)
    assert len(descs) == len(non_transient_data)

    for data_access, desc in zip(non_transient_data, descs):
        datadesc = desc or parent_sdfg.arrays[data_access]
        assert isinstance(parent_graph, dace.SDFGState), "Parent graph must be a SDFGState"
        inner_sdfg: dace.SDFG = nsdfg_node.sdfg

        # Skip if the connector already exists and is wired
        if (data_access in nsdfg_node.in_connectors
                and len(list(parent_graph.in_edges_by_connector(nsdfg_node, data_access))) > 0):
            continue

        # Remove conflicting symbols in nested SDFG
        if data_access in inner_sdfg.symbols:
            inner_sdfg.remove_symbol(data_access)

        # Add the data descriptor to the nested SDFG if missing
        if data_access not in inner_sdfg.arrays:
            copydesc = copy.deepcopy(datadesc)
            copydesc.transient = False
            inner_sdfg.add_datadesc(name=data_access, datadesc=copydesc)

        # Ensure the parent also has the data descriptor
        if data_access not in parent_sdfg.arrays:
            copydesc = copy.deepcopy(datadesc)
            copydesc.transient = False
            parent_sdfg.add_datadesc(name=data_access, datadesc=copydesc)

        # Collect enclosing map scopes to route data through
        parent_scopes = []
        cur_parent_scope = nsdfg_node
        scope_dict = parent_graph.scope_dict()
        while scope_dict[cur_parent_scope] is not None:
            parent_scopes.append(scope_dict[cur_parent_scope])
            cur_parent_scope = scope_dict[cur_parent_scope]

        # Helper: choose between full or exact-subset memlet
        def _get_memlet(it_id: int, data_access: str, datadesc: dace.data.Data):
            if add_with_exact_subset:
                return dace.memlet.Memlet(data=data_access, subset=copy.deepcopy(exact_subset))
            else:
                return dace.memlet.Memlet.from_array(data_access, datadesc)

        # --- Add input connection path ---
        an = parent_graph.add_access(data_access)
        src = an
        for it_id, parent_scope in enumerate(reversed(parent_scopes)):
            dst = parent_scope
            parent_graph.add_edge(
                src,
                None if isinstance(src, dace.nodes.AccessNode) else f"OUT_{data_access}",
                dst,
                data_access if isinstance(dst, dace.nodes.NestedSDFG) else f"IN_{data_access}",
                _get_memlet(it_id, data_access, datadesc),
            )
            # Ensure connectors exist
            if not isinstance(src, dace.nodes.AccessNode):
                src.add_out_connector(f"OUT_{data_access}", force=True)
            if isinstance(dst, dace.nodes.NestedSDFG):
                dst.add_in_connector(data_access, force=True)
            else:
                dst.add_in_connector(f"IN_{data_access}")
            src = parent_scope

        # Connect final edge to the NestedSDFG
        dst = nsdfg_node
        parent_graph.add_edge(
            src,
            None if isinstance(src, dace.nodes.AccessNode) else f"OUT_{data_access}",
            dst,
            data_access if isinstance(dst, dace.nodes.NestedSDFG) else f"IN_{data_access}",
            _get_memlet(it_id, data_access, datadesc),
        )
        if not isinstance(src, dace.nodes.AccessNode):
            src.add_out_connector(f"OUT_{data_access}", force=True)
        if isinstance(dst, dace.nodes.NestedSDFG):
            dst.add_in_connector(data_access, force=True)
        else:
            dst.add_in_connector(f"IN_{data_access}", force=True)

        # --- Optionally add output connection path ---
        if add_to_output_too:
            an = parent_graph.add_access(data_access)
            dst = an
            for it_id, parent_scope in enumerate(reversed(parent_scopes)):
                src = parent_graph.exit_node(parent_scope)
                parent_graph.add_edge(
                    src,
                    data_access if isinstance(src, dace.nodes.NestedSDFG) else f"OUT_{data_access}",
                    dst,
                    None if isinstance(dst, dace.nodes.AccessNode) else f"IN_{data_access}",
                    _get_memlet(it_id, data_access, datadesc),
                )
                if not isinstance(dst, dace.nodes.AccessNode):
                    dst.add_in_connector(f"IN_{data_access}", force=True)
                if isinstance(src, dace.nodes.NestedSDFG):
                    src.add_out_connector(data_access, force=True)
                else:
                    src.add_out_connector(f"OUT_{data_access}")
                dst = src
            src = nsdfg_node
            parent_graph.add_edge(
                src,
                data_access if isinstance(src, dace.nodes.NestedSDFG) else f"OUT_{data_access}",
                dst,
                None if isinstance(dst, dace.nodes.AccessNode) else f"IN_{data_access}",
                _get_memlet(it_id, data_access, datadesc),
            )
            if not isinstance(dst, dace.nodes.AccessNode):
                dst.add_in_connector(f"IN_{data_access}", force=True)
            src.add_out_connector(data_access if isinstance(src, dace.nodes.NestedSDFG) else f"OUT_{data_access}")

    # Re-propagate memlets when subsets are explicit
    if add_with_exact_subset:
        propagate_memlets_state(parent_graph.sdfg, parent_graph)

    # Add any free symbols from array shapes/strides to the nested SDFG
    new_symbols = set()
    for data_access, desc in zip(non_transient_data, descs):
        if desc is None:
            desc = parent_graph.sdfg.arrays[data_access]
        data_free_syms = set()
        for dim, stride in zip(desc.shape, desc.strides):
            dim_expr = dace.symbolic.SymExpr(dim)
            stride_expr = dace.symbolic.SymExpr(stride)
            if not isinstance(stride_expr, int):
                data_free_syms |= stride_expr.free_symbols
            if not isinstance(dim_expr, int):
                data_free_syms |= dim_expr.free_symbols
        new_symbols |= data_free_syms

    defined_syms = parent_graph.symbols_defined_at(nsdfg_node)
    for sym in new_symbols:
        if str(sym) not in nsdfg_node.sdfg.symbols:
            nsdfg_node.sdfg.add_symbol(sym, defined_syms[str(sym)])
        if str(sym) not in nsdfg_node.symbol_mapping:
            nsdfg_node.symbol_mapping[str(sym)] = str(sym)


def token_replace(code: str, src: str, dst: str) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', code)

    # Replace tokens that exactly match src
    tokens = [dst if token.strip() == src else token for token in tokens]

    # Recombine everything
    return ''.join(tokens).strip()


def token_match(string_to_check: str, pattern_str: str) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens}

    return pattern_str in tokens


def replace_length_one_arrays_with_scalars(sdfg: dace.SDFG, recursive: bool = True, transient_only: bool = False):
    scalarized_arrays = set()
    for arr_name, arr in [(k, v) for k, v in sdfg.arrays.items()]:
        if isinstance(arr, dace.data.Array) and (arr.shape == (1, ) or arr.shape == [
                1,
        ]):
            if (not transient_only) or arr.transient:
                sdfg.remove_data(arr_name, False)
                sdfg.add_scalar(name=arr_name,
                                dtype=arr.dtype,
                                storage=arr.storage,
                                transient=arr.transient,
                                lifetime=arr.lifetime,
                                debuginfo=arr.debuginfo,
                                find_new_name=False)
                scalarized_arrays.add(arr_name)
                print(f"Making {arr_name} into scalar")

    # Replace [0] accesses of scalars (formerly array ones) on interstate edges
    for edge in sdfg.all_interstate_edges():
        new_dict = dict()
        for k, v in edge.data.assignments.items():
            nv = v
            for scalar_name in scalarized_arrays:
                if f"{scalar_name}[0]" in nv:
                    nv = nv.replace(f"{scalar_name}[0]", scalar_name)
            new_dict[k] = nv
        edge.data.assignments = new_dict

    # Replace [0] accesses of scalars (formerly array ones) on IfBlocks
    for node in sdfg.all_control_flow_blocks():
        if isinstance(node, ConditionalBlock):
            for cond, body in node.branches:
                if cond is None:
                    continue
                nlc = cond.as_string if isinstance(cond, CodeBlock) else str(cond)
                for scalar_name in scalarized_arrays:
                    if f"{scalar_name}[0]" in nlc:
                        nlc = nlc.replace(f"{scalar_name}[0]", scalar_name)
                cond = CodeBlock(nlc, cond.language if isinstance(cond, CodeBlock) else dace.dtypes.Language.Python)

    # Replace [0] accesses of scalars (formerly array ones) on LoopRegions
    for node in sdfg.all_control_flow_regions():
        if isinstance(node, LoopRegion):
            nlc = node.loop_condition.as_string if isinstance(node.loop_condition, CodeBlock) else str(
                node.loop_condition)
            for scalar_name in scalarized_arrays:
                if f"{scalar_name}[0]" in nlc:
                    nlc = nlc.replace(f"{scalar_name}[0]", scalar_name)
            node.loop_condition = CodeBlock(
                nlc, node.loop_condition.language
                if isinstance(node.loop_condition, CodeBlock) else dace.dtypes.Language.Python)

    if recursive:
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    replace_length_one_arrays_with_scalars(node.sdfg, recursive=True, transient_only=True)


def connect_array_names(sdfg: dace.SDFG, local_storage: dace.dtypes.StorageType, src_storage: dace.dtypes.StorageType,
                        local_name_prefix: str):

    array_name_dict = dict()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                local_arr = state.sdfg.arrays[node.data]
                print(local_arr.storage)
                if local_arr.storage == local_storage:
                    assert len(state.in_edges(node)) <= 1
                    # Reads
                    for ie in state.in_edges(node):
                        if ie.data.data is not None and ie.data.data != node.data:
                            src_data = state.sdfg.arrays[ie.data.data]
                            print(src_data)
                            if src_data.storage == src_storage:
                                assert node.data not in array_name_dict
                                array_name_dict[node.data] = ie.data.data
                    # Writes
                    for oe in state.out_edges(node):
                        if oe.data.data is not None and oe.data.data != node.data:
                            dst_data = state.sdfg.arrays[oe.data.data]
                            print(dst_data)
                            if dst_data.storage == src_storage:
                                assert node.data not in array_name_dict
                                array_name_dict[node.data] = oe.data.data

    print(array_name_dict)
    repldict = {k: f"{local_name_prefix}{v}" for k, v in array_name_dict.items()}

    sdfg.replace_dict(repldict, replace_keys=True)
    sdfg.validate()

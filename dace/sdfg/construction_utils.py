from typing import Dict, Set, Union
import dace
import copy

from dace.sdfg.propagation import propagate_memlets_state


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


def get_missing_symbols(nsdfg_node: dace.nodes.NestedSDFG) -> Set[str]:
    """
    Detects symbols used in a nested SDFG that are not yet mapped from the parent.

    Args:
        nsdfg_node: The NestedSDFG node to check.

    Returns:
        A set of symbol names that are missing in the node's symbol mapping.

    Notes:
        - This function compares used symbols with available connectors and
          mapped symbols.
    """
    nsdfg = nsdfg_node.sdfg
    connectors = nsdfg_node.in_connectors.keys() | nsdfg_node.out_connectors.keys()
    symbols = set(k for k in nsdfg.used_symbols(all_symbols=False) if k not in connectors)
    missing_symbols = [s for s in symbols if s not in nsdfg_node.symbol_mapping]
    if missing_symbols:
        print(f"Missing symbols: {missing_symbols} for nsdfg: {nsdfg_node}")
    return set(missing_symbols)


def add_missing_symbols_to_nsdfgs(sdfg: dace.SDFG):
    """
    Recursively fixes missing symbol mappings for all nested SDFGs in the given SDFG.

    Args:
        sdfg: The root SDFG to process.

    Notes:
        - Calls `add_missing_symbols_to_nsdfg` for each nested SDFG encountered.
        - Recurses into deeper nested SDFGs as needed.
    """
    nsdfgs = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                add_missing_symbols_to_nsdfg(state, node)
    for nsdfg in nsdfgs:
        add_missing_symbols_to_nsdfgs(nsdfg)


def add_missing_symbols_to_nsdfg(parent_state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG):
    """
    Adds any symbols used inside a nested SDFG that are missing in its symbol mapping.

    Args:
        parent_state: The state that contains the NestedSDFG node.
        nsdfg: The NestedSDFG node to fix.

    Behavior:
        - Identifies missing symbols via `get_missing_symbols`.
        - Adds them both to the nested SDFG’s symbol table and its mapping from the parent.
    """
    missing_symbols = get_missing_symbols(nsdfg)
    for ms in missing_symbols:
        nsdfg.symbol_mapping[ms] = ms
        if ms not in nsdfg.sdfg.symbols:
            defined_syms = parent_state.symbols_defined_at(nsdfg)
            nsdfg.sdfg.add_symbol(ms, defined_syms[ms])

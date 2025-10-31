# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Dict, Set, Union
import copy
from dace.sdfg import ControlFlowRegion
from dace.sdfg.propagation import propagate_memlets_state
import copy
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion
from sympy import Function
import dace.sdfg.utils as sdutil
from dace.sdfg.tasklet_utils import token_replace_dict, extract_bracket_tokens, remove_bracket_tokens


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


def copy_graph_contents(old_graph: ControlFlowRegion,
                        new_graph: ControlFlowRegion) -> Dict[dace.nodes.Node, dace.nodes.Node]:
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
    assert isinstance(old_graph, ControlFlowRegion)
    assert isinstance(new_graph, ControlFlowRegion)

    node_map = dict()

    # Copy all nodes
    for n in old_graph.nodes():
        c_n = copy.deepcopy(n)
        node_map[n] = c_n
        new_graph.add_node(c_n, is_start_block=old_graph.start_block == n)

    # Copy all edges, reconnecting them to their new node counterparts
    for e in old_graph.edges():
        c_src = node_map[e.src]
        c_dst = node_map[e.dst]
        new_graph.add_edge(c_src, c_dst, copy.deepcopy(e.data))

    sdutil.set_nested_sdfg_parent_references(new_graph.sdfg)

    return node_map


def move_branch_cfg_up_discard_conditions(if_block: ConditionalBlock, body_to_take: ControlFlowRegion):
    """
    Moves a branch of a conditional block up in the control flow graph (CFG),
    replacing the conditional with the selected branch, discarding
    the conditional check and other branches.

    This operation:
    - Copies all nodes and edges from the selected branch (`body_to_take`) into
      the parent graph of the conditional.
    - Connects all incoming edges of the original conditional block to the
      start of the selected branch.
    - Connects all outgoing edges of the original conditional block to the
      end of the selected branch.
    - Removes the original conditional block from the graph.

    Parameters:
        -if_block : ConditionalBlock
            The conditional block in the CFG whose branch is to be promoted.
        -body_to_take : ControlFlowRegion
            The branch of the conditional block to be moved up. Must be one of the
            branches of `if_block`.
    """
    # Sanity check the ensure passed arguments are correct
    bodies = {b for _, b in if_block.branches}
    assert body_to_take in bodies
    assert isinstance(if_block, ConditionalBlock)

    graph = if_block.parent_graph

    node_map = dict()
    # Save end and start blocks for reconnections
    new_start_block = None
    new_end_block = None

    for node in body_to_take.nodes():
        # Copy over nodes
        copynode = copy.deepcopy(node)
        node_map[node] = copynode
        # Check if we need to have a new start state
        start_block_case = (body_to_take.start_block == node) and (graph.start_block == if_block)
        if body_to_take.start_block == node:
            assert new_start_block is None
            new_start_block = copynode
        if body_to_take.out_degree(node) == 0:
            assert new_end_block is None
            new_end_block = copynode
        graph.add_node(copynode, is_start_block=start_block_case)

    for edge in body_to_take.edges():
        src = node_map[edge.src]
        dst = node_map[edge.dst]
        graph.add_edge(src, dst, copy.deepcopy(edge.data))

    for ie in graph.in_edges(if_block):
        graph.add_edge(ie.src, new_start_block, copy.deepcopy(ie.data))
    for oe in graph.out_edges(if_block):
        graph.add_edge(new_end_block, oe.dst, copy.deepcopy(oe.data))

    graph.remove_node(if_block)


def insert_non_transient_data_through_parent_scopes(non_transient_data: Set[str],
                                                    nsdfg_node: 'dace.nodes.NestedSDFG',
                                                    parent_graph: 'dace.SDFGState',
                                                    parent_sdfg: 'dace.SDFG',
                                                    add_to_output_too: bool = False,
                                                    add_with_exact_subset: bool = False,
                                                    exact_subset: Union[None, dace.subsets.Range] = None,
                                                    nsdfg_connector_name: Union[str, None] = None):
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
        inner_data_access = data_access if nsdfg_connector_name is None else nsdfg_connector_name
        if inner_data_access not in inner_sdfg.arrays:
            copydesc = copy.deepcopy(datadesc)
            copydesc.transient = False
            inner_sdfg.add_datadesc(name=inner_data_access, datadesc=copydesc)

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

        state = {
            'cur_in_conn_name': f"IN_{data_access}_p",
            'cur_out_conn_name': f"OUT_{data_access}_p",
            'cur_name_set': False,
        }

        def _get_in_conn_name(dst, state=state):
            if state['cur_name_set'] is False:
                i = 0
                while (state['cur_in_conn_name'] in dst.in_connectors
                       or state['cur_out_conn_name'] in dst.out_connectors):
                    state['cur_in_conn_name'] = f"IN_{data_access}_p_{i}"
                    state['cur_out_conn_name'] = f"OUT_{data_access}_p_{i}"
                    i += 1
                state['cur_name_set'] = True

            inner_data_access = data_access if nsdfg_connector_name is None else nsdfg_connector_name

            if isinstance(dst, dace.nodes.AccessNode):
                return None
            elif isinstance(dst, dace.nodes.NestedSDFG):
                return inner_data_access
            else:
                return state['cur_in_conn_name']

        def _get_out_conn_name(src, state=state):
            if state['cur_name_set'] is False:
                i = 0
                while (state['cur_in_conn_name'] in src.in_connectors
                       or state['cur_out_conn_name'] in src.out_connectors):
                    state['cur_in_conn_name'] = f"IN_{data_access}_p_{i}"
                    state['cur_out_conn_name'] = f"OUT_{data_access}_p_{i}"
                    i += 1
                state['cur_name_set'] = True

            inner_data_access = data_access if nsdfg_connector_name is None else nsdfg_connector_name
            if isinstance(src, dace.nodes.AccessNode):
                return None
            elif isinstance(src, dace.nodes.NestedSDFG):
                return inner_data_access
            else:
                return state['cur_out_conn_name']

        an = parent_graph.add_access(data_access)
        src = an
        for it_id, parent_scope in enumerate(reversed(parent_scopes)):
            dst = parent_scope
            # Initialize state with a parent map
            _get_in_conn_name(dst)

            parent_graph.add_edge(
                src,
                _get_out_conn_name(src),
                dst,
                _get_in_conn_name(dst),
                _get_memlet(it_id, data_access, datadesc),
            )
            # Ensure connectors exist
            if not isinstance(src, dace.nodes.AccessNode):
                src.add_out_connector(_get_out_conn_name(src), force=True)
            if isinstance(dst, dace.nodes.NestedSDFG):
                dst.add_in_connector(_get_in_conn_name(dst), force=True)
            else:
                dst.add_in_connector(_get_in_conn_name(dst))
            src = parent_scope

        # Connect final edge to the NestedSDFG
        dst = nsdfg_node
        parent_graph.add_edge(
            src,
            _get_out_conn_name(src),
            dst,
            _get_in_conn_name(dst),
            _get_memlet(it_id, data_access, datadesc),
        )
        if not isinstance(src, dace.nodes.AccessNode):
            src.add_out_connector(_get_out_conn_name(src), force=True)
        if isinstance(dst, dace.nodes.NestedSDFG):
            dst.add_in_connector(_get_in_conn_name(dst), force=True)
        else:
            dst.add_in_connector(_get_in_conn_name(dst), force=True)

        # --- Optionally add output connection path ---
        if add_to_output_too:
            an = parent_graph.add_access(data_access)
            dst = an
            for it_id, parent_scope in enumerate(reversed(parent_scopes)):
                src = parent_graph.exit_node(parent_scope)
                parent_graph.add_edge(
                    src,
                    _get_out_conn_name(src),
                    dst,
                    _get_in_conn_name(dst),
                    _get_memlet(it_id, data_access, datadesc),
                )
                if not isinstance(dst, dace.nodes.AccessNode):
                    dst.add_in_connector(_get_in_conn_name(dst), force=True)
                if isinstance(src, dace.nodes.NestedSDFG):
                    src.add_out_connector(_get_out_conn_name(src), force=True)
                else:
                    src.add_out_connector(_get_out_conn_name(src), )
                dst = src
            src = nsdfg_node
            parent_graph.add_edge(
                src,
                _get_out_conn_name(src),
                dst,
                _get_in_conn_name(dst),
                _get_memlet(it_id, data_access, datadesc),
            )
            if not isinstance(dst, dace.nodes.AccessNode):
                dst.add_in_connector(f"IN_{data_access}_p", force=True)
            src.add_out_connector(_get_out_conn_name(dst))

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
            nsdfg_node.sdfg.add_symbol(str(sym), defined_syms[str(sym)])
        if str(sym) not in nsdfg_node.symbol_mapping:
            nsdfg_node.symbol_mapping[str(sym)] = str(sym)


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


def generate_assignment_as_tasklet_in_state(state: dace.SDFGState, lhs: str, rhs: str):
    rhs = rhs.strip()
    rhs_sym_expr = dace.symbolic.SymExpr(rhs).evalf()
    lhs = lhs.strip()
    lhs_sym_expr = dace.symbolic.SymExpr(lhs).evalf()

    in_connectors = dict()
    out_connectors = dict()

    # Get functions for indirect accesses
    i = 0
    for free_sym in rhs_sym_expr.free_symbols.union({f.func for f in rhs_sym_expr.atoms(Function)}):
        if str(free_sym) in state.sdfg.arrays:
            in_connectors[str(free_sym)] = f"_in_{free_sym}_{i}"
            i += 1
    for free_sym in lhs_sym_expr.free_symbols.union({f.func for f in lhs_sym_expr.atoms(Function)}):
        if str(free_sym) in state.sdfg.arrays:
            out_connectors[str(free_sym)] = f"_out_{free_sym}_{i}"
            i += 1

    if in_connectors == {} and out_connectors == {}:
        raise Exception("Generated tasklets result in no or out connectors")

    # Process interstate edge, extract brackets for access patterns
    in_access_exprs = extract_bracket_tokens(token_replace_dict(rhs, in_connectors))
    out_access_exprs = extract_bracket_tokens(token_replace_dict(lhs, out_connectors))
    lhs = remove_bracket_tokens(token_replace_dict(lhs, out_connectors))
    rhs = remove_bracket_tokens(token_replace_dict(rhs, in_connectors))

    # Ass tasklets
    t = state.add_tasklet(name=f"assign_{lhs}",
                          inputs=set(in_connectors.values()),
                          outputs=set(out_connectors.values()),
                          code=f"{lhs} = {rhs}")

    # Add connectors and accesses
    in_access_dict = dict()
    out_access_dict = dict()
    for k, v in in_connectors.items():
        in_access_dict[v] = state.add_access(k)
    for k, v in out_connectors.items():
        out_access_dict[v] = state.add_access(k)

    # Add in and out connections
    for k, v in in_access_dict.items():
        data_name = v.data
        access_str = in_access_exprs.get(k)
        if access_str is None:
            access_str = "0"
        state.add_edge(v, None, t, k, dace.memlet.Memlet(expr=f"{data_name}[{access_str}]"))
    for k, v in out_access_dict.items():
        data_name = v.data
        access_str = out_access_exprs.get(k)
        if access_str is None:
            access_str = "0"
        state.add_edge(t, k, v, None, dace.memlet.Memlet(expr=f"{data_name}[{access_str}]"))


def get_num_parent_map_scopes(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):
    return len(get_parent_maps(root_sdfg, node, parent_state))


def get_num_parent_map_and_loop_scopes(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):
    return len(get_parent_map_and_loop_scopes(root_sdfg, node, parent_state))


def get_parent_map_and_loop_scopes(root_sdfg: dace.SDFG, node: Union[dace.nodes.MapEntry, ControlFlowRegion,
                                                                     dace.nodes.Tasklet, ConditionalBlock],
                                   parent_state: dace.SDFGState):
    scope_dict = parent_state.scope_dict() if parent_state is not None else None
    num_parent_maps_and_loops = 0
    cur_node = node
    parent_scopes = list()

    def _get_parent_state(sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG):
        for n, g in sdfg.all_nodes_recursive():
            if n == nsdfg_node:
                return g
        return None

    if isinstance(cur_node, (dace.nodes.MapEntry, dace.nodes.Tasklet)):
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
                num_parent_maps_and_loops += 1
                parent_scopes.append(scope_dict[cur_node])
            cur_node = scope_dict[cur_node]

    parent_graph = parent_state.parent_graph if parent_state is not None else node.parent_graph
    parent_sdfg = parent_state.sdfg if parent_state is not None else node.parent_graph.sdfg
    while parent_graph != parent_sdfg:
        if isinstance(parent_graph, LoopRegion):
            num_parent_maps_and_loops += 1
            parent_scopes.append(parent_graph)
        parent_graph = parent_graph.parent_graph

    # Check parent nsdfg
    parent_nsdfg_node = parent_sdfg.parent_nsdfg_node
    parent_nsdfg_parent_state = _get_parent_state(root_sdfg, parent_nsdfg_node)

    while parent_nsdfg_node is not None and parent_nsdfg_parent_state is not None:
        scope_dict = parent_nsdfg_parent_state.scope_dict()
        cur_node = parent_nsdfg_node
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
                num_parent_maps_and_loops += 1
                parent_scopes.append(scope_dict[cur_node])
            cur_node = scope_dict[cur_node]

        parent_graph = parent_nsdfg_parent_state.parent_graph
        parent_sdfg = parent_graph.sdfg
        while parent_graph != parent_sdfg:
            if isinstance(parent_graph, LoopRegion):
                num_parent_maps_and_loops += 1
                parent_scopes.append(parent_graph)
            parent_graph = parent_graph.parent_graph

        parent_nsdfg_node = parent_sdfg.parent_nsdfg_node
        parent_nsdfg_parent_state = _get_parent_state(root_sdfg, parent_nsdfg_node)

    return parent_scopes


def get_parent_maps(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):

    def _get_parent_state(sdfg: dace.SDFG, nsdfg_node: dace.nodes.NestedSDFG):
        for n, g in sdfg.all_nodes_recursive():
            if n == nsdfg_node:
                return g
        return None

    maps = []
    scope_dict = parent_state.scope_dict()
    cur_node = node
    while scope_dict[cur_node] is not None:
        if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
            maps.append((cur_node, parent_state))
        cur_node = scope_dict[cur_node]

    parent_graph = parent_state.parent_graph
    while parent_graph != parent_state.sdfg:
        if isinstance(parent_graph, LoopRegion):
            pass
        parent_graph = parent_graph.parent_graph

    # Check parent nsdfg
    parent_nsdfg_node = parent_state.sdfg.parent_nsdfg_node
    parent_nsdfg_parent_state = _get_parent_state(root_sdfg, parent_nsdfg_node)

    while parent_nsdfg_node is not None:
        scope_dict = parent_nsdfg_parent_state.scope_dict()
        cur_node = parent_nsdfg_node
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
                maps.append((cur_node, parent_state))
            cur_node = scope_dict[cur_node]
        parent_nsdfg_node = parent_nsdfg_parent_state.sdfg.parent_nsdfg_node
        parent_nsdfg_parent_state = parent_state.sdfg.parent_graph

    return maps

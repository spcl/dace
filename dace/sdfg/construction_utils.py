import re
from typing import Dict, Set, Union
import dace
import copy

from dace.sdfg import ControlFlowRegion
from dace.sdfg.propagation import propagate_memlets_state
import copy
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion

import sympy
from sympy import symbols, Function

from sympy.printing.pycode import PythonCodePrinter
import dace.sdfg.utils as sdutil
from dace.transformation.passes import FuseStates


class BracketFunctionPrinter(PythonCodePrinter):

    def _print_Function(self, expr):
        name = self._print(expr.func)
        args = ", ".join([self._print(arg) for arg in expr.args])
        return f"{name}[{args}]"


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
    # Sanity check the ensure apssed arguments are correct
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


# Put map-body into NSDFG
# Convert Map to Loop
# Put map into NSDFG


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


def token_replace_dict(code: str, repldict: Dict[str, str]) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', code)

    # Replace tokens that exactly match src
    tokens = [repldict[token.strip()] if token.strip() in repldict else token for token in tokens]

    # Recombine everything
    return ''.join(tokens).strip()


def token_match(string_to_check: str, pattern_str: str) -> str:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens}

    return pattern_str in tokens


def token_split(string_to_check: str) -> Set[str]:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens}

    return tokens


def token_split_variable_names(string_to_check: str) -> Set[str]:
    # Split while keeping delimiters
    tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

    # Replace tokens that exactly match src
    tokens = {token.strip() for token in tokens if token not in ["[", "]", "(", ")"] and token.isidentifier()}

    return tokens


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


def tasklet_has_symbol(tasklet: dace.nodes.Tasklet, symbol_str: str) -> bool:
    if tasklet.code.language == dace.dtypes.Language.Python:
        try:
            sym_expr = dace.symbolic.SymExpr(tasklet.code.as_astring)
            return (symbol_str in {str(s) for s in sym_expr.free_symbols})
        except Exception as e:
            return token_match(tasklet.code.as_string, symbol_str)
    else:
        return token_match(tasklet.code.as_string, symbol_str)


def replace_code(code_str: str, code_lang: dace.dtypes.Language, repldict: Dict[str, str]) -> str:

    def _str_replace(lhs: str, rhs: str) -> str:
        code_str = token_replace_dict(rhs, repldict)
        return f"{lhs.strip()} = {code_str.strip()}"

    if code_lang == dace.dtypes.Language.Python:
        try:
            lhs, rhs = code_str.split(" = ")
            lhs = lhs.strip()
            rhs = rhs.strip()
        except Exception as e:
            try:
                new_rhs_sym_expr = dace.symbolic.SymExpr(code_str).subs(repldict)
                printer = BracketFunctionPrinter({'strict': False})
                cleaned_expr = printer.doprint(new_rhs_sym_expr).strip()
                assert isinstance(cleaned_expr, str)
                return f"{cleaned_expr}"
            except Exception as e:
                return _str_replace(code_str)
        try:
            new_rhs_sym_expr = dace.symbolic.SymExpr(rhs).subs(repldict)
            printer = BracketFunctionPrinter({'strict': False})
            cleaned_expr = printer.doprint(new_rhs_sym_expr).strip()
            assert isinstance(cleaned_expr, str)
            return f"{lhs.strip()} = {cleaned_expr}"
        except Exception as e:
            return _str_replace(rhs)
    else:
        return _str_replace(rhs)


def tasklet_replace_code(tasklet: dace.nodes.Tasklet, repldict: Dict[str, str]):
    new_code = replace_code(tasklet.code.as_string, tasklet.code.language, repldict)
    tasklet.code = CodeBlock(code=new_code, language=tasklet.code.language)


def extract_bracket_tokens(s: str) -> list[tuple[str, list[str]]]:
    """
    Extracts all contents inside [...] along with the token before the '[' as the name.

    Args:
        s (str): Input string.

    Returns:
        List of tuples: [(name_token, string inside brackes)]
    """
    results = []

    # Pattern to match <name>[content_inside]
    pattern = re.compile(r'(\b\w+)\[([^\]]*?)\]')

    for match in pattern.finditer(s):
        name = match.group(1)  # token before '['
        content = match.group(2).split()  # split content inside brackets into tokens

        results.append((name, " ".join(content)))

    return {k: v for (k, v) in results}


def remove_bracket_tokens(s: str) -> str:
    """
    Removes all [...] patterns from the string.

    Args:
        s (str): Input string.

    Returns:
        str: String with all [...] removed.
    """
    return re.sub(r'\[.*?\]', '', s)


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


def _find_parent_state(root_sdfg: dace.SDFG, node: dace.nodes.NestedSDFG):
    if node is not None:
        # Find parent state of that node
        for n, g in root_sdfg.all_nodes_recursive():
            if n == node:
                parent_state = g
                return parent_state
    return None


def get_num_parent_map_scopes(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):
    scope_dict = parent_state.scope_dict()
    num_parent_maps = 0
    cur_node = node
    while scope_dict[cur_node] is not None:
        if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
            num_parent_maps += 1
        cur_node = scope_dict[cur_node]

    # Check parent nsdfg
    parent_nsdfg_node = parent_state.sdfg.parent_nsdfg_node
    parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

    while parent_nsdfg_node is not None:
        scope_dict = parent_nsdfg_parent_state.scope_dict()
        cur_node = parent_nsdfg_node
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
                num_parent_maps += 1
            cur_node = scope_dict[cur_node]
        parent_nsdfg_node = parent_nsdfg_parent_state.sdfg.parent_nsdfg_node
        parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

    return num_parent_maps


def get_num_parent_map_and_loop_scopes(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):
    return len(get_parent_map_and_loop_scopes(root_sdfg, node, parent_state))


def get_parent_map_and_loop_scopes(root_sdfg: dace.SDFG, node: Union[dace.nodes.MapEntry, ControlFlowRegion,
                                                                     dace.nodes.Tasklet, ConditionalBlock],
                                   parent_state: dace.SDFGState):
    scope_dict = parent_state.scope_dict() if parent_state is not None else None
    num_parent_maps_and_loops = 0
    cur_node = node
    parent_scopes = list()

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
    parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

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
        parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

    return parent_scopes


def get_parent_maps(root_sdfg: dace.SDFG, node: dace.nodes.MapEntry, parent_state: dace.SDFGState):
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
    parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

    while parent_nsdfg_node is not None:
        scope_dict = parent_nsdfg_parent_state.scope_dict()
        cur_node = parent_nsdfg_node
        while scope_dict[cur_node] is not None:
            if isinstance(scope_dict[cur_node], dace.nodes.MapEntry):
                maps.append((cur_node, parent_state))
            cur_node = scope_dict[cur_node]
        parent_nsdfg_node = parent_nsdfg_parent_state.sdfg.parent_nsdfg_node
        parent_nsdfg_parent_state = _find_parent_state(root_sdfg, parent_nsdfg_node)

    return maps


def _find_new_name(base: str, existing_names: Set[str]) -> str:
    i = 0
    candidate = f"{base}_d_{i}"
    while candidate in existing_names:
        i += 1
        candidate = f"{base}_d_{i}"
    return candidate


def duplicate_memlets_sharing_single_in_connector(state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
    for out_conn in list(map_entry.out_connectors.keys()):
        out_edges_of_out_conn = set(state.out_edges_by_connector(map_entry, out_conn))
        if len(out_edges_of_out_conn) > 1:
            base_in_edge = out_edges_of_out_conn.pop()

            # Get all parent maps (including this)
            parent_maps: Set[dace.nodes.MapEntry] = {map_entry}
            sdict = state.scope_dict()
            parent_map = sdict[map_entry]
            while parent_map is not None:
                parent_maps.add(parent_map)
                parent_map = sdict[parent_map]

            # Need it to find unique names
            all_existing_connector_names = set()
            for map_entry in parent_maps:
                for in_conn in map_entry.in_connectors:
                    all_existing_connector_names.add(in_conn[len("IN_"):])
                for out_conn in map_entry.out_connectors:
                    all_existing_connector_names.add(out_conn[len("OUT_"):])

            # Base path
            memlet_paths = []
            path = state.memlet_path(base_in_edge)
            source_node = path[0].src
            memlet_paths.append(path)
            while sdict[source_node] is not None:
                if not isinstance(source_node, (dace.nodes.AccessNode, dace.nodes.MapEntry)):
                    print(source_node)
                    raise Exception(
                        f"In the path from map entry to the top level scope, only access nodes and other map entries may appear, got: {source_node}"
                    )
                in_edges = state.in_edges(source_node)
                if isinstance(source_node, dace.nodes.MapEntry) and len(in_edges) != 1:
                    in_edges = list(state.in_edges_by_connector(source_node, "IN_" + path[-1].src_conn[len("OUT_"):]))
                if isinstance(source_node, dace.nodes.AccessNode) and len(in_edges) != 1:
                    raise Exception(
                        "In the path from map entry to the top level scope, the intermediate access nodes need to have in and out degree (by connector) 1"
                    )

                in_edge = in_edges[0]
                path = state.memlet_path(in_edge)
                source_node = path[0].src
                memlet_paths.append(path)
                #print(source_node)

            # Need to duplicate the out edges
            for e in list(out_edges_of_out_conn):
                state.remove_edge(e)

            for edge_to_duplicate in out_edges_of_out_conn:
                base = edge_to_duplicate.src_conn[len("OUT_"):]
                new_connector_base = _find_new_name(base, all_existing_connector_names)
                all_existing_connector_names.add(new_connector_base)

                node_map = dict()
                for i, subpath in enumerate(memlet_paths):
                    for j, e in enumerate(reversed(subpath)):
                        # We work by adding an in edge
                        in_name = f"IN_{new_connector_base}"
                        out_name = f"OUT_{new_connector_base}"

                        if e.src_conn is not None:
                            out_conn = out_name if e.src_conn.startswith("OUT_") else e.src_conn
                        else:
                            out_conn = None

                        if e.dst_conn is not None:
                            if e.src == map_entry:
                                in_conn = edge_to_duplicate.dst_conn
                            else:
                                in_conn = in_name if e.dst_conn.startswith("IN_") else e.dst_conn
                        else:
                            in_conn = None

                        if isinstance(e.src, dace.nodes.MapEntry):
                            src_node = e.src
                        elif isinstance(e.src, dace.nodes.AccessNode):
                            if e.src in node_map:
                                src_node = node_map[e.src]
                            else:
                                a = state.add_access(e.src.data)
                                node_map[e.src] = a
                                src_node = a
                        else:
                            src_node = e.src

                        if isinstance(e.dst, dace.nodes.MapEntry):
                            dst_node = e.dst
                        elif isinstance(e.dst, dace.nodes.AccessNode):
                            if e.dst in node_map:
                                dst_node = node_map[e.dst]
                            else:
                                a = state.add_access(e.dst.data)
                                node_map[e.dst] = a
                                dst_node = a
                        else:
                            dst_node = e.dst

                        # Above the first map, always add the complete subset and then call memlet propagation
                        if e.src is map_entry:
                            data = copy.deepcopy(edge_to_duplicate.data)
                        else:
                            data = dace.memlet.Memlet.from_array(e.data.data, state.sdfg.arrays[e.data.data])

                        state.add_edge(src_node, out_conn, dst_node, in_conn, data)

                        if out_conn is not None and out_conn not in src_node.out_connectors:
                            src_node.add_out_connector(out_conn, force=True)
                        if in_conn is not None and in_conn not in dst_node.in_connectors:
                            dst_node.add_in_connector(in_conn, force=True)

                        # If we duplicate an access node, we should add correct dependency edges
                        if i == len(memlet_paths) - 1:
                            if j == len(subpath) - 1:
                                # Source node
                                origin_source_node = e.src
                                for ie in state.in_edges(origin_source_node):
                                    state.add_edge(ie.src, None, src_node, None, dace.memlet.Memlet(None))

    propagate_memlets_state(state.sdfg, state)

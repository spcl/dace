import ast
import copy
import re
from typing import Set
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, LoopRegion


def replace_length_one_arrays_with_scalars(sdfg: dace.SDFG, recursive: bool = True):
    scalarized_arrays = set()
    for arr_name, arr in [(k, v) for k, v in sdfg.arrays.items()]:
        if isinstance(arr, dace.data.Array) and arr.shape == (1, ):
            sdfg.remove_data(arr_name, False)
            sdfg.add_scalar(name=arr_name,
                            dtype=arr.dtype,
                            storage=arr.storage,
                            transient=arr.transient,
                            lifetime=arr.lifetime,
                            debuginfo=arr.debuginfo,
                            find_new_name=False)
            scalarized_arrays.add(arr_name)

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
                    replace_length_one_arrays_with_scalars(node.sdfg)


def array_is_used_in_the_sdfg(sdfg: dace.SDFG, arr_name: str):
    # Check access nodes
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == arr_name:
                return True
    # Check edges
    for edge in sdfg.all_interstate_edges():
        for k, v in edge.data.assignments.items():
            if k == arr_name or arr_name in v:
                return True
    # If
    for node in sdfg.all_control_flow_blocks():
        if isinstance(node, ConditionalBlock):
            for cond, body in node.branches:
                code_str = ""
                if cond is not None:
                    for code in cond.code:
                        if isinstance(code, list):
                            code_str = " ".join({str(s.as_string) for s in code})
                        elif isinstance(code, str):
                            code_str = code
                        elif isinstance(code, CodeBlock):
                            code_str = code.as_string
                        elif isinstance(code, ast.Expr):
                            code_str = ast.unparse(code)
                        else:
                            raise Exception(f"Unhandled case: type: {type(code)}")
                if arr_name in {s.strip() for s in re.split(r'[()\[\]\s]+', code_str) if code_str is not None}:
                    return True
    # Loop
    for node in sdfg.all_control_flow_regions():
        if isinstance(node, LoopRegion):
            if arr_name in {s.strip() for s in re.split(r'[()\[\]\s]+', node.loop_condition.as_string)}:
                return True
            if arr_name in {s.strip() for s in re.split(r'[()\[\]\s]+', node.init_statement.as_string)}:
                return True

    return False


def array_is_written_to_in_the_sdfg(sdfg: dace.SDFG, arr_name: str):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode) and node.data == arr_name:
                if state.in_degree(node) > 1:
                    non_none_data = {ie for ie in state.in_edges(node) if ie.data is not None}
                    if len(non_none_data) > 0:
                        return True
    return False


def remove_array_from_connectors(parent_state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG, arr_name: str):

    def _rm_memlet_tree(parent_state: dace.SDFGState, memlet_tree: dace.memlet.MemletTree):
        for tree_node in memlet_tree.traverse_children(True):
            edge = tree_node.edge
            parent_state.remove_edge(edge)
            if edge.src_conn is not None:
                edge.src.remove_out_connector(edge.src_conn)
            if edge.dst_conn is not None:
                edge.dst.remove_in_connector(edge.dst_conn)
            if parent_state.degree(edge.src) == 0:
                parent_state.remove_node(edge.src)
            if parent_state.degree(edge.dst) == 0:
                parent_state.remove_node(edge.dst)

    if arr_name in nsdfg.in_connectors:
        memlet_trees: Set[dace.memlet.MemletTree] = set()
        for ie in parent_state.in_edges_by_connector(nsdfg, arr_name):
            mtree = parent_state.memlet_tree(ie)
            memlet_trees.add(mtree)
        assert len(memlet_trees) == 1
        memlet_tree = memlet_trees.pop()
        _rm_memlet_tree(parent_state, memlet_tree)

    if arr_name in nsdfg.out_connectors:
        memlet_trees: Set[dace.memlet.MemletTree] = set()
        for oe in parent_state.out_edges_by_connector(nsdfg, arr_name):
            mtree = parent_state.memlet_tree(oe)
            memlet_trees.add(mtree)
        assert len(memlet_trees) == 1
        memlet_tree = memlet_trees.pop()
        _rm_memlet_tree(parent_state, memlet_tree)


def try_to_add_missing_arrays_to_nsdfgs(sdfg: dace.SDFG):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for arr_name, arr in node.sdfg.arrays.items():
                    if arr.transient is False:
                        #if array_is_used_in_the_sdfg(node.sdfg, arr_name):
                        # Add to input
                        if arr_name not in node.in_connectors and arr_name not in node.out_connectors:
                            assert state.scope_dict()[node] is None, f"Parent scopes are not supported by this function"

                            print(f"Add {arr_name} to parent nSDFG's in connectors")
                            node.add_in_connector(arr_name, force=True)
                            an = state.add_access(arr_name)

                            if arr_name in node.sdfg.arrays and arr_name not in state.sdfg.arrays:
                                print(f"Adding {arr_name} desc to parent SDFG because it is not available there")
                                cpdesc = copy.deepcopy(node.sdfg.arrays[arr_name])
                                state.sdfg.add_datadesc(arr_name, cpdesc)

                            state.add_edge(an, None, node, arr_name,
                                           dace.memlet.Memlet.from_array(arr_name, state.sdfg.arrays[arr_name]))

                            if array_is_written_to_in_the_sdfg(node.sdfg, arr_name):
                                print(f"{arr_name} is written to too, add to parent nSDFG's out connectors")
                                node.add_out_connector(arr_name, force=True)
                                an = state.add_access(arr_name)
                                if arr_name in node.sdfg.arrays and arr_name not in state.sdfg.arrays:
                                    cpdesc = copy.deepcopy(node.sdfg.arrays[arr_name])
                                    state.sdfg.add_datadesc(arr_name, cpdesc)
                                state.add_edge(node, arr_name, an, None,
                                               dace.memlet.Memlet.from_array(arr_name, state.sdfg.arrays[arr_name]))

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                try_to_add_missing_arrays_to_nsdfgs(node.sdfg)


def prune_unnused_arrays_from_nsdfgs(sdfg: dace.SDFG):

    def _arr_in_connectors(nsdfg: dace.nodes.NestedSDFG, arr_name: str):
        return arr_name in nsdfg.in_connectors or arr_name in nsdfg.out_connectors

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for arr_name in list(node.sdfg.arrays.keys()):
                    if not array_is_used_in_the_sdfg(node.sdfg, arr_name):
                        if _arr_in_connectors(node, arr_name):
                            print(f"Removing unused array from connectors first")
                            remove_array_from_connectors(state, node, arr_name)
                        print(f"Removing {arr_name} from {node}")
                        node.sdfg.remove_data(arr_name)

    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                prune_unnused_arrays_from_nsdfgs(node.sdfg)


def get_missing_symbols(nsdfg_node: dace.nodes.NestedSDFG) -> Set[str]:
    nsdfg = nsdfg_node.sdfg
    connectors = nsdfg_node.in_connectors.keys() | nsdfg_node.out_connectors.keys()
    symbols = set(k for k in nsdfg.used_symbols(all_symbols=False) if k not in connectors)
    missing_symbols = [s for s in symbols if s not in nsdfg_node.symbol_mapping]
    return set(missing_symbols)


def add_missing_symbols_to_symbol_maps_of_nsdfgs(sdfg: dace.SDFG):
    nsdfgs = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                nsdfg = node
                inner_sdfg = node.sdfg
                nsdfgs.add(inner_sdfg)
                missing_symbols = get_missing_symbols(nsdfg)
                for ms in missing_symbols:
                    print(f"Adding missing symbol {ms} to the symbol map of {nsdfg}")
                    nsdfg.symbol_mapping[ms] = ms

    for nsdfg in nsdfgs:
        add_missing_symbols_to_symbol_maps_of_nsdfgs(nsdfg)


def _connector_is_live_in_nsdfg(inner: dace.SDFG, name: str) -> bool:
    """Returns True if ``name`` is referenced anywhere inside ``inner`` --
    access nodes, interstate-edge assignments/conditions, conditional-block
    conditions, or loop conditions/updates/init statements. Conservative: a
    substring match over a tokenised representation is used for expressions,
    so false positives are possible but false negatives are not."""

    def _tokens(expr: str) -> Set[str]:
        return {s.strip() for s in re.split(r'[()\[\]\s,+\-*/%<>!=&|^~?:]+', expr) if s.strip()}

    if array_is_used_in_the_sdfg(inner, name):
        return True

    for e in inner.all_interstate_edges():
        data = e.data
        if name in data.assignments:
            return True
        for v in data.assignments.values():
            if name in _tokens(str(v)):
                return True
        cond = data.condition.as_string if data.condition is not None else ""
        if cond and name in _tokens(cond):
            return True

    for region in inner.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for attr in ("loop_condition", "update_statement", "init_statement"):
                code = getattr(region, attr, None)
                text = code.as_string if isinstance(code, CodeBlock) else (str(code) if code else "")
                if text and name in _tokens(text):
                    return True

    return False


def _prune_memlet_path(state: dace.SDFGState, edge):
    """Remove ``edge`` together with the full memlet path it belongs to,
    cleaning up connectors on any intermediate map entries/exits and any
    access-node taps that become orphan as a result."""
    for e in list(state.memlet_path(edge)):
        if e not in state.edges():
            continue
        state.remove_edge(e)
        if e.src_conn is not None:
            try:
                e.src.remove_out_connector(e.src_conn)
            except (KeyError, ValueError):
                pass
        if e.dst_conn is not None:
            try:
                e.dst.remove_in_connector(e.dst_conn)
            except (KeyError, ValueError):
                pass
        for ep in (e.src, e.dst):
            if (isinstance(ep, dace.nodes.AccessNode)
                    and ep in state.nodes()
                    and state.degree(ep) == 0):
                state.remove_node(ep)


def prune_unused_nsdfg_connectors(state: dace.SDFGState, nsdfg: dace.nodes.NestedSDFG) -> int:
    """Drop input/output connectors of ``nsdfg`` that are never accessed inside
    its inner SDFG.

    The surrounding memlet path (access-node taps outside any map scope, and
    the pairs of connectors on any enclosing map entry/exit) is walked outward
    and pruned along with the connector. Inner arrays that become unused as a
    result are also dropped from ``nsdfg.sdfg`` so validation doesn't trip on
    non-transient arrays without a feeding connector.

    Returns the number of connector names removed.
    """
    inner = nsdfg.sdfg
    names = set(nsdfg.in_connectors) | set(nsdfg.out_connectors)
    removed = 0
    for name in names:
        if _connector_is_live_in_nsdfg(inner, name):
            continue

        for ie in list(state.in_edges_by_connector(nsdfg, name)):
            _prune_memlet_path(state, ie)
        for oe in list(state.out_edges_by_connector(nsdfg, name)):
            _prune_memlet_path(state, oe)

        if name in nsdfg.in_connectors:
            nsdfg.remove_in_connector(name)
        if name in nsdfg.out_connectors:
            nsdfg.remove_out_connector(name)

        if name in inner.arrays and not _connector_is_live_in_nsdfg(inner, name):
            try:
                inner.remove_data(name, validate=False)
            except Exception:
                inner.arrays.pop(name, None)
        removed += 1
    return removed


def prune_unused_nsdfg_connectors_recursive(sdfg: dace.SDFG) -> int:
    """Apply :func:`prune_unused_nsdfg_connectors` to every ``NestedSDFG`` in
    the SDFG hierarchy, bottom-up so that outer nested SDFGs see already
    cleaned inner ones."""
    total = 0
    for state in sdfg.all_states():
        for node in list(state.nodes()):
            if isinstance(node, dace.nodes.NestedSDFG):
                total += prune_unused_nsdfg_connectors_recursive(node.sdfg)
                total += prune_unused_nsdfg_connectors(state, node)
    return total

import dace
import re
import ast
import copy
from typing import Set

from dace.properties import CodeBlock
from dace.sdfg.sdfg import ConditionalBlock
from dace.sdfg.state import LoopRegion

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
            if arr_name in {s.strip() for s in re.split(r'[()\[\]\s]+',node.loop_condition.as_string)}:
                return True
            if arr_name in {s.strip() for s in re.split(r'[()\[\]\s]+',node.init_statement.as_string)}:
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
        memlet_trees: Set[MemletTree] = set()
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
                            print(f"Add {arr_name} to parent nSDFG's in connectors")
                            node.add_in_connector(arr_name, force=True)
                            an = state.add_access(arr_name)
                            
                            if arr_name in node.sdfg.arrays and arr_name not in state.sdfg.arrays:
                                print(f"Adding {arr_name} desc to parent SDFG because it is not available there")
                                cpdesc = copy.deepcopy(node.sdfg.arrays[arr_name])
                                state.sdfg.add_datadesc(arr_name, cpdesc)

                            state.add_edge(
                                an,
                                None,
                                node,
                                arr_name,
                                dace.memlet.Memlet.from_array(arr_name, state.sdfg.arrays[arr_name])
                            )

                            if array_is_written_to_in_the_sdfg(node.sdfg, arr_name):
                                print(f"{arr_name} is written to too, add to parent nSDFG's out connectors")
                                node.add_out_connector(arr_name, force=True)
                                print(state.sdfg.arrays)
                                an = state.add_access(arr_name)
                                if arr_name in node.sdfg.arrays and arr_name not in state.sdfg.arrays:
                                    cpdesc = copy.deepcopy(node.sdfg.arrays[arr_name])
                                    state.sdfg.add_datadesc(arr_name, cpdesc)
                                state.add_edge(
                                    node,
                                    arr_name,
                                    an,
                                    None,
                                    dace.memlet.Memlet.from_array(arr_name, state.sdfg.arrays[arr_name])
                                )


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


def try_fix_mismatching_inout_connectors(sdfg: dace.SDFG):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                in_conns = set(node.in_connectors.keys())
                out_conns = set(node.out_connectors.keys())
                inout_conns = in_conns.intersection(out_conns)

                for inout_conn in inout_conns:
                    ies = list(state.in_edges_by_connector(node, inout_conn))
                    oes = list(state.out_edges_by_connector(node, inout_conn))
                    if len(ies) != 1 or len(oes) != 1:
                        continue

                    ie = ies[0]
                    oe = oes[0]

                    if ie.data != oe.data:
                        if (isinstance(state.sdfg.arrays[ie.data.data], dace.data.View) and
                            not (isinstance(state.sdfg.arrays[oe.data.data], dace.data.View))):
                            an = state.add_access(ie.data.data)
                            assert ie.data.data == oe.src_conn
                            state.add_edge(
                                node,
                                oe.src_conn,
                                an,
                                None,
                                copy.deepcopy(ie.data.data)
                            )
                            state.add_edge(
                                an,
                                "views",
                                oe.dst,
                                oe.dst_conn,
                                copy.deepcopy(oe.data)
                            )
                            state.remove_edge(oe)


# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains functions for ensuring SDFGs and nested SDFGs share the same data descriptors.
"""
from dace import SDFG, data, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation.helpers import unsqueeze_memlet
from typing import Dict, List, Set
import copy


def dealias_sdfg_recursive(sdfg: SDFG):
    """
    Renames all data containers in an SDFG tree (i.e., nested SDFGs) to use the same data descriptors
    as the top-level SDFG. This function takes care of offsetting memlets and internal
    uses of arrays such that there is one naming system, and no aliasing of managed memory.

    :param sdfg: The SDFG to operate on.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():
        dealias_sdfg(nsdfg)


def dealias_sdfg(sdfg: SDFG):
    """
    Renames all data containers in an SDFG to match the same data descriptors
    as its parent SDFG, if exists. This function takes care of offsetting memlets and internal
    uses of arrays such that there is one naming system, and no aliasing of managed memory.

    This function operates in-place.

    :param sdfg: The SDFG to operate on.
    """
    if sdfg.parent is None:
        return

    parent_sdfg = sdfg.parent_sdfg
    parent_state = sdfg.parent
    parent_node = sdfg.parent_nsdfg_node

    # Rename nested arrays that happen to have the same name with an unrelated parent array connected to the node
    parent_names = set()
    for edge in parent_state.all_edges(parent_node):
        if edge.data.data in parent_sdfg.arrays:
            parent_names.add(edge.data.data)
    inner_replacements: Dict[str, str] = {}
    for name, desc in sdfg.arrays.items():
        if name in parent_names:
            replace = False
            if desc.transient:
                replace = True
            else:
                for edge in parent_state.edges_by_connector(parent_node, name):
                    parent_name = edge.data.data
                    assert parent_name in parent_sdfg.arrays
                    if name != parent_name:
                        replace = True
                        break
            if replace:
                new_name = sdfg._find_new_name(name)
                inner_replacements[name] = new_name

    if inner_replacements:
        symbolic.safe_replace(inner_replacements, lambda d: replace_datadesc_names(sdfg, d), value_as_string=True)
        parent_node.in_connectors = {
            inner_replacements[c] if c in inner_replacements else c: t
            for c, t in parent_node.in_connectors.items()
        }
        parent_node.out_connectors = {
            inner_replacements[c] if c in inner_replacements else c: t
            for c, t in parent_node.out_connectors.items()
        }
        for e in parent_state.all_edges(parent_node):
            if e.src_conn in inner_replacements:
                e._src_conn = inner_replacements[e.src_conn]
            elif e.dst_conn in inner_replacements:
                e._dst_conn = inner_replacements[e.dst_conn]

    replacements: Dict[str, str] = {}
    inv_replacements: Dict[str, List[str]] = {}
    parent_edges: Dict[str, Memlet] = {}
    to_unsqueeze: Set[str] = set()

    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        for edge in parent_state.edges_by_connector(parent_node, name):
            parent_name = edge.data.data
            assert parent_name in parent_sdfg.arrays
            if name != parent_name:
                replacements[name] = parent_name
                parent_edges[name] = edge
                to_unsqueeze.add(parent_name)
                if parent_name in inv_replacements:
                    inv_replacements[parent_name].append(name)
                else:
                    inv_replacements[parent_name] = [name]
                break

    if to_unsqueeze:
        for parent_name in to_unsqueeze:
            parent_arr = parent_sdfg.arrays[parent_name]

            # Add new symbols from the parent data descriptor to the symbol mapping.
            previous_syms = set()
            for name in inv_replacements[parent_name]:
                child_arr = sdfg.arrays[name]
                previous_syms |= child_arr.used_symbols(all_symbols=True)
            new_syms = parent_arr.used_symbols(all_symbols=True) - previous_syms
            for sym in new_syms:
                if str(sym) not in sdfg.symbols:
                    sdfg.add_symbol(str(sym), parent_sdfg.symbols[str(sym)])
                    parent_node.symbol_mapping[str(sym)] = str(sym)

            if isinstance(parent_arr, data.ArrayView):
                parent_arr = parent_arr.as_array()
            elif isinstance(parent_arr, data.StructureView):
                parent_arr = parent_arr.as_structure()
            elif isinstance(parent_arr, data.ContainerView):
                parent_arr = parent_arr.as_array()
            child_names = inv_replacements[parent_name]
            for name in child_names:
                child_arr = copy.deepcopy(parent_arr)
                child_arr.transient = False
                sdfg.arrays[name] = child_arr
            for state in sdfg.states():
                for e in state.edges():
                    if e.data.is_empty():
                        continue
                    if not state.is_leaf_memlet(e):
                        continue

                    mpath = state.memlet_path(e)
                    src, dst = mpath[0].src, mpath[-1].dst

                    # We need to take directionality of the memlet into account and unsqueeze either to source or
                    # destination subset
                    if isinstance(src, nd.AccessNode) and src.data in child_names:
                        src_data = src.data
                        new_src_memlet = unsqueeze_memlet(e.data, parent_edges[src.data].data, use_src_subset=True)
                    else:
                        src_data = None
                        new_src_memlet = None
                        # We need to take directionality of the memlet into account
                    if isinstance(dst, nd.AccessNode) and dst.data in child_names:
                        dst_data = dst.data
                        new_dst_memlet = unsqueeze_memlet(e.data, parent_edges[dst.data].data, use_dst_subset=True)
                    else:
                        dst_data = None
                        new_dst_memlet = None

                    # NOTE: If new symbols appear in the Memlet, we need to add them to the symbol mapping.
                    # NOTE: We assume that these symbols are defined (in any sense) in the immediate parent scope.
                    # NOTE: Since these symbols appear in Memlets, we assume that they are integers.
                    previous_syms = e.data.used_symbols(all_symbols=True)
                    if new_src_memlet is not None:
                        new_syms = new_src_memlet.used_symbols(all_symbols=True) - previous_syms
                        for sym in new_syms:
                            if sym not in sdfg.symbols:
                                sdfg.add_symbol(str(sym), symbolic.DEFAULT_SYMBOL_TYPE)
                                parent_node.symbol_mapping[sym] = sym
                        e.data.src_subset = new_src_memlet.subset
                    if new_dst_memlet is not None:
                        new_syms = new_dst_memlet.used_symbols(all_symbols=True) - previous_syms
                        for sym in new_syms:
                            if sym not in sdfg.symbols:
                                sdfg.add_symbol(str(sym), symbolic.DEFAULT_SYMBOL_TYPE)
                                parent_node.symbol_mapping[sym] = sym
                        e.data.dst_subset = new_dst_memlet.subset
                    if e.data.data == src_data:
                        e.data.data = new_src_memlet.data
                    elif e.data.data == dst_data:
                        e.data.data = new_dst_memlet.data

            for e in sdfg.all_interstate_edges():
                repl_dict = dict()
                syms = e.data.read_symbols()
                for memlet in e.data.get_read_memlets(sdfg.arrays):
                    if memlet.data in child_names:
                        repl_dict[str(memlet)] = unsqueeze_memlet(memlet, parent_edges[memlet.data].data)
                        if memlet.data in syms:
                            syms.remove(memlet.data)
                for s in syms:
                    if s in parent_edges:
                        if s in sdfg.arrays:
                            repl_dict[s] = parent_edges[s].data.data
                        else:
                            repl_dict[s] = str(parent_edges[s].data)
                e.data.replace_dict(repl_dict)
            for name in child_names:
                edge = parent_edges[name]
                for e in parent_state.memlet_tree(edge):
                    if e.data.data == parent_name:
                        e.data.subset = subsets.Range.from_array(parent_arr)
                    else:
                        e.data.other_subset = subsets.Range.from_array(parent_arr)

    if replacements:
        symbolic.safe_replace(replacements, lambda d: replace_datadesc_names(sdfg, d), value_as_string=True)
        parent_node.in_connectors = {
            replacements[c] if c in replacements else c: t
            for c, t in parent_node.in_connectors.items()
        }
        parent_node.out_connectors = {
            replacements[c] if c in replacements else c: t
            for c, t in parent_node.out_connectors.items()
        }
        for e in parent_state.all_edges(parent_node):
            if e.src_conn in replacements:
                e._src_conn = replacements[e.src_conn]
            elif e.dst_conn in replacements:
                e._dst_conn = replacements[e.dst_conn]

        # Remove multiple edges to the same connectors
        for name in replacements.values():
            in_edges = list(parent_state.in_edges_by_connector(parent_node, name))
            out_edges = list(parent_state.out_edges_by_connector(parent_node, name))
            if len(in_edges) > 1:
                for edge in in_edges[1:]:
                    parent_state.remove_memlet_path(edge)
            if len(out_edges) > 1:
                for edge in out_edges[1:]:
                    parent_state.remove_memlet_path(edge)

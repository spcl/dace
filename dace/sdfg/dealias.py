# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains functions for ensuring SDFGs and nested SDFGs share the same data descriptors.
"""
from dace import SDFG, data, subsets, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes as nd
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation.helpers import unsqueeze_memlet
from typing import Dict, List, Set, Tuple
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


def integrate_nested_sdfg(sdfg: SDFG):
    """
    Integrates a nested SDFG into its parent SDFG, ensuring that all data descriptors that are connected to
    the nested SDFG are shared with the parent SDFG. This function adds data containers to the nested
    SDFG based on the edges connected to it, which match the data descriptors of the parent SDFG exactly.
    It then changes the data descriptors that are not transient to become ``View`` data descriptors (i.e., with the same
    properties that they had before, such as shape, dtype, and strides) using the ``View.view`` static function,
    and connects the views to the newly-added data descriptors. That is, for every access node that uses the newly
    redefined views, add a new access node that uses the newly-added data descriptor, and connect the two.
    After this operation, the nested SDFG is valid in the context of the parent SDFG, and subsequent transformations
    may be applied to remove the resultant views, if possible (not the responsibility of this function).

    Precondition: The nested SDFG node must already be connected within the parent SDFG state.

    :param sdfg: The SDFG to operate on.
    :note: This function operates in-place.
    """
    if sdfg.parent is None:
        return

    parent_sdfg = sdfg.parent_sdfg
    parent_state = sdfg.parent
    parent_node = sdfg.parent_nsdfg_node

    # Track which data containers need to be added and converted to views
    to_add_and_view: Dict[str,
                          Tuple[str,
                                data.Data]] = {}  # Maps connector name -> (parent data name, parent data descriptor)
    parent_mapping: Dict[str, str] = {}  # Maps connector name to parent data name

    # Collect all edges connected to the nested SDFG node
    for edge in parent_state.all_edges(parent_node):
        if edge.data.data in parent_sdfg.arrays:
            connector = edge.dst_conn if edge.dst == parent_node else edge.src_conn
            if connector and connector in sdfg.arrays:
                # Only process non-transient arrays
                if not sdfg.arrays[connector].transient:
                    # If the parent data descriptor is equivalent to the inner data descriptor, simply copy it
                    if parent_sdfg.arrays[edge.data.data] == sdfg.arrays[connector]:
                        sdfg.arrays[connector] = copy.deepcopy(parent_sdfg.arrays[edge.data.data])
                        sdfg.arrays[connector].transient = False
                        continue
                    to_add_and_view[connector] = (edge.data.data, parent_sdfg.arrays[edge.data.data], edge.data)

    # Process each data container that needs to be integrated
    visited: Set[str] = set()
    symbols_to_add: Set[str] = set()
    for inner_name, (parent_name, parent_desc, parent_memlet) in to_add_and_view.items():
        if inner_name in visited:
            continue
        visited.add(inner_name)

        # If the parent data descriptor is a view, we need to convert it to a regular data descriptor
        # so that it can be used as a non-transient data descriptor in the nested SDFG.
        if isinstance(parent_desc, data.View):
            if isinstance(parent_desc, data.Structure):
                parent_desc = parent_desc.as_structure()
            else:
                parent_desc = parent_desc.as_array()
        else:
            parent_desc = copy.deepcopy(parent_desc)
        parent_desc.transient = False

        # Add the parent data descriptor to the nested SDFG
        new_parent_name = sdfg.add_datadesc(parent_name, parent_desc, find_new_name=True)
        parent_mapping[inner_name] = new_parent_name
        if new_parent_name != parent_name:
            new_memlet = copy.deepcopy(parent_memlet)
            new_memlet.data = new_parent_name
            to_add_and_view[inner_name] = (new_parent_name, parent_desc, new_memlet)

        # Get the original data descriptor
        original_desc = sdfg.arrays[inner_name]

        # Create a view of the parent data with the same properties as the original
        view_desc = data.View.view(original_desc)

        # If there is a shape mismatch, try to adjust the view descriptor
        # using ND array program squeeze semantics.
        if len(view_desc.shape) != len(parent_desc.shape):
            try:
                unsqueezed_dims = unsqueeze_memlet(Memlet.from_array(inner_name, view_desc),
                                                   parent_memlet,
                                                   return_dims=True)
                # Every dimension that was squeezed should be removed from the view shape
                view_desc.strides = [
                    parent_desc.strides[i] for i in range(len(parent_desc.shape)) if i not in unsqueezed_dims
                ]
            except (ValueError, NotImplementedError):
                print("WARNING")
                # If unsqueezing fails, we keep the original view descriptor
                pass

        # Replace the original descriptor with the view
        sdfg.arrays[inner_name] = view_desc

    # For each state, add access nodes and connections
    for state in sdfg.all_states():
        # Find relevant access nodes
        for view_node in state.data_nodes():
            if view_node.data not in to_add_and_view:
                continue

            parent_name, parent_desc, parent_memlet = to_add_and_view[view_node.data]

            # Collect existing edges
            in_edges = list(state.in_edges(view_node))
            out_edges = list(state.out_edges(view_node))

            # Skip if no edges (isolated node)
            if not in_edges and not out_edges:
                continue

            # Create a new access node for the parent data
            parent_access = state.add_access(parent_name)

            # Rewire the graph based on access pattern
            if in_edges and out_edges:
                # Both read and write: need two view nodes
                # Create a new view node for the write path
                view_node_write = state.add_access(view_node.data)

                # Rewire: predecessors -> view_write -> parent -> view_read -> successors
                # Move all incoming edges to the write view node
                for e in in_edges:
                    state.add_edge(e.src, e.src_conn, view_node_write, e.dst_conn, e.data)
                    state.remove_edge(e)

                # Connect view_write -> parent
                state.add_edge(view_node_write, 'views', parent_access, None, copy.deepcopy(parent_memlet))

                # Connect parent -> view_read (original view_node)
                state.add_edge(parent_access, None, view_node, 'views', copy.deepcopy(parent_memlet))

            elif out_edges:
                # Read only: parent -> view -> successors
                state.add_edge(parent_access, None, view_node, 'views', copy.deepcopy(parent_memlet))

            else:  # in_edges only
                # Write only: predecessors -> view -> parent
                state.add_edge(view_node, 'views', parent_access, None, copy.deepcopy(parent_memlet))

    # Modify connector names on the nested SDFG node to match the parent SDFG
    parent_node.in_connectors = {
        parent_mapping[c] if c in parent_mapping else c: t
        for c, t in parent_node.in_connectors.items()
    }
    parent_node.out_connectors = {
        parent_mapping[c] if c in parent_mapping else c: t
        for c, t in parent_node.out_connectors.items()
    }

    # Update edges to use the new parent data names
    for edge in parent_state.all_edges(parent_node):
        if edge.dst is parent_node:
            if edge.dst_conn in parent_mapping:
                edge.dst_conn = parent_mapping[edge.dst_conn]
        elif edge.src is parent_node:
            if edge.src_conn in parent_mapping:
                edge.src_conn = parent_mapping[edge.src_conn]

    # Add remaining symbols to symbol mapping using symbols_defined_at
    symtypes = parent_state.symbols_defined_at(parent_node)
    for sym_name, sym_type in symtypes.items():
        if sym_name not in sdfg.symbols:
            # Add the symbol to the SDFG and the parent node's symbol mapping
            sdfg.add_symbol(sym_name, sym_type)
        parent_node.symbol_mapping[sym_name] = sym_name

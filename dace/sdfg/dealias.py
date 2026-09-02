# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
This module contains functions for ensuring SDFGs and nested SDFGs share the same data descriptors.
"""
from dace import data, dtypes, subsets, symbolic, utils
from dace.memlet import Memlet
from dace.sdfg import nodes as nd, utils as sdutil
from dace.sdfg.sdfg import SDFG
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation.helpers import unsqueeze_memlet
from typing import Dict, List, Optional, Set, Tuple
import ast
import copy


def names_in_subtree(sdfg: SDFG) -> Set[str]:
    """
    Collects every name that means something in an SDFG or in the SDFGs nested within it.

    A name minted for a container has to avoid all of them. Scope parameters and the symbols that
    control flow defines share a namespace with the containers once code is generated, and under
    the nested SDFG contract a descendant's non-transient names are the parent's names, so a name
    that is free here but taken further down still collides.

    :param sdfg: The SDFG at the root of the subtree to inspect.
    :return: The set of names that are already spoken for.
    """
    names: Set[str] = set()
    for nsdfg in sdfg.all_sdfgs_recursive():
        names |= set(nsdfg.arrays.keys())
        names |= set(nsdfg.symbols.keys())
        names |= set(nsdfg.constants_prop.keys())
        for state in nsdfg.states():
            for node in state.nodes():
                names.update(map(str, node.new_symbols(nsdfg, state, {}).keys()))
        for edge in nsdfg.all_interstate_edges():
            names |= set(edge.data.assignments.keys())
        for region in nsdfg.all_control_flow_regions():
            names.update(map(str, region.new_symbols({}).keys()))
    return names


def fold_views_at_nested_sdfgs(sdfg: SDFG) -> None:
    """
    Rewires nested SDFG edges that reach their data through a view so they address the container.

    A nested SDFG connected to a view of a container is, through that view, connected to the
    container. Dealiasing exists to leave a single naming system behind, so the chain of views is
    walked to the container that backs it and the accessed subsets are composed along the way; the
    view nodes that nothing else uses afterwards are dropped. Without this the views survive as
    nodes of their own, and every consumer downstream has to know how to look through them.

    :param sdfg: The SDFG to operate on.
    :note: This function operates in-place, and only on edges adjacent to a nested SDFG node.
    """
    for state in sdfg.states():
        candidates = []
        for node in state.nodes():
            if not isinstance(node, nd.NestedSDFG):
                continue
            for edge in state.all_edges(node):
                far = edge.src if edge.dst is node else edge.dst
                if isinstance(far, nd.AccessNode) and isinstance(far.desc(sdfg), data.View):
                    candidates.append((edge, far, edge.dst is node))

        touched = set()
        for edge, view_node, into_nested in candidates:
            # Compose the whole of what the view covers, not the part this edge happens to
            # access. The edge memlet states the window the connector stands for; the memlets
            # inside are narrowed against it afterwards, and composing the access here as well
            # would count it twice.
            window = Memlet.from_array(view_node.data, view_node.desc(sdfg))
            composed, backing = _compose_through_views(state, view_node, window)
            if backing is None:
                continue
            if into_nested:
                state.add_edge(backing, edge.src_conn, edge.dst, edge.dst_conn, composed)
            else:
                state.add_edge(edge.src, edge.src_conn, backing, edge.dst_conn, composed)
            state.remove_edge(edge)
            touched.add(view_node)

        # A view kept alive only by the edge that was just rewired is now dead, along with the
        # ``views`` edge binding it.
        for view_node in touched:
            while (view_node is not None and view_node in state.nodes() and state.degree(view_node) == 1):
                binding = state.all_edges(view_node)[0]
                nxt = binding.src if binding.dst is view_node else binding.dst
                state.remove_node(view_node)
                view_node = nxt if (isinstance(nxt, nd.AccessNode) and isinstance(nxt.desc(sdfg), data.View)) else None


def _compose_through_views(state, node, memlet: Memlet):
    """
    Expresses a memlet addressing a view in the coordinates of the container backing the chain.

    :param state: The state the view nodes live in.
    :param node: The view access node the memlet is attached to.
    :param memlet: The memlet, in ``node``'s coordinates.
    :return: The composed memlet and the access node of the backing container, or
             ``(memlet, None)`` if the chain cannot be composed.
    """
    result = copy.deepcopy(memlet)
    backing = None
    seen = set()
    while (isinstance(node, nd.AccessNode) and isinstance(node.desc(state.sdfg), data.View) and id(node) not in seen):
        seen.add(id(node))
        view_edge = sdutil.get_view_edge(state, node)
        viewed = sdutil.get_view_node(state, node)
        if view_edge is None or viewed is None:
            break
        try:
            composed = unsqueeze_memlet(result, view_edge.data)
        except (ValueError, NotImplementedError):
            break
        composed.data = viewed.data
        result = composed
        backing = viewed
        node = viewed
    return (result, backing) if backing is not None else (memlet, None)


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
            # The names coinciding does not mean there is nothing to do: a connector can carry
            # the container's name while still describing a narrower window of it, in which case
            # the memlets inside are in the window's coordinates and still have to be unsqueezed.
            if name != parent_name or not parent_sdfg.arrays[parent_name].is_equivalent(sdfg.arrays[name]):
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
        # Symbols used by the parent's data descriptors may be defined by an enclosing scope (e.g., map
        # parameters or dynamic map inputs) rather than by the parent SDFG's symbol repository, so the
        # scope-aware set of defined symbols is the correct source of truth here.
        defined_symbols = parent_state.symbols_defined_at(parent_node)

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
                    symtype = defined_symbols.get(str(sym), parent_sdfg.symbols.get(str(sym), None))
                    sdfg.add_symbol(str(sym), symtype or dtypes.typeclass(int))
                    parent_node.symbol_mapping[str(sym)] = symbolic.pystr_to_symbolic(str(sym))

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
                        try:
                            new_src_memlet = unsqueeze_memlet(e.data, parent_edges[src.data].data, use_src_subset=True)
                        except (ValueError, NotImplementedError):
                            # The access has no expression in the parent's coordinates -- a reshape,
                            # say. Leaving it names the same data through the container it already
                            # refers to, which is sound if less direct.
                            src_data = None
                            new_src_memlet = None
                    else:
                        src_data = None
                        new_src_memlet = None
                        # We need to take directionality of the memlet into account
                    if isinstance(dst, nd.AccessNode) and dst.data in child_names:
                        dst_data = dst.data
                        try:
                            new_dst_memlet = unsqueeze_memlet(e.data, parent_edges[dst.data].data, use_dst_subset=True)
                        except (ValueError, NotImplementedError):
                            dst_data = None
                            new_dst_memlet = None
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
                            if str(sym) not in sdfg.symbols:
                                sdfg.add_symbol(str(sym), symbolic.DEFAULT_SYMBOL_TYPE)
                                parent_node.symbol_mapping[str(sym)] = symbolic.pystr_to_symbolic(str(sym))
                        e.data.src_subset = new_src_memlet.subset
                    if new_dst_memlet is not None:
                        new_syms = new_dst_memlet.used_symbols(all_symbols=True) - previous_syms
                        for sym in new_syms:
                            if str(sym) not in sdfg.symbols:
                                sdfg.add_symbol(str(sym), symbolic.DEFAULT_SYMBOL_TYPE)
                                parent_node.symbol_mapping[str(sym)] = symbolic.pystr_to_symbolic(str(sym))
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
            # A dotted connector names a member of a structure. The member's descriptor is owned by
            # the structure that declares it rather than by ``arrays``, so it cannot be replaced or
            # shadowed by a view of its own; the structure holding it is what gets integrated.
            if connector and '.' not in connector and connector in sdfg.arrays:
                # Only process non-transient arrays
                if not sdfg.arrays[connector].transient:
                    # If the parent data descriptor is equivalent to the inner data descriptor, simply copy it
                    if parent_sdfg.arrays[edge.data.data].is_equivalent(sdfg.arrays[connector]):
                        # ``offset`` names the origin of the index space the memlets inside are
                        # written in -- the Fortran frontend uses it to keep one-based indices --
                        # and equivalence deliberately does not compare it. Adopting the parent's
                        # would silently move every one of those memlets by the difference.
                        # Only arrays carry a settable one; a scalar or a structure has no index
                        # space of its own to preserve.
                        old_desc = sdfg.arrays[connector]
                        inner_offset = (copy.deepcopy(old_desc.offset) if isinstance(old_desc, data.Array) else None)
                        sdfg.arrays[connector] = copy.deepcopy(parent_sdfg.arrays[edge.data.data])
                        if inner_offset is not None and isinstance(sdfg.arrays[connector], data.Array):
                            sdfg.arrays[connector].offset = inner_offset

                        # Make non-reference descriptor
                        if isinstance(
                                sdfg.arrays[connector],
                            (data.ArrayView, data.ContainerView, data.ArrayReference, data.ContainerArrayReference)):
                            sdfg.arrays[connector] = sdfg.arrays[connector].as_array()
                        elif isinstance(sdfg.arrays[connector], (data.StructureView, data.StructureReference)):
                            sdfg.arrays[connector] = sdfg.arrays[connector].as_structure()

                        sdfg.arrays[connector].transient = False
                        continue
                    to_add_and_view[connector] = (edge.data.data, parent_sdfg.arrays[edge.data.data], edge.data)

    parent_names: Set[str] = set()  # The names of the parent containers being integrated
    parent_symbols: Set[str] = set()  # The symbols their descriptors and memlets are written in
    for parent_name, parent_desc, parent_memlet in to_add_and_view.values():
        # The parent data name itself may not alias internally-defined names
        parent_names.add(parent_name)
        for sym in parent_desc.used_symbols(all_symbols=True):
            parent_symbols.add(str(sym))
        for sym in parent_memlet.used_symbols(all_symbols=True):
            parent_symbols.add(str(sym))

    if parent_names or parent_symbols:
        # Rename internal names that would clash with the symbols introduced from the parent. Since the mapping is
        # the identity, free symbols used inside are considered the same symbol and are not renamed. A connector
        # about to be integrated keeps its name for the view that replaces it, so a fresh name only has to be
        # minted for the container behind it -- but a connector named like one of the symbols still shadows that
        # symbol, and is renamed like any other container that clashes with it.
        introduced = (parent_names - set(to_add_and_view.keys())) | parent_symbols
        renamed = remove_symbol_aliases(sdfg, {sym: sym for sym in introduced})

        # ``remove_symbol_aliases`` renames the containers inside the nested SDFG. A connector among
        # them is also the name of a connector on the node and of the edges reaching it, so those
        # have to follow it.
        conn_renames = {old: new for old, new in renamed.items() if old in to_add_and_view}
        if conn_renames:
            for old, new in conn_renames.items():
                to_add_and_view[new] = to_add_and_view.pop(old)
            parent_node.in_connectors = {conn_renames.get(c, c): t for c, t in parent_node.in_connectors.items()}
            parent_node.out_connectors = {conn_renames.get(c, c): t for c, t in parent_node.out_connectors.items()}
            for edge in parent_state.all_edges(parent_node):
                if edge.dst is parent_node and edge.dst_conn in conn_renames:
                    edge._dst_conn = conn_renames[edge.dst_conn]
                if edge.src is parent_node and edge.src_conn in conn_renames:
                    edge._src_conn = conn_renames[edge.src_conn]

    # Process each data container that needs to be integrated. The names already spoken for in the
    # subtree are collected once and extended as names are minted, rather than re-walking the tree
    # for every container.
    taken_names = names_in_subtree(sdfg)
    visited: Set[str] = set()
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

        # Add the parent data descriptor to the nested SDFG. ``find_new_name`` would only avoid
        # this SDFG's own containers, symbols and constants, so a fallback name could still land
        # on a map parameter or on a name a descendant already uses.
        new_parent_name = utils.find_new_name(parent_name, taken_names)
        new_parent_name = sdfg.add_datadesc(new_parent_name, parent_desc, find_new_name=True)
        taken_names.add(new_parent_name)
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
        if len(view_desc.shape) < len(parent_desc.shape):
            try:
                unsqueezed_dims = unsqueeze_memlet(Memlet.from_array(inner_name, view_desc),
                                                   parent_memlet,
                                                   return_dims=True)
                # Every dimension that was squeezed should be removed from the view shape. A
                # dimension the parent memlet walks with a step covers that many elements of the
                # parent per element of the view, so the step multiplies the stride -- without it
                # a view of ``A[..., 1:N-1:2]`` reads consecutive elements instead of every other.
                steps = [r[2] for r in parent_memlet.subset.ranges]
                view_desc.strides = [
                    parent_desc.strides[i] * steps[i] for i in range(len(parent_desc.shape)) if i not in unsqueezed_dims
                ]
            except (ValueError, NotImplementedError):
                # If unsqueezing fails, we keep the original view descriptor
                pass
        elif len(view_desc.shape) > len(parent_desc.shape):
            # View has more dimensions than parent, let passes try to eliminate the view
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
        # Skip parent symbols that are shadowed by unrelated internal data containers or constants
        if sym_name in sdfg.arrays or sym_name in sdfg.constants_prop:
            continue
        if sym_name not in sdfg.symbols:
            # Add the symbol to the SDFG and the parent node's symbol mapping
            sdfg.add_symbol(sym_name, sym_type)
        parent_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)

    # Containers read only by meta code never receive a ``views`` edge above, so redirect those
    # accesses to the parent container they alias.
    redirect_meta_accesses(sdfg, to_add_and_view)


def redirect_meta_accesses(sdfg: SDFG, integrated: Dict[str, Tuple[str, data.Data, Memlet]]) -> Set[str]:
    """
    Rewrites meta accesses of freshly integrated containers so that they address the parent container.

    Integration expresses the narrowing of a connector as a ``View``, and a view only takes effect
    through the ``views`` edge of an access node. A container that is read solely by meta code -- a
    branch condition, a loop condition, an interstate-edge condition or assignment -- never gets such
    an edge, so code generation has no pointer to emit for it and the generated program does not even
    compile. Those accesses are therefore rewritten to address the parent container directly, with
    the accessed subset composed into the parent's.

    :param sdfg: The nested SDFG being integrated.
    :param integrated: The containers that were integrated, mapping the inner name to the parent
                       name, the parent descriptor and the memlet connecting the two.
    :return: The names of the inner containers whose meta accesses were rewritten.
    :note: This function operates in-place.
    """
    # Avoid import loops
    from dace.frontend.python import astutils
    from dace.sdfg.memlet_utils import MemletReplacer

    if not integrated:
        return set()

    rewritten: Set[str] = set()

    def process(memlet: Memlet) -> Optional[Memlet]:
        entry = integrated.get(memlet.data)
        if entry is None:
            return None
        parent_name, _, parent_memlet = entry
        try:
            new_memlet = unsqueeze_memlet(memlet, parent_memlet)
        except (ValueError, NotImplementedError):
            # The access cannot be expressed in the parent's coordinate system; leave it alone so
            # that the resulting SDFG fails loudly rather than silently addressing the wrong data.
            return None
        new_memlet.data = parent_name
        rewritten.add(memlet.data)
        return new_memlet

    replacer = MemletReplacer(sdfg.arrays, process, set(integrated.keys()))

    for edge in sdfg.all_interstate_edges():
        if not edge.data.is_unconditional():
            for stmt in edge.data.condition.code:
                replacer.visit(stmt)
        for name, assignment in list(edge.data.assignments.items()):
            edge.data.assignments[name] = astutils.unparse(replacer.visit(ast.parse(assignment)))

    for region in sdfg.all_control_flow_regions():
        for code in region.get_meta_codeblocks():
            if code.code is None or isinstance(code.code, str):
                continue
            for stmt in code.code:
                replacer.visit(stmt)

    # Views that only existed for the sake of a meta access are now unused
    for name in rewritten:
        if any(node.data == name for state in sdfg.all_states() for node in state.data_nodes()):
            continue
        try:
            sdfg.remove_data(name, validate=True)
        except ValueError:
            # Still referenced somewhere; keeping it is harmless
            pass

    return rewritten


def remove_symbol_aliases(sdfg: SDFG, symbol_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Ensures that symbols introduced into the SDFG through the values of ``symbol_mapping`` will not
    alias unrelated names inside the SDFG. Only names that genuinely clash are renamed:

      * Symbols that are (re)defined inside the SDFG (e.g., map parameters, interstate edge
        assignments, loop variables) and match an introduced symbol.
      * Data container or constant names that match an introduced symbol.
      * Symbols used inside the SDFG that match an introduced symbol but are not keys of the
        mapping. Keys are either identity-mapped (i.e., the same symbol as the parent's) or
        replaced separately by the caller, so they do not alias.

    :param sdfg: The SDFG to operate on.
    :param symbol_mapping: A dictionary mapping SDFG symbols to symbolic expressions in the parent scope.
    :return: A dictionary mapping original names to their new names, if any renaming was necessary.
    """
    if not symbol_mapping:
        return {}
    # The following symbols will be introduced into the SDFG and are at risk of aliasing internal names.
    target_symbols = {
        str(s)
        for s in set().union(*(symbolic.free_symbols_and_functions(v) for v in symbol_mapping.values()))
    }
    if not target_symbols:
        return {}

    # Names that are (re)defined inside the SDFG always clash with introduced symbols
    defined_symbols: Set[str] = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            defined_symbols.update(map(str, node.new_symbols(sdfg, state, {}).keys()))
    for edge in sdfg.all_interstate_edges():
        defined_symbols.update(edge.data.assignments.keys())
    for region in sdfg.all_control_flow_regions():
        defined_symbols.update(map(str, region.new_symbols({}).keys()))

    used_symbols = set(map(str, sdfg.used_symbols(all_symbols=True)))
    # Free symbols of this SDFG are resolved through the parent node's symbol mapping, and any of them
    # that the mapping does not list is implicitly identity-mapped (see ``SDFGState.add_nested_sdfg``).
    # Such a symbol therefore *is* the parent's symbol of the same name and must never be renamed away.
    free_symbols = set(map(str, sdfg.free_symbols))

    clashing: Set[str] = set()
    for sym in target_symbols:
        if sym in defined_symbols or sym in sdfg.arrays or sym in sdfg.constants_prop:
            clashing.add(sym)
        elif sym in used_symbols and sym not in symbol_mapping and sym not in free_symbols:
            clashing.add(sym)

    if not clashing:
        return {}

    # A rename target must also avoid every name that is spoken for further down the tree: a
    # grandchild's map parameter is as much of a clash as one in this SDFG.
    taken_names = (used_symbols | defined_symbols | target_symbols | set(sdfg.arrays.keys())
                   | set(sdfg.constants_prop.keys()) | set(symbol_mapping.keys()) | names_in_subtree(sdfg))
    repl_dict: Dict[str, str] = {}
    for sym in clashing:
        new_name = data.find_new_name(sym, taken_names)
        repl_dict[sym] = new_name
        taken_names.add(new_name)

    sdfg.replace_dict(repl_dict)
    symbolic.safe_replace(repl_dict, lambda d: _replace_dict_keys(symbol_mapping, d))
    if sdfg.parent_nsdfg_node is not None:
        symbolic.safe_replace(repl_dict, lambda d: _replace_dict_keys(sdfg.parent_nsdfg_node.symbol_mapping, d))
    return repl_dict


def _replace_dict_keys(target_dict: Dict[str, symbolic.SymbolicType], d: Dict[str, str]):
    """
    Helper function to replace keys in a dictionary with other keys.

    :param d: The dictionary to replace keys in.
    :return: A new dictionary with keys replaced by the new keys in the input dictionary.
    """
    tmp: Dict[str, symbolic.SymbolicType] = copy.copy(target_dict)
    target_dict.clear()
    for key, value in tmp.items():
        new_key = d.get(key, key)  # Replace key if it exists in d, otherwise keep original
        target_dict[str(new_key)] = value

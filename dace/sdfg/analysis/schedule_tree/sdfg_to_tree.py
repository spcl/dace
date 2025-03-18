# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import copy
from typing import Dict, List, Optional, Set
import dace
from dace import data, subsets, symbolic
from dace.codegen import control_flow as cf
from dace.sdfg.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import ControlFlowBlock, SDFGState
from dace.sdfg import utils as sdutil, graph as gr, nodes as nd
from dace.sdfg.replace import replace_datadesc_names
from dace.frontend.python.astutils import negate_expr
from dace.sdfg.analysis.schedule_tree import treenodes as tn, passes as stpasses
from dace.transformation.passes.analysis import StateReachability
from dace.transformation.helpers import unsqueeze_memlet
from dace.properties import CodeBlock
from dace.memlet import Memlet

import networkx as nx
import time
import sys

NODE_TO_SCOPE_TYPE = {
    dace.nodes.MapEntry: tn.MapScope,
    dace.nodes.ConsumeEntry: tn.ConsumeScope,
    dace.nodes.PipelineEntry: tn.PipelineScope,
}


def dealias_sdfg(sdfg: SDFG):
    """
    Renames all data containers in an SDFG tree (i.e., nested SDFGs) to use the same data descriptors
    as the top-level SDFG. This function takes care of offsetting memlets and internal
    uses of arrays such that there is one naming system, and no aliasing of managed memory.

    This function operates in-place.

    :param sdfg: The SDFG to operate on.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():

        if not nsdfg.parent:
            continue

        replacements: Dict[str, str] = {}
        inv_replacements: Dict[str, List[str]] = {}
        parent_edges: Dict[str, Memlet] = {}
        to_unsqueeze: Set[str] = set()

        parent_sdfg = nsdfg.parent_sdfg
        parent_state = nsdfg.parent
        parent_node = nsdfg.parent_nsdfg_node

        for name, desc in nsdfg.arrays.items():
            if desc.transient:
                continue
            for edge in parent_state.edges_by_connector(parent_node, name):
                parent_name = edge.data.data
                if parent_name == name or parent_name in nsdfg.arrays:
                    continue
                assert parent_name in parent_sdfg.arrays
                replacements[name] = parent_name
                parent_edges[name] = edge
                if parent_name in inv_replacements:
                    inv_replacements[parent_name].append(name)
                    to_unsqueeze.add(parent_name)
                else:
                    inv_replacements[parent_name] = [name]
                break

        if to_unsqueeze:
            for parent_name in to_unsqueeze:
                parent_arr = parent_sdfg.arrays[parent_name]
                if isinstance(parent_arr, data.View):
                    parent_arr = parent_arr.as_array()
                elif isinstance(parent_arr, data.StructureView):
                    parent_arr = parent_arr.as_structure()
                elif isinstance(parent_arr, data.ContainerView):
                    parent_arr = copy.deepcopy(parent_arr.stype)
                child_names = inv_replacements[parent_name]
                for name in child_names:
                    child_arr = copy.deepcopy(parent_arr)
                    child_arr.transient = False
                    nsdfg.arrays[name] = child_arr
                for state in nsdfg.states():
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

                        if new_src_memlet is not None:
                            e.data.src_subset = new_src_memlet.subset
                        if new_dst_memlet is not None:
                            e.data.dst_subset = new_dst_memlet.subset
                        if e.data.data == src_data:
                            e.data.data = new_src_memlet.data
                        elif e.data.data == dst_data:
                            e.data.data = new_dst_memlet.data

                for e in nsdfg.all_interstate_edges():
                    repl_dict = dict()
                    syms = e.data.read_symbols()
                    for memlet in e.data.get_read_memlets(nsdfg.arrays):
                        if memlet.data in child_names:
                            repl_dict[str(memlet)] = unsqueeze_memlet(memlet, parent_edges[memlet.data].data)
                            if memlet.data in syms:
                                syms.remove(memlet.data)
                    for s in syms:
                        if s in parent_edges:
                            if s in nsdfg.arrays:
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
            struct_outside_replacements: Dict[str, str] = {}
            cleaned_replacements = {}
            for k, val in replacements.items():
                if '.' in val:
                    if '.' not in k:
                        struct_outside_replacements[k] = val
                    continue
                else:
                    cleaned_replacements[k] = val
            replacements = cleaned_replacements

            symbolic.safe_replace(replacements, lambda d: replace_datadesc_names(nsdfg, d), value_as_string=True)
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

            '''
            if struct_outside_replacements:
                p_scope = parent_state.scope_dict()[parent_node]
                in_map = not isinstance(p_scope, SDFGState)
                for k, val in struct_outside_replacements.items():
                    needs_input = k in parent_node.in_connectors
                    needs_output = k in parent_node.out_connectors
                    root_container = val.split('.')[0]
                    if (root_container not in nsdfg.arrays or
                        (needs_input and root_container not in parent_node.in_connectors) or
                        (needs_output and root_container not in parent_node.out_connectors)):
                        nsdfg.add_datadesc(root_container, parent_sdfg.arrays[root_container])
                        if needs_input:
                            parent_node.add_in_connector(root_container, parent_sdfg.arrays[root_container].dtype,
                                                         force=True)
                            read = parent_state.add_access(root_container)
                            if in_map:
                                parent_state.add_memlet_path(read, p_scope, parent_node,
                                                             memlet=Memlet(expr=root_container + '[0]'),
                                                             dst_conn=root_container,
                                                             propagate=False)
                            else:
                                parent_state.add_memlet_path(read, parent_node,
                                                             memlet=Memlet(expr=root_container + '[0]'),
                                                             dst_conn=root_container,
                                                             propagate=False)
                        if needs_output:
                            parent_node.add_out_connector(root_container, parent_sdfg.arrays[root_container].dtype,
                                                          force=True)
                            write = parent_state.add_access(root_container)
                            if in_map:
                                parent_state.add_memlet_path(parent_node, p_scope, write,
                                                             memlet=Memlet(expr=root_container + '[0]'),
                                                             src_conn=root_container,
                                                             propagate=False)
                            else:
                                parent_state.add_memlet_path(parent_node, write,
                                                             memlet=Memlet(expr=root_container + '[0]'),
                                                             src_conn=root_container,
                                                             propagate=False)
                    else:
                        continue
                replacements = struct_outside_replacements
                symbolic.safe_replace(replacements, lambda d: replace_datadesc_names(nsdfg, d, True),
                                      value_as_string=True)
                for k in struct_outside_replacements.keys():
                    del nsdfg.arrays[k]
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
                        '''


def normalize_memlet(sdfg: SDFG, state: SDFGState, original: gr.MultiConnectorEdge[Memlet], data: str) -> Memlet:
    """
    Normalizes a memlet to a given data descriptor.

    :param sdfg: The SDFG.
    :param state: The state.
    :param original: The original memlet.
    :param data: The data descriptor.
    :return: A new memlet.
    """
    # Shallow copy edge
    edge = gr.MultiConnectorEdge(original.src, original.src_conn, original.dst, original.dst_conn,
                                 copy.deepcopy(original.data), original.key)
    edge.data.try_initialize(sdfg, state, edge)

    if '.' in edge.data.data and edge.data.data.startswith(data + '.'):
        return edge.data
    if edge.data.data == data:
        return edge.data

    memlet = edge.data
    if memlet._is_data_src:
        new_subset, new_osubset = memlet.get_dst_subset(edge, state), memlet.get_src_subset(edge, state)
    else:
        new_subset, new_osubset = memlet.get_src_subset(edge, state), memlet.get_dst_subset(edge, state)

    memlet.data = data
    memlet.subset = new_subset
    memlet.other_subset = new_osubset
    memlet._is_data_src = True
    return memlet


def replace_memlets(sdfg: SDFG, input_mapping: Dict[str, Memlet], output_mapping: Dict[str, Memlet]):
    """
    Replaces all uses of data containers in memlets and interstate edges in an SDFG.
    :param sdfg: The SDFG.
    :param input_mapping: A mapping from internal data descriptor names to external input memlets.
    :param output_mapping: A mapping from internal data descriptor names to external output memlets.
    """
    for state in sdfg.states():
        for e in state.edges():
            mpath = state.memlet_path(e)
            src = mpath[0].src
            dst = mpath[-1].dst
            memlet = e.data
            if isinstance(src, dace.nodes.AccessNode) and src.data in input_mapping:
                src_data = src.data
                src_memlet = unsqueeze_memlet(memlet, input_mapping[src.data], use_src_subset=True)
            else:
                src_data = None
                src_memlet = None
            if isinstance(dst, dace.nodes.AccessNode) and dst.data in output_mapping:
                dst_data = dst.data
                dst_memlet = unsqueeze_memlet(memlet, output_mapping[dst.data], use_dst_subset=True)
            else:
                dst_data = None
                dst_memlet = None

            # Other cases (code->code)
            if src_data is None and dst_data is None:
                if e.data.data in input_mapping:
                    memlet = unsqueeze_memlet(memlet, input_mapping[e.data.data])
                elif e.data.data in output_mapping:
                    memlet = unsqueeze_memlet(memlet, output_mapping[e.data.data])
                e.data = memlet
            else:
                if src_memlet is not None:
                    memlet.src_subset = src_memlet.subset
                if dst_memlet is not None:
                    memlet.dst_subset = dst_memlet.subset
                if memlet.data == src_data:
                    memlet.data = src_memlet.data
                elif memlet.data == dst_data:
                    memlet.data = dst_memlet.data

    for e in sdfg.edges():
        repl_dict = dict()
        syms = e.data.read_symbols()
        for memlet in e.data.get_read_memlets(sdfg.arrays):
            if memlet.data in input_mapping or memlet.data in output_mapping:
                # If array name is both in the input connectors and output connectors with different
                # memlets, this is undefined behavior. Prefer output
                if memlet.data in input_mapping:
                    mapping = input_mapping
                if memlet.data in output_mapping:
                    mapping = output_mapping

                repl_dict[str(memlet)] = str(unsqueeze_memlet(memlet, mapping[memlet.data]))
                if memlet.data in syms:
                    syms.remove(memlet.data)
        for s in syms:
            if s in input_mapping:
                if s in sdfg.arrays:
                    repl_dict[s] = input_mapping[s].data
                else:
                    repl_dict[s] = str(input_mapping[s])

        # Manual replacement with strings
        # TODO(later): Would be MUCH better to use MemletReplacer / e.data.replace_dict(repl_dict, replace_keys=False)
        for find, replace in repl_dict.items():
            for k, v in e.data.assignments.items():
                if find in v:
                    e.data.assignments[k] = v.replace(find, replace)
            condstr = e.data.condition.as_string
            if find in condstr:
                e.data.condition.as_string = condstr.replace(find, replace)


def remove_name_collisions(sdfg: SDFG):
    """
    Removes name collisions in nested SDFGs by renaming states, data containers, and symbols.

    :param sdfg: The SDFG.
    """
    state_names_seen = set()
    identifiers_seen = set()

    for nsdfg in sdfg.all_sdfgs_recursive():
        # Rename duplicate states
        for state in nsdfg.states():
            if state.label in state_names_seen:
                state.label = data.find_new_name(state.label, state_names_seen)
            state_names_seen.add(state.label)

        replacements: Dict[str, str] = {}
        parent_node = nsdfg.parent_nsdfg_node

        # Preserve top-level SDFG names
        do_not_replace = False
        if not parent_node:
            do_not_replace = True

        # Rename duplicate data containers
        for name, desc in nsdfg.arrays.items():
            if name in identifiers_seen:
                if not desc.transient or do_not_replace:
                    continue

                new_name = data.find_new_name(name, identifiers_seen)
                replacements[name] = new_name
                name = new_name
            identifiers_seen.add(name)

        # Rename duplicate top-level symbols
        for name in nsdfg.get_all_toplevel_symbols():
            # Will already be renamed during conversion
            if parent_node is not None and name in parent_node.symbol_mapping:
                continue

            if name in identifiers_seen and not do_not_replace:
                new_name = data.find_new_name(name, identifiers_seen)
                replacements[name] = new_name
                name = new_name
            identifiers_seen.add(name)

        # Rename duplicate constants
        for name in nsdfg.constants_prop.keys():
            if name in identifiers_seen and not do_not_replace:
                new_name = data.find_new_name(name, identifiers_seen)
                replacements[name] = new_name
                name = new_name
            identifiers_seen.add(name)

        # If there is a name collision, replace all uses of the old names with the new names
        if replacements:
            nsdfg.replace_dict(replacements)


def _make_view_node(state: SDFGState, edge: gr.MultiConnectorEdge[Memlet], view_name: str,
                    viewed_name: str) -> tn.ViewNode:
    """
    Helper function to create a view schedule tree node from a memlet edge.
    """
    sdfg = state.parent
    normalized = normalize_memlet(sdfg, state, edge, viewed_name)
    view_node = tn.ViewNode(target=view_name,
                            source=viewed_name,
                            memlet=normalized,
                            src_desc=sdfg.arrays[viewed_name],
                            view_desc=sdfg.arrays[view_name])
    view_node.sdfg = state.sdfg
    return view_node


def replace_symbols_until_set(nsdfg: dace.nodes.NestedSDFG):
    """
    Replaces symbol values in a nested SDFG until their value has been reset. This is used for matching symbol
    namespaces between an SDFG and a nested SDFG.
    """
    mapping = nsdfg.symbol_mapping
    sdfg = nsdfg.sdfg
    reachable_states = StateReachability().apply_pass(sdfg, {})[sdfg.cfg_id]
    redefined_symbols: Dict[SDFGState, Set[str]] = defaultdict(set)

    # Collect redefined symbols
    for e in sdfg.edges():
        redefined = e.data.assignments.keys()
        redefined_symbols[e.dst] |= redefined
        for reachable in reachable_states[e.dst]:
            redefined_symbols[reachable] |= redefined

    # Replace everything but the redefined symbols
    for state in sdfg.nodes():
        per_state_mapping = {k: v for k, v in mapping.items() if k not in redefined_symbols[state]}
        symbolic.safe_replace(per_state_mapping, state.replace_dict)
        for e in sdfg.out_edges(state):
            symbolic.safe_replace(per_state_mapping, lambda d: e.data.replace_dict(d, replace_keys=False))


def prepare_schedule_tree_edges(state: SDFGState) -> Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode]:
    """
    Creates a dictionary mapping edges to their corresponding schedule tree nodes, if relevant.
    This handles view edges, reference sets, and dynamic map inputs.

    :param state: The state.
    """
    result: Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode] = {}
    scope_to_edges: Dict[nd.EntryNode, List[gr.MultiConnectorEdge[Memlet]]] = defaultdict(list)
    edges_to_ignore = set()
    sdfg = state.parent

    for edge in state.edges():
        if edge in edges_to_ignore or edge in result:
            continue
        if edge.data.is_empty():  # Ignore empty memlets
            edges_to_ignore.add(edge)
            continue

        # Part of a memlet path - only consider innermost memlets
        mtree = state.memlet_tree(edge)
        all_edges = set(e for e in mtree)
        leaves = set(mtree.leaves())
        edges_to_ignore.update(all_edges - leaves)

        # For every tree leaf, create a copy/view/reference set node as necessary
        for e in leaves:
            if e in edges_to_ignore or e in result:
                continue

            # 1. Check for views
            if isinstance(e.src, dace.nodes.AccessNode):
                desc = e.src.desc(sdfg)
                if isinstance(desc, dace.data.View):
                    vedge = sdutil.get_view_edge(state, e.src)
                    if e is vedge:
                        viewed_node = sdutil.get_view_node(state, e.src)
                        result[e] = _make_view_node(state, e, e.src.data, viewed_node.data)
                        scope = state.entry_node(e.dst if mtree.downwards else e.src)
                        scope_to_edges[scope].append(e)
                        continue
            if isinstance(e.dst, dace.nodes.AccessNode):
                desc = e.dst.desc(sdfg)
                if isinstance(desc, dace.data.View):
                    vedge = sdutil.get_view_edge(state, e.dst)
                    if e is vedge:
                        viewed_node = sdutil.get_view_node(state, e.dst)
                        result[e] = _make_view_node(state, e, e.dst.data, viewed_node.data)
                        scope = state.entry_node(e.dst if mtree.downwards else e.src)
                        scope_to_edges[scope].append(e)
                        continue

            # 2. Check for reference sets
            if isinstance(e.dst, dace.nodes.AccessNode) and e.dst_conn == 'set':
                assert isinstance(e.dst.desc(sdfg), dace.data.Reference)

                # Determine source
                if isinstance(mtree.root().edge.src, dace.nodes.CodeNode):
                    src_desc = mtree.root().edge.src
                else:
                    src_desc = sdfg.arrays[e.data.data]
                result[e] = tn.RefSetNode(target=e.dst.data,
                                          memlet=e.data,
                                          src_desc=src_desc,
                                          ref_desc=sdfg.arrays[e.dst.data])
                result[e].sdfg = state.sdfg
                scope = state.entry_node(e.dst if mtree.downwards else e.src)
                scope_to_edges[scope].append(e)
                continue

            # 3. Check for copies
            # Get both ends of the memlet path
            mpath = state.memlet_path(e)
            src = mpath[0].src
            dst = mpath[-1].dst
            if not isinstance(src, dace.nodes.AccessNode):
                continue
            if not isinstance(dst, (dace.nodes.AccessNode, dace.nodes.EntryNode)):
                continue

            # If the edge destination is the innermost node, it is a downward-pointing path
            is_target_dst = e.dst is dst

            innermost_node = dst if is_target_dst else src
            outermost_node = src if is_target_dst else dst

            # Normalize memlets to their innermost node, or source->destination if it is a same-scope edge
            if e.src is src and e.dst is dst:
                outermost_node = src
                innermost_node = dst

            if isinstance(dst, dace.nodes.EntryNode):
                # Special case: dynamic map range has no data
                result[e] = tn.DynScopeCopyNode(target=e.dst_conn, memlet=e.data)
            else:
                target_name = innermost_node.data
                new_memlet = normalize_memlet(sdfg, state, e, outermost_node.data)
                result[e] = tn.CopyNode(target=target_name, memlet=new_memlet)
            result[e].sdfg = state.sdfg

            scope = state.entry_node(e.dst if mtree.downwards else e.src)
            scope_to_edges[scope].append(e)

    return result, scope_to_edges


def state_schedule_tree(state: SDFGState) -> List[tn.ScheduleTreeNode]:
    """
    Use scope-aware topological sort to get nodes by scope and return the schedule tree of this state.

    :param state: The state.
    :return: A string for the whole state
    """
    result: List[tn.ScheduleTreeNode] = []
    sdfg = state.parent

    edge_to_stree: Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode]
    scope_to_edges: Dict[nd.EntryNode, List[gr.MultiConnectorEdge[Memlet]]]
    edge_to_stree, scope_to_edges = prepare_schedule_tree_edges(state)
    edges_to_ignore = set()

    # Handle all unscoped edges to generate output views
    views = _generate_views_in_scope(scope_to_edges[None], edge_to_stree, sdfg, state)
    result.extend(views)

    scopes: List[List[tn.ScheduleTreeNode]] = []
    for node in sdutil.scope_aware_topological_sort(state):
        if isinstance(node, dace.nodes.EntryNode):
            # Handle dynamic scope inputs
            for e in state.in_edges(node):
                if e in edges_to_ignore:
                    continue
                if e in edge_to_stree:
                    result.append(edge_to_stree[e])
                    edges_to_ignore.add(e)

            # Handle all scoped edges to generate (views)
            views = _generate_views_in_scope(scope_to_edges[node], edge_to_stree, sdfg, state)
            result.extend(views)

            # Create scope node and add to stack
            scopes.append(result)
            subnodes = []
            scope_node: tn.DataflowScope = NODE_TO_SCOPE_TYPE[type(node)](state=state, node=node, children=subnodes)
            scope_node.sdfg = state.sdfg
            result.append(scope_node)
            result = subnodes
        elif isinstance(node, dace.nodes.ExitNode):
            result = scopes.pop()
        elif isinstance(node, dace.nodes.NestedSDFG):
            nested_array_mapping_input = {}
            nested_array_mapping_output = {}
            generated_nviews = set()

            # Replace symbols and memlets in nested SDFGs to match the namespace of the parent SDFG
            replace_symbols_until_set(node)

            # Create memlets for nested SDFG mapping, or nview schedule nodes if slice cannot be determined
            for e in state.all_edges(node):
                conn = e.dst_conn if e.dst is node else e.src_conn
                if e.data.is_empty() or not conn:
                    continue
                res = sdutil.map_view_to_array(node.sdfg.arrays[conn], sdfg.arrays[e.data.data], e.data.subset)
                no_mapping = False
                if res is None:
                    no_mapping = True
                else:
                    mapping, expanded, squeezed = res
                    if expanded:  # "newaxis" slices will be seen as views (for now)
                        no_mapping = True
                    else:
                        if e.dst is node:
                            nested_array_mapping_input[conn] = e.data
                        else:
                            nested_array_mapping_output[conn] = e.data

                if no_mapping:  # Must use view (nview = nested SDFG view)
                    if conn not in generated_nviews:
                        nview_node = tn.NView(target=conn,
                                              source=e.data.data,
                                              memlet=e.data,
                                              src_desc=sdfg.arrays[e.data.data],
                                              view_desc=node.sdfg.arrays[conn])
                        nview_node.sdfg = node.sdfg
                        result.append(nview_node)
                        generated_nviews.add(conn)

            replace_memlets(node.sdfg, nested_array_mapping_input, nested_array_mapping_output)

            # Insert the nested SDFG flattened
            nested_stree = as_schedule_tree(node.sdfg, in_place=True, toplevel=False)
            result.extend(nested_stree.children)
        elif isinstance(node, dace.nodes.Tasklet):
            in_memlets = {e.dst_conn: e.data for e in state.in_edges(node) if e.dst_conn}
            out_memlets = {e.src_conn: e.data for e in state.out_edges(node) if e.src_conn}
            tasklet_node = tn.TaskletNode(node=node, in_memlets=in_memlets, out_memlets=out_memlets)
            tasklet_node.sdfg = state.sdfg
            result.append(tasklet_node)
        elif isinstance(node, dace.nodes.LibraryNode):
            # NOTE: LibraryNodes do not necessarily have connectors
            if node.in_connectors:
                in_memlets = {e.dst_conn: e.data for e in state.in_edges(node) if e.dst_conn}
            else:
                in_memlets = set([e.data for e in state.in_edges(node)])
            if node.out_connectors:
                out_memlets = {e.src_conn: e.data for e in state.out_edges(node) if e.src_conn}
            else:
                out_memlets = set([e.data for e in state.out_edges(node)])
            libnode = tn.LibraryCall(node=node, in_memlets=in_memlets, out_memlets=out_memlets)
            libnode.sdfg = state.sdfg
            result.append(libnode)
        elif isinstance(node, dace.nodes.AccessNode):
            # If one of the neighboring edges has a schedule tree node attached to it, use that
            # (except for views, which were generated above)
            for e in state.all_edges(node):
                if e in edges_to_ignore:
                    continue
                if e in edge_to_stree:
                    if isinstance(edge_to_stree[e], tn.ViewNode):
                        continue
                    result.append(edge_to_stree[e])
                    edges_to_ignore.add(e)

    assert len(scopes) == 0

    return result


def _generate_views_in_scope(edges: List[gr.MultiConnectorEdge[Memlet]],
                             edge_to_stree: Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode], sdfg: SDFG,
                             state: SDFGState) -> List[tn.ScheduleTreeNode]:
    """
    Generates all view and reference set edges in the correct order. This function is intended to be used
    at the beginning of a scope.
    """
    result: List[tn.ScheduleTreeNode] = []

    # Make a dependency graph of all the views
    g = nx.DiGraph()
    node_to_stree = {}
    for e in edges:
        if e not in edge_to_stree:
            continue
        st = edge_to_stree[e]
        if not isinstance(st, tn.ViewNode):
            continue
        g.add_edge(st.source, st.target)
        node_to_stree[st.target] = st

    # Traverse in order and deduplicate
    already_generated = set()
    for n in nx.topological_sort(g):
        if n in node_to_stree and n not in already_generated:
            result.append(node_to_stree[n])
            already_generated.add(n)

    return result


def as_schedule_tree(sdfg: SDFG, in_place: bool = False, toplevel: bool = True) -> tn.ScheduleTreeScope:
    """
    Converts an SDFG into a schedule tree. The schedule tree is a tree of nodes that represent the execution order of
    the SDFG.
    Each node in the tree can either represent a single statement (symbol assignment, tasklet, copy, library node, etc.)
    or a ``ScheduleTreeScope`` block (map, for-loop, pipeline, etc.) that contains other nodes.

    It can be used to generate code from an SDFG, or to perform schedule transformations on the SDFG. For example,
    erasing an empty if branch, or merging two consecutive for-loops. The SDFG can then be reconstructed via the
    ``from_schedule_tree`` function.

    :param sdfg: The SDFG to convert.
    :param in_place: If True, the SDFG is modified in-place. Otherwise, a copy is made. Note that the SDFG might not be
                     usable after the conversion if ``in_place`` is True!
    :return: A schedule tree representing the given SDFG.
    """
    from dace.transformation import helpers as xfh  # Avoid import loop

    if not in_place:
        sdfg = copy.deepcopy(sdfg)

    # Prepare SDFG for conversion
    #############################

    # Split edges with assignments and conditions
    xfh.split_interstate_edges(sdfg)

    # Replace code->code edges with data<->code edges
    xfh.replace_code_to_code_edges(sdfg)

    if toplevel:  # Top-level SDFG preparation (only perform once)
        dealias_sdfg(sdfg)
        # Handle name collisions (in arrays, state labels, symbols)
        remove_name_collisions(sdfg)

    #############################

    # Create initial tree from CFG
    cfg: cf.ControlFlow = cf.structured_control_flow_tree_with_regions(sdfg, lambda _: '')

    def transition_totree(node: cf.ControlFlow, parent: cf.GeneralBlock,
                          block: ControlFlowBlock) -> List[tn.ScheduleTreeNode]:
        result: List[tn.ScheduleTreeNode] = []
        for e in block.parent_graph.out_edges(block):
            edge_body = []

            if e not in parent.assignments_to_ignore:
                for aname, aval in e.data.assignments.items():
                    assign_node = tn.AssignNode(name=aname,
                                                value=CodeBlock(aval),
                                                edge=InterstateEdge(assignments={aname: aval}))
                    assign_node.sdfg = block.sdfg
                    edge_body.append(assign_node)

            if not parent.sequential:
                if e not in parent.gotos_to_ignore:
                    goto_node = tn.GotoNode(target=e.dst.label)
                    goto_node.sdfg = block.sdfg
                    edge_body.append(goto_node)
                else:
                    if e in parent.gotos_to_break:
                        break_node = tn.BreakNode()
                        break_node.sdfg = block.sdfg
                        edge_body.append(break_node)
                    elif e in parent.gotos_to_continue:
                        continue_node = tn.ContinueNode()
                        continue_node.sdfg = block.sdfg
                        edge_body.append(continue_node)
            else:
                # If the next state is not the expected target (loop-back edge, next state),
                # emit goto
                expected_transition = False
                if isinstance(parent, (cf.ForScope, cf.WhileScope)) and e.dst is parent.guard:
                    expected_transition = True
                elif isinstance(parent, cf.DoWhileScope) and e.dst is parent.body[0]:
                    expected_transition = True
                else:
                    next_block = cf.find_next_block(node)
                    # Next state in block or first state in next CF block
                    if next_block is not None:
                        if isinstance(next_block, cf.GeneralLoopScope):  # Special case for control flow regions
                            if e.dst is next_block.loop:
                                expected_transition = True
                        elif next_block.first_block is e.dst:
                            expected_transition = True

                if not expected_transition and e not in parent.gotos_to_ignore:
                    goto_node = tn.GotoNode(target=e.dst.label)
                    goto_node.sdfg = block.sdfg
                    edge_body.append(goto_node)

            if e not in parent.gotos_to_ignore and not e.data.is_unconditional():
                if sdfg.out_degree(block) == 1 and parent.sequential:
                    # Conditional state in sequential block! Add "if not condition goto exit"
                    state_if_node = tn.StateIfScope(condition=CodeBlock(negate_expr(e.data.condition)),
                                                    children=[tn.GotoNode(target=None)])
                    state_if_node.sdfg = block.sdfg
                    result.append(state_if_node)
                    result.extend(edge_body)
                else:
                    # Add "if condition" with the body above
                    state_if_node = tn.StateIfScope(condition=e.data.condition, children=edge_body) 
                    state_if_node.sdfg = block.sdfg
                    result.append(state_if_node)
            else:
                result.extend(edge_body)
        return result

    # Traverse said tree (also into states) to create the schedule tree
    def totree(node: cf.ControlFlow, parent: cf.GeneralBlock = None) -> List[tn.ScheduleTreeNode]:
        result: List[tn.ScheduleTreeNode] = []
        if isinstance(node, cf.GeneralBlock):
            subnodes: List[tn.ScheduleTreeNode] = []
            for n in node.elements:
                subnodes.extend(totree(n, node))
            if not node.sequential:
                # Nest in general block
                gblock = tn.GBlock(children=subnodes)
                gblock.sdfg = node.first_block.sdfg
                result = [gblock]
            else:
                # Use the sub-nodes directly
                result = subnodes

        elif isinstance(node, cf.BasicCFBlock):
            result = state_schedule_tree(node.state)

            # Add interstate assignments unrelated to structured control flow
            if parent is not None:
                result.extend(transition_totree(node, parent, node.state))
        elif isinstance(node, cf.ForScope):
            for_node = tn.ForScope(header=node, children=totree(node.body))
            for_node = node.first_block.sdfg
            result.append(for_node)
        elif isinstance(node, cf.IfScope):
            if_node = tn.IfScope(condition=node.condition, children=totree(node.body))
            if_node.sdfg = node.first_block.sdfg
            result.append(if_node)
            if node.orelse is not None:
                else_node = tn.ElseScope(children=totree(node.orelse))
                else_node.sdfg = node.first_block.sdfg
                result.append(else_node)
        elif isinstance(node, cf.IfElseChain):
            # Add "if" for the first condition, "elif"s for the rest
            if_node = tn.IfScope(condition=node.body[0][0], children=totree(node.body[0][1]))
            if_node.sdfg = node.first_block.sdfg
            result.append(if_node)
            for cond, body in node.body[1:]:
                elif_node = tn.ElifScope(condition=cond, children=totree(body))
                elif_node.sdfg = node.first_block.sdfg
                result.append(elif_node)
            # "else goto exit"
            goto_node = tn.GotoNode(target=None)
            goto_node.sdfg = node.first_block.sdfg
            else_node = tn.ElseScope(children=[goto_node])
            else_node.sdfg = node.first_block.sdfg
            result.append(else_node)
        elif isinstance(node, cf.WhileScope):
            while_node = tn.WhileScope(header=node, children=totree(node.body))
            while_node.sdfg = node.first_block.sdfg
            result.append(while_node)
        elif isinstance(node, cf.DoWhileScope):
            do_while_node = tn.DoWhileScope(header=node, children=totree(node.body))
            do_while_node.sdfg = node.sdfg
            result.append(do_while_node)
        elif isinstance(node, cf.GeneralLoopScope):
            loop_node = tn.GeneralLoopScope(header=node, children=totree(node.body))
            loop_node.sdfg = node.loop.sdfg
            result.append(loop_node)

            # Add interstate assignments unrelated to structured control flow
            if parent is not None:
                result.extend(transition_totree(node, parent, node.loop))
        elif isinstance(node, cf.GeneralConditionalScope):
            if_node = tn.IfScope(condition=node.conditional.branches[0][0],
                                 children=totree(node.branch_bodies[0][1]),
                                 cond_block=node.conditional)
            if_node.sdfg = node.conditional.sdfg
            result.append(if_node)
            for i, (cond, _) in enumerate(node.conditional.branches[1:]):
                if cond is not None:
                    elif_node = tn.ElifScope(condition=cond, children=totree(node.branch_bodies[i][1]))
                    elif_node = node.conditional.sdfg
                    result.append(elif_node)
                else:
                    else_node = tn.ElseScope(children=totree(node.branch_bodies[i][1]))
                    else_node.sdfg = node.conditional.sdfg
                    result.append(else_node)

            # Add interstate assignments unrelated to structured control flow
            if parent is not None:
                result.extend(transition_totree(node, parent, node.conditional))
        elif isinstance(node, cf.ReturnCFBlock):
            goto_node = tn.GotoNode(target=None)
            goto_node.sdfg = node.block.sdfg
            result.append(goto_node)
        else:
            # e.g., "SwitchCaseScope"
            raise tn.UnsupportedScopeException(type(node).__name__)

        if node.first_block is not None:
            first_state_node = tn.StateLabel(state=node.first_block)
            first_state_node.sdfg = node.first_block.sdfg
            result = [first_state_node] + result

        return result

    # Recursive traversal of the control flow tree
    result = tn.ScheduleTreeScope(children=totree(cfg))
    result.sdfg = sdfg

    # Clean up tree
    stpasses.remove_unused_and_duplicate_labels(result)

    return result


if __name__ == '__main__':
    s = time.time()
    sdfg = SDFG.from_file(sys.argv[1])
    print('Loaded SDFG in', time.time() - s, 'seconds')
    s = time.time()
    stree = as_schedule_tree(sdfg, in_place=True)
    print('Created schedule tree in', time.time() - s, 'seconds')

    with open('output_stree.txt', 'w') as fp:
        fp.write(stree.as_string(-1) + '\n')

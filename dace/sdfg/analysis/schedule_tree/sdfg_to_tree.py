# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Dict, List, Set
import dace
from dace import symbolic, data
from dace.codegen import control_flow as cf
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import utils as sdutil, graph as gr
from dace.frontend.python.astutils import negate_expr
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.properties import CodeBlock
from dace.memlet import Memlet

import time
import sys


def normalize_memlet(sdfg: SDFG, state: SDFGState, original: gr.MultiConnectorEdge[Memlet], data: str) -> Memlet:
    """
    Normalizes a memlet to a given data descriptor.
    
    :param sdfg: The SDFG.
    :param state: The state.
    :param original: The original memlet.
    :param data: The data descriptor.
    :return: A new memlet.
    """
    edge = copy.deepcopy(original)
    edge.data.try_initialize(sdfg, state, edge)

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
    return memlet


def replace_memlets(sdfg: SDFG, array_mapping: Dict[str, Memlet]):
    """
    Replaces all uses of data containers in memlets and interstate edges in an SDFG.
    :param sdfg: The SDFG.
    :param array_mapping: A mapping from internal data descriptor names to external memlets.
    """
    # TODO replace, normalize, and compose
    pass


def remove_name_collisions(sdfg: SDFG):
    """
    Removes name collisions in nested SDFGs by renaming states, data containers, and symbols.

    :param sdfg: The SDFG.
    """
    state_names_seen = set()
    identifiers_seen = set()

    for nsdfg in sdfg.all_sdfgs_recursive():
        # Rename duplicate states
        for state in nsdfg.nodes():
            if state.label in state_names_seen:
                state.set_label(data.find_new_name(state.label, state_names_seen))
            state_names_seen.add(state.label)

        replacements: Dict[str, str] = {}
        parent_node = nsdfg.parent_nsdfg_node

        # Rename duplicate data containers
        for name, desc in nsdfg.arrays.items():
            # Will already be renamed during conversion
            if parent_node is not None and not desc.transient:
                continue

            if name in identifiers_seen:
                new_name = data.find_new_name(name, identifiers_seen)
                replacements[name] = new_name
                name = new_name
            identifiers_seen.add(name)

        # Rename duplicate symbols
        for name in nsdfg.get_all_symbols():
            # Will already be renamed during conversion
            if parent_node is not None and name in parent_node.symbol_mapping:
                continue

            if name in identifiers_seen:
                new_name = data.find_new_name(name, identifiers_seen)
                replacements[name] = new_name
                name = new_name
            identifiers_seen.add(name)

        # Rename duplicate constants
        for name in nsdfg.constants_prop.keys():
            if name in identifiers_seen:
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
    return tn.ViewNode(target=view_name,
                       source=viewed_name,
                       memlet=normalized,
                       src_desc=sdfg.arrays[viewed_name],
                       view_desc=sdfg.arrays[view_name])


def prepare_schedule_tree_edges(state: SDFGState) -> Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode]:
    """
    Creates a dictionary mapping edges to their corresponding schedule tree nodes, if relevant.

    :param state: The state.
    """
    result: Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode] = {}
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
                        continue
            if isinstance(e.dst, dace.nodes.AccessNode):
                desc = e.dst.desc(sdfg)
                if isinstance(desc, dace.data.View):
                    vedge = sdutil.get_view_edge(state, e.dst)
                    if e is vedge:
                        viewed_node = sdutil.get_view_node(state, e.dst)
                        result[e] = _make_view_node(state, e, e.dst.data, viewed_node.data)
                        continue

            # 2. Check for reference sets
            if isinstance(e.dst, dace.nodes.AccessNode) and e.dst_conn == 'set':
                assert isinstance(e.dst.desc(sdfg), dace.data.Reference)
                result[e] = tn.RefSetNode(target=e.data.data,
                                          memlet=e.data,
                                          src_desc=sdfg.arrays[e.data.data],
                                          ref_desc=sdfg.arrays[e.dst.data])
                continue

            # 3. Check for copies
            # Get both ends of the memlet path
            mpath = state.memlet_path(e)
            src = mpath[0].src
            dst = mpath[-1].dst
            if not isinstance(src, dace.nodes.AccessNode):
                continue
            if not isinstance(dst, dace.nodes.AccessNode):
                continue

            # If the edge destination is the innermost node, it is a downward-pointing path
            is_target_dst = e.dst is dst

            innermost_node = dst if is_target_dst else src
            outermost_node = src if is_target_dst else dst

            # Normalize memlets to their innermost node, or source->destination if it is a same-scope edge
            if e.src is src and e.dst is dst:
                outermost_node = src
                innermost_node = dst

            new_memlet = normalize_memlet(sdfg, state, e, outermost_node.data)
            result[e] = tn.CopyNode(target=innermost_node.data, memlet=new_memlet)

    return result


def state_schedule_tree(state: SDFGState) -> List[tn.ScheduleTreeNode]:
    """
    Use scope-aware topological sort to get nodes by scope and return the schedule tree of this state.

    :param state: The state.
    :return: A string for the whole state
    """
    result: List[tn.ScheduleTreeNode] = []
    NODE_TO_SCOPE_TYPE = {
        dace.nodes.MapEntry: tn.MapScope,
        dace.nodes.ConsumeEntry: tn.ConsumeScope,
        dace.nodes.PipelineEntry: tn.PipelineScope,
    }
    sdfg = state.parent

    edge_to_stree: Dict[gr.MultiConnectorEdge[Memlet], tn.ScheduleTreeNode] = prepare_schedule_tree_edges(state)
    edges_to_ignore = set()

    scopes: List[List[tn.ScheduleTreeNode]] = []
    for node in sdutil.scope_aware_topological_sort(state):
        if isinstance(node, dace.nodes.EntryNode):
            scopes.append(result)
            subnodes = []
            result.append(NODE_TO_SCOPE_TYPE[type(node)](node=node, children=subnodes))
            result = subnodes
        elif isinstance(node, dace.nodes.ExitNode):
            result = scopes.pop()
        elif isinstance(node, dace.nodes.NestedSDFG):
            nested_array_mapping = {}

            # Replace symbols and memlets in nested SDFGs to match the namespace of the parent SDFG
            # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
            symbolic.safe_replace(node.symbol_mapping, node.sdfg.replace_dict)
            replace_memlets(node.sdfg, nested_array_mapping)

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
                        nested_array_mapping[conn] = e.data

                if no_mapping:  # Must use view (nview = nested SDFG view)
                    result.append(
                        tn.NView(target=conn,
                                 source=e.data.data,
                                 memlet=e.data,
                                 src_desc=sdfg.arrays[e.data.data],
                                 view_desc=node.sdfg.arrays[conn]))


            # Insert the nested SDFG flattened
            nested_stree = as_schedule_tree(node.sdfg, in_place=True, toplevel=False)
            result.extend(nested_stree.children)
        elif isinstance(node, dace.nodes.Tasklet):
            in_memlets = {e.dst_conn: e.data for e in state.in_edges(node) if e.dst_conn}
            out_memlets = {e.src_conn: e.data for e in state.out_edges(node) if e.src_conn}
            result.append(tn.TaskletNode(node=node, in_memlets=in_memlets, out_memlets=out_memlets))
        elif isinstance(node, dace.nodes.LibraryNode):
            in_memlets = {e.dst_conn: e.data for e in state.in_edges(node) if e.dst_conn}
            out_memlets = {e.src_conn: e.data for e in state.out_edges(node) if e.src_conn}
            result.append(tn.LibraryCall(node=node, in_memlets=in_memlets, out_memlets=out_memlets))
        elif isinstance(node, dace.nodes.AccessNode):
            # If one of the neighboring edges has a schedule tree node attached to it, use that
            for e in state.all_edges(node):
                if e in edges_to_ignore:
                    continue
                if e in edge_to_stree:
                    result.append(edge_to_stree[e])
                    edges_to_ignore.add(e)

    assert len(scopes) == 0

    return result


def as_schedule_tree(sdfg: SDFG, in_place: bool = False, toplevel: bool = True) -> tn.ScheduleTreeScope:
    """
    Converts an SDFG into a schedule tree. The schedule tree is a tree of nodes that represent the execution order of the SDFG.
    Each node in the tree can either represent a single statement (symbol assignment, tasklet, copy, library node, etc.) or
    a ``ScheduleTreeScope`` block (map, for-loop, pipeline, etc.) that contains other nodes.
    
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
        # Handle name collisions (in arrays, state labels, symbols)
        remove_name_collisions(sdfg)

    #############################

    # Create initial tree from CFG
    cfg: cf.ControlFlow = cf.structured_control_flow_tree(sdfg, lambda _: '')

    # Traverse said tree (also into states) to create the schedule tree
    def totree(node: cf.ControlFlow, parent: cf.GeneralBlock = None) -> List[tn.ScheduleTreeNode]:
        result: List[tn.ScheduleTreeNode] = []
        if isinstance(node, cf.GeneralBlock):
            subnodes: List[tn.ScheduleTreeNode] = []
            for n in node.elements:
                subnodes.extend(totree(n, node))
            if not node.sequential:
                # Nest in general block
                result = [tn.GBlock(children=subnodes)]
            else:
                # Use the sub-nodes directly
                result = subnodes

        elif isinstance(node, cf.SingleState):
            result = state_schedule_tree(node.state)

            # Add interstate assignments unrelated to structured control flow
            if parent is not None:
                for e in sdfg.out_edges(node.state):
                    edge_body = []

                    if e not in parent.assignments_to_ignore:
                        for aname, aval in e.data.assignments.items():
                            edge_body.append(tn.AssignNode(name=aname, value=CodeBlock(aval)))

                    if not parent.sequential:
                        if e not in parent.gotos_to_ignore:
                            edge_body.append(tn.GotoNode(target=e.dst.label))
                        else:
                            if e in parent.gotos_to_break:
                                edge_body.append(tn.BreakNode())
                            elif e in parent.gotos_to_continue:
                                edge_body.append(tn.ContinueNode())

                    if e not in parent.gotos_to_ignore and not e.data.is_unconditional():
                        if sdfg.out_degree(node.state) == 1 and parent.sequential:
                            # Conditional state in sequential block! Add "if not condition goto exit"
                            result.append(
                                tn.StateIfScope(condition=CodeBlock(negate_expr(e.data.condition)),
                                                children=[tn.GotoNode(target=None)]))
                            result.extend(edge_body)
                        else:
                            # Add "if condition" with the body above
                            result.append(tn.StateIfScope(condition=e.data.condition, children=edge_body))
                    else:
                        result.extend(edge_body)

        elif isinstance(node, cf.ForScope):
            result.append(tn.ForScope(header=node, children=totree(node.body)))
        elif isinstance(node, cf.IfScope):
            result.append(tn.IfScope(condition=node.condition, children=totree(node.body)))
            if node.orelse is not None:
                result.append(tn.ElseScope(children=totree(node.orelse)))
        elif isinstance(node, cf.IfElseChain):
            # Add "if" for the first condition, "elif"s for the rest
            result.append(tn.IfScope(condition=node.body[0][0], children=totree(node.body[0][1])))
            for cond, body in node.body[1:]:
                result.append(tn.ElifScope(condition=cond, children=totree(body)))
            # "else goto exit"
            result.append(tn.ElseScope(children=[tn.GotoNode(target=None)]))
        elif isinstance(node, cf.WhileScope):
            result.append(tn.WhileScope(header=node, children=totree(node.body)))
        elif isinstance(node, cf.DoWhileScope):
            result.append(tn.DoWhileScope(header=node, children=totree(node.body)))
        else:
            # e.g., "SwitchCaseScope"
            raise tn.UnsupportedScopeException(type(node).__name__)

        if node.first_state is not None:
            result = [tn.StateLabel(state=node.first_state)] + result

        return result

    # Recursive traversal of the control flow tree
    result = tn.ScheduleTreeScope(children=totree(cfg))

    # Clean up tree
    remove_unused_and_duplicate_labels(result)

    return result


def remove_unused_and_duplicate_labels(stree: tn.ScheduleTreeScope):

    class FindGotos(tn.ScheduleNodeVisitor):

        def __init__(self):
            self.gotos: Set[str] = set()

        def visit_GotoNode(self, node: tn.GotoNode):
            if node.target is not None:
                self.gotos.add(node.target)

    class RemoveLabels(tn.ScheduleNodeTransformer):

        def __init__(self, labels_to_keep: Set[str]) -> None:
            self.labels_to_keep = labels_to_keep
            self.labels_seen = set()

        def visit_StateLabel(self, node: tn.StateLabel):
            if node.state.name not in self.labels_to_keep:
                return None
            if node.state.name in self.labels_seen:
                return None
            self.labels_seen.add(node.state.name)
            return node

    fg = FindGotos()
    fg.visit(stree)
    return RemoveLabels(fg.gotos).visit(stree)


if __name__ == '__main__':
    s = time.time()
    sdfg = SDFG.from_file(sys.argv[1])
    print('Loaded SDFG in', time.time() - s, 'seconds')
    s = time.time()
    stree = as_schedule_tree(sdfg, in_place=True)
    print('Created schedule tree in', time.time() - s, 'seconds')
    with open('output_stree.txt', 'w') as fp:
        fp.write(stree.as_string(-1) + '\n')

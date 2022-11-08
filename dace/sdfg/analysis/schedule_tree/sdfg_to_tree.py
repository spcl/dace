# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Dict, List, Set
import dace
from dace.codegen import control_flow as cf
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.sdfg import utils as sdutil
import time
from dace.frontend.python.astutils import negate_expr
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.properties import CodeBlock


def state_schedule_tree(state: SDFGState, array_mapping: Dict[str, dace.Memlet]) -> List[tn.ScheduleTreeNode]:
    """
    Use scope_tree to get nodes by scope. Traverse all scopes and return a string for each scope.
    :return: A string for the whole state
    """
    result: List[tn.ScheduleTreeNode] = []
    NODE_TO_SCOPE_TYPE = {
        dace.nodes.MapEntry: tn.MapScope,
        dace.nodes.ConsumeEntry: tn.ConsumeScope,
        dace.nodes.PipelineEntry: tn.PipelineScope,
    }
    sdfg = state.parent

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
                        dname = e.data.data
                        if dname in array_mapping:  # Trace through recursive nested SDFGs
                            dname = array_mapping[dname].data  # TODO slice tracing

                        # TODO: Add actual slice
                        new_memlet = copy.deepcopy(e.data)
                        new_memlet.data = dname
                        nested_array_mapping[conn] = new_memlet

                if no_mapping:  # Must use view (nview = nested SDFG view)
                    result.append(
                        tn.NView(target=conn,
                                 source=e.data.data,
                                 memlet=e.data,
                                 src_desc=sdfg.arrays[e.data.data],
                                 view_desc=node.sdfg.arrays[conn]))

            # Insert the nested SDFG flattened
            nested_stree = as_schedule_tree(node.sdfg, nested_array_mapping)
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
            # Check type
            desc = node.desc(sdfg)
            vedge = None
            if isinstance(desc, dace.data.View):
                vedge = sdutil.get_view_edge(state, node)

            # Access nodes are only printed with corresponding memlets
            for e in state.all_edges(node):
                if e.data.is_empty():
                    continue
                conn = e.dst_conn if e.dst is node else e.src_conn

                # Reference + "set" connector
                if conn == 'set':
                    result.append(
                        tn.RefSetNode(target=node.data,
                                      memlet=e.data,
                                      src_desc=sdfg.arrays[e.data.data],
                                      ref_desc=sdfg.arrays[node.data]))
                    continue
                # View edge
                if e is vedge:
                    subset = e.data.get_src_subset(e, state) if e.dst is node else e.data.get_dst_subset(e, state)
                    vnode = sdutil.get_view_node(state, node)
                    new_memlet = copy.deepcopy(e.data)
                    new_memlet.data = node.data
                    new_memlet.subset = subset
                    new_memlet.other_subset = None
                    result.append(
                        tn.ViewNode(target=vnode.data,
                                    source=node.data,
                                    memlet=new_memlet,
                                    src_desc=sdfg.arrays[vnode.data],
                                    view_desc=sdfg.arrays[node.data]))
                    continue

                # Check if an incoming or outgoing memlet is a leaf (since the copy will be done at
                # the innermost level) and leads to access node (otherwise taken care of in another node)
                mpath = state.memlet_path(e)
                if len(mpath) == 1 and e.dst is node:
                    # Special case: only annotate source in a simple copy
                    continue
                if e.dst is node and mpath[-1] is e:
                    other = mpath[0].src
                    if not isinstance(other, dace.nodes.AccessNode):
                        continue
                    result.append(tn.CopyNode(target=node.data, memlet=e.data))
                    continue
                if e.src is node and mpath[0] is e:
                    other = mpath[-1].dst
                    if not isinstance(other, dace.nodes.AccessNode):
                        continue
                    result.append(tn.CopyNode(target=other.data, memlet=e.data))
                    continue

    assert len(scopes) == 0

    return result


def as_schedule_tree(sdfg: SDFG, array_mapping: Dict[str, dace.Memlet] = None) -> tn.ScheduleTreeScope:
    """
    Converts an SDFG into a schedule tree. The schedule tree is a tree of nodes that represent the execution order of the SDFG.
    Each node in the tree can either represent a single statement (symbol assignment, tasklet, copy, library node, etc.) or
    a ``ScheduleTreeScope`` block (map, for-loop, pipeline, etc.) that contains other nodes.
    
    It can be used to generate code from an SDFG, or to perform schedule transformations on the SDFG. For example,
    erasing an empty if branch, or merging two consecutive for-loops. The SDFG can then be reconstructed via the 
    ``from_schedule_tree`` function.
    
    :param sdfg: The SDFG to convert.
    :param array_mapping: (Internal, should be left empty) A mapping from array names to memlets.
    :return: A schedule tree representing the given SDFG.
    """

    from dace.transformation import helpers as xfh  # Avoid import loop
    array_mapping = array_mapping or {}

    # Split edges with assignments and conditions
    xfh.split_interstate_edges(sdfg)

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
            result = state_schedule_tree(node.state, array_mapping)

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
                                                children=tn.GotoNode(target=None)))
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
    remove_unused_labels(result)

    return result


def remove_unused_labels(stree: tn.ScheduleTreeScope):
    class FindGotos(tn.ScheduleNodeVisitor):
        def __init__(self):
            self.gotos: Set[str] = set()

        def visit_GotoNode(self, node: tn.GotoNode):
            if node.target is not None:
                self.gotos.add(node.target)

    class RemoveLabels(tn.ScheduleNodeTransformer):
        def __init__(self, labels_to_keep: Set[str]) -> None:
            self.labels_to_keep = labels_to_keep

        def visit_StateLabel(self, node: tn.StateLabel):
            if node.state.name not in self.labels_to_keep:
                return None
            return node

    fg = FindGotos()
    fg.visit(stree)
    return RemoveLabels(fg.gotos).visit(stree)


if __name__ == '__main__':
    stree = as_schedule_tree(sdfg)
    with open('output_stree.txt', 'w') as fp:
        fp.write(stree.as_string(-1))

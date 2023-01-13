# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Assortment of passes for schedule trees.
"""

from dace import data as dt, Memlet, subsets as sbs, symbolic as sym
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dataclasses import dataclass
from typing import Set


def remove_unused_and_duplicate_labels(stree: tn.ScheduleTreeScope):
    """
    Removes unused and duplicate labels from the schedule tree.

    :param stree: The schedule tree to remove labels from.
    """

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


def remove_empty_scopes(stree: tn.ScheduleTreeScope):
    """
    Removes empty scopes from the schedule tree.

    :warning: This pass is not safe to use for for-loops, as it will remove indices that may be used after the loop.
    """

    class RemoveEmptyScopes(tn.ScheduleNodeTransformer):

        def visit(self, node: tn.ScheduleTreeNode):
            if not isinstance(node, tn.ScheduleTreeScope):
                return super().visit(node)

            if len(node.children) == 0:
                return None

            return self.generic_visit(node)

    return RemoveEmptyScopes().visit(stree)


def wcr_to_reduce(stree: tn.ScheduleTreeScope):
    """
    Converts WCR assignments to reductions.

    :param stree: The schedule tree to remove WCR assignments from.
    """

    class WCRToReduce(tn.ScheduleNodeTransformer):

        def visit(self, node: tn.ScheduleTreeNode):

            if isinstance(node, tn.TaskletNode):

                wcr_found = False
                for _, memlet in node.out_memlets.items():
                    if memlet.wcr:
                        wcr_found = True
                        break

                if wcr_found:

                    loop_found = False
                    rng = None
                    idx = None
                    parent = node.parent
                    while parent:
                        if isinstance(parent, (tn.MapScope, tn.ForScope)):
                            loop_found = True
                            rng = parent.node.map.range
                            break
                        parent = parent.parent
                    
                    if loop_found:

                        for conn, memlet in node.out_memlets.items():
                            if memlet.wcr:

                                scope = node.parent
                                while memlet.data not in scope.containers:
                                    scope = scope.parent
                                desc = scope.containers[memlet.data]

                                shape = rng.size() + list(desc.shape) if not isinstance(desc, dt.Scalar) else rng.size()
                                parent.containers[f'{memlet.data}_arr'] = dt.Array(desc.dtype, shape, transient=True)
                                
                                indices = [(sym.pystr_to_symbolic(s), sym.pystr_to_symbolic(s), 1) for s in parent.node.map.params]
                                if not isinstance(desc, dt.Scalar):
                                    indices.extend(memlet.subset.ranges)
                                memlet.subset = sbs.Range(indices)
                                
                                from dace.libraries.standard import Reduce
                                rednode = Reduce(memlet.wcr)
                                libcall = tn.LibraryCall(rednode, {Memlet.from_array(f'{memlet.data}_arr', parent.containers[f'{memlet.data}_arr'])}, {Memlet.from_array(memlet.data, desc)})
                                
                                memlet.data = f'{memlet.data}_arr'
                                memlet.wcr = None
                    
                                parent.children.append(libcall)

                
            return self.generic_visit(node)
    
    return WCRToReduce().visit(stree)


def canonicalize_if(tree: tn.ScheduleTreeScope):
    """
    Canonicalizes sequences of if-elif-else scopes to sequences of if scopes.
    """

    from dace.sdfg.nodes import CodeBlock

    class CanonicalizeIf(tn.ScheduleNodeTransformer):

        def visit(self, node: tn.ScheduleTreeNode):

            if isinstance(node, (tn.ElifScope, tn.ElseScope)):         
                parent = node.parent
                assert node in parent.children
                node_idx = parent.children.index(node)

                conditions = []
                for curr_node in reversed(parent.children[:node_idx]):
                    conditions.append(curr_node.condition)
                    if isinstance(curr_node, tn.IfScope):
                        break
                condition = f"not ({' or '.join([f'({c.as_string})' for c in conditions])})"
                if isinstance(node, tn.ElifScope):
                    condition = f"{condition} and {node.condition.as_string}"
                new_node = tn.IfScope(node.sdfg, node.top_level, node.children, CodeBlock(condition))
                new_node.parent = parent
            else:
                new_node = node

            return self.generic_visit(new_node)

    return CanonicalizeIf().visit(tree)


def fission_scopes(node: tn.ScheduleTreeScope):

    from dace.sdfg.analysis.schedule_tree.transformations import loop_fission, if_fission

    @dataclass
    class FissionScopes(tn.ScheduleNodeTransformer):

        tree: tn.ScheduleTreeScope

        def visit_IfScope(self, node: tn.IfScope):
            return if_fission(node, assume_canonical=True, distribute=True)

        def visit_ForScope(self, node: tn.ForScope):
            return loop_fission(node, self.tree)

        def visit_MapScope(self, node: tn.MapScope):
            return loop_fission(node, self.tree)
        
        def visit(self, node: tn.ScheduleTreeNode):
            node = self.generic_visit(node)
            if isinstance(node, (tn.IfScope, tn.ForScope, tn.MapScope)):
                return super().visit(node)
            return node

    return FissionScopes(node).visit(node)


def validate(node: tn.ScheduleTreeNode) -> bool:

    if isinstance(node, tn.ScheduleTreeScope):
        if any(child.parent is not node for child in node.children):
            return False
        if all(validate(child) for child in node.children):
            return True
        else:
            return False
        
    return True

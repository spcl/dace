# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Assortment of passes for schedule trees.
"""

from dace import data as dt, Memlet, subsets as sbs, symbolic as sym
from dace.sdfg.analysis.schedule_tree import treenodes as tn
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


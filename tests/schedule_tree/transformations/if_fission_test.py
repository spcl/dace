# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests fission of If scopes. """
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree import passes, transformations as ttrans, treenodes as tnodes
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree, as_sdfg


class IfCounter(tnodes.ScheduleNodeVisitor):

    if_count: int
    elif_count: int
    else_count: int

    def __init__(self):
        self.if_count = 0
        self.elif_count = 0
        self.else_count = 0
    
    def visit_IfScope(self, node: tnodes.IfScope):
        self.if_count += 1
        self.generic_visit(node)
    
    def visit_ElifScope(self, node: tnodes.ElifScope):
        self.elif_count += 1
        self.generic_visit(node)
    
    def visit_ElseScope(self, node: tnodes.ElseScope):
        self.else_count += 1
        self.generic_visit(node)


def test_if_fission():

    @dace.program
    def ifelifelse(c: dace.int64):
        out0, out1 = 0, 0
        if c < 0:
            out0 = -5
            out1 = -10
        if c == 0:
            pass
        if c > 0:
            out0 = 5
            out1 = 10
        return out0, out1

    sdfg_pre = ifelifelse.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    ifcounter_pre = IfCounter()
    ifcounter_pre.visit(tree)
    if ifcounter_pre.elif_count > 0 or ifcounter_pre.else_count > 0:
        passes.canonicalize_if(tree)
        ifcounter_post = IfCounter()
        ifcounter_post.visit(tree)
        assert ifcounter_post.elif_count == 0
        assert ifcounter_post.else_count == 0

    for child in list(tree.children):
        if isinstance(child, tnodes.IfScope):
            ttrans.if_fission(child)

    sdfg_post = as_sdfg(tree)

    for c in (-100, 0, 100):
        ref = ifelifelse.f(c)
        val0 = sdfg_pre(c=c)
        val1 = sdfg_post(c=c)
        assert val0 == ref
        assert val1 == ref


if __name__ == "__main__":
    test_if_fission()

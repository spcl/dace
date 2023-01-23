# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests canonicalization of If/Elif/Else scopes. """
import dace
import numpy as np
from dace.sdfg.analysis.schedule_tree import passes, treenodes as tnodes
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


def test_ifelifelse_canonicalization():

    @dace.program
    def ifelifelse(c: dace.int64):
        out = 0
        if c < 0:
            out = c - 1
        elif c == 0:
            pass
        else:
            out = c % 2
        return out

    sdfg_pre = ifelifelse.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    ifcounter_pre = IfCounter()
    ifcounter_pre.visit(tree)
    ifcount = ifcounter_pre.if_count + ifcounter_pre.elif_count + ifcounter_pre.else_count

    passes.canonicalize_if(tree)
    ifcounter_post = IfCounter()
    ifcounter_post.visit(tree)
    assert ifcounter_post.if_count == ifcount
    assert ifcounter_post.elif_count == 0
    assert ifcounter_post.else_count == 0

    sdfg_post = as_sdfg(tree)

    for c in (-100, 0, 100):
        ref = ifelifelse.f(c)
        val0 = sdfg_pre(c=c)
        val1 = sdfg_post(c=c)
        assert val0[0] == ref
        assert val1[0] == ref


def test_ifelifelse_canonicalization2():

    @dace.program
    def ifelifelse2(c: dace.int64):
        out = 0
        if c < 0:
            if c < -100:
                out = c + 1
            elif c < -50:
                out = c + 2
            else:
                out = c + 3
        elif c == 0:
            pass
        else:
            if c > 100:
                out = c % 2
            elif c > 50:
                out = c % 3
            else:
                out = c % 4
        return out

    sdfg_pre = ifelifelse2.to_sdfg()
    tree = as_schedule_tree(sdfg_pre)
    ifcounter_pre = IfCounter()
    ifcounter_pre.visit(tree)
    ifcount = ifcounter_pre.if_count + ifcounter_pre.elif_count + ifcounter_pre.else_count

    passes.canonicalize_if(tree)
    ifcounter_post = IfCounter()
    ifcounter_post.visit(tree)
    assert ifcounter_post.if_count == ifcount
    assert ifcounter_post.elif_count == 0
    assert ifcounter_post.else_count == 0

    sdfg_post = as_sdfg(tree)

    for c in (-200, -70, -20, 0, 15, 67, 122):
        ref = ifelifelse2.f(c)
        val0 = sdfg_pre(c=c)
        val1 = sdfg_post(c=c)
        assert val0[0] == ref
        assert val1[0] == ref


if __name__ == "__main__":
    test_ifelifelse_canonicalization()
    test_ifelifelse_canonicalization2()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for calling plain-Python objects with a ``__call__`` method (auto-wrapped
by preprocessing into a bound-method :class:`DaceProgram` with ``methodobj``/
``objname`` set) from an unrolled ``for`` loop over a compile-time list of such
objects, and the corresponding ``dace.nounroll`` fallback.

Regression test for the nextgen inliner's ``_prepare_callee`` not injecting
the bound method object into the callee's globals, which made every
``self.attr`` access inside the callee unresolvable and degrade to a Python
callback.
"""
import numpy as np

import dace
from dace import nounroll, unroll
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


class ProgramB:

    def __init__(self, parameter):
        self.parameter: float = parameter

    def __call__(self, A, B):
        A[:] = B[:] + self.parameter


B1 = ProgramB(0.1)
B2 = ProgramB(0.2)
Bs = [B1, B2]
nB = 2


def test_unroll_object_calls():

    @dace.program
    def unrolltest(array_a: dace.float64[10], array_b: dace.float64[10]):
        B1(array_a, array_b)
        B2(array_a, array_b)
        for n in unroll(range(nB)):
            Bs[n](array_a, array_b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    tree = nextgen.parse_program(unrolltest, a, b)

    scopes = _nodes_of_type(tree, tn.FunctionCallScope)
    assert len(scopes) == 4
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_unroll_object_specialization():

    @dace.program
    def unrolltest(array_a: dace.float64[10], array_b: dace.float64[10]):
        B1(array_a, array_b)
        B2(array_a, array_b)
        for n in unroll(range(nB)):
            Bs[n](array_a, array_b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    tree = nextgen.parse_program(unrolltest, a, b)

    # Both instances specialized separately: their distinct parameters must
    # both appear among the lowered tasklet codes (rather than one shadowing
    # the other). ScheduleTreeRoot.as_string() only prints memlet summaries
    # for tasklets ("out = tasklet(in)"), not their code, so the code has to
    # be read off the tasklet nodes directly (as in inlining_test.py).
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    codes = [tasklet.node.code.as_string for tasklet in tasklets]
    assert any('0.1' in code for code in codes)
    assert any('0.2' in code for code in codes)


def test_unroll_object_execution():

    @dace.program
    def unrolltest(array_a: dace.float64[10], array_b: dace.float64[10]):
        B1(array_a, array_b)
        B2(array_a, array_b)
        for n in unroll(range(nB)):
            Bs[n](array_a, array_b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    tree = nextgen.parse_program(unrolltest, a, b)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    sdfg = tree.as_sdfg()

    array_a = np.random.rand(10)
    array_b = np.random.rand(10)
    sdfg(array_a=array_a, array_b=array_b)

    # Four sequential updates of A from B: A = B+0.1, A = B+0.2, A = B+0.1 (n=0),
    # A = B+0.2 (n=1) -- only the last one survives.
    expected = array_b + 0.2
    assert np.allclose(array_a, expected)


def test_nounroll_object_fallback():

    @dace.program
    def nounrolltest(array_a: dace.float64[10], array_b: dace.float64[10]):
        for n in nounroll(range(nB)):
            Bs[n](array_a, array_b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    tree = nextgen.parse_program(nounrolltest, a, b)

    loops = _nodes_of_type(tree, tn.LoopScope)
    assert len(loops) >= 1

    # The fallback callback must live inside the (non-unrolled) loop.
    found_callback_in_loop = False
    for loop in loops:
        if any(isinstance(node, tn.PythonCallbackNode) for node in loop.preorder_traversal()):
            found_callback_in_loop = True
    assert found_callback_in_loop


if __name__ == '__main__':
    test_unroll_object_calls()
    test_unroll_object_specialization()
    test_unroll_object_execution()
    test_nounroll_object_fallback()

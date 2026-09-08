# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for closure-array support in the next-generation frontend: external
arrays referenced by a program (or by inlined callees) register as
non-transient containers, deduplicated by their source qualified name.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def _closure_containers(tree: tn.ScheduleTreeRoot):
    return [
        name for name, descriptor in tree.containers.items()
        if not descriptor.transient and name not in tree.arg_names and not name.startswith('__return')
    ]


def test_closure_array_top_level():
    external = np.ones(12)

    @dace.program
    def uses_external(A: dace.float64[12]):
        A[:] = external + 1.0

    tree = nextgen.parse_program(uses_external)
    closure_names = _closure_containers(tree)
    assert len(closure_names) == 1
    assert tuple(tree.containers[closure_names[0]].shape) == (12, )
    # The external array participates in dataflow, not a callback
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert any(memlet.data == closure_names[0] for tasklet in tasklets for memlet in tasklet.in_memlets.values())
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_closure_array_in_callee():
    external = np.ones(12)

    @dace.program
    def callee(Y: dace.float64[12]):
        Y[:] = external * 2.0

    @dace.program
    def caller(A: dace.float64[12]):
        callee(A)

    tree = nextgen.parse_program(caller)
    assert len(_nodes_of_type(tree, tn.FunctionCallScope)) == 1
    closure_names = _closure_containers(tree)
    assert len(closure_names) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_closure_array_shared_between_caller_and_callee():
    external = np.ones(12)

    @dace.program
    def callee(Y: dace.float64[12]):
        Y[:] = Y + external

    @dace.program
    def caller(A: dace.float64[12], B: dace.float64[12]):
        B[:] = external
        callee(A)

    tree = nextgen.parse_program(caller)
    # The same external array maps to a single repository container even when
    # referenced from both the caller and an inlined callee
    closure_names = _closure_containers(tree)
    assert len(closure_names) == 1
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


if __name__ == '__main__':
    test_closure_array_top_level()
    test_closure_array_in_callee()
    test_closure_array_shared_between_caller_and_callee()

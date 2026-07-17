# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the parallel schedule-tree lowering path: every ``to_sdfg`` also
builds a schedule tree through the next-generation frontend and checks it
(:meth:`DaceProgram._run_parallel_schedule_tree_lowering_checks`). Programs
that used to exercise the old staged frontend's diagnostics (statement nodes,
reference assignments, PythonClass containers) must now lower cleanly.
"""
import dace
import numpy as np

from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def test_parallel_path_object_attribute_assignment():

    class AttrHolder:

        def __init__(self):
            self.arr = np.zeros(4, dtype=np.float64)

    attr_holder = AttrHolder()

    @dace.program
    def prog(A: dace.float64[4], out: dace.float64[4]):
        attr_holder.arr = A
        out[:] = attr_holder.arr

    # The parallel next-gen lowering runs inside to_sdfg and must not raise
    sdfg = prog.to_sdfg(simplify=False)
    assert sdfg is not None

    # The next-gen tree contains no legacy StatementNodes (verified contract)
    tree = nextgen.parse_program(prog)
    assert not any(isinstance(node, tn.StatementNode) for node in tree.preorder_traversal())


def test_parallel_path_pythonclass_argument():

    class Holder:
        scalar: dace.float64

    PythonHolder = dace.data.PythonClass.from_class(Holder)

    @dace.program
    def prog(holder: PythonHolder, A: dace.float64[4], out: dace.float64[4]):
        holder.new_data = A
        out[:] = holder.new_data[:]

    sdfg = prog.to_sdfg(simplify=False)
    assert sdfg is not None


if __name__ == '__main__':
    test_parallel_path_object_attribute_assignment()
    test_parallel_path_pythonclass_argument()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for declared-type semantics in the nextgen frontend: a type annotation
(``b: dace.float64``) fixes the descriptor of the container the name is bound
to, and the values later assigned to that name convert INTO it instead of
re-typing the name.

The distinction matters beyond dtypes: re-typing a name means rebinding it to a
fresh container, and a rebinding that happens inside a loop is loop-carried,
which forces the whole loop into a Python callback.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _containers(program, *args):
    """The (name -> descriptor) repository of ``program``'s schedule tree."""
    root = nextgen.parse_program(program, *args)
    return root, root.containers


def _callbacks(root: tn.ScheduleTreeRoot):
    return [node for node in root.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_declaration_types_the_name():
    """``b: dace.float64`` holds a float64, even when assigned an integer."""

    @dace.program
    def annotated(a: dace.float64[20], t: dace.int64):
        b: dace.float64
        for i in dace.map[0:20]:
            b = t
            a[i] = b

    root, containers = _containers(annotated, np.zeros(20), 5)
    assert containers['b'].dtype == dace.float64
    # A declared name is not rebound by the values assigned to it, so no
    # ``b_0``/``b_1`` versions exist and the loop carries nothing.
    assert not [name for name in containers if name.startswith('b_')]
    assert not _callbacks(root)


def test_declared_accumulator_updates_in_place():
    """An annotated accumulator keeps its declared type and its container."""

    @dace.program
    def reduce_rows(x: dace.float32[20, 20], y: dace.float32[20]):
        for i in dace.map[0:20]:
            ytmp: dace.float32 = 0
            for j in dace.map[0:20] @ dace.ScheduleType.Sequential:
                ytmp += x[i, j]

            y[i] = ytmp

    root, containers = _containers(reduce_rows, np.zeros((20, 20), np.float32), np.zeros(20, np.float32))
    assert containers['ytmp'].dtype == dace.float32
    assert not [name for name in containers if name.startswith('ytmp_')]
    assert not _callbacks(root)


def test_declaration_execution():
    """The declared type is the one that runs (integer value, float storage)."""

    @dace.program
    def annotated(a: dace.float64[20], t: dace.int64):
        b: dace.float64
        for i in dace.map[0:20]:
            b = t
            a[i] = b

    a = np.random.rand(20)
    annotated(a, 5)
    assert np.allclose(a, 5.0)


def test_shape_change_still_rebinds():
    """A declaration does not force values of a different shape into it."""

    @dace.program
    def reshaped(a: dace.float64[20]):
        b: dace.float64[10] = np.zeros([10], dace.float64)
        b = np.zeros([20], dace.float64)
        a[:] = b

    root, containers = _containers(reshaped, np.zeros(20))
    rebound = [name for name in containers if name == 'b' or name.startswith('b_')]
    assert len(rebound) > 1, f'expected a rebinding for the shape change, got {rebound}'


if __name__ == '__main__':
    test_declaration_types_the_name()
    test_declared_accumulator_updates_in_place()
    test_declaration_execution()
    test_shape_change_still_rebinds()

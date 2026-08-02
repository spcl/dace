# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that callables marked with ``@dace.always_inline`` are folded to
constants during preprocessing, rather than degrading to Python callbacks or
to inlined nested programs in the next-generation frontend.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.common import DaceSyntaxError
from dace.sdfg.analysis.schedule_tree import treenodes as tn


class _TracerRegistry:
    """Models an externally-defined registry object keyed by field name."""

    def __init__(self, *names: str):
        self._mapping = {name: index for index, name in enumerate(names)}

    @dace.always_inline
    def index(self, name: str) -> int:
        return self._mapping[name]


_tracers = _TracerRegistry('vapor', 'liquid', 'ice')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_always_inline_no_callback():

    @dace.program
    def tester(A: dace.float64[10]):
        A[_tracers.index('liquid')] = 1.0

    tree = nextgen.parse_program(tester)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    # The index is folded into the memlet, and no nested call is emitted for the method
    assert not _nodes_of_type(tree, tn.FunctionCallScope)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert str(next(iter(tasklets[0].out_memlets.values())).subset) == '1'


def test_always_inline_execution():

    @dace.program
    def tester(A: dace.float64[10]):
        A[_tracers.index('ice')] = 3.0

    A = np.zeros(10)
    tester(A)
    expected = np.zeros(10)
    expected[2] = 3.0
    assert np.allclose(A, expected)


def test_always_inline_non_constant_argument():

    @dace.program
    def tester(A: dace.float64[10], i: dace.int64):
        A[_tracers.index(i)] = 1.0

    with pytest.raises(DaceSyntaxError):
        nextgen.parse_program(tester, np.zeros(10), 1)


if __name__ == '__main__':
    test_always_inline_no_callback()
    test_always_inline_execution()
    test_always_inline_non_constant_argument()

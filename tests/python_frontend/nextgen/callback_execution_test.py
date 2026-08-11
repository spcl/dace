# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Execution tests for programs that keep statements in the Python interpreter.

A callback is a *call argument* of the compiled program (a function pointer in
the C signature), and the next-generation frontend synthesizes the callable
itself instead of taking it from the caller's closure. The tree therefore
carries its callbacks (``ScheduleTreeRoot.callback_objects``) and hands them to
the SDFG it converts into (``SDFG.callback_objects``), which is what makes a
program with a callback callable without the caller reconstructing anything.
"""
import numpy as np
import pytest

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: Side-effect sink for callbacks under test (module-level so the callbacks
#: resolve it as a global, like real user code does).
RECORD = []


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def note(value):
    RECORD.append(float(value))


def test_tree_carries_its_callbacks():

    @dace.program
    def program(A: dace.float64[4]):
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(program)
    names = [node.outlined_function_name for node in _callbacks(tree)]
    assert names
    assert sorted(tree.callback_objects) == sorted(names)
    assert all(callable(value) for value in tree.callback_objects.values())


def test_converted_sdfg_carries_its_callbacks():

    @dace.program
    def program(A: dace.float64[4]):
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(program)
    sdfg = tree.as_sdfg()
    assert sorted(sdfg.callback_objects) == sorted(tree.callback_objects)
    # The callback is a genuine argument of the compiled program, and it is the
    # SDFG's own object that fills it in.
    assert all(name in sdfg.symbols for name in sdfg.callback_objects)


def test_callback_runs_and_program_computes():
    RECORD.clear()

    @dace.program
    def program(A: dace.float64[4]):
        note(A[2])
        A[0] = A[0] + 1.0

    A = np.array([1.0, 2.0, 3.0, 4.0])
    nextgen.parse_program(program).as_sdfg()(A=A)

    assert RECORD == [3.0]
    assert np.allclose(A, [2.0, 2.0, 3.0, 4.0])


def test_program_whose_only_argument_is_a_callback():
    """No arrays, no symbols: the callback is the entire call signature."""
    RECORD.clear()

    @dace.program
    def program():
        note(7.0)

    nextgen.parse_program(program).as_sdfg()()

    assert RECORD == [7.0]


def test_callback_in_a_loop_runs_every_iteration():
    RECORD.clear()

    @dace.program
    def program(A: dace.float64[3]):
        for i in range(3):
            note(A[i])

    nextgen.parse_program(program).as_sdfg()(A=np.array([1.0, 2.0, 3.0]))

    assert RECORD == [1.0, 2.0, 3.0]


def test_callback_in_a_branch_runs_conditionally():
    RECORD.clear()

    @dace.program
    def program(A: dace.float64[2], flag: dace.int32):
        if flag > 0:
            note(A[0])

    sdfg = nextgen.parse_program(program).as_sdfg()
    sdfg(A=np.array([5.0, 6.0]), flag=0)
    assert RECORD == []
    sdfg(A=np.array([5.0, 6.0]), flag=1)
    assert RECORD == [5.0]


def test_callback_reached_through_an_attribute():
    """
    Preprocessing rewrites ``RECORD.append(...)`` to a call of a sanitized name
    that is a global of no scope, so the callable only exists in the closure the
    frontend detected it in. The callback's execution namespace has to carry it.
    """
    RECORD.clear()

    @dace.program
    def program(A: dace.float64[2]):
        RECORD.append(A[1])

    nextgen.parse_program(program).as_sdfg()(A=np.array([1.0, 8.0]))

    assert RECORD == [8.0]


def test_callback_inside_an_inlined_callee():
    """The callee's own detected callbacks come from the callee's closure."""
    RECORD.clear()

    @dace.program
    def callee(A: dace.float64[3]):
        note(A[0])
        A[1] = 5.0

    @dace.program
    def program(A: dace.float64[3]):
        callee(A)
        A[2] = 9.0

    A = np.array([1.0, 2.0, 3.0])
    nextgen.parse_program(program).as_sdfg()(A=A)

    assert RECORD == [1.0]
    assert np.allclose(A, [1.0, 5.0, 9.0])


def test_returning_a_callback_result():
    """The value a callback produces is returned like any other value."""

    @dace.program
    def program(A: dace.float64[5]):
        return sorted(A)

    result = nextgen.parse_program(program).as_sdfg()(A=np.array([3.0, 1.0, 2.0, 5.0, 4.0]))

    assert list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_explicit_callback_argument_wins():
    """A callable passed at call time overrides the SDFG's own object."""
    RECORD.clear()
    replacements = []

    @dace.program
    def program(A: dace.float64[2]):
        note(A[0])

    tree = nextgen.parse_program(program)
    name, = tree.callback_objects
    sdfg = tree.as_sdfg()
    sdfg(A=np.array([9.0, 0.0]), **{name: lambda value: replacements.append(float(value))})

    assert RECORD == []
    assert replacements == [9.0]


def test_deep_copied_sdfg_keeps_its_callbacks():
    import copy

    @dace.program
    def program(A: dace.float64[2]):
        print(A)

    sdfg = nextgen.parse_program(program).as_sdfg()
    duplicate = copy.deepcopy(sdfg)

    assert duplicate.callback_objects == sdfg.callback_objects
    # A shared dictionary would leak edits between copies
    duplicate.callback_objects.clear()
    assert sdfg.callback_objects


def test_serialization_drops_callbacks():
    """Live objects do not round-trip through JSON, by construction."""

    @dace.program
    def program(A: dace.float64[2]):
        print(A)

    sdfg = nextgen.parse_program(program).as_sdfg()
    assert sdfg.callback_objects
    assert dace.SDFG.from_json(sdfg.to_json()).callback_objects == {}


def test_missing_callback_is_reported_as_a_callback():
    """
    Callbacks are runtime objects: an SDFG that lost them (serialization is the
    real case) must say what the missing argument is, not just name it.
    """

    @dace.program
    def program(A: dace.float64[2]):
        print(A)

    sdfg = nextgen.parse_program(program).as_sdfg()
    name, = sdfg.callback_objects
    sdfg.callback_objects = {}
    with pytest.raises(KeyError, match=f'{name}.*callback'):
        sdfg(A=np.array([1.0, 2.0]))


if __name__ == '__main__':
    test_tree_carries_its_callbacks()
    test_converted_sdfg_carries_its_callbacks()
    test_callback_runs_and_program_computes()
    test_program_whose_only_argument_is_a_callback()
    test_callback_in_a_loop_runs_every_iteration()
    test_callback_in_a_branch_runs_conditionally()
    test_callback_reached_through_an_attribute()
    test_callback_inside_an_inlined_callee()
    test_returning_a_callback_result()
    test_explicit_callback_argument_wins()
    test_deep_copied_sdfg_keeps_its_callbacks()
    test_serialization_drops_callbacks()
    test_missing_callback_is_reported_as_a_callback()

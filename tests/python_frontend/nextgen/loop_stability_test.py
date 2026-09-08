# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the loop-entry stability rule (``rules.control_flow.
_lower_loop_with_stability_check``): a loop whose body rebinds a name bound
before it would need a merge at the loop head, which the binding design avoids
by re-lowering the loop as one interpreter callback.

The rule is right, but most of what it used to reject was not a merge at all:

- a one-element array accumulator (``total += x[i]`` on a ``float64[1]``) is
  the SAME container, not a new one -- this one was a silent miscompilation
  rather than a fallback, since the rebound name meant the parameter was never
  written,
- ``i += 1`` on a typed integer stays that type (NumPy's NEP 50 weak scalar
  promotion), rather than widening to int64 and rebinding,
- a sequential loop's iteration variable is the loop's own binding, and Python
  leaves it visible afterwards,
- a ``dace.map`` parameter is scoped to the map, so a LATER scope may reuse the
  name,
- a compile-time value the body materializes just has to be a container before
  the loop starts.

What still needs a merge -- a container rebound to a differently shaped one --
must still fall back, which the last test checks.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_one_element_accumulator_writes_through():
    """``total += x[i]`` on a ``float64[1]`` parameter. This asserts EXECUTION:
    the rebinding it used to cause was callback-free and silently wrong, the
    parameter staying untouched while a fresh scalar accumulated."""

    @dace.program
    def accumulate(x: dace.float64[16], total: dace.float64[1]):
        for i in range(16):
            total += x[i]

    x, total = np.random.rand(16), np.zeros(1)
    tree = nextgen.parse_program(accumulate, x, total)
    assert not _callbacks(tree)
    assert not [name for name in tree.containers if name.startswith('total_')]

    tree.as_sdfg().compile()(x=x, total=total)
    assert np.allclose(total, x.sum())


def test_typed_counter_keeps_its_dtype():
    """``i += 1`` on an ``int32`` counter: the literal is weak (NEP 50), so the
    name keeps one container instead of being rebound to an int64 one."""

    @dace.program
    def counter(A: dace.int32[10]):
        i = np.int32(1)
        while i < 10:
            A[i] = i
            i += 1

    tree = nextgen.parse_program(counter, np.zeros(10, dtype=np.int32))
    assert not _callbacks(tree)
    assert tree.containers['i'].dtype == dace.int32

    A = np.zeros(10, dtype=np.int32)
    tree.as_sdfg().compile()(A=A)
    assert np.array_equal(A, np.array([0] + list(range(1, 10)), dtype=np.int32))


def test_weak_literal_does_not_widen_but_a_float_literal_does():
    """The two halves of the weak-promotion rule, on descriptors."""

    @dace.program
    def kinds(a: dace.float32[4], b: dace.int32[4], out32: dace.float32[4], out64: dace.float64[4]):
        out32[:] = a * 2  # int literal, lower kind: stays float32
        out64[:] = b * 2.0  # float literal, higher kind: promotes to float64

    tree = nextgen.parse_program(kinds, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.int32),
                                 np.zeros(4, dtype=np.float32), np.zeros(4))
    assert not _callbacks(tree)

    a, b = np.random.rand(4).astype(np.float32), np.arange(4, dtype=np.int32)
    out32, out64 = np.zeros(4, dtype=np.float32), np.zeros(4)
    tree.as_sdfg().compile()(a=a, b=b, out32=out32, out64=out64)
    assert np.allclose(out32, a * 2) and np.allclose(out64, b * 2.0)


def test_iteration_variable_may_shadow_a_previous_binding():
    """``i = -1`` before the loop, then ``for i in range(...)``: the loop owns
    that name, and Python leaves its last value visible afterwards."""

    @dace.program
    def shadowing(A: dace.float64[8], out: dace.float64[1]):
        i = -1
        for i in range(8):
            A[i] += 1
        if i > 3:
            out[0] = 1.0

    A, out = np.zeros(8), np.zeros(1)
    tree = nextgen.parse_program(shadowing, A, out)
    assert not _callbacks(tree)

    tree.as_sdfg().compile()(A=A, out=out)
    assert np.allclose(A, 1.0) and out[0] == 1.0


def test_map_parameter_does_not_outlive_its_scope():
    """A map parameter is scoped to the map, so a later scope may bind the same
    name to something else -- which used to look loop-carried to that scope and
    roll it back to a callback."""

    @dace.program
    def two_scopes(A: dace.int64[8], B: dace.int64[8]):
        for i in dace.map[0:8]:
            A[i] = i
        for j in dace.map[0:8]:
            i = 1
            while i < 3:
                i = i + 1
            B[j] = A[j] + i

    A, B = np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64)
    tree = nextgen.parse_program(two_scopes, A, B)
    assert not _callbacks(tree)

    tree.as_sdfg().compile()(A=A, B=B)
    assert np.array_equal(A, np.arange(8)) and np.array_equal(B, np.arange(8) + 3)


def test_compile_time_value_is_promoted_before_the_loop():
    """A name bound to a compile-time value that the body materializes: the
    promotion is emitted ahead of the loop and the body re-lowered against it."""

    @dace.program
    def promoted(A: dace.int64[8]):
        n = len(A)
        for k in range(8):
            n = n + 1
            A[k] = n

    A = np.zeros(8, dtype=np.int64)
    tree = nextgen.parse_program(promoted, A)
    assert not _callbacks(tree)

    tree.as_sdfg().compile()(A=A)
    assert np.array_equal(A, np.arange(9, 17))


def test_genuine_rebinding_still_falls_back():
    """A name rebound to a differently SHAPED container really does need a
    merge at the loop head, and still degrades to a callback."""

    @dace.program
    def reshaping(A: dace.float64[8]):
        acc = A[0:2]
        for k in range(4):
            acc = A[0:4]  # A different shape every iteration but the first
            A[k] = acc[0]

    tree = nextgen.parse_program(reshaping, np.zeros(8))
    callbacks = _callbacks(tree)
    assert len(callbacks) == 1
    assert 'loop-stability' in callbacks[0].reason


if __name__ == '__main__':
    test_one_element_accumulator_writes_through()
    test_typed_counter_keeps_its_dtype()
    test_weak_literal_does_not_widen_but_a_float_literal_does()
    test_iteration_variable_may_shadow_a_previous_binding()
    test_map_parameter_does_not_outlive_its_scope()
    test_compile_time_value_is_promoted_before_the_loop()
    test_genuine_rebinding_still_falls_back()

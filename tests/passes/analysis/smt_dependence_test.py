# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Verdicts of the SMT dependence oracle.

The oracle answers three questions for ``LoopToMap``: is a write injective over the iteration
domain, and does a read/write pair carry a RAW, a WAR, or nothing. Every case below is one the
affine classifier cannot decide, and each asserts the exact verdict rather than "not None" --
a wrong verdict here is a miscompile (``RAW`` reported as ``none`` parallelizes a recurrence),
not a missed optimization.
"""
import pytest

import sympy as sp

from dace import symbolic
from dace.transformation.passes.analysis import smt_dependence

pytestmark = pytest.mark.skipif(not smt_dependence.has_z3(), reason='needs z3')

I = symbolic.pystr_to_symbolic('i')
N = symbolic.pystr_to_symbolic('N')


def test_injective_writes_are_proven():
    """``2*i`` and ``i*i`` touch a distinct element per iteration on ``[0, N]``."""
    assert smt_dependence.prove_injective_write(2 * I, 'i', 0, N, 1) is True
    assert smt_dependence.prove_injective_write(I * I, 'i', 0, N, 1) is True


def test_colliding_write_is_refused():
    """``i % 2`` maps every iteration onto two slots, so the oracle must NOT certify it."""
    assert smt_dependence.prove_injective_write(sp.Mod(I, 2), 'i', 0, N, 1) is not True


def test_read_behind_the_write_is_a_raw():
    """``A[i-1]`` read against ``A[i]`` written: iteration ``k`` reads what ``k-1`` wrote."""
    assert smt_dependence.classify_read_write_pair(I - 1, I, 'i', 0, N) == 'RAW'


def test_read_ahead_of_the_write_is_a_war():
    """``A[i+1]`` read against ``A[i]`` written: the alias only ever runs read-before-write."""
    assert smt_dependence.classify_read_write_pair(I + 1, I, 'i', 0, N) == 'WAR'


def test_disjoint_accesses_carry_nothing():
    """The read sits past the end of the write's range for every in-domain iteration."""
    assert smt_dependence.classify_read_write_pair(I + N + 1, I, 'i', 0, N) == 'none'


def test_a_guard_can_break_the_raw_chain():
    """Guarding the read with ``i > N`` empties it of in-domain iterations, so no RAW remains
    -- the shape the oracle exists for, where the guard and not the subscript decides."""
    guard = sp.StrictGreaterThan(I, N)
    assert smt_dependence.classify_read_write_pair(I - 1, I, 'i', 0, N, read_guard=guard) == 'none'


def test_an_untranslatable_expression_is_inconclusive():
    """An opaque function is not modeled; the caller must fall back, not get a verdict."""
    opaque = sp.Function('f')(I)
    assert smt_dependence.classify_read_write_pair(opaque, I, 'i', 0, N) is None
    assert smt_dependence.prove_injective_write(opaque, 'i', 0, N, 1) is not True


if __name__ == '__main__':
    test_injective_writes_are_proven()
    test_colliding_write_is_refused()
    test_read_behind_the_write_is_a_raw()
    test_read_ahead_of_the_write_is_a_war()
    test_disjoint_accesses_carry_nothing()
    test_a_guard_can_break_the_raw_chain()
    test_an_untranslatable_expression_is_inconclusive()

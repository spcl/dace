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


def test_read_ahead_excludes_the_readers_own_iteration():
    """``prove_read_ahead`` is what a snapshot rewrite needs, and no verdict from
    :func:`classify_read_write_pair` implies it -- not even the strongest one.

    ``A[i]`` read against ``A[i]`` written classifies ``'none'``: two DIFFERENT iterations never
    touch a common element. The read still aliases the write of its OWN iteration, so redirecting
    it to a pre-loop snapshot would hand it the stale original instead of the value that iteration
    just produced. Only the read-ahead question excludes that, and it does."""
    assert smt_dependence.classify_read_write_pair(I, I, 'i', 0, N) == 'none'
    assert smt_dependence.prove_read_ahead(I, I, 'i', 0, N) is not True
    assert smt_dependence.prove_read_ahead(I + 1, I, 'i', 0, N) is True


def test_read_ahead_reads_the_guard():
    """A non-affine subscript with no guard could land anywhere, including on an element an
    earlier iteration wrote; the same subscript under ``P > i`` cannot. The guard, not the
    subscript, is what makes the second one provable -- the ``A[IDX[i]*IDX[i-1]]`` shape."""
    p = symbolic.pystr_to_symbolic('P')
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N) is not True
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N, read_guard=sp.StrictGreaterThan(p, I)) is True


def test_dace_connectives_reach_the_solver():
    """``pystr_to_symbolic`` keeps ``and`` / ``or`` / ``not`` as DaCe function nodes rather than
    folding them to sympy's, so a guard collected off a ``ConditionalBlock`` arrives in that form.
    It used to fall through the translator to ``None``, which silently disarmed every
    guard-dependent proof -- the failure mode is a REFUSAL, so nothing ever flagged it."""
    guard = symbolic.pystr_to_symbolic('(P > i) and (P < N)')
    assert str(guard.func) == 'AND', guard
    p = symbolic.pystr_to_symbolic('P')
    assert smt_dependence.prove_read_ahead(p, I, 'i', 0, N, read_guard=guard) is True


if __name__ == '__main__':
    test_injective_writes_are_proven()
    test_colliding_write_is_refused()
    test_read_behind_the_write_is_a_raw()
    test_read_ahead_of_the_write_is_a_war()
    test_disjoint_accesses_carry_nothing()
    test_a_guard_can_break_the_raw_chain()
    test_an_untranslatable_expression_is_inconclusive()

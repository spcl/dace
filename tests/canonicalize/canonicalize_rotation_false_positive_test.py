# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shapes a loop-carried ROTATION rewrite must REFUSE.

Rotation substitution replaces reads of a carried scalar with a shifted array read
(``x == b[i-1]``), deletes the update, and peels the first iteration -- turning a sequential
loop into a DOALL. It is valid only when the carried value HAS that closed form.

Every kernel below is surface-identical to a rotation -- one scalar, written on every iteration,
which is why ``LoopToMap`` refuses at ``loop_to_map.py:671`` -- but has no shifted-read closed
form. Substituting anyway is a SILENT miscompile: the loop parallelizes and the numbers are
wrong, with no error raised.

These assert numerics against a sequential reference, so they are green today (nothing rewrites
them) and go red the moment a rotation pass over-fires. That is the point: a guard is only worth
having if it fails when the guard it guards is breached.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
_LEN = 64


def _run(program, **arrays):
    """Canonicalize and execute ``program``, returning its output array."""
    sdfg = program.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, validate_all=False)
    sdfg(N=_LEN, **arrays)
    return sdfg


# -- the carried value depends on itself: a REDUCTION, not a delay line -------------------


@dace.program
def accumulate(a: dace.float64[N], b: dace.float64[N]):
    x = 0.0
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = x + b[i]  # RHS names x -> no shifted-read closed form exists


def test_accumulation_is_not_a_rotation():
    b = np.arange(_LEN, dtype=np.float64) + 1.0
    got = np.zeros(_LEN)
    _run(accumulate, a=got, b=b.copy())

    want, x = np.empty(_LEN), 0.0
    for i in range(_LEN):
        want[i] = (b[i] + x) * 0.5
        x = x + b[i]
    # A rotation rewrite would substitute b[i-1] for x, computing (b[i]+b[i-1])*0.5.
    assert np.allclose(got, want, rtol=0, atol=0), 'an accumulation was rewritten as a rotation'


# -- read AFTER the write: x is b[i], not b[i-1] ------------------------------------------


@dace.program
def read_after_write(a: dace.float64[N], b: dace.float64[N]):
    x = 0.0
    for i in range(N):
        x = b[i]
        a[i] = (b[i] + x) * 0.5  # x is THIS iteration's value


def test_read_after_write_is_not_shifted():
    b = np.arange(_LEN, dtype=np.float64) + 1.0
    got = np.zeros(_LEN)
    _run(read_after_write, a=got, b=b.copy())
    # Off by one iteration if the rewrite assumes read-before-write.
    want = (b + b) * 0.5
    assert np.allclose(got, want, rtol=0, atol=0), 'a post-update read was shifted by one iteration'


# -- the update is CONDITIONAL: the carried value is the last taken write, not b[i-1] -----


@dace.program
def conditional_update(a: dace.float64[N], b: dace.float64[N]):
    x = 0.0
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        if b[i] > 8.0:
            x = b[i]  # only SOME iterations write -> x trails an unknown distance


def test_conditional_update_is_not_a_rotation():
    b = np.arange(_LEN, dtype=np.float64) + 1.0
    got = np.zeros(_LEN)
    _run(conditional_update, a=got, b=b.copy())

    want, x = np.empty(_LEN), 0.0
    for i in range(_LEN):
        want[i] = (b[i] + x) * 0.5
        if b[i] > 8.0:
            x = b[i]
    assert np.allclose(got, want, rtol=0, atol=0), 'a conditionally-updated carry was rewritten'


# -- the rotated source IS the written array: substitution changes which version is read ---


@dace.program
def in_place(a: dace.float64[N]):
    x = 0.0
    for i in range(N):
        old = a[i]
        a[i] = (a[i] + x) * 0.5
        x = old  # reads a[i] BEFORE this iteration overwrote it


def test_in_place_rotation_reads_the_pre_write_version():
    a = np.arange(_LEN, dtype=np.float64) + 1.0
    got = a.copy()
    _run(in_place, a=got)

    want, x = a.copy(), 0.0
    for i in range(_LEN):
        old = want[i]
        want[i] = (want[i] + x) * 0.5
        x = old
    # Substituting a[i-1] would read the value iteration i-1 WROTE, not the one it read.
    assert np.allclose(got, want, rtol=0, atol=0), 'in-place carry read the post-write version'


# -- the carry escapes the loop: deleting the update must still leave its final value ------


@dace.program
def carry_escapes(a: dace.float64[N], b: dace.float64[N], out: dace.float64[1]):
    x = 0.0
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = b[i]
    out[0] = x  # the value the sequential loop left behind


def test_carry_live_after_loop_is_materialised():
    b = np.arange(_LEN, dtype=np.float64) + 1.0
    got, out = np.zeros(_LEN), np.zeros(1)
    _run(carry_escapes, a=got, b=b.copy(), out=out)
    assert out[0] == b[_LEN - 1], 'the carried value was not materialised after the loop'


# -- a genuine rotation must NOT be lifted to an accumulation (WCR) ------------------------


@dace.program
def genuine_rotation(a: dace.float64[N], b: dace.float64[N]):
    x = b[N - 1]
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


def test_genuine_rotation_is_not_lifted_to_wcr():
    sdfg = genuine_rotation.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, validate_all=False)
    wcrs = [
        e.data.wcr for state in sdfg.states() for e in state.edges() if e.data is not None and e.data.wcr is not None
    ]
    assert not wcrs, f'rotation lifted to a reduction: {wcrs}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

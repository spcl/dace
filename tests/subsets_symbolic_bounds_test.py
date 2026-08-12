# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Range`` must hold symbolic bounds no matter which path writes them."""

import pytest

from dace import subsets, symbolic


def test_setitem_keeps_bounds_symbolic():
    """A raw Python ``int`` assigned through ``__setitem__`` must not survive as one.

    ``Range.__init__`` coerces every bound, ``ndrange()`` is annotated ``SymbolicType``, and
    callers act on that: ``LoopToMap._check_range`` calls ``.match()`` on a bound, other passes
    call ``.subs()`` or ``.free_symbols``. When the write path stored the tuple verbatim, an
    ``int`` bound raised ``'int' object has no attribute 'match'`` far from the assignment that
    caused it -- and ``LoopToMap`` swallows that as a warning, so the loop silently stayed
    sequential instead of being refused for a stated reason.
    """
    rng = subsets.Range([('i', 'i', 1)])

    # The property callers depend on is the sympy interface, not "has free symbols":
    # ``issymbolic(Integer(2))`` is False, yet ``Integer(2).match(...)`` is exactly what
    # ``_check_range`` needs and a Python ``int`` cannot do.
    rng[0] = (2, 5, 1)
    for bound in rng.ndrange()[0]:
        assert not isinstance(bound, int), f'{bound!r} is a raw int'
        bound.match(symbolic.pystr_to_symbolic('__unused'))

    # Slice assignment goes through the same path.
    rng[0:1] = [(0, 7, 1)]
    for bound in rng.ndrange()[0]:
        assert not isinstance(bound, int)
        bound.match(symbolic.pystr_to_symbolic('__unused'))

    # Strings and already-symbolic values keep working.
    rng[0] = ('N', symbolic.pystr_to_symbolic('M + 1'), 1)
    start, end, _ = rng.ndrange()[0]
    assert str(start) == 'N' and str(end) == 'M + 1'

    # A malformed range is caught where it is written, not frames later.
    with pytest.raises(ValueError):
        rng[0] = (1, 2)


if __name__ == '__main__':
    test_setitem_keeps_bounds_symbolic()

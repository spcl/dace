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
    rng = subsets.Range([("i", "i", 1)])

    # The property callers depend on is the sympy interface, not "has free symbols":
    # ``issymbolic(Integer(2))`` is False, yet ``Integer(2).match(...)`` is exactly what
    # ``_check_range`` needs and a Python ``int`` cannot do.
    rng[0] = (2, 5, 1)
    for bound in rng.ndrange()[0]:
        assert not isinstance(bound, int), f"{bound!r} is a raw int"
        bound.match(symbolic.pystr_to_symbolic("__unused"))

    # Slice assignment goes through the same path.
    rng[0:1] = [(0, 7, 1)]
    for bound in rng.ndrange()[0]:
        assert not isinstance(bound, int)
        bound.match(symbolic.pystr_to_symbolic("__unused"))

    # Strings and already-symbolic values keep working.
    rng[0] = ("N", symbolic.pystr_to_symbolic("M + 1"), 1)
    start, end, _ = rng.ndrange()[0]
    assert str(start) == "N" and str(end) == "M + 1"

    # A malformed range is caught where it is written, not frames later.
    with pytest.raises(ValueError):
        rng[0] = (1, 2)


def test_replace_renames_a_subscripted_container():
    """``Subscript.free_symbols`` leaves out the container, so ``replace`` must still reach ``idx`` in ``idx[i]``."""
    for old, new in ((symbolic.symbol("idx"), symbolic.symbol("jdx")), ("idx", "jdx")):
        rng = subsets.Range.from_string("idx[i]:idx[i] + 4")
        rng.replace({old: new})
        start, end, step = rng.ndrange()[0]
        assert symbolic.arrays(start) == symbolic.arrays(end) == {"jdx"}


def test_replace_matches_a_full_subs():
    """Dropping unreachable entries must not change the result, including entries reached only through the value of
    another entry, which ``subs`` may apply in either order."""
    cases = ({"N": "M", "M": "K", "Z": "0"}, {"M": "K", "N": "M"}, {"N": "M + P", "P": "2"}, {"Z": "0"})
    for case in cases:
        repl = {symbolic.symbol(k): symbolic.pystr_to_symbolic(v) for k, v in case.items()}
        rng = subsets.Range.from_string("0:N, 0:P")
        expected = [tuple(b.subs(repl) for b in bounds) for bounds in rng.ndrange()]
        rng.replace(repl)
        assert rng.ndrange() == expected, case


if __name__ == "__main__":
    test_setitem_keeps_bounds_symbolic()
    test_replace_renames_a_subscripted_container()
    test_replace_matches_a_full_subs()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""In-memory registry of runtime assumptions a canonicalization pass relied on.

Some parallelization rewrites are only value-preserving under a relation the
compiler cannot prove but the caller is expected to honour. Splitting a
modular-wrap write ``a[(i + K) % N]`` into two affine halves, for instance, is
sound only when the wrap offset is smaller than the modulus (``K < N``): that is
what makes the far segment's index fold to ``i + K - N`` land in ``[0, N - 1]``.
Rather than bake such a relation in silently, the rewriting pass RECORDS it here
and the terminal :class:`AssumeSymbolConstraints` pass emits a runtime trap that
aborts if it is violated -- the same abort-only discipline as the
nonnegative-symbol guard (see [[feedback_symbols_nonnegative_canonicalization]]).

The registry lives on the SDFG object (a plain attribute, not a serialized
property): an assumption is recorded and emitted within a single in-memory
canonicalize run, so it never needs to survive a save/load. A relation is a
sympy ``Boolean`` (e.g. ``StrictLessThan(K, N)``) over the SDFG's free symbols.
"""
from typing import List

from dace import symbolic
from dace.sdfg import SDFG

#: Attribute under which recorded assumptions hang off the SDFG's ``__dict__``.
_ATTR = '_tracked_assumptions'


def record_assumption(sdfg: SDFG, relation) -> None:
    """Record ``relation`` (a sympy boolean that must hold at runtime) on ``sdfg``.

    Deduped, and a relation that simplifies to a constant ``True`` is dropped (it
    always holds, so there is nothing to guard). A relation that simplifies to a
    constant ``False`` is kept: it means the rewrite that recorded it is unsound
    for every input, which the emitted always-trapping guard makes loud rather
    than silent.
    """
    relation = symbolic.simplify(relation)
    if relation == True:  # noqa: E712 -- sympy ``S.true`` compares equal to ``True``
        return
    store = vars(sdfg).get(_ATTR)
    if store is None:
        store = []
        vars(sdfg)[_ATTR] = store
    if relation not in store:
        store.append(relation)


def tracked_assumptions(sdfg: SDFG) -> List:
    """The relations recorded on ``sdfg`` via :func:`record_assumption` (a copy)."""
    return list(vars(sdfg).get(_ATTR, ()))


__all__ = ['record_assumption', 'tracked_assumptions']

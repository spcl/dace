# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""In-memory registry of runtime assumptions a canonicalization pass relied on."""

from dace import symbolic
from dace.sdfg import SDFG

#: Attribute under which recorded assumptions hang off the SDFG's ``__dict__``.
ATTR = "_tracked_assumptions"


def record_assumption(sdfg: SDFG, relation: symbolic.SymbolicType) -> None:
    """Record ``relation`` (a sympy boolean that must hold at runtime) on ``sdfg``."""
    relation = symbolic.simplify(relation)
    if relation == True:  # noqa: E712 -- sympy ``S.true`` compares equal to ``True``
        return
    store = vars(sdfg).get(ATTR)
    if store is None:
        store = []
        vars(sdfg)[ATTR] = store
    if relation not in store:
        store.append(relation)


def tracked_assumptions(sdfg: SDFG) -> list[symbolic.SymbolicType]:
    """The relations recorded on ``sdfg`` via :func:`record_assumption` (a copy)."""
    return list(vars(sdfg).get(ATTR, ()))


__all__ = ["record_assumption", "tracked_assumptions"]

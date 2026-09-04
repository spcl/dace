# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe's ``OrderedSet``: insertion-ordered iteration, but set equality."""
from typing import Any

from ordered_set import OrderedSet as SequenceOrderedSet


class OrderedSet(SequenceOrderedSet):
    """``ordered_set.OrderedSet`` with ``==`` comparing membership instead of order.

    Deterministic iteration is why DaCe uses this container everywhere: insertion order stands
    in for a canonical order so codegen does not move with ``PYTHONHASHSEED``. That order is a
    property of how a set was BUILT, not of what it holds, so two sets carrying the same nodes
    are the same set. Upstream disagrees -- it is a ``Sequence``, so ``==`` between two of them
    compares element order -- which turned invariant checks into assertions on the order the
    two sides happened to be accumulated in.

    Comparison against a genuine sequence keeps the upstream order-sensitive meaning: a ``list``
    is ordered by nature, and code comparing against one is asking about order.
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SequenceOrderedSet):
            return len(self) == len(other) and all(item in self for item in other)
        return super().__eq__(other)

    # Python clears the inherited `__hash__` on any class that defines `__eq__`; the base is
    #  already unhashable (it is mutable), so state that rather than leaving it implicit.
    __hash__ = None

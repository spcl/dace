# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe's ``OrderedSet``: insertion-ordered iteration, but set equality."""
from typing import Any, Iterable

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

    def update(self, sequence: Iterable[Any]) -> int:
        """Upstream ``update`` (same order, same returned index) without a method call per element.

        Merging one ordered set into another is the hot case -- block reachability unions whole
        closures per block -- so a set of the same family is merged by its index map in C; any
        other iterable takes one local loop instead of ``add`` per item."""
        mapping = self.map
        items = self.items
        if type(sequence) in ORDERED_SET_TYPES:
            if not sequence.items:
                return 0
            if not items:
                items.extend(sequence.items)
                mapping.update(sequence.map)
                return len(items) - 1
            # Scan the incoming items only: a keys-view difference walks the (large) target map as well.
            appended = [item for item in sequence.items if item not in mapping]
            if appended:
                mapping.update(zip(appended, range(len(items), len(items) + len(appended))))
                items.extend(appended)
            return mapping[sequence.items[-1]]
        item_index = 0
        try:
            for item in sequence:
                index = mapping.get(item)
                if index is None:
                    index = len(items)
                    mapping[item] = index
                    items.append(item)
                item_index = index
        except TypeError:
            raise ValueError("Argument needs to be an iterable, got %s" % type(sequence))
        return item_index


#: Exact types whose index map ``OrderedSet.update`` merges directly. A type test, not ``isinstance``: the
#: base is an ABC, and its ``__instancecheck__`` costs more than the merge it guards. Other subclasses take the
#: element loop, which is the same result.
ORDERED_SET_TYPES = (OrderedSet, SequenceOrderedSet)

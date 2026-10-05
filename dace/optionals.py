# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Narrowing of optional values; a leaf module so that every layer of DaCe can import it."""
from typing import TypeVar

T = TypeVar('T')


def required(value: T | None) -> T:
    """``value`` with ``None`` excluded from its type.

    Marks the places where DaCe's optional fields (``Memlet.data``, ``Memlet.subset``, scope lookups, ...) are known
    to be set; a ``None`` fails right here instead of at the first attribute access.

    :param value: A value that is not ``None`` at this point.
    :returns: The same object.
    :raises TypeError: If ``value`` is ``None``.
    """
    if value is None:
        raise TypeError('expected a value, got None')
    return value

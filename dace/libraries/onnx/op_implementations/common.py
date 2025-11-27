# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Common utilities and helper functions for ONNX pure implementations.
"""


def iterables_equal(a, b) -> bool:
    """ Return whether the two iterables ``a`` and ``b`` are equal. """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))

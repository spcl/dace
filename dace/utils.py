# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility functions for DaCe.

This module provides general utility functions that are used across various parts of DaCe.
"""

import math
from typing import Iterable, Sequence, Union

import sympy

# Type alias for numeric or symbolic values
NumericType = Union[int, float, sympy.Basic]


def prod(sequence: Iterable[NumericType], start: NumericType = 1) -> NumericType:
    """
    Computes the product of a sequence of numbers or symbolic expressions.

    This function handles both numeric values and SymPy symbolic expressions,
    making it suitable for use with DaCe's symbolic shape calculations.

    :param sequence: An iterable of numbers or symbolic expressions.
    :param start: The starting value for the product (default: 1).
    :return: The product of all elements in the sequence, multiplied by start.
             Returns start if the sequence is empty.
    """
    result = start
    for item in sequence:
        result = result * item
    return result


def find_new_name(name: str, existing_names: Sequence[str]) -> str:
    """
    Returns a name that matches the given ``name`` as a prefix, but does not
    already exist in the given existing name set. The behavior is typically
    to append an underscore followed by a unique (increasing) number. If the
    name does not already exist in the set, it is returned as-is.

    :param name: The given name to find.
    :param existing_names: The set of existing names.
    :return: A new name that is not in existing_names.
    """
    if name not in existing_names:
        return name
    cur_offset = 0
    new_name = name + '_' + str(cur_offset)
    while new_name in existing_names:
        cur_offset += 1
        new_name = name + '_' + str(cur_offset)
    return new_name


def deduplicate(iterable):
    """ Removes duplicates in the passed iterable. """
    return type(iterable)([i for i in sorted(set(iterable), key=lambda x: iterable.index(x))])

# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Utility functions for DaCe.

This module provides general utility functions that are used across various parts of DaCe.
"""

import functools
from typing import Sequence


def prod(sequence):
    """
    Computes the product of a sequence of numbers.

    :param sequence: A sequence of numbers.
    :return: The product of all elements in the sequence.
    """
    return functools.reduce(lambda a, b: a * b, sequence, 1)


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

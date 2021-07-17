# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A single generator that creates an N-dimensional for loop in Python. """
import itertools
import numpy as np
from typing import List, Tuple, Union

# Python 3 compatibility for xrange
try:
    xxrange = xrange
except NameError:
    xxrange = range


def slicetoxrange(s):
    """ Helper function that turns a slice into a range (for iteration). """
    if isinstance(s, int):
        return xxrange(s, s + 1)

    ifnone = lambda a, b: b if a is None else a

    return xxrange(ifnone(s.start, 0), s.stop, ifnone(s.step, 1))


def NDLoop(ndslice, internal_function, *args, **kwargs):
    """ Wrapped generator that calls an internal function in an N-dimensional 
        for-loop in Python. 
        :param ndslice: Slice or list of slices (`slice` objects) to loop over.
        :param internal_function: Function to call in loop.
        :param *args: Arguments to `internal_function`.
        :param **kwargs: Keyword arguments to `internal_function`.
        :return: N-dimensional loop index generator.
    """
    if isinstance(ndslice, int) or isinstance(ndslice, slice):
        ndxrange = (slicetoxrange(ndslice), )
    else:
        ndxrange = tuple(slicetoxrange(d) for d in ndslice)
    for indices in itertools.product(*ndxrange):
        internal_function(*(indices + args), **kwargs)


def ndrange(slice_list: Union[Tuple[slice], slice]):
    """ Generator that creates an N-dimensional for loop in Python. 
        :param slice_list: Slice or list of slices (as tuples or `slice`s)
                          to loop over.
        :return: N-dimensional loop index generator.
    """
    if not isinstance(slice_list, (tuple, list)):
        yield from slicetoxrange(slice_list)
    else:
        ndxrange = tuple(slicetoxrange(d) for d in slice_list)
        for indices in itertools.product(*ndxrange):
            yield indices

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Types and wrappers used in DaCe's Python frontend. """
from __future__ import print_function
import numpy
import itertools
from collections import deque
from typing import Deque, Generic, Type, TypeVar

from dace import dtypes, symbolic
T = TypeVar('T')


def ndarray(shape, dtype=numpy.float64, *args, **kwargs):
    """ Returns a numpy ndarray where all symbols have been evaluated to
        numbers and types are converted to numpy types. """
    repldict = {sym: sym.get() for sym in symbolic.symlist(shape).values()}
    new_shape = [
        int(s.subs(repldict) if symbolic.issymbolic(s) else s) for s in shape
    ]
    new_dtype = dtype.type if isinstance(dtype, dtypes.typeclass) else dtype
    return numpy.ndarray(shape=new_shape, dtype=new_dtype, *args, **kwargs)

stream: Type[Deque[T]] = deque

class stream_array(Generic[T]):
    """ Stream array object in Python. """
    def __init__(self, dtype, shape):
        from dace import data

        self._type = dtype
        self._shape = shape
        self.descriptor = data.Stream(dtype, 0, shape, True)
        self.queue_array = numpy.ndarray(shape, dtype=deque)
        for i in itertools.product(*(range(s) for s in shape)):
            self.queue_array[i] = stream()

    @property
    def shape(self):
        return self.shape

    def __getitem__(self, key) -> Deque[T]:
        return self.queue_array.__getitem__(key)

    def __getslice__(self, *args) -> Deque[T]:
        return self.queue_array.__getslice__(*args)



def scalar(dtype=dtypes.float32):
    """ Convenience function that defines a scalar (array of size 1). """
    return ndarray([1], dtype)


def define_local(dimensions, dtype=dtypes.float32):
    """ Defines a transient array in a DaCe program. """
    return ndarray(dimensions, dtype=dtype)


def define_local_scalar(dtype=dtypes.float32):
    """ Defines a transient scalar (array of size 1) in a DaCe program. """
    return ndarray([1], dtype=dtype)


def define_stream(dtype=dtypes.float32, buffer_size=1):
    """ Defines a local stream in a DaCe program. """
    return define_streamarray([1], dtype=dtype, buffer_size=buffer_size)


def define_streamarray(dimensions, dtype=dtypes.float32, buffer_size=1):
    """ Defines a local stream array in a DaCe program. """
    return stream_array(dtype, dimensions)

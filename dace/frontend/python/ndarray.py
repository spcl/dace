""" Array types and wrappers used in DaCe's Python frontend. """
from __future__ import print_function
import ctypes
import enum
import inspect
import numpy
import itertools
from collections import deque

from dace import symbolic, types

###########################################################
# NDArray type


class ndarray(numpy.ndarray):
    """ An N-dimensional array wrapper around `numpy.ndarray` that enables 
        symbolic sizes. """

    def __new__(cls,
                shape,
                dtype=types.float32,
                materialize_func=None,
                allow_conflicts=False,
                *args,
                **kwargs):
        """ Initializes a DaCe ND-array.
            @param shape: The array shape (may contain symbols).
            @param dtype: The array data type.
            @param materialize_func: An optional string that contains a method
                                     to materialize array contents on demand.
                                     If not None, the array is not allocated 
                                     within the DaCe program.
            @param allow_conflicts: If True, suppresses warnings on conflicting
                                    array writes in DaCe programs without a 
                                    matching conflict resolution memlet.
        """
        # Avoiding import loops
        from dace import data

        tmpshape = shape
        shape = [symbolic.eval(s, 0) for s in shape]

        kwargs.update({'dtype': dtype.type})

        res = numpy.ndarray.__new__(cls, shape, *args, **kwargs)
        res._symlist = symbolic.symlist(tmpshape)
        for _, sym in res._symlist.items():
            sym._arrays_to_update.append(res)

        if not isinstance(dtype, types.typeclass):
            dtype = types.typeclass(dtype.type)

        res.descriptor = data.Array(
            dtype,
            tmpshape,
            materialize_func=materialize_func,
            transient=False,
            allow_conflicts=allow_conflicts)
        return res

    def update_resolved_symbol(self, sym):
        """ Notifies an array that a symbol has been resolved so that it
            can be resized. """
        self.resize(
            [symbolic.eval(s, 0) for s in self.descriptor.shape],
            refcheck=False)
        self._symlist = symbolic.symlist(self.descriptor.shape)

    def missing_syms(self):
        return ','.join(
            [s for s, v in self._symlist.items() if not v.is_initialized()])

    def __setitem__(self, key, value):
        if self.descriptor.materialize_func is not None:
            raise PermissionError(
                "You cannot write into an Immaterial storage.")
        return numpy.ndarray.__setitem__(self, key, value)

    def __getitem__(self, key):
        if 0 in self.shape:
            self.update_resolved_symbol(None)
        if 0 in self.shape:
            raise IndexError(
                'Cannot create sub-array, not all symbols are set " "(missing symbols: %s)'
                % self.missing_syms())
        return numpy.ndarray.__getitem__(self, key)

    # Python 2.x compatibility
    def __getslice__(self, *args):
        if 0 in self.shape:
            raise IndexError(
                'Cannot create sub-array, not all symbols are set (missing symbols: %s)'
                % self.missing_syms())
        return numpy.ndarray.__getslice__(self, *args)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        from dace import data

        # Create a new descriptor
        self.descriptor = data.Array(
            types.typeclass(obj.dtype.type),
            obj.shape,
            materialize_func=None,
            transient=False,
            allow_conflicts=False)

        self._symlist = {}

    def __lshift__(self, other):
        pass

    def __rshift__(self, other):
        pass

    def __hash__(self):
        return hash(self.data.tobytes())

    def __call__(self, *args):
        return self


class transient(ndarray):
    """ Transient DaCe array subclass. """

    def __new__(cls, *args, **kwargs):
        res = ndarray.__new__(cls, *args, **kwargs)
        res.descriptor.transient = True
        return res


class stream(object):
    """ Stream array object in Python. Mostly used in the Python SDFG 
        simulator. """

    def __init__(self, dtype, shape):
        from dace import data

        self._type = dtype
        self._shape = shape
        self.descriptor = data.Stream(dtype, 1, 0, shape, True)
        self.queue_array = numpy.ndarray(shape, dtype=deque)
        for i in itertools.product(*(range(s) for s in shape)):
            self.queue_array[i] = deque()

    @property
    def shape(self):
        return self.shape

    def __getitem__(self, key):
        return self.queue_array.__getitem__(key)

    def __getslice__(self, *args):
        return self.queue_array.__getslice__(*args)


def scalar(dtype=types.float32, allow_conflicts=False):
    """ Convenience function that defines a scalar (array of size 1). """
    return ndarray([1], dtype, allow_conflicts=allow_conflicts)


def define_local(dimensions, dtype=types.float32, allow_conflicts=False):
    """ Defines a transient array in a DaCe program. """
    return transient(dimensions, dtype=dtype, allow_conflicts=allow_conflicts)


def define_local_scalar(dtype=types.float32, allow_conflicts=False):
    """ Defines a transient scalar (array of size 1) in a DaCe program. """
    return transient([1], dtype=dtype, allow_conflicts=allow_conflicts)


def define_stream(dtype=types.float32, buffer_size=0):
    """ Defines a local stream in a DaCe program. """
    return define_streamarray([1], dtype=dtype, buffer_size=buffer_size)


def define_streamarray(dimensions, dtype=types.float32, buffer_size=0):
    """ Defines a local stream array in a DaCe program. """
    return stream(dtype, dimensions)


def asarray(array):
    """ Converts an existing Numpy NDArray to DaCe NDArray. """
    obj = numpy.asarray(array).view(ndarray)
    return obj

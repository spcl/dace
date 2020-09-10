# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import re, json
import copy as cp
import sympy as sp
import numpy
from typing import Set

import dace.dtypes as dtypes
from dace.codegen import cppunparse
from dace import symbolic, serialize
from dace.properties import (Property, make_properties, DictProperty,
                             ReferenceProperty, ShapeProperty, SubsetProperty,
                             SymbolicProperty, TypeClassProperty,
                             DebugInfoProperty, CodeProperty, ListProperty)


def create_datadescriptor(obj):
    """ Creates a data descriptor from various types of objects.
        @see: dace.data.Data
    """
    from dace import dtypes  # Avoiding import loops
    if isinstance(obj, Data):
        return obj

    try:
        return obj.descriptor
    except AttributeError:
        if isinstance(obj, numpy.ndarray):
            return Array(dtype=dtypes.typeclass(obj.dtype.type),
                         shape=obj.shape)
        if symbolic.issymbolic(obj):
            return Scalar(symbolic.symtype(obj))
        if isinstance(obj, dtypes.typeclass):
            return Scalar(obj)
        return Scalar(dtypes.typeclass(type(obj)))


@make_properties
class Data(object):
    """ Data type descriptors that can be used as references to memory.
        Examples: Arrays, Streams, custom arrays (e.g., sparse matrices).
    """

    dtype = TypeClassProperty(default=dtypes.int32)
    shape = ShapeProperty(default=[])
    transient = Property(dtype=bool, default=False)
    storage = Property(dtype=dtypes.StorageType,
                       desc="Storage location",
                       choices=dtypes.StorageType,
                       default=dtypes.StorageType.Default,
                       from_string=lambda x: dtypes.StorageType[x])
    lifetime = Property(dtype=dtypes.AllocationLifetime,
                        desc='Data allocation span',
                        choices=dtypes.AllocationLifetime,
                        default=dtypes.AllocationLifetime.Scope,
                        from_string=lambda x: dtypes.AllocationLifetime[x])
    location = DictProperty(
        key_type=str,
        value_type=symbolic.pystr_to_symbolic,
        desc='Full storage location identifier (e.g., rank, GPU ID)')
    debuginfo = DebugInfoProperty(allow_none=True)

    def __init__(self, dtype, shape, transient, storage, location, lifetime,
                 debuginfo):
        self.dtype = dtype
        self.shape = shape
        self.transient = transient
        self.storage = storage
        self.location = location if location is not None else {}
        self.lifetime = lifetime
        self.debuginfo = debuginfo
        self._validate()

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol,
                                  symbolic.sympy.Basic)) for s in self.shape):
            raise TypeError('Shape must be a list or tuple of integer values '
                            'or symbols')
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @property
    def toplevel(self):
        return self.lifetime is not dtypes.AllocationLifetime.Scope

    def copy(self):
        raise RuntimeError(
            'Data descriptors are unique and should not be copied')

    def is_equivalent(self, other):
        """ Check for equivalence (shape and type) of two data descriptors. """
        raise NotImplementedError

    def as_arg(self, with_types=True, for_call=False, name=None):
        """Returns a string for a C++ function signature (e.g., `int *A`). """
        raise NotImplementedError

    @property
    def free_symbols(self) -> Set[symbolic.SymbolicType]:
        """ Returns a set of undefined symbols in this data descriptor. """
        result = set()
        for s in self.shape:
            if isinstance(s, sp.Basic):
                result |= set(s.free_symbols)
        return result

    def __repr__(self):
        return 'Abstract Data Container, DO NOT USE'

    @property
    def veclen(self):
        return self.dtype.veclen if hasattr(self.dtype, "veclen") else 1

    
    @property
    def ctype(self):
        return self.dtype.ctype


@make_properties
class Scalar(Data):
    """ Data descriptor of a scalar value. """

    allow_conflicts = Property(dtype=bool, default=False)

    def __init__(self,
                 dtype,
                 transient=False,
                 storage=dtypes.StorageType.Default,
                 allow_conflicts=False,
                 location=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):
        self.allow_conflicts = allow_conflicts
        shape = [1]
        super(Scalar, self).__init__(dtype, shape, transient, storage, location,
                                     lifetime, debuginfo)

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Scalar":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Scalar(dtypes.int8)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        # Check validity now
        ret.validate()
        return ret

    def __repr__(self):
        return 'Scalar (dtype=%s)' % self.dtype

    def clone(self):
        return Scalar(self.dtype, self.transient, self.storage,
                      self.allow_conflicts, self.location, self.lifetime,
                      self.debuginfo)

    @property
    def strides(self):
        return [1]

    @property
    def total_size(self):
        return 1

    @property
    def offset(self):
        return [0]

    def is_equivalent(self, other):
        if not isinstance(other, Scalar):
            return False
        if self.dtype != other.type:
            return False
        return True

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call:
            return name
        return self.dtype.as_arg(name)

    def sizes(self):
        return None

    def covers_range(self, rng):
        if len(rng) != 1:
            return False

        rng = rng[0]

        try:
            if (rng[1] - rng[0]) > rng[2]:
                return False
        except TypeError:  # cannot determine truth value of Relational
            pass
            #print('WARNING: Cannot evaluate relational expression %s, assuming true.' % ((rng[1] - rng[0]) > rng[2]),
            #      'If this expression is false, please refine symbol definitions in the program.')

        return True


def _prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


@make_properties
class Array(Data):
    """ Array/constant descriptor (dimensions, type and other properties). """

    # Properties
    allow_conflicts = Property(
        dtype=bool,
        default=False,
        desc='If enabled, allows more than one '
        'memlet to write to the same memory location without conflict '
        'resolution.')

    strides = ShapeProperty(
        # element_type=symbolic.pystr_to_symbolic,
        desc='For each dimension, the number of elements to '
        'skip in order to obtain the next element in '
        'that dimension.')

    total_size = SymbolicProperty(
        default=1,
        desc='The total allocated size of the array. Can be used for'
        ' padding.')

    offset = ListProperty(element_type=symbolic.pystr_to_symbolic,
                          desc='Initial offset to translate all indices by.')

    may_alias = Property(dtype=bool,
                         default=False,
                         desc='This pointer may alias with other pointers in '
                         'the same function')

    alignment = Property(dtype=int,
                         default=0,
                         desc='Allocation alignment in bytes (0 uses '
                         'compiler-default)')

    def __init__(self,
                 dtype,
                 shape,
                 transient=False,
                 allow_conflicts=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 strides=None,
                 offset=None,
                 may_alias=False,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 alignment=0,
                 debuginfo=None,
                 total_size=None):

        super(Array, self).__init__(dtype, shape, transient, storage, location,
                                    lifetime, debuginfo)

        if shape is None:
            raise IndexError('Shape must not be None')

        self.allow_conflicts = allow_conflicts
        self.may_alias = may_alias
        self.alignment = alignment

        if strides is not None:
            self.strides = cp.copy(strides)
        else:
            self.strides = [_prod(shape[i + 1:]) for i in range(len(shape))]

        self.total_size = total_size or _prod(shape)

        if offset is not None:
            self.offset = cp.copy(offset)
        else:
            self.offset = [0] * len(shape)

        self.validate()

    def __repr__(self):
        return 'Array (dtype=%s, shape=%s)' % (self.dtype, self.shape)

    def clone(self):
        return Array(self.dtype, self.shape, self.transient,
                     self.allow_conflicts, self.storage, self.location,
                     self.strides, self.offset, self.may_alias, self.lifetime,
                     self.alignment, self.debuginfo, self.total_size)

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        # Take care of symbolic expressions
        attrs['strides'] = list(map(str, attrs['strides']))

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Array":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Array(dtypes.int8, ())
        serialize.set_properties_from_json(ret, json_obj, context=context)
        # TODO: This needs to be reworked (i.e. integrated into the list property)
        ret.strides = list(map(symbolic.pystr_to_symbolic, ret.strides))

        # Check validity now
        ret.validate()
        return ret

    def validate(self):
        super(Array, self).validate()
        if len(self.strides) != len(self.shape):
            raise TypeError('Strides must be the same size as shape')

        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol,
                                  symbolic.sympy.Basic)) for s in self.strides):
            raise TypeError('Strides must be a list or tuple of integer '
                            'values or symbols')

        if len(self.offset) != len(self.shape):
            raise TypeError('Offset must be the same size as shape')

    def covers_range(self, rng):
        if len(rng) != len(self.shape):
            return False

        for s, (rb, re, rs) in zip(self.shape, rng):
            # Shape has to be positive
            if isinstance(s, sp.Basic):
                olds = s
                if 'positive' in s.assumptions0:
                    s = sp.Symbol(str(s), **s.assumptions0)
                else:
                    s = sp.Symbol(str(s), positive=True, **s.assumptions0)
                if isinstance(rb, sp.Basic):
                    rb = rb.subs({olds: s})
                if isinstance(re, sp.Basic):
                    re = re.subs({olds: s})
                if isinstance(rs, sp.Basic):
                    rs = rs.subs({olds: s})

            try:
                if rb < 0:  # Negative offset
                    return False
            except TypeError:  # cannot determine truth value of Relational
                pass
                #print('WARNING: Cannot evaluate relational expression %s, assuming true.' % (rb > 0),
                #      'If this expression is false, please refine symbol definitions in the program.')
            try:
                if re > s:  # Beyond shape
                    return False
            except TypeError:  # cannot determine truth value of Relational
                pass
                #print('WARNING: Cannot evaluate relational expression %s, assuming true.' % (re < s),
                #      'If this expression is false, please refine symbol definitions in the program.')

        return True

    # Checks for equivalent shape and type
    def is_equivalent(self, other):
        if not isinstance(other, Array):
            return False

        # Test type
        if self.dtype != other.dtype:
            return False

        # Test dimensionality
        if len(self.shape) != len(other.shape):
            return False

        # Test shape
        for dim, otherdim in zip(self.shape, other.shape):
            # Any other case (constant vs. constant), check for equality
            if otherdim != dim:
                return False
        return True

    def as_arg(self, with_types=True, for_call=False, name=None):
        arrname = name

        if not with_types or for_call:
            return arrname
        if self.may_alias:
            return str(self.dtype.ctype) + ' *' + arrname
        return str(self.dtype.ctype) + ' * __restrict__ ' + arrname

    def sizes(self):
        return [
            d.name if isinstance(d, symbolic.symbol) else str(d)
            for d in self.shape
        ]

    @property
    def free_symbols(self):
        result = super().free_symbols
        for s in self.strides:
            if isinstance(s, sp.Expr):
                result |= set(s.free_symbols)
        if isinstance(self.total_size, sp.Expr):
            result |= set(self.total_size.free_symbols)
        for o in self.offset:
            if isinstance(o, sp.Expr):
                result |= set(o.free_symbols)

        return result


@make_properties
class Stream(Data):
    """ Stream (or stream array) data descriptor. """

    # Properties
    offset = ListProperty(element_type=symbolic.pystr_to_symbolic)
    buffer_size = SymbolicProperty(desc="Size of internal buffer.", default=0)

    def __init__(self,
                 dtype,
                 buffer_size,
                 shape=None,
                 transient=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 offset=None,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 debuginfo=None):

        if shape is None:
            shape = (1, )

        self.buffer_size = buffer_size

        if offset is not None:
            if len(offset) != len(shape):
                raise TypeError('Offset must be the same size as shape')
            self.offset = cp.copy(offset)
        else:
            self.offset = [0] * len(shape)

        super(Stream, self).__init__(dtype, shape, transient, storage, location,
                                     lifetime, debuginfo)

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Stream":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Stream(dtypes.int8, 1)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        # Check validity now
        ret.validate()
        return ret

    def __repr__(self):
        return 'Stream (dtype=%s, shape=%s)' % (self.dtype, self.shape)

    @property
    def total_size(self):
        return _prod(self.shape)

    @property
    def strides(self):
        return [_prod(self.shape[i + 1:]) for i in range(len(self.shape))]

    def clone(self):
        return Stream(self.dtype, self.buffer_size, self.shape, self.transient,
                      self.storage, self.location, self.offset, self.lifetime,
                      self.debuginfo)

    # Checks for equivalent shape and type
    def is_equivalent(self, other):
        if not isinstance(other, Stream):
            return False

        # Test type
        if self.dtype != other.dtype:
            return False

        # Test dimensionality
        if len(self.shape) != len(other.shape):
            return False

        # Test shape
        for dim, otherdim in zip(self.shape, other.shape):
            if dim != otherdim:
                return False
        return True

    def as_arg(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call: return name
        if self.storage in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared
        ]:
            return 'dace::GPUStream<%s, %s> %s' % (str(
                self.dtype.ctype), 'true' if sp.log(
                    self.buffer_size, 2).is_Integer else 'false', name)

        return 'dace::Stream<%s> %s' % (str(self.dtype.ctype), name)

    def sizes(self):
        return [
            d.name if isinstance(d, symbolic.symbol) else str(d)
            for d in self.shape
        ]

    def size_string(self):
        return (" * ".join(
            [cppunparse.pyexpr2cpp(symbolic.symstr(s)) for s in self.shape]))

    def is_stream_array(self):
        return _prod(self.shape) != 1

    def covers_range(self, rng):
        if len(rng) != len(self.shape):
            return False

        for s, (rb, re, rs) in zip(self.shape, rng):
            # Shape has to be positive
            if isinstance(s, sp.Basic):
                olds = s
                if 'positive' in s.assumptions0:
                    s = sp.Symbol(str(s), **s.assumptions0)
                else:
                    s = sp.Symbol(str(s), positive=True, **s.assumptions0)
                if isinstance(rb, sp.Basic):
                    rb = rb.subs({olds: s})
                if isinstance(re, sp.Basic):
                    re = re.subs({olds: s})
                if isinstance(rs, sp.Basic):
                    rs = rs.subs({olds: s})

            try:
                if rb < 0:  # Negative offset
                    return False
            except TypeError:  # cannot determine truth value of Relational
                pass
                #print('WARNING: Cannot evaluate relational expression %s, assuming true.' % (rb > 0),
                #      'If this expression is false, please refine symbol definitions in the program.')
            try:
                if re > s:  # Beyond shape
                    return False
            except TypeError:  # cannot determine truth value of Relational
                pass
                #print('WARNING: Cannot evaluate relational expression %s, assuming true.' % (re < s),
                #      'If this expression is false, please refine symbol definitions in the program.')

        return True

    @property
    def free_symbols(self):
        result = super().free_symbols
        if isinstance(self.buffer_size, sp.Expr):
            result |= set(self.buffer_size.free_symbols)
        for o in self.offset:
            if isinstance(o, sp.Expr):
                result |= set(o.free_symbols)

        return result

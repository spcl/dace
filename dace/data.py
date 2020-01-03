import functools
import re, json
import copy as cp
import sympy as sp
import numpy

import dace, dace.dtypes as dtypes
from dace.codegen import cppunparse
from dace import symbolic
from dace.properties import (Property, make_properties, ReferenceProperty,
                             ShapeProperty, SubsetProperty, SymbolicProperty,
                             TypeClassProperty, DebugInfoProperty,
                             CodeProperty, ListProperty)


def validate_name(name):
    if not isinstance(name, str):
        return False
    if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]*$', name) is None:
        return False
    return True


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
            return Array(
                dtype=dtypes.typeclass(obj.dtype.type), shape=obj.shape)
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
    storage = Property(
        dtype=dace.dtypes.StorageType,
        desc="Storage location",
        choices=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.Default,
        from_string=lambda x: dtypes.StorageType[x])
    location = Property(
        dtype=str,  # Dict[str, symbolic]
        desc='Full storage location identifier (e.g., rank, GPU ID)',
        default='')
    toplevel = Property(
        dtype=bool, desc="Allocate array outside of state", default=False)
    debuginfo = DebugInfoProperty(allow_none=True)

    def __init__(self, dtype, shape, transient, storage, location, toplevel,
                 debuginfo):
        self.dtype = dtype
        self.shape = shape
        self.transient = transient
        self.storage = storage
        self.location = location
        self.toplevel = toplevel
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
        attrs = dace.serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    def copy(self):
        raise RuntimeError(
            'Data descriptors are unique and should not be copied')

    def is_equivalent(self, other):
        """ Check for equivalence (shape and type) of two data descriptors. """
        raise NotImplementedError

    def signature(self, with_types=True, for_call=False, name=None):
        """Returns a string for a C++ function signature (e.g., `int *A`). """
        raise NotImplementedError

    def __repr__(self):
        return 'Abstract Data Container, DO NOT USE'


@make_properties
class Scalar(Data):
    """ Data descriptor of a scalar value. """

    allow_conflicts = Property(dtype=bool, default=False)

    def __init__(self,
                 dtype,
                 transient=False,
                 storage=dace.dtypes.StorageType.Default,
                 allow_conflicts=False,
                 location='',
                 toplevel=False,
                 debuginfo=None):
        self.allow_conflicts = allow_conflicts
        shape = [1]
        super(Scalar, self).__init__(dtype, shape, transient, storage,
                                     location, toplevel, debuginfo)

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Scalar":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Scalar(dace.dtypes.int8)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

        # Check validity now
        ret.validate()
        return ret

    def __repr__(self):
        return 'Scalar (dtype=%s)' % self.dtype

    def clone(self):
        return Scalar(self.dtype, self.transient, self.storage,
                      self.allow_conflicts, self.location, self.toplevel,
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

    def signature(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call: return name
        if isinstance(self.dtype, dace.callback):
            assert name is not None
            return self.dtype.signature(name)
        return str(self.dtype.ctype) + ' ' + name

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


def set_materialize_func(obj, val):
    """ Change the storage type of an array with a materialize function to
        immaterial.
    """
    if val is not None:
        if (obj.storage != dace.dtypes.StorageType.Default
                and obj.storage != dace.dtypes.StorageType.Immaterial):
            raise ValueError("Immaterial array must have immaterial storage, "
                             "but has: {}".format(storage))
        obj.storage = dace.dtypes.StorageType.Immaterial
    obj._materialize_func = val


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

    # TODO: Should we use a Code property here?
    materialize_func = Property(
        dtype=str, allow_none=True, setter=set_materialize_func)

    strides = ListProperty(
        element_type=symbolic.pystr_to_symbolic,
        desc='For each dimension, the number of elements to '
        'skip in order to obtain the next element in '
        'that dimension.')

    total_size = SymbolicProperty(
        default=1,
        desc='The total allocated size of the array. Can be used for'
        ' padding.')

    offset = ListProperty(
        element_type=symbolic.pystr_to_symbolic,
        desc='Initial offset to translate all indices by.')

    may_alias = Property(
        dtype=bool,
        default=False,
        desc='This pointer may alias with other pointers in '
        'the same function')

    def __init__(self,
                 dtype,
                 shape,
                 materialize_func=None,
                 transient=False,
                 allow_conflicts=False,
                 storage=dace.dtypes.StorageType.Default,
                 location='',
                 strides=None,
                 offset=None,
                 may_alias=False,
                 toplevel=False,
                 debuginfo=None,
                 total_size=None):

        super(Array, self).__init__(dtype, shape, transient, storage, location,
                                    toplevel, debuginfo)

        if shape is None:
            raise IndexError('Shape must not be None')

        self.allow_conflicts = allow_conflicts
        self.materialize_func = materialize_func
        self.may_alias = may_alias

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
        return Array(self.dtype, self.shape, self.materialize_func,
                     self.transient, self.allow_conflicts, self.storage,
                     self.location, self.strides, self.offset, self.may_alias,
                     self.toplevel, self.debuginfo, self.total_size)

    def to_json(self):
        attrs = dace.serialize.all_properties_to_json(self)

        # Take care of symbolic expressions
        attrs['strides'] = list(map(str, attrs['strides']))

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Array":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Array(dace.dtypes.int8, ())
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
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
                                  symbolic.sympy.Basic))
               for s in self.strides):
            raise TypeError('Strides must be a list or tuple of integer '
                            'values or symbols')

        if len(self.offset) != len(self.shape):
            raise TypeError('Offset must be the same size as shape')

    def covers_range(self, rng):
        if len(rng) != len(self.shape):
            return False

        for s, (rb, re, rs) in zip(self.shape, rng):
            # Shape has to be positive
            if isinstance(s, sympy.Basic):
                olds = s
                if 'positive' in s.assumptions0:
                    s = sympy.Symbol(str(s), **s.assumptions0)
                else:
                    s = sympy.Symbol(str(s), positive=True, **s.assumptions0)
                if isinstance(rb, sympy.Basic):
                    rb = rb.subs({olds: s})
                if isinstance(re, sympy.Basic):
                    re = re.subs({olds: s})
                if isinstance(rs, sympy.Basic):
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
            # If both are symbols, ensure equality
            if symbolic.issymbolic(dim) and symbolic.issymbolic(otherdim):
                if dim != otherdim:
                    return False

            # If one is a symbol and the other is a constant
            # make sure they are equivalent
            elif symbolic.issymbolic(otherdim):
                if symbolic.eval(otherdim) != dim:
                    return False
            elif symbolic.issymbolic(dim):
                if symbolic.eval(dim) != otherdim:
                    return False
            else:
                # Any other case (constant vs. constant), check for equality
                if otherdim != dim:
                    return False
        return True

    def signature(self, with_types=True, for_call=False, name=None):
        arrname = name
        if self.materialize_func is not None:
            if for_call:
                return 'nullptr'
            if not with_types:
                return arrname
            arrname = '/* ' + arrname + ' (immaterial) */'

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

    # OPERATORS
    #def __add__(self, other):
    #    return (self, None)


@make_properties
class Stream(Data):
    """ Stream (or stream array) data descriptor. """

    # Properties
    offset = ListProperty(element_type=symbolic.pystr_to_symbolic)
    buffer_size = SymbolicProperty(desc="Size of internal buffer.", default=0)
    veclen = Property(
        dtype=int, desc="Vector length. Memlets must adhere to this.")

    def __init__(self,
                 dtype,
                 veclen,
                 buffer_size,
                 shape=None,
                 transient=False,
                 storage=dace.dtypes.StorageType.Default,
                 location='',
                 offset=None,
                 toplevel=False,
                 debuginfo=None):

        if shape is None:
            shape = (1, )

        self.veclen = veclen
        self.buffer_size = buffer_size

        if offset is not None:
            if len(offset) != len(shape):
                raise TypeError('Offset must be the same size as shape')
            self.offset = cp.copy(offset)
        else:
            self.offset = [0] * len(shape)

        super(Stream, self).__init__(dtype, shape, transient, storage,
                                     location, toplevel, debuginfo)

    def to_json(self):
        attrs = dace.serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Stream":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Stream(dace.dtypes.int8, 1, 1)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

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
        return Stream(self.dtype, self.veclen, self.buffer_size, self.shape,
                      self.transient, self.storage, self.location, self.offset,
                      self.toplevel, self.debuginfo)

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
            # If both are symbols, ensure equality
            if symbolic.issymbolic(dim) and symbolic.issymbolic(otherdim):
                if dim != otherdim:
                    return False

            # If one is a symbol and the other is a constant
            # make sure they are equivalent
            elif symbolic.issymbolic(otherdim):
                if symbolic.eval(otherdim) != dim:
                    return False
            elif symbolic.issymbolic(dim):
                if symbolic.eval(dim) != otherdim:
                    return False
            else:
                # Any other case (constant vs. constant), check for equality
                if otherdim != dim:
                    return False
        return True

    def signature(self, with_types=True, for_call=False, name=None):
        if not with_types or for_call: return name
        if self.storage in [
                dace.dtypes.StorageType.GPU_Global,
                dace.dtypes.StorageType.GPU_Shared,
                dace.dtypes.StorageType.GPU_Stack
        ]:
            return 'dace::GPUStream<%s, %s> %s' % (
                str(self.dtype.ctype), 'true'
                if sp.log(self.buffer_size, 2).is_Integer else 'false', name)

        return 'dace::Stream<%s> %s' % (str(self.dtype.ctype), name)

    def sizes(self):
        return [
            d.name if isinstance(d, symbolic.symbol) else str(d)
            for d in self.shape
        ]

    def size_string(self):
        return (" * ".join([
            cppunparse.pyexpr2cpp(dace.symbolic.symstr(s)) for s in self.shape
        ]))

    def is_stream_array(self):
        return _prod(self.shape) != 1

    def covers_range(self, rng):
        if len(rng) != len(self.shape):
            return False

        for s, (rb, re, rs) in zip(self.shape, rng):
            # Shape has to be positive
            if isinstance(s, sympy.Basic):
                olds = s
                if 'positive' in s.assumptions0:
                    s = sympy.Symbol(str(s), **s.assumptions0)
                else:
                    s = sympy.Symbol(str(s), positive=True, **s.assumptions0)
                if isinstance(rb, sympy.Basic):
                    rb = rb.subs({olds: s})
                if isinstance(re, sympy.Basic):
                    re = re.subs({olds: s})
                if isinstance(rs, sympy.Basic):
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

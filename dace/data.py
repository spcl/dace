# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import functools
import re
import copy as cp
import sympy as sp
import numpy
from numbers import Number
from typing import Set, Sequence, Tuple

import dace.dtypes as dtypes
from dace.codegen import cppunparse
from dace import symbolic, serialize
from dace.properties import (EnumProperty, Property, make_properties, DictProperty, ReferenceProperty, ShapeProperty,
                             SubsetProperty, SymbolicProperty, TypeClassProperty, DebugInfoProperty, CodeProperty,
                             ListProperty)


def create_datadescriptor(obj):
    """ Creates a data descriptor from various types of objects.
        @see: dace.data.Data
    """
    from dace import dtypes  # Avoiding import loops
    if isinstance(obj, Data):
        return obj
    elif hasattr(obj, '__descriptor__'):
        return obj.__descriptor__()
    elif hasattr(obj, 'descriptor'):
        return obj.descriptor
    elif isinstance(obj, (list, tuple, numpy.ndarray)):
        if isinstance(obj, (list, tuple)):  # Lists and tuples are cast to numpy
            obj = numpy.array(obj)

        if obj.dtype.fields is not None:  # Struct
            dtype = dtypes.struct('unnamed', **{k: dtypes.typeclass(v[0].type) for k, v in obj.dtype.fields.items()})
        else:
            dtype = dtypes.typeclass(obj.dtype.type)
        return Array(dtype=dtype, strides=tuple(s // obj.itemsize for s in obj.strides), shape=obj.shape)
    # special case for torch tensors. Maybe __array__ could be used here for a more
    # general solution, but torch doesn't support __array__ for cuda tensors.
    elif type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor":
        try:
            # If torch is importable, define translations between typeclasses and torch types. These are reused by daceml.
            # conversion happens here in pytorch:
            # https://github.com/pytorch/pytorch/blob/143ef016ee1b6a39cf69140230d7c371de421186/torch/csrc/utils/tensor_numpy.cpp#L237
            import torch
            TYPECLASS_TO_TORCH_DTYPE = {
                dtypes.bool_: torch.bool,
                dtypes.int8: torch.int8,
                dtypes.int16: torch.int16,
                dtypes.int32: torch.int32,
                dtypes.int64: torch.int64,
                dtypes.uint8: torch.uint8,
                dtypes.float16: torch.float16,
                dtypes.float32: torch.float32,
                dtypes.float64: torch.float64,
                dtypes.complex64: torch.complex64,
                dtypes.complex128: torch.complex128,
            }

            TORCH_DTYPE_TO_TYPECLASS = {v: k for k, v in TYPECLASS_TO_TORCH_DTYPE.items()}

            return Array(dtype=TORCH_DTYPE_TO_TYPECLASS[obj.dtype], strides=obj.stride(), shape=tuple(obj.shape))
        except ImportError:
            raise ValueError("Attempted to convert a torch.Tensor, but torch could not be imported")
    elif dtypes.is_gpu_array(obj):
        interface = obj.__cuda_array_interface__
        dtype = dtypes.typeclass(numpy.dtype(interface['typestr']).type)
        itemsize = numpy.dtype(interface['typestr']).itemsize
        if len(interface['shape']) == 0:
            return Scalar(dtype, storage=dtypes.StorageType.GPU_Global)
        return Array(dtype=dtype,
                     shape=interface['shape'],
                     strides=(tuple(s // itemsize for s in interface['strides']) if interface['strides'] else None),
                     storage=dtypes.StorageType.GPU_Global)
    elif symbolic.issymbolic(obj):
        return Scalar(symbolic.symtype(obj))
    elif isinstance(obj, dtypes.typeclass):
        return Scalar(obj)
    elif (obj is int or obj is float or obj is complex or obj is bool or obj is None):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, type) and issubclass(obj, numpy.number):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, (Number, numpy.number, numpy.bool, numpy.bool_)):
        return Scalar(dtypes.typeclass(type(obj)))
    elif callable(obj):
        # Cannot determine return value/argument types from function object
        return Scalar(dtypes.callback(None))
    elif isinstance(obj, str):
        return Scalar(dtypes.string())

    raise TypeError(f'Could not create a DaCe data descriptor from object {obj}. '
                    'If this is a custom object, consider creating a `__descriptor__` '
                    'adaptor method to the type hint or object itself.')


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


def _prod(sequence):
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


@make_properties
class Data(object):
    """ Data type descriptors that can be used as references to memory.
        Examples: Arrays, Streams, custom arrays (e.g., sparse matrices).
    """

    dtype = TypeClassProperty(default=dtypes.int32, choices=dtypes.Typeclasses)
    shape = ShapeProperty(default=[])
    transient = Property(dtype=bool, default=False)
    storage = EnumProperty(dtype=dtypes.StorageType, desc="Storage location", default=dtypes.StorageType.Default)
    lifetime = EnumProperty(dtype=dtypes.AllocationLifetime,
                            desc='Data allocation span',
                            default=dtypes.AllocationLifetime.Scope)
    location = DictProperty(key_type=str, value_type=str, desc='Full storage location identifier (e.g., rank, GPU ID)')
    debuginfo = DebugInfoProperty(allow_none=True)

    def __init__(self, dtype, shape, transient, storage, location, lifetime, debuginfo):
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
        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)) for s in self.shape):
            raise TypeError('Shape must be a list or tuple of integer values ' 'or symbols')
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @property
    def toplevel(self):
        return self.lifetime is not dtypes.AllocationLifetime.Scope

    def copy(self):
        raise RuntimeError('Data descriptors are unique and should not be copied')

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

    def strides_from_layout(
        self,
        *dimensions: int,
        alignment: symbolic.SymbolicType = 1,
        only_first_aligned: bool = False,
    ) -> Tuple[Tuple[symbolic.SymbolicType], symbolic.SymbolicType]:
        """
        Returns the absolute strides and total size of this data descriptor,
        according to the given dimension ordering and alignment.
        :param dimensions: A sequence of integers representing a permutation
                           of the descriptor's dimensions.
        :param alignment: Padding (in elements) at the end, ensuring stride
                          is a multiple of this number. 1 (default) means no
                          padding.
        :param only_first_aligned: If True, only the first dimension is padded
                                   with ``alignment``. Otherwise all dimensions
                                   are.
        :return: A 2-tuple of (tuple of strides, total size).
        """
        # Verify dimensions
        if tuple(sorted(dimensions)) != tuple(range(len(self.shape))):
            raise ValueError('Every dimension must be given and appear once.')
        if (alignment < 1) == True or (alignment < 0) == True:
            raise ValueError('Invalid alignment value')

        strides = [1] * len(dimensions)
        total_size = 1
        first = True
        for dim in dimensions:
            strides[dim] = total_size
            if not only_first_aligned or first:
                dimsize = (((self.shape[dim] + alignment - 1) // alignment) * alignment)
            else:
                dimsize = self.shape[dim]
            total_size *= dimsize
            first = False

        return (tuple(strides), total_size)

    def set_strides_from_layout(self,
                                *dimensions: int,
                                alignment: symbolic.SymbolicType = 1,
                                only_first_aligned: bool = False):
        """
        Sets the absolute strides and total size of this data descriptor,
        according to the given dimension ordering and alignment.
        :param dimensions: A sequence of integers representing a permutation
                           of the descriptor's dimensions.
        :param alignment: Padding (in elements) at the end, ensuring stride
                          is a multiple of this number. 1 (default) means no
                          padding.
        :param only_first_aligned: If True, only the first dimension is padded
                                   with ``alignment``. Otherwise all dimensions
                                   are.
        """
        strides, totalsize = self.strides_from_layout(*dimensions,
                                                      alignment=alignment,
                                                      only_first_aligned=only_first_aligned)
        self.strides = strides
        self.total_size = totalsize


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
        super(Scalar, self).__init__(dtype, shape, transient, storage, location, lifetime, debuginfo)

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Scalar":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Scalar(dtypes.int8)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    def __repr__(self):
        return 'Scalar (dtype=%s)' % self.dtype

    def clone(self):
        return Scalar(self.dtype, self.transient, self.storage, self.allow_conflicts, self.location, self.lifetime,
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
        if self.dtype != other.dtype:
            return False
        return True

    def as_arg(self, with_types=True, for_call=False, name=None):
        if self.storage is dtypes.StorageType.GPU_Global:
            return Array(self.dtype, [1]).as_arg(with_types, for_call, name)
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


@make_properties
class Array(Data):
    """ Array/constant descriptor (dimensions, type and other properties). """

    # Properties
    allow_conflicts = Property(dtype=bool,
                               default=False,
                               desc='If enabled, allows more than one '
                               'memlet to write to the same memory location without conflict '
                               'resolution.')

    strides = ShapeProperty(
        # element_type=symbolic.pystr_to_symbolic,
        desc='For each dimension, the number of elements to '
        'skip in order to obtain the next element in '
        'that dimension.')

    total_size = SymbolicProperty(default=0, desc='The total allocated size of the array. Can be used for' ' padding.')

    offset = ShapeProperty(desc='Initial offset to translate all indices by.')

    may_alias = Property(dtype=bool,
                         default=False,
                         desc='This pointer may alias with other pointers in '
                         'the same function')

    alignment = Property(dtype=int, default=0, desc='Allocation alignment in bytes (0 uses ' 'compiler-default)')

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

        super(Array, self).__init__(dtype, shape, transient, storage, location, lifetime, debuginfo)

        if shape is None:
            raise IndexError('Shape must not be None')

        self.allow_conflicts = allow_conflicts
        self.may_alias = may_alias
        self.alignment = alignment

        if strides is not None:
            self.strides = cp.copy(strides)
        else:
            self.strides = [_prod(shape[i + 1:]) for i in range(len(shape))]

        if strides is not None and shape is not None and total_size is None:
            # Compute the minimal total_size that could be used with strides and shape
            self.total_size = sum(((shp - 1) * s for shp, s in zip(shape, strides))) + 1
        else:
            self.total_size = total_size or _prod(shape)

        if offset is not None:
            self.offset = cp.copy(offset)
        else:
            self.offset = [0] * len(shape)

        self.validate()

    def __repr__(self):
        return '%s (dtype=%s, shape=%s)' % (type(self).__name__, self.dtype, self.shape)

    def clone(self):
        return type(self)(self.dtype, self.shape, self.transient, self.allow_conflicts, self.storage, self.location,
                          self.strides, self.offset, self.may_alias, self.lifetime, self.alignment, self.debuginfo,
                          self.total_size)

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls(dtypes.int8, ())
        serialize.set_properties_from_json(ret, json_obj, context=context)

        # Default shape-related properties
        if not ret.offset:
            ret.offset = [0] * len(ret.shape)
        if not ret.strides:
            # Default strides are C-ordered
            ret.strides = [_prod(ret.shape[i + 1:]) for i in range(len(ret.shape))]
        if ret.total_size == 0:
            ret.total_size = _prod(ret.shape)

        # Check validity now
        ret.validate()
        return ret

    def validate(self):
        super(Array, self).validate()
        if len(self.strides) != len(self.shape):
            raise TypeError('Strides must be the same size as shape')

        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)) for s in self.strides):
            raise TypeError('Strides must be a list or tuple of integer ' 'values or symbols')

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
        return [d.name if isinstance(d, symbolic.symbol) else str(d) for d in self.shape]

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

        super(Stream, self).__init__(dtype, shape, transient, storage, location, lifetime, debuginfo)

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls(dtypes.int8, 1)
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    def __repr__(self):
        return '%s (dtype=%s, shape=%s)' % (type(self).__name__, self.dtype, self.shape)

    @property
    def total_size(self):
        return _prod(self.shape)

    @property
    def strides(self):
        return [_prod(self.shape[i + 1:]) for i in range(len(self.shape))]

    def clone(self):
        return type(self)(self.dtype, self.buffer_size, self.shape, self.transient, self.storage, self.location,
                          self.offset, self.lifetime, self.debuginfo)

    # Checks for equivalent shape and type
    def is_equivalent(self, other):
        if not isinstance(other, type(self)):
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
        if self.storage in [dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared]:
            return 'dace::GPUStream<%s, %s> %s' % (str(
                self.dtype.ctype), 'true' if sp.log(self.buffer_size, 2).is_Integer else 'false', name)

        return 'dace::Stream<%s> %s' % (str(self.dtype.ctype), name)

    def sizes(self):
        return [d.name if isinstance(d, symbolic.symbol) else str(d) for d in self.shape]

    def size_string(self):
        return (" * ".join([cppunparse.pyexpr2cpp(symbolic.symstr(s)) for s in self.shape]))

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


@make_properties
class View(Array):
    """ 
    Data descriptor that acts as a reference (or view) of another array. Can
    be used to reshape or reinterpret existing data without copying it.

    To use a View, it needs to be referenced in an access node that is directly
    connected to another access node. The rules for deciding which access node
    is viewed are:
      * If there is one edge (in/out) that leads (via memlet path) to an access
        node, and the other side (out/in) has a different number of edges.
      * If there is one incoming and one outgoing edge, and one leads to a code
        node, the one that leads to an access node is the viewed data.
      * If both sides lead to access nodes, if one memlet's data points to the 
        view it cannot point to the viewed node.
      * If both memlets' data are the respective access nodes, the access 
        node at the highest scope is the one that is viewed.
      * If both access nodes reside in the same scope, the input data is viewed.

    Other cases are ambiguous and will fail SDFG validation.

    In the Python frontend, ``numpy.reshape`` and ``numpy.ndarray.view`` both
    generate Views.
    """
    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for ' 'Views')

    def as_array(self):
        copy = cp.deepcopy(self)
        copy.__class__ = Array
        return copy

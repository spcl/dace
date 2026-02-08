# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Core data descriptor classes.

This module contains the base ``Data`` class and all core descriptor classes:
``Scalar``, ``Array``, ``ContainerArray``, ``Stream``, ``Structure``,
``View``, ``Reference``, and their subclasses.
"""
import copy as cp
import ctypes
import dataclasses

from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import sympy as sp

try:
    from numpy.typing import ArrayLike
except (ModuleNotFoundError, ImportError):
    ArrayLike = Any

from dace import dtypes, serialize, symbolic
from dace.properties import (DebugInfoProperty, DictProperty, EnumProperty, ListProperty, NestedDataClassProperty,
                             OrderedDictProperty, Property, ShapeProperty, SymbolicProperty, TypeClassProperty,
                             make_properties)
from dace.utils import prod

# Backward compatibility alias
_prod = prod


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return [(k, serialize.to_json(v)) for k, v in arrays.items()]


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return OrderedDict((k, serialize.from_json(v, context)) for k, v in obj)


@make_properties
class Data:
    """ Data type descriptors that can be used as references to memory.
        Examples: Arrays, Streams, custom arrays (e.g., sparse matrices).
    """

    def _transient_setter(self, value):
        self._transient = value
        if isinstance(self, Structure):
            for _, v in self.members.items():
                if isinstance(v, Data):
                    v.transient = value

    dtype = TypeClassProperty(default=dtypes.int32)
    shape = ShapeProperty(default=[])
    transient = Property(dtype=bool, default=False, setter=_transient_setter)
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

    def __call__(self):
        # This method is implemented to support type hints
        return self

    def validate(self):
        """ Validate the correctness of this object.
            Raises an exception on error. """
        self._validate()

    # Validation of this class is in a separate function, so that this
    # class can call `_validate()` without calling the subclasses'
    # `validate` function.
    def _validate(self):
        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)) for s in self.shape):
            raise TypeError('Shape must be a list or tuple of integer values '
                            'or symbols')
        if any((shp < 0) == True for shp in self.shape):
            raise TypeError(f'Found negative shape in Data, its shape was {self.shape}')
        return True

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @property
    def toplevel(self):
        return self.lifetime is not dtypes.AllocationLifetime.Scope

    def is_equivalent(self, other):
        """ Check for equivalence (shape and type) of two data descriptors. """
        raise NotImplementedError

    def __eq__(self, other):
        # Evaluate equivalence using serialized value
        return serialize.dumps(self) == serialize.dumps(other)

    def __hash__(self):
        # Compute hash using serialized value (i.e., with all properties included)
        return hash(serialize.dumps(self))

    def as_arg(self, with_types=True, for_call=False, name=None):
        """Returns a string for a C++ function signature (e.g., `int *A`). """
        raise NotImplementedError

    def as_python_arg(self, with_types=True, for_call=False, name=None):
        """Returns a string for a Data-Centric Python function signature (e.g., `A: dace.int32[M]`). """
        raise NotImplementedError

    def used_symbols(self, all_symbols: bool) -> Set[symbolic.SymbolicType]:
        """
        Returns a set of symbols that are used by this data descriptor.

        :param all_symbols: Include not-strictly-free symbols that are used by this data descriptor,
                            e.g., shape and size of a global array.
        :return: A set of symbols that are used by this data descriptor. NOTE: The results are symbolic
                 rather than a set of strings.
        """
        result = set()
        if (self.transient and not isinstance(self, (View, Reference))) or all_symbols:
            for s in self.shape:
                if isinstance(s, sp.Basic):
                    result |= set(s.free_symbols)
        return result

    @property
    def free_symbols(self) -> Set[symbolic.SymbolicType]:
        """ Returns a set of undefined symbols in this data descriptor. """
        return self.used_symbols(all_symbols=True)

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

    def __matmul__(self, storage: dtypes.StorageType):
        """
        Syntactic sugar for specifying the storage of a data descriptor.
        This enables controlling the storage location as follows:

        .. code-block:: python

            @dace
            def add(X: dace.float32[10, 10] @ dace.StorageType.GPU_Global):
                return X + 1
        """
        new_desc = cp.deepcopy(self)
        new_desc.storage = storage
        return new_desc


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

    @property
    def start_offset(self):
        return 0

    @property
    def alignment(self):
        return 0

    @property
    def optional(self) -> bool:
        return False

    @property
    def pool(self) -> bool:
        return False

    @property
    def may_alias(self) -> bool:
        return False

    def is_equivalent(self, other):
        # Special case: array of size 1
        if isinstance(other, Array) and other.shape == (1, ) and other.dtype == self.dtype:
            return True

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

    def as_python_arg(self, with_types=True, for_call=False, name=None):
        if self.storage is dtypes.StorageType.GPU_Global:
            return Array(self.dtype, [1]).as_python_arg(with_types, for_call, name)
        if not with_types or for_call:
            return name
        return f"{name}: {dtypes.TYPECLASS_TO_STRING[self.dtype].replace('::', '.')}"

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
    """
    Array data descriptor. This object represents a multi-dimensional data container in SDFGs that can be accessed and
    modified. The definition does not contain the actual array, but rather a description of how to construct it and
    how it should behave.

    The array definition is flexible in terms of data allocation, it allows arbitrary multidimensional, potentially
    symbolic shapes (e.g., an array with size ``N+1 x M`` will have ``shape=(N+1, M)``), of arbitrary data
    typeclasses (``dtype``). The physical data layout of the array is controlled by several properties:

       * The ``strides`` property determines the ordering and layout of the dimensions --- it specifies how many
         elements in memory are skipped whenever one element in that dimension is advanced. For example, the contiguous
         dimension always has a stride of ``1``; a C-style MxN array will have strides ``(N, 1)``, whereas a
         FORTRAN-style array of the same size will have ``(1, M)``. Strides can be larger than the shape, which allows
         post-padding of the contents of each dimension.
       * The ``start_offset`` property is a number of elements to pad the beginning of the memory buffer with. This is
         used to ensure that a specific index is aligned as a form of pre-padding (that element may not necessarily be
         the first element, e.g., in the case of halo or "ghost cells" in stencils).
       * The ``total_size`` property determines how large the total allocation size is. Normally, it is the product of
         the ``shape`` elements, but if pre- or post-padding is involved it may be larger.
       * ``alignment`` provides alignment guarantees (in bytes) of the first element in the allocated array. This is
         used by allocators in the code generator to ensure certain addresses are expected to be aligned, e.g., for
         vectorization.
       * Lastly, a property called ``offset`` controls the logical access of the array, i.e., what would be the first
         element's index after padding and alignment. This mimics a language feature prominent in scientific languages
         such as FORTRAN, where one could set an array to begin with 1, or any arbitrary index. By default this is set
         to zero.

    To summarize with an example, a two-dimensional array with pre- and post-padding looks as follows:

    .. code-block:: text

        [xxx][          |xx]
             [          |xx]
             [          |xx]
             [          |xx]
             ---------------
             [xxxxxxxxxxxxx]

        shape = (4, 10)
        strides = (12, 1)
        start_offset = 3
        total_size = 63   [= 3 + 12 * 5]
        offset = (0, 0, 0)


    Notice that the last padded row does not appear in strides, but is a consequence of ``total_size`` being larger.


    Apart from memory layout, other properties of ``Array`` help the data-centric transformation infrastructure make
    decisions about the array. ``allow_conflicts`` states that warnings should not be printed if potential conflicted
    acceses (e.g., data races) occur. ``may_alias`` inhibits transformations that may assume that this array does not
    overlap with other arrays in the same context (e.g., function).
    """

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

    total_size = SymbolicProperty(default=0, desc='The total allocated size of the array. Can be used for padding.')

    offset = ShapeProperty(desc='Initial offset to translate all indices by.')

    may_alias = Property(dtype=bool,
                         default=False,
                         desc='This pointer may alias with other pointers in the same function')

    alignment = Property(dtype=int, default=0, desc='Allocation alignment in bytes (0 uses compiler-default)')

    start_offset = Property(dtype=int, default=0, desc='Allocation offset elements for manual alignment (pre-padding)')
    optional = Property(dtype=bool,
                        default=None,
                        allow_none=True,
                        desc='Specifies whether this array may have a value of None. '
                        'If False, the array must not be None. If option is not set, '
                        'it is inferred by other properties and the OptionalArrayInference pass.')
    pool = Property(dtype=bool, default=False, desc='Hint to the allocator that using a memory pool is preferred')

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
                 total_size=None,
                 start_offset=None,
                 optional=None,
                 pool=False):

        super(Array, self).__init__(dtype, shape, transient, storage, location, lifetime, debuginfo)

        self.allow_conflicts = allow_conflicts
        self.may_alias = may_alias
        self.alignment = alignment

        if start_offset is not None:
            self.start_offset = start_offset
        self.optional = optional
        if optional is None and self.transient:
            self.optional = False
        self.pool = pool

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

        self._packed_c_strides = None
        self._packed_fortran_strides = None

        self.validate()

    def __repr__(self):
        return '%s (dtype=%s, shape=%s)' % (type(self).__name__, self.dtype, self.shape)

    def clone(self):
        return type(self)(self.dtype, self.shape, self.transient, self.allow_conflicts, self.storage, self.location,
                          self.strides, self.offset, self.may_alias, self.lifetime, self.alignment, self.debuginfo,
                          self.total_size, self.start_offset, self.optional, self.pool)

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
        if len(self.offset) != len(self.shape):
            raise TypeError('Offset must be the same size as shape')

        if any(not isinstance(s, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic)) for s in self.strides):
            raise TypeError('Strides must be a list or tuple of integer values or symbols')
        if any(not isinstance(off, (int, symbolic.SymExpr, symbolic.symbol, symbolic.sympy.Basic))
               for off in self.offset):
            raise TypeError('Offset must be a list or tuple of integer values or symbols')

        # Actually it would be enough to only enforce the non negativity only if the shape is larger than one.
        if any((stride < 0) == True for stride in self.strides):
            raise TypeError(f'Found negative strides in array, they were {self.strides}')
        if (self.total_size < 0) == True:
            raise TypeError(f'The total size of an array must be positive but it was negative {self.total_size}')

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
        # Special case: Scalar
        if isinstance(other, Scalar) and self.shape == (1, ) and self.dtype == other.dtype:
            return True

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

        # Test strides
        for stride, otherstride in zip(self.strides, other.strides):
            if otherstride != stride:
                return False

        # Test total size
        # if self.total_size != other.total_size:
        #     return False

        # Test offset
        # for off, otheroff in zip(self.offset, other.offset):
        #     if otheroff != off:
        #         return False

        return True

    def as_arg(self, with_types=True, for_call=False, name=None):
        arrname = name

        if not with_types or for_call:
            return arrname
        if self.may_alias:
            return str(self.dtype.ctype) + ' *' + arrname
        return str(self.dtype.ctype) + ' * __restrict__ ' + arrname

    def as_python_arg(self, with_types=True, for_call=False, name=None):
        arrname = name

        if not with_types or for_call:
            return arrname
        return f"{arrname}: {dtypes.TYPECLASS_TO_STRING[self.dtype].replace('::', '.')}{list(self.shape)}"

    def sizes(self):
        return [d.name if isinstance(d, symbolic.symbol) else str(d) for d in self.shape]

    def used_symbols(self, all_symbols: bool) -> Set[symbolic.SymbolicType]:
        result = super().used_symbols(all_symbols)
        for s in self.strides:
            if isinstance(s, sp.Expr):
                result |= set(s.free_symbols)
        for o in self.offset:
            if isinstance(o, sp.Expr):
                result |= set(o.free_symbols)
        if (self.transient and not isinstance(self, (View, Reference))) or all_symbols:
            if isinstance(self.total_size, sp.Expr):
                result |= set(self.total_size.free_symbols)
        return result

    @property
    def free_symbols(self):
        return self.used_symbols(all_symbols=True)

    def _set_shape_dependent_properties(self, shape, strides, total_size, offset):
        """
        Used to set properties which depend on the shape of the array
        either to their default value, which depends on the shape, or
        if explicitely provided to the given value. For internal use only.
        """
        if shape is None:
            raise IndexError('Shape must not be None')

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

        # Clear cached values and recompute
        self._packed_c_strides = None
        self._packed_fortran_strides = None
        self._packed_c_strides = self._get_packed_c_strides()
        self._packed_fortran_strides = self._get_packed_fortran_strides()

    def set_shape(
        self,
        new_shape,
        strides=None,
        total_size=None,
        offset=None,
    ):
        """
        Updates the shape of an array.
        """
        self.shape = new_shape
        self._set_shape_dependent_properties(new_shape, strides, total_size, offset)
        self.validate()

    def _get_packed_fortran_strides(self) -> Tuple[int]:
        """Compute packed strides for Fortran-style (column-major) layout."""
        # Strides increase along the leading dimensions
        if self._packed_fortran_strides is None:
            strides = [1]
            accum = 1
            # Iterate in reversed order except the first dimension
            for s in self.shape[:-1]:
                accum *= s
                strides.append(accum)
            self._packed_fortran_strides = tuple(strides)
        return self._packed_fortran_strides

    def _get_packed_c_strides(self) -> Tuple[int]:
        """Compute packed strides for C-style (row-major) layout."""
        # Strides increase along the trailing dimensions
        if self._packed_c_strides is None:
            strides = [1]
            accum = 1
            # Iterate in reversed order except the first dimension
            for s in reversed(self.shape[1:]):
                accum *= s
                strides.insert(0, accum)
            self._packed_c_strides = tuple(strides)
        return self._packed_c_strides

    def is_packed_fortran_strides(self) -> bool:
        """Return True if strides match Fortran-contiguous (column-major) layout."""
        strides = self._get_packed_fortran_strides()
        return tuple(strides) == tuple(self.strides)

    def is_packed_c_strides(self) -> bool:
        """Return True if strides match Fortran-contiguous (row-major) layout."""
        strides = self._get_packed_c_strides()
        return tuple(strides) == tuple(self.strides)


@make_properties
class ContainerArray(Array):
    """ An array that may contain other data containers (e.g., Structures, other arrays). """

    stype = NestedDataClassProperty(allow_none=True, default=None)

    def __init__(self,
                 stype: Data,
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
                 total_size=None,
                 start_offset=None,
                 optional=None,
                 pool=False):

        self.stype = stype
        if stype:
            if isinstance(stype, Structure):
                dtype = stype.dtype
            else:
                dtype = dtypes.pointer(stype.dtype)
        else:
            dtype = dtypes.pointer(dtypes.typeclass(None))  # void*
        super(ContainerArray,
              self).__init__(dtype, shape, transient, allow_conflicts, storage, location, strides, offset, may_alias,
                             lifetime, alignment, debuginfo, total_size, start_offset, optional, pool)

    @classmethod
    def from_json(cls, json_obj, context=None):
        # Create dummy object
        ret = cls(None, ())
        serialize.set_properties_from_json(ret, json_obj, context=context)

        # Default shape-related properties
        if not ret.offset:
            ret.offset = [0] * len(ret.shape)
        if not ret.strides:
            # Default strides are C-ordered
            ret.strides = [_prod(ret.shape[i + 1:]) for i in range(len(ret.shape))]
        if ret.total_size == 0:
            ret.total_size = _prod(ret.shape)

        return ret


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

    @property
    def start_offset(self):
        return 0

    @property
    def optional(self) -> bool:
        return False

    @property
    def may_alias(self) -> bool:
        return False

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

        # Test buffer size
        if self.buffer_size != other.buffer_size:
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

    def used_symbols(self, all_symbols: bool) -> Set[symbolic.SymbolicType]:
        result = super().used_symbols(all_symbols)
        if (self.transient or all_symbols) and isinstance(self.buffer_size, sp.Expr):
            result |= set(self.buffer_size.free_symbols)
        for o in self.offset:
            if isinstance(o, sp.Expr):
                result |= set(o.free_symbols)

        return result

    @property
    def free_symbols(self):
        return self.used_symbols(all_symbols=True)


@make_properties
class Structure(Data):
    """ Base class for structures. """

    members = OrderedDictProperty(default=OrderedDict(),
                                  desc="Dictionary of structure members",
                                  from_json=_arrays_from_json,
                                  to_json=_arrays_to_json)
    name = Property(dtype=str, desc="Structure type name")

    def __init__(self,
                 members: Union[Dict[str, Data], List[Tuple[str, Data]]],
                 name: str = 'Structure',
                 transient: bool = False,
                 storage: dtypes.StorageType = dtypes.StorageType.Default,
                 location: Dict[str, str] = None,
                 lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope,
                 debuginfo: dtypes.DebugInfo = None):

        self.members = OrderedDict(members)
        for k, v in self.members.items():
            if isinstance(v, dtypes.typeclass):
                v = Scalar(v)
                self.members[k] = v
            v.transient = transient

        self.name = name
        fields_and_types = OrderedDict()
        symbols = set()
        for k, v in self.members.items():
            if isinstance(v, Structure):
                symbols |= v.free_symbols
                fields_and_types[k] = (v.dtype, str(v.total_size))
            elif isinstance(v, Array):
                symbols |= v.free_symbols
                fields_and_types[k] = (dtypes.pointer(v.dtype), str(_prod(v.shape)))
            elif isinstance(v, Scalar):
                symbols |= v.free_symbols
                fields_and_types[k] = v.dtype
            elif isinstance(v, dtypes.typeclass):
                fields_and_types[k] = v
            elif isinstance(v, (sp.Basic, symbolic.SymExpr)):
                symbols |= v.free_symbols
                fields_and_types[k] = symbolic.symtype(v)
            elif isinstance(v, (int, np.integer)):
                fields_and_types[k] = dtypes.typeclass(type(v))
            else:
                raise TypeError(f"Attribute {k}'s value {v} has unsupported type: {type(v)}")

        # NOTE: We will not store symbols in the dtype for now, but leaving it as a comment to investigate later.
        # NOTE: See discussion about data/object symbols.
        # for s in symbols:
        #     if str(s) in fields_and_types:
        #         continue
        #     if hasattr(s, "dtype"):
        #         fields_and_types[str(s)] = s.dtype
        #     else:
        #         fields_and_types[str(s)] = dtypes.int32

        dtype = dtypes.pointer(dtypes.struct(name, **fields_and_types))
        dtype.base_type.__descriptor__ = self
        shape = (1, )
        super(Structure, self).__init__(dtype, shape, transient, storage, location, lifetime, debuginfo)

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'Structure':
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Structure({})
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @staticmethod
    def from_dataclass(cls, **overrides) -> 'Structure':
        """
        Creates a Structure data descriptor from a dataclass instance.

        :param cls: The dataclass to convert.
        :param overrides: Optional overrides for the structure fields.
        :return: A Structure data descriptor.
        """
        members = {}
        for field in dataclasses.fields(cls):
            # Recursive structures
            if dataclasses.is_dataclass(field.type):
                members[field.name] = Structure.from_dataclass(field.type)
                continue
            members[field.name] = field.type

        members.update(overrides)
        return Structure(members, name=cls.__name__)

    @property
    def total_size(self):
        return -1

    @property
    def offset(self):
        return [0]

    @property
    def start_offset(self):
        return 0

    @property
    def strides(self):
        return [1]

    @property
    def free_symbols(self) -> Set[symbolic.SymbolicType]:
        """ Returns a set of undefined symbols in this data descriptor. """
        result = set()
        for k, v in self.members.items():
            result |= v.free_symbols
        return result

    def __repr__(self):
        return f"{self.name} ({', '.join([f'{k}: {v}' for k, v in self.members.items()])})"

    def as_arg(self, with_types=True, for_call=False, name=None):
        if self.storage is dtypes.StorageType.GPU_Global:
            return Array(self.dtype, [1]).as_arg(with_types, for_call, name)
        if not with_types or for_call:
            return name
        return self.dtype.as_arg(name)

    def __getitem__(self, s):
        """ This is syntactic sugar that allows us to define an array type
            with the following syntax: ``Structure[N,M]``
            :return: A ``data.ContainerArray`` data descriptor.
        """
        if isinstance(s, list) or isinstance(s, tuple):
            return ContainerArray(self, tuple(s))
        return ContainerArray(self, (s, ))

    # NOTE: Like Scalars?
    @property
    def may_alias(self) -> bool:
        return False

    # TODO: Can Structures be optional?
    @property
    def optional(self) -> bool:
        return False

    def keys(self):
        result = self.members.keys()
        for k, v in self.members.items():
            if isinstance(v, Structure):
                result |= set(map(lambda x: f"{k}.{x}", v.keys()))
        return result

    def clone(self):
        return Structure(self.members, self.name, self.transient, self.storage, self.location, self.lifetime,
                         self.debuginfo)

    # NOTE: Like scalars?
    @property
    def pool(self) -> bool:
        return False

    def make_argument(self, **fields) -> ctypes.Structure:
        """
        Creates a structure instance from the given field values, which can be used as
        an argument for DaCe programs.

        :param fields: Dictionary of field names to values.
        :return: A ctypes Structure instance.
        """
        # Import here to avoid circular import
        from dace.data.ctypes_interop import make_ctypes_argument
        struct_type: dtypes.struct = self.dtype.base_type
        struct_ctype = struct_type.as_ctypes()

        def _make_arg(arg: Any, expected_type: Data, name: str) -> Any:
            if isinstance(expected_type, Structure):
                return ctypes.pointer(expected_type.make_argument_from_object(arg))
            return make_ctypes_argument(arg, expected_type, name)

        args = {
            field_name: _make_arg(field_value, self.members[field_name], field_name)
            for field_name, field_value in fields.items() if field_name in self.members
        }

        struct_instance = struct_ctype(**args)
        return struct_instance

    def make_argument_from_object(self, obj) -> ctypes.Structure:
        """
        Creates a structure instance from the given object, which can be used as
        an argument for DaCe programs. If the object has attributes matching the field names,
        those attributes are used as field values. Other attributes are ignored.

        :param obj: Object containing field values.
        :return: A ctypes Structure instance.
        """
        return self.make_argument(**{field_name: getattr(obj, field_name) for field_name in self.members})


class View:
    """
    Data descriptor that acts as a static reference (or view) of another data container.
    Can be used to reshape or reinterpret existing data without copying it.

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
    """

    @staticmethod
    def view(viewed_container: Data, debuginfo=None):
        """
        Create a new View of the specified data container.

        :param viewed_container: The data container properties of this view
        :param debuginfo: Specific source line information for this view, if
                          different from ``viewed_container``.
        :return: A new subclass of View with the appropriate viewed container
                 properties, e.g., ``StructureView`` for a ``Structure``.
        """
        debuginfo = debuginfo or viewed_container.debuginfo
        # Construct the right kind of view from the input data container
        if isinstance(viewed_container, Structure):
            result = StructureView(members=cp.deepcopy(viewed_container.members),
                                   name=viewed_container.name,
                                   storage=viewed_container.storage,
                                   location=viewed_container.location,
                                   lifetime=dtypes.AllocationLifetime.Scope,
                                   debuginfo=debuginfo)
        elif isinstance(viewed_container, ContainerArray):
            result = ContainerView(stype=cp.deepcopy(viewed_container.stype),
                                   shape=viewed_container.shape,
                                   allow_conflicts=viewed_container.allow_conflicts,
                                   storage=viewed_container.storage,
                                   location=viewed_container.location,
                                   strides=viewed_container.strides,
                                   offset=viewed_container.offset,
                                   may_alias=viewed_container.may_alias,
                                   lifetime=dtypes.AllocationLifetime.Scope,
                                   alignment=viewed_container.alignment,
                                   debuginfo=debuginfo,
                                   total_size=viewed_container.total_size,
                                   start_offset=viewed_container.start_offset,
                                   optional=viewed_container.optional,
                                   pool=viewed_container.pool)
        elif isinstance(viewed_container, (Array, Scalar)):
            result = ArrayView(dtype=viewed_container.dtype,
                               shape=viewed_container.shape,
                               allow_conflicts=viewed_container.allow_conflicts,
                               storage=viewed_container.storage,
                               location=viewed_container.location,
                               strides=viewed_container.strides,
                               offset=viewed_container.offset,
                               may_alias=viewed_container.may_alias,
                               lifetime=dtypes.AllocationLifetime.Scope,
                               alignment=viewed_container.alignment,
                               debuginfo=debuginfo,
                               total_size=viewed_container.total_size,
                               start_offset=viewed_container.start_offset,
                               optional=viewed_container.optional,
                               pool=viewed_container.pool)
        else:
            # In undefined cases, make a container array view of size 1
            result = ContainerView(cp.deepcopy(viewed_container), [1], debuginfo=debuginfo)

        # Views are always transient
        result.transient = True
        return result


class Reference:
    """
    Data descriptor that acts as a dynamic reference of another data descriptor. It can be used just like a regular
    data descriptor, except that it could be set to an arbitrary container (or subset thereof) at runtime. To set a
    reference, connect another access node to it and use the "set" connector.

    In order to enable data-centric analysis and optimizations, avoid using References as much as possible.
    """

    @staticmethod
    def view(viewed_container: Data, debuginfo=None):
        """
        Create a new Reference of the specified data container.

        :param viewed_container: The data container properties of this reference.
        :param debuginfo: Specific source line information for this reference, if
                          different from ``viewed_container``.
        :return: A new subclass of View with the appropriate viewed container
                 properties, e.g., ``StructureReference`` for a ``Structure``.
        """
        result = cp.deepcopy(viewed_container)

        # Assign the right kind of reference from the input data container
        # NOTE: The class assignment below is OK since the Reference class is a subclass of the instance,
        # and those should not have additional fields.
        if isinstance(viewed_container, ContainerArray):
            result.__class__ = ContainerArrayReference
        elif isinstance(viewed_container, Structure):
            result.__class__ = StructureReference
        elif isinstance(viewed_container, Array):
            result.__class__ = ArrayReference
        elif isinstance(viewed_container, Scalar):
            result = ArrayReference(dtype=viewed_container.dtype,
                                    shape=[1],
                                    storage=viewed_container.storage,
                                    lifetime=viewed_container.lifetime,
                                    alignment=viewed_container.alignment,
                                    debuginfo=viewed_container.debuginfo,
                                    total_size=1,
                                    start_offset=0,
                                    optional=viewed_container.optional,
                                    pool=False,
                                    byval=False)
        else:  # In undefined cases, make a container array reference of size 1
            result = ContainerArrayReference(result, [1], debuginfo=debuginfo)

        if debuginfo is not None:
            result.debuginfo = debuginfo

        # References are always transient
        result.transient = True
        result.lifetime = dtypes.AllocationLifetime.Scope
        return result


@make_properties
class ArrayView(Array, View):
    """
    Data descriptor that acts as a static reference (or view) of another array. Can
    be used to reshape or reinterpret existing data without copying it.

    In the Python frontend, ``numpy.reshape`` and ``numpy.ndarray.view`` both
    generate ArrayViews.
    """

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for Views')

    def as_array(self):
        copy = cp.deepcopy(self)
        copy.__class__ = Array
        return copy


@make_properties
class StructureView(Structure, View):
    """
    Data descriptor that acts as a view of another structure.
    """

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'StructureView':
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = StructureView({})
        serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for Views')

    def as_structure(self):
        copy = cp.deepcopy(self)
        copy.__class__ = Structure
        return copy


@make_properties
class ContainerView(ContainerArray, View):
    """
    Data descriptor that acts as a view of another container array. Can
    be used to access nested container types without a copy.
    """

    def __init__(self,
                 stype: Data,
                 shape=None,
                 transient=True,
                 allow_conflicts=False,
                 storage=dtypes.StorageType.Default,
                 location=None,
                 strides=None,
                 offset=None,
                 may_alias=False,
                 lifetime=dtypes.AllocationLifetime.Scope,
                 alignment=0,
                 debuginfo=None,
                 total_size=None,
                 start_offset=None,
                 optional=None,
                 pool=False):
        shape = [1] if shape is None else shape
        super().__init__(stype, shape, transient, allow_conflicts, storage, location, strides, offset, may_alias,
                         lifetime, alignment, debuginfo, total_size, start_offset, optional, pool)

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for ContainerViews')

    def as_array(self):
        copy = cp.deepcopy(self)
        copy.__class__ = ContainerArray
        return copy


@make_properties
class ArrayReference(Array, Reference):
    """
    Data descriptor that acts as a dynamic reference of another array. See ``Reference`` for more information.

    In order to enable data-centric analysis and optimizations, avoid using References as much as possible.
    """

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for References')

    def as_array(self):
        copy = cp.deepcopy(self)
        copy.__class__ = Array
        return copy


@make_properties
class StructureReference(Structure, Reference):
    """
    Data descriptor that acts as a dynamic reference of another Structure. See ``Reference`` for more information.

    In order to enable data-centric analysis and optimizations, avoid using References as much as possible.
    """

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for References')

        if 'set' in self.members:
            raise NameError('A structure that is referenced may not contain a member called "set" (reserved keyword).')

    def as_structure(self):
        copy = cp.deepcopy(self)
        copy.__class__ = Structure
        return copy


@make_properties
class ContainerArrayReference(ContainerArray, Reference):
    """
    Data descriptor that acts as a dynamic reference of another data container array. See ``Reference`` for more
    information.

    In order to enable data-centric analysis and optimizations, avoid using References as much as possible.
    """

    def validate(self):
        super().validate()

        # We ensure that allocation lifetime is always set to Scope, since the
        # view is generated upon "allocation"
        if self.lifetime != dtypes.AllocationLifetime.Scope:
            raise ValueError('Only Scope allocation lifetime is supported for References')

    def as_array(self):
        copy = cp.deepcopy(self)
        copy.__class__ = ContainerArray
        return copy

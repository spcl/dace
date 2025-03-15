# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import aenum
import copy as cp
import ctypes
import functools

from abc import ABC, abstractmethod
from collections import OrderedDict
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy
import sympy as sp

try:
    from numpy.typing import ArrayLike
except (ModuleNotFoundError, ImportError):
    ArrayLike = Any

import dace.dtypes as dtypes
from dace import serialize, symbolic
from dace.codegen import cppunparse
from dace.properties import (DebugInfoProperty, DictProperty, EnumProperty, ListProperty, NestedDataClassProperty,
                             OrderedDictProperty, Property, ShapeProperty, SymbolicProperty, TypeClassProperty,
                             make_properties)


def create_datadescriptor(obj, no_custom_desc=False):
    """ Creates a data descriptor from various types of objects.

        :see: dace.data.Data
    """
    from dace import dtypes  # Avoiding import loops
    if isinstance(obj, Data):
        return obj
    elif not no_custom_desc and hasattr(obj, '__descriptor__'):
        return obj.__descriptor__()
    elif not no_custom_desc and hasattr(obj, 'descriptor'):
        return obj.descriptor
    elif type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor":
        # special case for torch tensors. Maybe __array__ could be used here for a more
        # general solution, but torch doesn't support __array__ for cuda tensors.
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

            storage = dtypes.StorageType.GPU_Global if obj.device.type == 'cuda' else dtypes.StorageType.Default

            return Array(dtype=TORCH_DTYPE_TO_TYPECLASS[obj.dtype],
                         strides=obj.stride(),
                         shape=tuple(obj.shape),
                         storage=storage)
        except ImportError:
            raise ValueError("Attempted to convert a torch.Tensor, but torch could not be imported")
    elif dtypes.is_array(obj) and (hasattr(obj, '__array_interface__') or hasattr(obj, '__cuda_array_interface__')):
        if dtypes.is_gpu_array(obj):
            interface = obj.__cuda_array_interface__
            storage = dtypes.StorageType.GPU_Global
        else:
            interface = obj.__array_interface__
            storage = dtypes.StorageType.Default

        if hasattr(obj, 'dtype') and obj.dtype.fields is not None:  # Struct
            dtype = dtypes.struct('unnamed', **{k: dtypes.typeclass(v[0].type) for k, v in obj.dtype.fields.items()})
        else:
            if numpy.dtype(interface['typestr']).type is numpy.void:  # Struct from __array_interface__
                if 'descr' in interface:
                    dtype = dtypes.struct('unnamed', **{
                        k: dtypes.typeclass(numpy.dtype(v).type)
                        for k, v in interface['descr']
                    })
                else:
                    raise TypeError(f'Cannot infer data type of array interface object "{interface}"')
            else:
                dtype = dtypes.typeclass(numpy.dtype(interface['typestr']).type)
        itemsize = numpy.dtype(interface['typestr']).itemsize
        if len(interface['shape']) == 0:
            return Scalar(dtype, storage=storage)
        return Array(dtype=dtype,
                     shape=interface['shape'],
                     strides=(tuple(s // itemsize for s in interface['strides']) if interface['strides'] else None),
                     storage=storage)
    elif isinstance(obj, (list, tuple)):
        # Lists and tuples are cast to numpy
        obj = numpy.array(obj)

        if obj.dtype.fields is not None:  # Struct
            dtype = dtypes.struct('unnamed', **{k: dtypes.typeclass(v[0].type) for k, v in obj.dtype.fields.items()})
        else:
            dtype = dtypes.typeclass(obj.dtype.type)
        return Array(dtype=dtype, strides=tuple(s // obj.itemsize for s in obj.strides), shape=obj.shape)
    elif type(obj).__module__ == "cupy" and type(obj).__name__ == "ndarray":
        # special case for CuPy and HIP, which does not support __cuda_array_interface__
        storage = dtypes.StorageType.GPU_Global
        dtype = dtypes.typeclass(obj.dtype.type)
        itemsize = obj.itemsize
        return Array(dtype=dtype, shape=obj.shape, strides=tuple(s // itemsize for s in obj.strides), storage=storage)
    elif symbolic.issymbolic(obj):
        return Scalar(symbolic.symtype(obj))
    elif isinstance(obj, dtypes.typeclass):
        return Scalar(obj)
    elif (obj is int or obj is float or obj is complex or obj is bool or obj is None):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, type) and issubclass(obj, numpy.number):
        return Scalar(dtypes.typeclass(obj))
    elif isinstance(obj, (Number, numpy.number, numpy.bool_)):
        return Scalar(dtypes.typeclass(type(obj)))
    elif obj is type(None):
        # NoneType is void *
        return Scalar(dtypes.pointer(dtypes.typeclass(None)))
    elif isinstance(obj, str) or obj is str:
        return Scalar(dtypes.string)
    elif callable(obj):
        # Cannot determine return value/argument types from function object
        return Scalar(dtypes.callback(None))

    raise TypeError(f'Could not create a DaCe data descriptor from object {obj}. '
                    'If this is a custom object, consider creating a `__descriptor__` '
                    'adaptor method to the type hint or object itself.')


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

    dtype = TypeClassProperty(default=dtypes.int32, choices=dtypes.Typeclasses)
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


def _arrays_to_json(arrays):
    if arrays is None:
        return None
    return [(k, serialize.to_json(v)) for k, v in arrays.items()]


def _arrays_from_json(obj, context=None):
    if obj is None:
        return {}
    return OrderedDict((k, serialize.from_json(v, context)) for k, v in obj)


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
            elif isinstance(v, (sp.Basic, symbolic.SymExpr)):
                symbols |= v.free_symbols
                fields_and_types[k] = symbolic.symtype(v)
            elif isinstance(v, (int, numpy.integer)):
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


class TensorIterationTypes(aenum.AutoNumberEnum):
    """
    Types of tensor iteration capabilities.

    Value (Coordinate Value Iteration) allows to directly iterate over
    coordinates such as when using the Dense index type.

    Position (Coordinate Position Iteratation) iterates over coordinate
    positions, at which the actual coordinates lie. This is for example the case
    with a compressed index, in which the pos array enables one to iterate over
    the positions in the crd array that hold the actual coordinates.
    """
    Value = ()
    Position = ()


class TensorAssemblyType(aenum.AutoNumberEnum):
    """
    Types of possible assembly strategies for the individual indices.

    NoAssembly: Assembly is not possible as such.

    Insert: index allows inserting elements at random (e.g. Dense)

    Append: index allows appending to a list of existing coordinates. Depending
    on append order, this affects whether the index is ordered or not. This
    could be changed by sorting the index after assembly
    """
    NoAssembly = ()
    Insert = ()
    Append = ()


class TensorIndex(ABC):
    """
    Abstract base class for tensor index implementations.
    """

    @property
    @abstractmethod
    def iteration_type(self) -> TensorIterationTypes:
        """
        Iteration capability supported by this index.

        See TensorIterationTypes for reference.
        """
        pass

    @property
    @abstractmethod
    def locate(self) -> bool:
        """
        True if the index supports locate (aka random access), False otw.
        """
        pass

    @property
    @abstractmethod
    def assembly(self) -> TensorAssemblyType:
        """
        What assembly type is supported by the index.

        See TensorAssemblyType for reference.
        """
        pass

    @property
    @abstractmethod
    def full(self) -> bool:
        """
        True if the level is full, False otw.

        A level is considered full if it encompasses all valid coordinates along
        the corresponding tensor dimension.
        """
        pass

    @property
    @abstractmethod
    def ordered(self) -> bool:
        """
        True if the level is ordered, False otw.

        A level is ordered when all coordinates that share the same ancestor are
        ordered by increasing value (e.g. in typical CSR).
        """
        pass

    @property
    @abstractmethod
    def unique(self) -> bool:
        """
        True if coordinate in the level are unique, False otw.

        A level is considered unique if no collection of coordinates that share
        the same ancestor contains duplicates. In CSR this is True, in COO it is
        not.
        """
        pass

    @property
    @abstractmethod
    def branchless(self) -> bool:
        """
        True if the level doesn't branch, false otw.

        A level is considered branchless if no coordinate has a sibling (another
        coordinate with same ancestor) and all coordinates in parent level have
        a child. In other words if there is a bijection between the coordinates
        in this level and the parent level. An example of the is the Singleton
        index level in the COO format.
        """
        pass

    @property
    @abstractmethod
    def compact(self) -> bool:
        """
        True if the level is compact, false otw.

        A level is compact if no two coordinates are separated by an unlabled
        node that does not encode a coordinate. An example of a compact level
        can be found in CSR, while the DIA formats range and offset levels are
        not compact (they have entries that would coorespond to entries outside
        the tensors index range, e.g. column -1).
        """
        pass

    @abstractmethod
    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        """
        Generates the fields needed for the index.

        :return: a Dict of fields that need to be present in the struct
        """
        pass

    def to_json(self):
        attrs = serialize.all_properties_to_json(self)

        retdict = {"type": type(self).__name__, "attributes": attrs}

        return retdict

    @classmethod
    def from_json(cls, json_obj, context=None):

        # Selecting proper subclass
        if json_obj['type'] == "TensorIndexDense":
            self = TensorIndexDense.__new__(TensorIndexDense)
        elif json_obj['type'] == "TensorIndexCompressed":
            self = TensorIndexCompressed.__new__(TensorIndexCompressed)
        elif json_obj['type'] == "TensorIndexSingleton":
            self = TensorIndexSingleton.__new__(TensorIndexSingleton)
        elif json_obj['type'] == "TensorIndexRange":
            self = TensorIndexRange.__new__(TensorIndexRange)
        elif json_obj['type'] == "TensorIndexOffset":
            self = TensorIndexOffset.__new__(TensorIndexOffset)
        else:
            raise TypeError(f"Invalid data type, got: {json_obj['type']}")

        serialize.set_properties_from_json(self, json_obj['attributes'], context=context)

        return self


@make_properties
class TensorIndexDense(TensorIndex):
    """
    Dense tensor index.

    Levels of this type encode the the coordinate in the interval [0, N), where
    N is the size of the corresponding dimension. This level doesn't need any
    index structure beyond the corresponding dimension size.
    """

    _ordered = Property(dtype=bool, default=False)
    _unique = Property(dtype=bool)

    @property
    def iteration_type(self) -> TensorIterationTypes:
        return TensorIterationTypes.Value

    @property
    def locate(self) -> bool:
        return True

    @property
    def assembly(self) -> TensorAssemblyType:
        return TensorAssemblyType.Insert

    @property
    def full(self) -> bool:
        return True

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return True

    def __init__(self, ordered: bool = True, unique: bool = True):
        self._ordered = ordered
        self._unique = unique

    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        return {}

    def __repr__(self) -> str:
        s = "Dense"

        non_defaults = []
        if not self._ordered:
            non_defaults.append("¬O")
        if not self._unique:
            non_defaults.append("¬U")

        if len(non_defaults) > 0:
            s += f"({','.join(non_defaults)})"

        return s


@make_properties
class TensorIndexCompressed(TensorIndex):
    """
    Tensor level that stores coordinates in segmented array.

    Levels of this type are compressed using a segented array. The pos array
    holds the start and end positions of the segment in the crd (coordinate)
    array that holds the child coordinates corresponding the parent.
    """

    _full = Property(dtype=bool, default=False)
    _ordered = Property(dtype=bool, default=False)
    _unique = Property(dtype=bool, default=False)

    @property
    def iteration_type(self) -> TensorIterationTypes:
        return TensorIterationTypes.Position

    @property
    def locate(self) -> bool:
        return False

    @property
    def assembly(self) -> TensorAssemblyType:
        return TensorAssemblyType.Append

    @property
    def full(self) -> bool:
        return self._full

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return True

    def __init__(self, full: bool = False, ordered: bool = True, unique: bool = True):
        self._full = full
        self._ordered = ordered
        self._unique = unique

    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        return {
            f"idx{lvl}_pos": dtypes.int32[dummy_symbol],  # TODO (later) choose better length
            f"idx{lvl}_crd": dtypes.int32[dummy_symbol],  # TODO (later) choose better length
        }

    def __repr__(self) -> str:
        s = "Compressed"

        non_defaults = []
        if self._full:
            non_defaults.append("F")
        if not self._ordered:
            non_defaults.append("¬O")
        if not self._unique:
            non_defaults.append("¬U")

        if len(non_defaults) > 0:
            s += f"({','.join(non_defaults)})"

        return s


@make_properties
class TensorIndexSingleton(TensorIndex):
    """
    Tensor index that encodes a single coordinate per parent coordinate.

    Levels of this type hold exactly one coordinate for every coordinate in the
    parent level. An example can be seen in the COO format, where every
    coordinate but the first is encoded in this manner.
    """

    _full = Property(dtype=bool, default=False)
    _ordered = Property(dtype=bool, default=False)
    _unique = Property(dtype=bool, default=False)

    @property
    def iteration_type(self) -> TensorIterationTypes:
        return TensorIterationTypes.Position

    @property
    def locate(self) -> bool:
        return False

    @property
    def assembly(self) -> TensorAssemblyType:
        return TensorAssemblyType.Append

    @property
    def full(self) -> bool:
        return self._full

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return True

    @property
    def compact(self) -> bool:
        return True

    def __init__(self, full: bool = False, ordered: bool = True, unique: bool = True):
        self._full = full
        self._ordered = ordered
        self._unique = unique

    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        return {
            f"idx{lvl}_crd": dtypes.int32[dummy_symbol],  # TODO (later) choose better length
        }

    def __repr__(self) -> str:
        s = "Singleton"

        non_defaults = []
        if self._full:
            non_defaults.append("F")
        if not self._ordered:
            non_defaults.append("¬O")
        if not self._unique:
            non_defaults.append("¬U")

        if len(non_defaults) > 0:
            s += f"({','.join(non_defaults)})"

        return s


@make_properties
class TensorIndexRange(TensorIndex):
    """
    Tensor index that encodes a interval of coordinates for every parent.

    The interval is computed from an offset for each parent together with the
    tensor dimension size of this level (M) and the parent level (N) parents
    corresponding tensor. Given the parent coordinate i, the level encodes the
    range of coordinates between max(0, -offset[i]) and min(N, M - offset[i]).
    """

    _ordered = Property(dtype=bool, default=False)
    _unique = Property(dtype=bool, default=False)

    @property
    def iteration_type(self) -> TensorIterationTypes:
        return TensorIterationTypes.Value

    @property
    def locate(self) -> bool:
        return False

    @property
    def assembly(self) -> TensorAssemblyType:
        return TensorAssemblyType.NoAssembly

    @property
    def full(self) -> bool:
        return False

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return False

    def __init__(self, ordered: bool = True, unique: bool = True):
        self._ordered = ordered
        self._unique = unique

    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        return {
            f"idx{lvl}_offset": dtypes.int32[dummy_symbol],  # TODO (later) choose better length
        }

    def __repr__(self) -> str:
        s = "Range"

        non_defaults = []
        if not self._ordered:
            non_defaults.append("¬O")
        if not self._unique:
            non_defaults.append("¬U")

        if len(non_defaults) > 0:
            s += f"({','.join(non_defaults)})"

        return s


@make_properties
class TensorIndexOffset(TensorIndex):
    """
    Tensor index that encodes the next coordinates as offset from parent.

    Given a parent coordinate i and an offset index k, the level encodes the
    coordinate j = i + offset[k].
    """

    _ordered = Property(dtype=bool, default=False)
    _unique = Property(dtype=bool, default=False)

    @property
    def iteration_type(self) -> TensorIterationTypes:
        return TensorIterationTypes.Position

    @property
    def locate(self) -> bool:
        return False

    @property
    def assembly(self) -> TensorAssemblyType:
        return TensorAssemblyType.NoAssembly

    @property
    def full(self) -> bool:
        return False

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return True

    @property
    def compact(self) -> bool:
        return False

    def __init__(self, ordered: bool = True, unique: bool = True):
        self._ordered = ordered
        self._unique = unique

    def fields(self, lvl: int, dummy_symbol: symbolic.SymExpr) -> Dict[str, Data]:
        return {
            f"idx{lvl}_offset": dtypes.int32[dummy_symbol],  # TODO (later) choose better length
        }

    def __repr__(self) -> str:
        s = "Offset"

        non_defaults = []
        if not self._ordered:
            non_defaults.append("¬O")
        if not self._unique:
            non_defaults.append("¬U")

        if len(non_defaults) > 0:
            s += f"({','.join(non_defaults)})"

        return s


@make_properties
class Tensor(Structure):
    """
    Abstraction for Tensor storage format.

    This abstraction is based on [https://doi.org/10.1145/3276493].
    """

    value_dtype = TypeClassProperty(default=dtypes.int32, choices=dtypes.Typeclasses)
    tensor_shape = ShapeProperty(default=[])
    indices = ListProperty(element_type=TensorIndex)
    index_ordering = ListProperty(element_type=symbolic.SymExpr)
    value_count = SymbolicProperty(default=0)

    def __init__(self,
                 value_dtype: dtypes.Typeclasses,
                 tensor_shape,
                 indices: List[Tuple[TensorIndex, Union[int, symbolic.SymExpr]]],
                 value_count: symbolic.SymExpr,
                 name: str,
                 transient: bool = False,
                 storage: dtypes.StorageType = dtypes.StorageType.Default,
                 location: Dict[str, str] = None,
                 lifetime: dtypes.AllocationLifetime = dtypes.AllocationLifetime.Scope,
                 debuginfo: dtypes.DebugInfo = None):
        """
        Constructor for Tensor storage format.

        Below are examples of common matrix storage formats:

        .. code-block:: python

            M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))

            csr = dace.data.Tensor(
                dace.float32,
                (M, N),
                [(dace.data.Dense(), 0), (dace.data.Compressed(), 1)],
                nnz,
                "CSR_Matrix",
            )

            csc = dace.data.Tensor(
                dace.float32,
                (M, N),
                [(dace.data.Dense(), 1), (dace.data.Compressed(), 0)],
                nnz,
                "CSC_Matrix",
            )

            coo = dace.data.Tensor(
                dace.float32,
                (M, N),
                [
                    (dace.data.Compressed(unique=False), 0),
                    (dace.data.Singleton(), 1),
                ],
                nnz,
                "CSC_Matrix",
            )

            num_diags = dace.symbol('num_diags')  # number of diagonals stored

            diag = dace.data.Tensor(
                dace.float32,
                (M, N),
                [
                    (dace.data.Dense(), num_diags),
                    (dace.data.Range(), 0),
                    (dace.data.Offset(), 1),
                ],
                nnz,
                "DIA_Matrix",
            )

        Below you can find examples of common 3rd order tensor storage formats:

        .. code-block:: python

            I, J, K, nnz = (dace.symbol(s) for s in ('I', 'J', 'K', 'nnz'))

            coo = dace.data.Tensor(
                dace.float32,
                (I, J, K),
                [
                    (dace.data.Compressed(unique=False), 0),
                    (dace.data.Singleton(unique=False), 1),
                    (dace.data.Singleton(), 2),
                ],
                nnz,
                "COO_3D_Tensor",
            )

            csf = dace.data.Tensor(
                dace.float32,
                (I, J, K),
                [
                    (dace.data.Compressed(), 0),
                    (dace.data.Compressed(), 1),
                    (dace.data.Compressed(), 2),
                ],
                nnz,
                "CSF_3D_Tensor",
            )

        :param value_type: data type of the explicitly stored values.
        :param tensor_shape: logical shape of tensor (#rows, #cols, etc...)
        :param indices:
            a list of tuples, each tuple represents a level in the tensor
            storage hirachy, specifying the levels tensor index type, and the
            corresponding dimension this level encodes (as index of the
            tensor_shape tuple above). The order of the dimensions may differ
            from the logical shape of the tensor, e.g. as seen in the CSC
            format. If an index's dimension is unrelated to the tensor shape
            (e.g. in diagonal format where the first index's dimension is the
            number of diagonals stored), a symbol can be specified instead.
        :param value_count: number of explicitly stored values.
        :param name: name of resulting struct.
        :param others: See Structure class for remaining arguments
        """

        self.value_dtype = value_dtype
        self.tensor_shape = tensor_shape
        self.value_count = value_count

        indices, index_ordering = zip(*indices)
        self.indices, self.index_ordering = list(indices), list(index_ordering)

        num_dims = len(tensor_shape)
        dimension_order = [idx for idx in self.index_ordering if isinstance(idx, int)]

        # all tensor dimensions must occure exactly once in indices
        if not sorted(dimension_order) == list(range(num_dims)):
            raise TypeError((f"All tensor dimensions must be refferenced exactly once in "
                             f"tensor indices. (referenced dimensions: {dimension_order}; "
                             f"tensor dimensions: {list(range(num_dims))})"))

        # assembling permanent and index specific fields
        fields = dict(
            order=Scalar(dtypes.int32),
            dim_sizes=dtypes.int32[num_dims],
            value_count=value_count,
            values=dtypes.float32[value_count],
        )

        for (lvl, index) in enumerate(indices):
            fields.update(index.fields(lvl, value_count))

        super(Tensor, self).__init__(fields, name, transient, storage, location, lifetime, debuginfo)

    def __repr__(self):
        return f"{self.name} (dtype: {self.value_dtype}, shape: {list(self.tensor_shape)}, indices: {self.indices})"

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != 'Tensor':
            raise TypeError("Invalid data type")

        # Create dummy object
        tensor = Tensor.__new__(Tensor)
        serialize.set_properties_from_json(tensor, json_obj, context=context)

        return tensor


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
        return (" * ".join([cppunparse.pyexpr2cpp(symbolic.symstr(s, cpp_mode=True)) for s in self.shape]))

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
                                   lifetime=viewed_container.lifetime,
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
                                   lifetime=viewed_container.lifetime,
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
                               lifetime=viewed_container.lifetime,
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


def make_array_from_descriptor(descriptor: Array,
                               original_array: Optional[ArrayLike] = None,
                               symbols: Optional[Dict[str, Any]] = None) -> ArrayLike:
    """
    Creates an array that matches the given data descriptor, and optionally copies another array to it.

    :param descriptor: The data descriptor to create the array from.
    :param original_array: An optional array to fill the content of the return value with.
    :param symbols: An optional symbol mapping between symbol names and their values. Used for creating arrays
                    with symbolic sizes.
    :return: A NumPy-compatible array (CuPy for GPU storage) with the specified size and strides.
    """
    import numpy as np

    symbols = symbols or {}

    free_syms = set(map(str, descriptor.free_symbols)) - symbols.keys()
    if free_syms:
        raise NotImplementedError(f'Cannot make Python references to arrays with undefined symbolic sizes: {free_syms}')

    if descriptor.storage == dtypes.StorageType.GPU_Global:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError):
            raise NotImplementedError('GPU memory can only be allocated in Python if cupy is installed')

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = cp.ndarray(shape=[total_size], dtype=dtype)
            view = cp.ndarray(shape=shape,
                              dtype=dtype,
                              memptr=buffer.data,
                              strides=[s * dtype.itemsize for s in strides])
            return view

        def copy_array(dst, src):
            dst[:] = cp.asarray(src)

    elif descriptor.storage == dtypes.StorageType.FPGA_Global:
        raise TypeError('Cannot allocate FPGA array in Python')
    else:

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = np.ndarray([total_size], dtype=dtype)
            view = np.ndarray(shape, dtype, buffer=buffer, strides=[s * dtype.itemsize for s in strides])
            return view

        def copy_array(dst, src):
            dst[:] = src

    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    evaluated_shape = tuple(symbolic.evaluate(s, symbols) for s in descriptor.shape)
    evaluated_size = symbolic.evaluate(descriptor.total_size, symbols)
    evaluated_strides = tuple(symbolic.evaluate(s, symbols) for s in descriptor.strides)
    view = create_array(evaluated_shape, npdtype, evaluated_size, evaluated_strides)
    if original_array is not None:
        copy_array(view, original_array)

    return view


def make_reference_from_descriptor(descriptor: Array,
                                   original_array: ctypes.c_void_p,
                                   symbols: Optional[Dict[str, Any]] = None) -> ArrayLike:
    """
    Creates an array that matches the given data descriptor from the given pointer. Shares the memory
    with the argument (does not create a copy).

    :param descriptor: The data descriptor to create the array from.
    :param original_array: The array whose memory the return value would be used in.
    :param symbols: An optional symbol mapping between symbol names and their values. Used for referencing arrays
                    with symbolic sizes.
    :return: A NumPy-compatible array (CuPy for GPU storage) with the specified size and strides, sharing memory
             with the pointer specified in ``original_array``.
    """
    import numpy as np
    symbols = symbols or {}

    original_array: int = ctypes.cast(original_array, ctypes.c_void_p).value

    free_syms = set(map(str, descriptor.free_symbols)) - symbols.keys()
    if free_syms:
        raise NotImplementedError(f'Cannot make Python references to arrays with undefined symbolic sizes: {free_syms}')

    if descriptor.storage == dtypes.StorageType.GPU_Global:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError):
            raise NotImplementedError('GPU memory can only be referenced in Python if cupy is installed')

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = dtypes.ptrtocupy(original_array, descriptor.dtype.as_ctypes(), (total_size, ))
            view = cp.ndarray(shape=shape,
                              dtype=dtype,
                              memptr=buffer.data,
                              strides=[s * dtype.itemsize for s in strides])
            return view

    elif descriptor.storage == dtypes.StorageType.FPGA_Global:
        raise TypeError('Cannot reference FPGA array in Python')
    else:

        def create_array(shape: Tuple[int], dtype: np.dtype, total_size: int, strides: Tuple[int]) -> ArrayLike:
            buffer = dtypes.ptrtonumpy(original_array, descriptor.dtype.as_ctypes(), (total_size, ))
            view = np.ndarray(shape, dtype, buffer=buffer, strides=[s * dtype.itemsize for s in strides])
            return view

    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    evaluated_shape = tuple(symbolic.evaluate(s, symbols) for s in descriptor.shape)
    evaluated_size = symbolic.evaluate(descriptor.total_size, symbols)
    evaluated_strides = tuple(symbolic.evaluate(s, symbols) for s in descriptor.strides)
    return create_array(evaluated_shape, npdtype, evaluated_size, evaluated_strides)

# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tensor data descriptors for sparse tensor formats.

This module contains classes for representing various sparse tensor storage formats
based on the abstraction described in [https://doi.org/10.1145/3276493].
"""
import enum

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from dace import dtypes, serialize, symbolic
from dace.data.core import Data, Scalar, Structure
from dace.properties import ListProperty, Property, ShapeProperty, SymbolicProperty, TypeClassProperty, make_properties


class TensorIterationTypes(enum.Enum):
    """
    Types of tensor iteration capabilities.

    Value (Coordinate Value Iteration) allows to directly iterate over
    coordinates such as when using the Dense index type.

    Position (Coordinate Position Iteratation) iterates over coordinate
    positions, at which the actual coordinates lie. This is for example the case
    with a compressed index, in which the pos array enables one to iterate over
    the positions in the crd array that hold the actual coordinates.
    """
    Value = enum.auto()
    Position = enum.auto()


class TensorAssemblyType(enum.Enum):
    """
    Types of possible assembly strategies for the individual indices.

    NoAssembly: Assembly is not possible as such.

    Insert: index allows inserting elements at random (e.g. Dense)

    Append: index allows appending to a list of existing coordinates. Depending
    on append order, this affects whether the index is ordered or not. This
    could be changed by sorting the index after assembly
    """
    NoAssembly = enum.auto()
    Insert = enum.auto()
    Append = enum.auto()


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

    value_dtype = TypeClassProperty(default=dtypes.int32)
    tensor_shape = ShapeProperty(default=[])
    indices = ListProperty(element_type=TensorIndex)
    index_ordering = ListProperty(element_type=symbolic.SymExpr)
    value_count = SymbolicProperty(default=0)

    def __init__(self,
                 value_dtype: dtypes.typeclass,
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

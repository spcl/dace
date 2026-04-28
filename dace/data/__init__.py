# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Data descriptors for DaCe.

This package contains classes for describing data containers (arrays, scalars, streams, etc.)
that can be used in SDFGs. The classes in this package are used to specify the shape, type,
and storage location of data, as well as other properties that affect code generation.

For backward compatibility, all classes and functions are re-exported from the top-level
`dace.data` module.
"""

# Core data descriptors
from dace.data.core import (
    Data,
    Scalar,
    Array,
    ContainerArray,
    Stream,
    Structure,
    View,
    Reference,
    ArrayView,
    StructureView,
    ContainerView,
    ArrayReference,
    StructureReference,
    ContainerArrayReference,
)

# Import prod from utils and expose as _prod for backward compatibility
from dace.utils import prod as _prod

# Tensor/sparse tensor support
from dace.data.tensor import (
    TensorIterationTypes,
    TensorAssemblyType,
    TensorIndex,
    TensorIndexDense,
    TensorIndexCompressed,
    TensorIndexSingleton,
    TensorIndexRange,
    TensorIndexOffset,
    Tensor,
)

# Convenience aliases for tensor indices
Dense = TensorIndexDense
Compressed = TensorIndexCompressed
Singleton = TensorIndexSingleton
Range = TensorIndexRange
Offset = TensorIndexOffset

# ML-related data descriptors
from dace.data.ml import ParameterArray

# Descriptor creation and array creation from descriptors
from dace.data.creation import (
    create_datadescriptor,
    make_array_from_descriptor,
    make_reference_from_descriptor,
)

# Ctypes interoperability
from dace.data.ctypes_interop import make_ctypes_argument

# Import utility function from utils (for backward compatibility)
from dace.utils import find_new_name

__all__ = [
    # Core classes
    'Data',
    'Scalar',
    'Array',
    'ContainerArray',
    'Stream',
    'Structure',
    'View',
    'Reference',
    'ArrayView',
    'StructureView',
    'ContainerView',
    'ArrayReference',
    'StructureReference',
    'ContainerArrayReference',
    # Tensor support
    'TensorIterationTypes',
    'TensorAssemblyType',
    'TensorIndex',
    'TensorIndexDense',
    'TensorIndexCompressed',
    'TensorIndexSingleton',
    'TensorIndexRange',
    'TensorIndexOffset',
    'Tensor',
    # Tensor aliases
    'Dense',
    'Compressed',
    'Singleton',
    'Range',
    'Offset',
    # ML descriptors
    'ParameterArray',
    # Functions
    'create_datadescriptor',
    'make_array_from_descriptor',
    'make_reference_from_descriptor',
    'make_ctypes_argument',
    'find_new_name',
]

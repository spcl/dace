# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    Type compatibility: This module offers a function `assert_type_compatibility` that ensures SVE instructions can work with the given datatypes.
    It throws an IncompatibleTypeError which provides some information.

    It is called almost everywhere in the SVEUnparser, because it checks whether:
        - The types can be represented in SVE
        - An instruction on them can be used (e.g. svint32_t and svfloat32_t in the same context are fundamentally incompatible)
        - Any `None` occurs, which is a hint for failed inference
        - Pointers are mixed with vectors or scalars
"""

import dace
import dace.dtypes as dtypes
from dace.codegen.targets.sve import util as util
import collections


class IncompatibleTypeError(Exception):
    def __init__(self, message, types):
        super().__init__(f'{message}; given: {types}')


def assert_type_compatibility(defined_symbols: collections.OrderedDict, types: tuple):
    """
    This method ensures that SVE can work with the given types.
    This is sometimes more, sometimes less restrictive than C standards.
    """

    # Sanity check for any failed inference
    if None in types:
        raise IncompatibleTypeError('`None` was given', types)

    # Find all unique vector, pointer and scalar types
    # TODO: Better way to determine uniqueness
    vec_types = list(set([t for t in types if isinstance(t, dtypes.vector)]))
    ptr_types = list(set([t for t in types if isinstance(t, dtypes.pointer)]))
    scal_types = list(set([t for t in types if not isinstance(t, (dtypes.vector, dtypes.pointer))]))

    # Check if we can represent the types in SVE
    for t in types:
        if util.get_base_type(t).type not in util.TYPE_TO_SVE:
            raise IncompatibleTypeError('Not available in SVE', types)

    # Check if we have different vector types (would require casting, not implemented yet)
    if len(vec_types) > 1:
        raise IncompatibleTypeError('Vectors of different type', types)

    # Ensure no mixing of pointers and vectors/scalars ever occurs (totally incompatible)
    if (len(vec_types) != 0 or len(scal_types) != 0) and len(ptr_types) != 0:
        raise IncompatibleTypeError('Vectors/scalars are incompatible with pointers', types)

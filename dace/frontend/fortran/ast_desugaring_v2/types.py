# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Any, Type

import numpy as np
from fparser.two.Fortran2003 import Real_Literal_Constant, Signed_Real_Literal_Constant, Int_Literal_Constant, \
    Signed_Int_Literal_Constant, Logical_Literal_Constant

# Type Aliases for fparser node types are defined in other modules where they are used.
# This file is for our own custom types.

# A SPEC is a tuple of strings that uniquely identifies an object in the AST.
# For example, ('my_program', 'my_module', 'my_subroutine', 'my_variable')
SPEC = Tuple[str, ...]

# A SPEC_TABLE maps a SPEC to the fparser node that defines it.
# Forward-referencing 'NAMED_STMTS_OF_INTEREST_TYPES' from utils as a string.
SPEC_TABLE = Dict[SPEC, 'NAMED_STMTS_OF_INTEREST_TYPES']

# Type Aliases for numpy types used in constant evaluation
NUMPY_INTS_TYPES = Union[np.int8, np.int16, np.int32, np.int64]
NUMPY_INTS = (np.int8, np.int16, np.int32, np.int64)
NUMPY_REALS = (np.float32, np.float64)
NUMPY_REALS_TYPES = Union[np.float32, np.float64]
NUMPY_TYPES = Union[NUMPY_INTS_TYPES, NUMPY_REALS_TYPES, np.bool_]

# Type Aliases for fparser literal nodes.
LITERAL_TYPES = Union[Real_Literal_Constant, Signed_Real_Literal_Constant, Int_Literal_Constant,
                      Signed_Int_Literal_Constant, Logical_Literal_Constant]
LITERAL_CLASSES = (Real_Literal_Constant, Signed_Real_Literal_Constant, Int_Literal_Constant,
                   Signed_Int_Literal_Constant, Logical_Literal_Constant)


class TYPE_SPEC:
    """
    A simplified representation of a Fortran variable's type and attributes.

    This class parses the attribute string of a variable declaration and provides
    a structured way to access its properties like shape, intent, etc.
    """
    NO_ATTRS = ''

    def __init__(self, spec: Union[str, SPEC], attrs: str = NO_ATTRS, is_arg: bool = False):
        """
        Initializes a TYPE_SPEC object.

        :param spec: The base type specifier, can be a string or a SPEC tuple.
        :param attrs: A string containing the variable's attributes (e.g., 'DIMENSION(..)', 'INTENT(IN)').
        :param is_arg: A boolean indicating if the variable is a subroutine/function argument.
        """
        if isinstance(spec, str):
            spec = (spec, )
        self.spec: SPEC = spec
        self.shape: Tuple[str, ...] = self._parse_shape(attrs)
        self.optional: bool = 'OPTIONAL' in attrs
        self.pointer: bool = 'POINTER' in attrs
        self.inp: bool = 'INTENT(IN)' in attrs or 'INTENT(INOUT)' in attrs
        self.out: bool = 'INTENT(OUT)' in attrs or 'INTENT(INOUT)' in attrs
        self.alloc: bool = 'ALLOCATABLE' in attrs
        self.const: bool = 'PARAMETER' in attrs
        self.keyword: Optional[str] = None
        if is_arg and not self.inp and not self.out:
            # If it's an argument and intent is not specified, it can be both input and output.
            self.inp, self.out = True, True

    @staticmethod
    def _parse_shape(attrs: str) -> Tuple[str, ...]:
        """
        Parses the DIMENSION attribute to extract the shape of the variable.

        :param attrs: The attribute string.
        :return: A tuple of strings representing the dimensions.
        """
        if 'DIMENSION' not in attrs:
            return tuple()
        parts = []
        dims = attrs.split('DIMENSION')[1]
        assert dims[0] == '('
        paren_count, part_start = 1, 1
        for i in range(1, len(dims)):
            if dims[i] == '(':
                paren_count += 1
            elif dims[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    parts.append(dims[part_start:i])
                    break
            elif dims[i] == ',':
                if paren_count == 1:
                    parts.append(dims[part_start:i])
                    part_start = i + 1
        return tuple(p.strip().lower() for p in parts)

    def __repr__(self):
        attrs = []
        if self.pointer:
            attrs.append("*")
        if self.shape:
            attrs.append(f"shape={self.shape}")
        if self.optional:
            attrs.append("optional")
        if not attrs:
            return f"{self.spec}"
        return f"{self.spec}[{' | '.join(attrs)}]"

    def to_decl(self, var: str) -> str:
        """
        Generates a Fortran declaration string for a variable with this type spec.

        :param var: The name of the variable.
        :return: A Fortran declaration string.
        """
        TYPE_MAP = {
            'INTEGER1': 'INTEGER(kind=1)',
            'INTEGER2': 'INTEGER(kind=2)',
            'INTEGER4': 'INTEGER(kind=4)',
            'INTEGER8': 'INTEGER(kind=8)',
            'INTEGER': 'INTEGER(kind=4)',
            'REAL4': 'REAL(kind=4)',
            'REAL8': 'REAL(kind=8)',
            'REAL': 'REAL(kind=4)',
            'LOGICAL': 'LOGICAL',
        }
        typ = self.spec[-1]
        typ = TYPE_MAP.get(typ, f"type({typ})")

        bits: List[str] = [typ]
        if self.alloc:
            bits.append('allocatable')
        if self.optional:
            bits.append('optional')
        if self.inp and self.out:
            bits.append('intent(inout)')
        elif self.inp:
            bits.append('intent(in)')
        elif self.out:
            bits.append('intent(out)')
        if self.const:
            bits.append('parameter')
        bits_str: str = ', '.join(bits)
        shape_str: str = ', '.join(self.shape) if self.shape else ''
        shape_str = f"({shape_str})" if shape_str else ''
        return f"{bits_str} :: {var}{shape_str}"


@dataclass
class ConstTypeInjection:
    """
    Represents the injection of a constant value for a component within a derived type.
    This is used to replace a component's value everywhere a certain derived type is used,
    optionally within a specific scope.
    """
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    type_spec: SPEC  # The root config derived type's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


@dataclass
class ConstInstanceInjection:
    """
    Represents the injection of a constant value for a component of a specific variable instance.
    This is used to replace a component's value for a particular variable, not all instances of its type.
    """
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    root_spec: SPEC  # The root config object's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


ConstInjection = Union[ConstTypeInjection, ConstInstanceInjection]


def numpy_type_to_literal(val: NUMPY_TYPES) -> LITERAL_TYPES:
    """
    Converts a numpy scalar value (int, float, or bool) into its corresponding
    fparser literal node representation.

    :param val: The numpy scalar value.
    :return: An fparser node representing the literal.
    """
    if isinstance(val, np.bool_):
        val = Logical_Literal_Constant('.true.' if val else '.false.')
    elif isinstance(val, NUMPY_INTS):
        bytez = _count_bytes(type(val))
        if val < 0:
            val = Signed_Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
        else:
            val = Int_Literal_Constant(f"{val}" if bytez == 4 else f"{val}_{bytez}")
    elif isinstance(val, NUMPY_REALS):
        bytez = _count_bytes(type(val))
        valstr = str(val)
        if bytez == 8:
            if 'e' in valstr:
                valstr = valstr.replace('e', 'D')
            else:
                valstr = f"{valstr}D0"
        if val < 0:
            val = Signed_Real_Literal_Constant(valstr)
        else:
            val = Real_Literal_Constant(valstr)
    return val


def _count_bytes(t: Type[NUMPY_TYPES]) -> int:
    """
    Returns the number of bytes for a given numpy numeric type.

    :param t: The numpy type.
    :return: The size of the type in bytes.
    """
    if t is np.int8: return 1
    if t is np.int16: return 2
    if t is np.int32: return 4
    if t is np.int64: return 8
    if t is np.float32: return 4
    if t is np.float64: return 8
    if t is np.bool_: return 1
    raise ValueError(f"{t} is not an expected type; expected {NUMPY_TYPES}")

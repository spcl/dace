# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Any

import numpy as np

# Type Aliases for fparser node types are defined in other modules where they are used.
# This file is for our own custom types.

# A SPEC is a tuple of strings that uniquely identifies an object in the AST.
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


class TYPE_SPEC:
    """A simplified representation of a Fortran variable's type and attributes."""
    NO_ATTRS = ''

    def __init__(self, spec: Union[str, SPEC], attrs: str = NO_ATTRS, is_arg: bool = False):
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
            self.inp, self.out = True, True

    @staticmethod
    def _parse_shape(attrs: str) -> Tuple[str, ...]:
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

    def to_decl(self, var: str):
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
    """Injects a constant value for a component within a specific derived type."""
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    type_spec: SPEC  # The root config derived type's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


@dataclass
class ConstInstanceInjection:
    """Injects a constant value for a component of a specific variable instance."""
    scope_spec: Optional[SPEC]  # Only replace within this scope object.
    root_spec: SPEC  # The root config object's spec (w.r.t. where it is defined)
    component_spec: SPEC  # A tuple of strings that identifies the targeted component
    value: Any  # Literal value to substitute with.


ConstInjection = Union[ConstTypeInjection, ConstInstanceInjection]

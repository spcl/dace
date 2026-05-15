"""Outer Fortran interface  --  the caller-facing surface of the entry
subroutine, snapshotted from HLFIR BEFORE any normalising pass
(``hlfir-flatten-structs`` in particular) runs.

We populate these from the bridge's new
``HLFIRModule.get_fortran_interface(entry)`` entry point  --  it walks
``hlfir.declare`` ops on the entry function in the untransformed
module and pulls out:

- each dummy arg's Fortran name / type / rank / intent,
- the full layout of every derived type referenced by a struct dummy
  (so we know what ``st%u`` / ``st%v`` / etc. look like from the
  caller's side),
- the module each derived type is defined in (so the generated
  wrapper can emit ``use <mod>, only: <type_name>``).

No fparser dependency  --  HLFIR's types carry all of this; mangled
names like ``_QM<mod>T<tname>`` let us recover module origins.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Member:
    """One field of a Fortran derived type."""
    name: str  # 'u'
    fortran_type: str  # 'real(c_double)' | 'complex(c_double)' | 'integer(c_int)'
    rank: int
    # Symbolic / literal extents as they appear in the struct declaration.
    # For assumed-shape inside structs we fall back to '?' and let the
    # wrapper use ``size(st%u, dim=d)`` at call time.
    shape: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DerivedType:
    """Layout of one Fortran derived type referenced by the entry."""
    name: str  # 't_state'
    module: Optional[str]  # 'mo_state' if defined in a module
    members: Tuple[Member, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OriginalArg:
    """One dummy argument of the entry subroutine, outer view."""
    name: str  # 'st'  --  Fortran-source name
    fortran_type: str  # 'real(c_double)' / 'type(t_state)' / 'logical' / ...
    rank: int
    shape: Tuple[str, ...] = field(default_factory=tuple)
    intent: str = ''  # 'in' | 'out' | 'inout' | ''
    # When fortran_type == 'type(<name>)', this points at the
    # DerivedType entry in ``OriginalInterface.struct_types``.
    struct_type: Optional[str] = None


@dataclass(frozen=True)
class OriginalInterface:
    """Caller-facing surface of the entry subroutine plus every
    derived type referenced by its dummies (transitively)."""
    entry: str  # 'compute_tendencies'
    args: Tuple[OriginalArg, ...]
    struct_types: Dict[str, DerivedType] = field(default_factory=dict)
    # Modules the wrapper needs to ``use <mod>, only: <syms>`` so the
    # derived types resolve when gfortran compiles the binding.
    used_modules: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

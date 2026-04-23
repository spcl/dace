"""Fortran binding emission for HLFIR-built SDFGs.

Peer of ``builder/`` / ``intrinsics/`` — NOT nested inside either.
This package runs AFTER the SDFG is built, reads the frozen
signature snapshot + the original Fortran interface, and emits a
``<entry>_bindings.f90`` wrapper that calls the compiled SDFG's C
ABI while preserving the user's Fortran-facing interface.

Public surface:
    FrozenArg, FrozenSignature, SignatureDriftError
        — from .frozen_signature
    OriginalInterface, OriginalArg, DerivedType, Member
        — from .fortran_interface
    decide_strategy, AliasStrategy, ComplexSplitStrategy,
      ExplicitCopyStrategy
        — from .layout_match
    emit_bindings(frozen, iface, out_path)
        — from .emit_bindings
"""
from __future__ import annotations

from dace.frontend.hlfir.bindings.frozen_signature import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)
from dace.frontend.hlfir.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
)
from dace.frontend.hlfir.bindings.layout_match import (
    AliasStrategy,
    ComplexSplitStrategy,
    ExplicitCopyStrategy,
    decide_strategy,
)
from dace.frontend.hlfir.bindings.emit_bindings import emit_bindings

__all__ = [
    "FrozenArg",
    "FrozenSignature",
    "SignatureDriftError",
    "OriginalInterface",
    "OriginalArg",
    "DerivedType",
    "Member",
    "AliasStrategy",
    "ComplexSplitStrategy",
    "ExplicitCopyStrategy",
    "decide_strategy",
    "emit_bindings",
]

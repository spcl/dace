"""Fortran binding emission for HLFIR-built SDFGs.

Peer of ``builder/`` / ``intrinsics/`` under ``dace/frontend/hlfir/``.
Runs AFTER the SDFG is built, consuming three inputs:

- ``FrozenSignature``  --  the SDFG's argument list snapshotted at
  build time (drift-checked at codegen).
- ``OriginalInterface``  --  the caller-facing Fortran surface of the
  entry subroutine.
- ``FlattenPlan``  --  record of every AoS -> SoA unpack performed by
  ``hlfir-flatten-structs``.

And producing one ``<entry>_bindings.f90`` module that preserves the
user's Fortran interface, aliases zero-copy where layouts agree, and
generates do-loop copy-in / copy-out where recipes demand it.

Public surface:
    FrozenArg / FrozenSignature / SignatureDriftError
         --  signature freezing + drift check
    OriginalInterface / OriginalArg / DerivedType / Member
         --  outer Fortran-facing surface
    FlattenRecipe / FlattenEntry / FlattenPlan
         --  the AoS->SoA plan from hlfir-flatten-structs
    emit_bindings(frozen, iface, plan, out_path)
         --  the top-level emitter
"""
from __future__ import annotations

from dace.frontend.hlfir.bindings.emit_bindings import emit_bindings
from dace.frontend.hlfir.bindings.flatten_plan import (
    FlattenEntry,
    FlattenPlan,
    FlattenRecipe,
    strip_index_args,
    substitute_indices,
)
from dace.frontend.hlfir.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
)
from dace.frontend.hlfir.bindings.frozen_signature import (
    FrozenArg,
    FrozenSignature,
    SignatureDriftError,
)

__all__ = [
    # Frozen signature
    "FrozenArg",
    "FrozenSignature",
    "SignatureDriftError",
    # Outer interface
    "OriginalInterface",
    "OriginalArg",
    "DerivedType",
    "Member",
    # Flatten plan
    "FlattenRecipe",
    "FlattenEntry",
    "FlattenPlan",
    "substitute_indices",
    "strip_index_args",
    # Emitter
    "emit_bindings",
]

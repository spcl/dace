"""Thin coordinator — turns ``(FrozenSignature, OriginalInterface,
FlattenPlan)`` into a ``<entry>_bindings.f90`` file.

The real work is in sibling modules:
    * ``flatten_plan.py``    — data model for the pass's output.
    * ``loop_copy.py``       — per-recipe renderers.
    * ``block_builders.py``  — one builder per Fortran section +
                               ``assemble_module`` stitcher.

``emit_bindings`` is ~10 lines of orchestration.  No Fortran-
construction logic lives here; it all routes through the named
builders so each concern is test-isolated.
"""
from __future__ import annotations

from pathlib import Path

from dace.frontend.hlfir.bindings.block_builders import (
    assemble_module,
    build_c_interface,
    build_finalize,
    build_handle_state,
    build_wrapper_body,
    build_wrapper_head,
    build_wrapper_tail,
)
from dace.frontend.hlfir.bindings.flatten_plan import FlattenPlan
from dace.frontend.hlfir.bindings.fortran_interface import OriginalInterface
from dace.frontend.hlfir.bindings.frozen_signature import FrozenSignature


def emit_bindings(
    frozen: FrozenSignature,
    iface: OriginalInterface,
    plan: FlattenPlan,
    out_path: str,
) -> Path:
    """Emit a Fortran binding module for the built SDFG.

    Args:
        frozen:   ``FrozenSignature`` snapshot — the SDFG-facing arg
                  list + free symbols.  Comes from ``SDFGBuilder.build()``
                  and is verified at codegen to catch drift.
        iface:    ``OriginalInterface`` — the caller-facing Fortran
                  surface of the entry subroutine (dummies + struct
                  layouts, snapshotted from HLFIR pre-pass).
        plan:     ``FlattenPlan`` — the record of every unpack
                  ``hlfir-flatten-structs`` performed.  One source
                  of truth for copy-in / copy-out code.
        out_path: Where to write ``<entry>_bindings.f90``.

    Returns:
        The path as a ``Path`` object (same as ``out_path`` just
        materialised).

    Side effect:
        Creates ``out_path``'s parent directory if it doesn't
        exist; overwrites any existing file at ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = {
        'c_interface': build_c_interface(frozen, iface),
        'handle_state': build_handle_state(iface),
        'wrapper_head': build_wrapper_head(frozen, iface, plan),
        'wrapper_body': build_wrapper_body(frozen, iface, plan),
        'wrapper_tail': build_wrapper_tail(frozen, iface, plan),
        'finalize': build_finalize(iface),
    }
    out_path.write_text(assemble_module(iface, frozen, blocks))
    return out_path

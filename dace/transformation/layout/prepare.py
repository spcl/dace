# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared preprocessing for layout transformations: establishes the normal form (no stray views,
implicit copies, or narrow nested-SDFG memlets; loops parallelized to maps) layout passes assume."""
import warnings

from dace import SDFG, data
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.layout.untile_loops_and_blocks import UntileLoopsAndBlocks
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.remove_views import RemoveViews


def normalize_to_packed_c(sdfg: SDFG) -> None:
    """Warn if any plain array is not packed C/Fortran-contiguous; a real relayout is a ``LayoutChange``, not done here."""
    for _, _, desc in sdfg.arrays_recursive():
        if not isinstance(desc, data.Array) or isinstance(desc, (data.View, data.ContainerArray)):
            continue
        if desc.is_packed_c_strides() or desc.is_packed_fortran_strides():
            continue
        warnings.warn(
            f"prepare_for_layout: array with non-packed strides {desc.strides} (shape "
            f"{desc.shape}); the layout algebra assumes a packed C/Fortran representation. "
            f"A relayout to the normal form must go through a LayoutChange node.",
            stacklevel=2,
        )


def prepare_for_layout(sdfg: SDFG, target: str = 'cpu', validate: bool = True) -> SDFG:
    """Normalize ``sdfg`` in place into the precondition layout passes assume."""
    # Must run before canonicalize -- its UntileLoops would collapse the tile loop first. No-op if not blocked+tiled.
    UntileLoopsAndBlocks().apply_pass(sdfg, {})

    # Parallelize first (loop-to-map inside canonicalize), before widening nested-SDFG memlets below.
    canonicalize(sdfg, target=target, validate=validate)

    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, validate=validate)

    # Re-clean views reintroduced by expansion; views feeding library nodes are preserved.
    RemoveViews().apply_pass(sdfg, {})

    # Lift implicit copies to CopyLibraryNode -- layout passes can't rewrite other_subset directly.
    InsertExplicitCopies().apply_pass(sdfg, {})

    normalize_to_packed_c(sdfg)

    if validate:
        sdfg.validate()
    return sdfg

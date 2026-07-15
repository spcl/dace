# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared preprocessing for the layout transformations.

Every layout transformation (Permute, Block, Unblock, Zip, Unzip, Pad, Shuffle)
assumes the SDFG is in a normal form:

  * Core Dialect: no views (except views feeding library nodes), no WCR memlets,
    no ``other_subset`` memlets, no implicit AccessNode->AccessNode copies, no
    streams.
  * Loops parallelized to maps where legal (so access patterns live on maps).
  * Nested-SDFG in/out memlets widened to full-array subsets, so a layout rewrite
    can recurse into a nested SDFG with full-shape arguments.

``prepare_for_layout`` runs that pipeline once so every layout pass sees the same
normal form. It mirrors the user's stated order: parallelize first (inside
canonicalize), then expand nested-SDFG inputs.
"""
import warnings

from dace import SDFG, data
from dace.sdfg.core_dialect import warn_if_not_core_dialect
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.remove_views import RemoveViews


def normalize_to_packed_c(sdfg: SDFG) -> None:
    """Establish the packed-stride normal form the layout algebra assumes.

    Layout is carried by materialized shape / dimension order on PACKED arrays (C- or
    Fortran-contiguous), never by arbitrary strides. Post-``RemoveViews`` every plain array
    should already be packed (DaCe allocates C-contiguous by default). This checks the
    invariant and warns on any non-packed array -- a true relayout of such an array to the
    packed normal form is a ``LayoutChange`` (a real transpose), not a silent stride reset,
    so it is out of scope here. Structures / container arrays / views are skipped.
    """
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
    """Normalize ``sdfg`` in place into the precondition layout passes assume.

    :param sdfg: the SDFG to normalize.
    :param target: ``'cpu'`` or ``'gpu'`` -- forwarded to ``canonicalize``.
    :param validate: validate after each major step.
    :return: the same ``sdfg``, normalized.
    """
    # 1. Canonicalize: front-stage RemoveViews cleans views, loop nests are
    #    normalized (Truemper), einsum/reduce are lifted, and LoopToMap stages
    #    parallelize. This is the "parallelize first" step.
    canonicalize(sdfg, target=target, validate=validate)

    # 2. Widen narrowed nested-SDFG in/out memlets to full-array subsets.
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs, validate=validate)

    # 3. Re-clean views reintroduced by expansion. Views feeding library nodes
    #    are preserved by RemoveViews' library-node-operand exception, which is
    #    exactly the "only views feeding library nodes are allowed" invariant.
    RemoveViews().apply_pass(sdfg, {})

    # 4. Establish the packed-stride normal form the layout algebra assumes.
    normalize_to_packed_c(sdfg)

    # 5. Report any residual core-dialect violations (never raises).
    warn_if_not_core_dialect(sdfg, source='prepare_for_layout')

    if validate:
        sdfg.validate()
    return sdfg

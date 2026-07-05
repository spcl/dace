# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``reduce_at_output`` map marker.

A masked in-body reduction (``if cond: acc <op>= x``) is normalized to a tileable
single-state select body + map-exit WCR by
:class:`~dace.transformation.passes.vectorization.normalize_map_reduction.NormalizeMapReduction`,
then lifted to a ``Reduce`` library node by ``LiftMapReductionToReduce`` before the
``no_wcr_inside_nested_sdfgs`` invariant runs -- so no in-NSDFG WCR survives to be
whitelisted. The old tagging pass (``MarkReduceAtOutput``) and its consumers
(``_FlattenTileBodyNesting``, ``SpliceReductionTileFold``) are retired; only this
marker constant remains, kept because
:mod:`~dace.transformation.passes.vectorization.utils.pass_invariants` still imports
it for the (now-dormant) tagged-map skip branches.
"""

#: Suffix a map's label carries when its body reduction is lowered via the
#: ``reduce_at_output`` boundary path. No pass appends it today (the marker skip in
#: ``pass_invariants`` is dormant), but the constant is preserved for that import.
REDUCE_AT_OUTPUT_MARKER = "__reduce_out"

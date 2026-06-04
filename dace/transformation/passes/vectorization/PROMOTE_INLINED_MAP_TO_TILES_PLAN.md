# PromoteInlinedMapToTiles — slicing plan

## Goal

The multi-dim K=2 path "always inlines the SDFG and widens all scalars
and stages every global write through intermediate full tiles". After
inlining, the body NSDFG is gone -- every former internal scalar +
tasklet sits directly in the outer Map scope. The existing
``PromoteNSDFGBodyToTiles`` does the same lowering for the body-NSDFG
case; this plan ports the lowering to the inlined outer-state case.

Three rewrites the new pass must do, in order:

1. **Widen** every body scalar (or length-1 Array) transient to a
   ``(W_0, ..., W_{K-1})`` Array, and rewrite every memlet that
   references it.
2. **Rewrite tasklets** to tile lib nodes (``TileBinop`` / ``TileUnop``
   / ``TileMerge`` / constant-store -> ``TileBroadcastSymbol``).
3. **Rewrite Array <-> tile-transient copy edges** to ``TileLoad`` /
   ``TileStore`` -- the standalone pass we just landed
   (:class:`RewriteArrayScalarToTileOp`) handles this step; the new
   pass just calls it.

Slice 0 below extracts the body-agnostic helpers from the descent so
the new pass and the existing descent can share them; slices 1-4 add
the new pass.

## Slicing (5 commits, each bounded ~400 LoC)

### Slice 0 — Extract body-agnostic helpers (refactor, no behavior change)

Pull these out of ``promote_nsdfg_body_to_tiles.py`` into a new
``utils/promote_helpers.py`` (or extend an existing utils module):

* ``_classify_box_for_widths(subset, arr, widths, iter_vars) ->
  TileAccessClassification`` -- the perfect-box classifier with the
  K=0 / length-1 special case. Currently inline in ``_box_classification``.
* ``_lane_index_expr(iter_var, all_iter_vars) -> str`` -- already
  module-level in ``emit_tile_ops.py``; just import from there.
* ``_operand_classifier(token, iter_vars, ...)`` -- the "Symbol /
  Scalar / Tile / NDTile" classification logic from ``_promote_binops``,
  generalised to take "is this a global / outer-scope symbol?" as a
  callback rather than hardcoding ``nsdfg_node.in_connectors`` /
  ``nsdfg_node.symbol_mapping``.

Net: ``promote_nsdfg_body_to_tiles.py`` shrinks by ~150 LoC; the
shared utils gain a ~250 LoC module. ``PromoteNSDFGBodyToTiles``
delegates to the utils; descent tests stay green.

**Commit message**: ``vec(K-dim): extract body-agnostic
promote helpers``

### Slice 1 — Pass scaffold + scalar widening

New file ``promote_inlined_map_to_tiles.py``. Contains:

* ``PromoteInlinedMapToTiles(ppl.Pass)`` class. Walks every tile-
  tagged map in the SDFG; for each map without a body NSDFG inside it
  (i.e. ``InlineSDFGs`` already ran), processes its scope.
* ``_widen_body_scalars(state, map_entry, spec)``: for every scalar
  or length-1 Array transient WHOSE access nodes are entirely within
  the Map scope, widen the descriptor to ``Array(shape=widths,
  dtype=orig_dtype, transient=True, storage=Register)`` and rewrite
  every memlet that references it from ``scalar[0]`` (or analogous)
  to a full-tile subset ``scalar[0:W_0, ..., 0:W_{K-1}]``.

The widening alone is meaningless without tasklet rewrite (the
tasklet body still expects scalar reads), so this slice does NOT yet
ship a tasklet rewrite -- the widened SDFG won't compile until
slice 2 lands. Tests assert the descriptor + memlet rewrite in
isolation; numerical equivalence comes in slice 4.

Net: new 300 LoC pass file; 5 unit tests.

**Commit message**: ``vec(K-dim): PromoteInlinedMapToTiles scaffold
+ scalar widening``

### Slice 2 — Binop / unop tasklet -> TileBinop / TileUnop

Add ``_promote_binops_outer`` and ``_promote_unops_outer`` to the
new pass. Logic mirrors the descent's ``_promote_binops`` /
``_promote_unops`` but calls into the body-agnostic
``_operand_classifier`` from slice 0 (with the inlined-state version
of the "is this from outside the map?" predicate).

Scope of THIS commit:
* CONTIGUOUS-box operands only (gather / strided deferred).
* No masks (will land with slice 4 once we wire mask threading).
* Symbol / Scalar / Tile operand kinds. ``NDTile`` deferred.

Net: ~400 LoC; 8 unit tests on hand-built fixtures.

**Commit message**: ``vec(K-dim): PromoteInlinedMapToTiles binop +
unop tasklet rewrite``

### Slice 3 — Constant-store + merge tasklet rewrites

* ``_promote_const_stores_outer``: ``_out = 0.0`` style tasklets
  become ``TileBroadcastSymbol``.
* ``_promote_merges_outer``: ``out = cond_true if cond else cond_false``
  tasklets become ``TileMerge``.

Net: ~250 LoC; 4 unit tests.

**Commit message**: ``vec(K-dim): PromoteInlinedMapToTiles const-
store + merge rewrites``

### Slice 4 — Wire to pipeline + integration test

Add a new ``VectorizeCPUMultiDim`` knob: ``inline_body_nsdfgs:
bool = False``. When True, the orchestrator inserts ``InlineSDFGs``
(with the right scope) after ``MarkTileDims`` and before the existing
``PromoteNSDFGBodyToTiles``. The new ``PromoteInlinedMapToTiles``
pass runs instead of ``PromoteNSDFGBodyToTiles`` when this knob is on.

Plus: thread the mask through the outer-state lib nodes (call the
existing ``_mask_name_for_map``) and run the existing
``RewriteArrayScalarToTileOp`` after the tasklet rewrites to handle
the Array <-> tile copy edges.

Integration test: build a small K=2 kernel
(``B[i,j] = A[i,j] + 1.0``), apply the pipeline with
``inline_body_nsdfgs=True``, assert numerical equivalence vs.
gfortran-compiled reference. This is the cloudsc-style "K=2 always-
inline" path's end-to-end smoke test.

Net: ~150 LoC orchestrator wiring + ~100 LoC integration test.

**Commit message**: ``vec(K-dim): wire PromoteInlinedMapToTiles into
VectorizeCPUMultiDim`` (single commit pushing the path end-to-end
green)

## What this plan deliberately does NOT cover

* Gather / scatter outer-state promotion (will reuse the descent's
  ``fan_out_tile_gather_index_symbols`` / scatter machinery in a
  later slice).
* SSA correction for read-write-in-same-state arrays (the descent's
  ``_ssa_correct_rmw_reads``). Outer-state equivalent is the same
  algorithm but tracks AccessNodes at outer scope.
* Loop-invariant write -> reduction node insertion (separate, smaller
  slice that can land before or after slice 4).
* Three-path orchestrator split (legacy / multi-dim K=1 / multi-dim
  K=2). Adds the K=2 path in slice 4; the legacy + K=1 paths stay
  exactly where they are.
* Diagonal K=1 stays diagonal / K=2 mixed -> gather-required box
  (this is a classifier-level decision that the existing per-dim
  classifier already encodes; the outer-state pass consumes the
  classifier output unchanged).

# Tilification Transformation -- Design Specification

**Status**: Draft for Tuesday freeze.
**Scope**: Lower a K-dim register tile (K in {1, 2, 3}) onto AVX-512 / NEON /
SVE / scalar / cuTile back ends.
**Layout**: sections 2-10 are the contract; section 11 audits the current codebase
against it; section 12 lists the gaps and the migration order; section 13 holds the
schedule.

---

## 1. Goals

1. **One canonical body shape**: the tiled loop body is a NestedSDFG with
   full-array subsets on every connector. No pointer offsetting at the
   call site.
2. **One operand contract per lib node**: each library-node input is either
   a *full tile* (shape `(K_0, ..., K_{K-1})`) or a *scalar* (length-1 / true
   `Scalar`). Nothing in between.
3. **One classification surface**: a per-dimension access lattice
   (section 4) plus a single composition rule (section 5). All higher-level emission
   decisions read from the lattice; the lattice falls back to GATHER when
   it cannot decide (section 4.2).
4. **One mask shape**: every lib node accepts an optional `(K_0, ...,
   K_{K-1})` mask; per-dim masking is uniformly expressed by broadcasting
   into the K-dim mask (section 7).
5. **One remainder rule**: K boundary regions (not 2^K - 1), produced by
   the corner-absorbing peel (section 8).
6. **Three gather forms** (full / per-dim / partial) on the same node,
   with affine fallback on omitted dims (section 9).

---

## 2. Iteration Space, Function Body, And The Call Convention

### 2.1 Notation

Fix the following throughout:

- `D` -- total dimensionality of the kernel's loop nest.
- `K` -- number of *tiled* dims; `K in {1, 2, 3}`; `K <= D`.
- `d_0, ..., d_{K-1}` -- the **tile dim indices** (a subset of
  `{0, ..., D-1}`), ordered innermost-last so `d_{K-1}` is the
  innermost.
- `W_p = widths[p]` -- tile width on the `p`-th tile dim
  (`p in {0, ..., K-1}`), innermost-last.
- `N_q` -- global upper bound on the `q`-th kernel dim
  (`q in {0, ..., D-1}`); the loop space is `[0, N_q)` per dim.
- `M_q = floor(N_q / W_p) * W_p` when `q = d_p` (main-tile boundary on a
  tiled dim); `M_q = N_q` when `q` is untiled.
- `i_d` -- the outer Map's iter-var bound to kernel dim `d`.
- `l_p in [0, W_p)` -- lane index within the tile on tile dim `p`.

### 2.2 Iteration space

The outer Map's iteration space is the product

```
I_outer = product_{q in untiled} [0, N_q) x product_{p=0}^{K-1} [0, M_{d_p}) step W_p
```

i.e. untiled dims iterate one element at a time over their full
extent; tiled dims iterate in steps of `W_p` over the *main* range
`[0, M_{d_p})`. The remainder regions (section 8) cover the tail strips
`[M_{d_p}, N_{d_p})` separately.

The **inner lane space** of the body, fixed at every outer iteration,
is

```
I_inner = product_{p=0}^{K-1} [0, W_p)
```

-- a closed product the lib nodes lower as register tiles. Outer
iter-vars enter the body as compile-time symbols; the lane indices
`l_p` are *not* in the body's symbol scope (they are materialised by
the lib-node expansions).

### 2.3 Layout and stride invariants

The classifier and the lib-node expansions **always read array
strides from the descriptor** (`arr.strides`). The strides record is
the single source of truth -- nothing in the lowering pipeline
recomputes strides from shape or hard-codes a layout.

**Supported layouts (locked -- only C or Fortran, no others)**:

| Layout | Contiguous dim | Stride pattern |
|-----------|----------------|---------------------------------------------------------|
| **C** | rightmost | `strides[-1] = 1`, `strides[d] = product_{d' > d} shape[d']` |
| **Fortran**| leftmost | `strides[0] = 1`, `strides[d] = product_{d' < d} shape[d']` |

Any other stride pattern (strided views over a larger buffer,
broadcasted strides, ragged layouts, non-packed strides) is
**rejected at `validate()` time** with a clear error naming the
array and the offending stride. This restriction is deliberate -- it
keeps the lib-node expansions backed by a small fixed intrinsic set.
Lift only when a kernel demands a third layout.

**How the lib node consumes strides**:

- `TileLoad` / `TileStore` reads `arr.strides` at expansion time and
  emits the flat offset from the classifier's per-dim `dim_strides[p]`
  + `replicate_factor[p]` + `src_dims` permutation.
- `TileLoad` / `TileStore` reads `arr.strides` to decode the
  index tile into a flat offset (full form) or to compose per-dim
  index components (per-dim / partial form).
- **Tile dim <-> array dim mapping** is explicit via `src_dims` /
  `dst_dims`. Default (`None`) means "innermost K source array dims
  map to the K tile dims in order" -- equivalent to C-layout
  innermost-K binding. Explicit permutation handles Fortran /
  transposed inputs without copying the source.
- **AFFINE stride coefficient `s * W_p <= N_{d_p}`**: the tile must
  stay within the array on its strided dim. The classifier admits
  any constant `s >= 1`; the expansion folds `s > 1` into
  `dim_strides[p]` without falling back to GATHER (AVX-512 / SVE /
  cuTile express the strided load natively).
- **`dim_strides[p] = 0`** is reserved for CONSTANT dims -- the lib
  node internally splats the single value across all `W_p` lanes;
  no source memory is touched per-lane.

### 2.4 Function body -- the tiled NestedSDFG

The tiled body is a single `NestedSDFG` node sitting inside the outer
Map's scope. It satisfies the **full-subset call convention**:

| Connector role | Outer memlet on the connector edge | Inner SDFG reference |
|----------------------|------------------------------------|-----------------------------|
| Input array `A` | `A[0:N_0, 0:N_1, ..., 0:N_{D-1}]` | inner array `A` shape |
| | (full extent on every dim) | `(N_0, N_1, ..., N_{D-1})` |
| Output array `B` | `B[0:N_0, ..., 0:N_{D-1}]` | inner array `B` shape |
| | (full extent on every dim) | `(N_0, N_1, ..., N_{D-1})` |
| Loop iter-var symbol | passed via `symbol_mapping` | inner symbol `i_d` in `[i_d, i_d + W_p)` for tile dim `d_p` |

**Invariants** (load-bearing for the rest of the design):

1. **No pointer offsetting at the call site.** Every connector memlet
   is the full array; per-tile addressing is purely symbolic inside
   the body.
2. **Inner arrays mirror outer arrays.** The inner descriptor for `A`
   has shape `(N_0, ..., N_{D-1})` and stride / layout identical to the
   outer descriptor. The body's memlets address sub-regions of the
   *inner* `A` using `i_d`-relative subsets
   `A[i_d_0:i_d_0+W_0, ..., i_d_{K-1}:i_d_{K-1}+W_{K-1}]` on the tiled
   dims and the same shape as the outer access on untiled dims.
3. **Lane indices are body-private.** `l_p` never appears in
   `symbol_mapping`. The lib nodes materialise lane indices internally;
   the body sees only `i_d` and the tile-shaped transients.
4. **Multi-dim keeps the NestedSDFG.** For `K >= 2`, the body **always
   remains a NestedSDFG**; inlining is not part of the multi-dim path.
   This guarantees that the staging contract (section 3.3) has a fixed
   structural surface to operate on. K=1 may inline as a legacy
   compatibility option; multi-dim does not.

**Rationale**: the full-subset convention eliminates the "connector
array shape mismatch" family of bugs that the descent and the
inlined-outer-state port both fought. The boundary is established
by [`ExpandNestedSDFGInputs`](../../interstate/expand_nested_sdfg_inputs.py)
(already invoked by `VectorizeCPUMultiDim`); all analysis runs
inside the body NSDFG.

---

## 3. Inside The Body -- Tile Transients And The Scalar Exception

### 3.1 Staging rule (inside-body)

> Inside the body NSDFG, every non-transient memlet read / write is
> staged through a fresh transient. The shape of the transient is
> chosen from the per-tile classification of the memlet's subset:

| Memlet's per-dim classification | Read side | Write side |
|---|---|---|
| **CONSTANT on every tile dim** | direct `AN -> AN(Scalar)` copy. Consumed via the `Scalar` operand kind (section 6.2; hardware splat at codegen). No `TileLoad` lib node. | direct producer `-> AN` copy (NO `TileStore`). Symmetric to the read path -- the loop-invariant write target is just a single-element store; the existing producer edge handles it. |
| At least one dim in `{LINEAR, AFFINE, REPLICATE, MODULAR}` | bridge `Array(shape=widths, storage=Register, transient=True)` + `TileLoad`. | bridge `Array(shape=widths)` + `TileStore`. |
| Any dim is `GATHER` | bridge `Array(shape=widths)` + `TileLoad` with `gather_dims` + `_idx_<d>` connectors. | bridge `Array(shape=widths)` + `TileStore` with `gather_dims` (scatter; **deferred**). |

Per user direction 2026-06-09: the CONSTANT row is a symmetric pair -- "scalar
load stays as a copy" and "constant-only write is the same". Neither side
goes through a lib node; both are handled at staging time as direct AN-edge
copies. The lib node and the per-arch lowering never see these patterns.

The boundary outside the NSDFG stays at full-array subsets
(established by `ExpandNestedSDFGInputs`); the inner descriptor
mirrors the outer shape; analysis runs inside.

### 3.2 Consequence

Each lib node sees only a full tile or a scalar. Partial-subset /
strided-lane complexity stays in the staging pass; the per-dim
lattice (section 4) is a closed surface.

### 3.3 No `AN(non-transient) -> NSDFG -> AN(non-transient)`

After the multi-dim pipeline runs, no non-transient AccessNode
appears in the middle of the body's dataflow. Every non-transient
inner AN is either:

* **A producer** of a staged scalar / tile transient (read side), or
* **A consumer** of a staged scalar / tile transient (write side).

Refusal: when staging cannot fold a producer / consumer pair
(cross-state RMW chain rejecting the linear-accumulation invariant,
overlapping subsets the classifier cannot resolve), the pass raises
`NotImplementedError` naming the array and the offending state.

Pipeline order (canonical walker-primary, all K) -- current as of 2026-06-10:

```
WCRToAugAssign                          (canonicalise WCR -> RMW where applicable)
   -> LoopToMap + RefineNestedAccess    (parallelise + tighten body NSDFGs)
   -> normalize_loop_nests              (Inline + MapCollapse; no propagation)
   -> Front prep:
       ConvertLengthOneArraysToScalars + NormalizeWCRSource + BypassTrivialAssignTasklets
   -> Branch lowering:
       SameWriteSetIfElseToITECFG + BranchNormalization
       (+ EliminateBranches + LowerITEToFpFactor for branch_mode='fp_factor')
   -> LowerInterstateConditionalAssignmentsToTasklets
   -> Tasklet preprocessing:
       RemoveFPTypeCasts, RemoveIntTypeCasts, PowerOperatorExpansion,
       SplitTasklets, RemoveMathCall, RemoveEmptyStates
   -> Tile shaping (must run in this order):
       SplitMapForTileRemainder      (optional per remainder_strategy)
       NestInnermostMapBodyIntoNSDFG (ALWAYS-ON; the walker needs a body NSDFG)
       _RunExpandNestedSDFGInputs    (embedded Pass; widens boundary memlets to
                                      full source-array subset; runs AFTER Nest)
       MarkTileDims                  (tag the outer map with TileDimSpec)
       StrideMapByTileWidths         (step 1 -> W; iter_var means "tile start")
       InferBodyTransientShapes      (proactive widening: non-transient AN edge
                                      memlets ``A[ii]`` -> ``A[ii:ii+W]`` for
                                      non-CONSTANT classifications; length-1
                                      intermediate transients -> ``(widths,)``
                                      when chain reads a tile source; CONSTANT /
                                      Scalar / Symbol stay as-is.)
       GenerateTileIterationMask     (emits TileMaskGen + _tile_iter_mask
                                      INSIDE the body NSDFG so the walker /
                                      converter can find + wire it)
   -> Walker + converter:
       PreparePerLaneIndices         (per-source-dim _idx_<k> materialisation)
       StageInsideBody               (CONSTANT / Tile / Gather dispatch per AN;
                                      finds the mask and wires has_mask=True +
                                      _mask onto TileLoad / TileStore)
       ConvertTaskletsToTileOps      (replaces in-body tasklets with Tile lib
                                      nodes; finds the mask and wires has_mask
                                      + _mask onto Tile{Binop, Unop, ITE, Reduce})
   -> Library node expansion + audit:
       sdfg.expand_library_nodes()
       ClearPerLaneIndexSymbols      (section 10.6 post-emit audit)
```

**Key ordering invariants** (per user direction 2026-06-09 / 2026-06-10):

* `NestInnermostMapBodyIntoNSDFG` is **always-on**. The walker traverses body
  NSDFGs; without one it has nothing to do.

* `_RunExpandNestedSDFGInputs` (a tiny `ppl.Pass` wrapper around
  `sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs)`) runs
  **AFTER Nest** and **BEFORE any tile-shaping pass**. Embedding it as a
  Pass in the list keeps the standard `ppl.Pipeline.apply_pass` machinery
  in charge of execution order. An earlier implementation used a
  two-half pipeline split with a depgraph-invalidation hack; that
  silently skipped the entire walker tail.

* `StrideMapByTileWidths` runs **BEFORE `InferBodyTransientShapes`**.
  After stride, `ii` means "start of a W-element tile"; the widening
  pass replaces single-element `A[ii]` with `A[ii:ii+W]` based on
  that semantics.

* `GenerateTileIterationMask` runs **AFTER `InferBodyTransientShapes`**.
  The mask reflects the final tile shape; emitting before widening
  would lock the mask shape into a shape that doesn't match the
  widened lib-node operands.

* `GenerateTileIterationMask` emits `TileMaskGen` + the
  `_tile_iter_mask` AccessNode **INSIDE the body NSDFG** (not in the
  outer scope). The walker / converter find the mask via
  `inner_sdfg.arrays['_tile_iter_mask']` and wire `_mask` connectors
  on every lib node they mint.

**SDFG-scheduling invariant** (load-bearing for correctness): every
consumer of a transient must read from the SAME AccessNode that the
producer writes to. Creating fresh `inner_state.add_access(name)`
calls per consumer produces orphan AccessNodes that DaCe's scheduler
orders independently from the producer -- the consumers may run
before the producer, producing zero / uninitialised reads. This
applies to:

* The `_tile_iter_mask` AccessNode: all walker + converter `_mask`
  edges reuse the AccessNode that `TileMaskGen._o` writes to (helper
  `_find_mask_producer_an` / `_find_mask_an`).

* The tile-bridge transient AccessNodes: the rewire helpers
  (`_rewire_consumers_to_bridge` / `_rewire_producers_to_bridge`)
  reuse the AccessNode that `TileLoad._dst` writes to / that
  `TileStore._src` reads from (helper `_find_existing_bridge_an`).

The legacy descent (``PromoteNSDFGBodyToTiles``, 2996 LoC) and legacy
``EmitTileOps`` (1470 LoC) were deleted during the walker-primary
migration. The historical inlined-outer-state path was dropped in
tier S of [MULTI_DIM_CODE_AUDIT.md](MULTI_DIM_CODE_AUDIT.md).

### 3.6 AN -> AN no-tasklet copy contract

Every staged copy between a non-transient and a transient is a
direct `AccessNode -> AccessNode` edge -- no `_out = _in` assign
tasklet in between.

The cleaning pass [BypassTrivialAssignTasklets](bypass_trivial_assign_tasklets.py)
runs early in the pipeline so the classifier and the staging pass
see direct edges everywhere.

### 3.7 Reading the AN-side subset

For any AN-incident edge, the subset belonging to that AN is read
via a **single three-way helper**, not by ad-hoc `edge.data.subset`
reads (which silently pick the wrong array when `data` points at
the other side):

```python
def an_side_subset(edge, an, sdfg) -> Range:
    if edge.data.data == an.data:
        return edge.data.subset
    if edge.data.other_subset is not None:
        return edge.data.other_subset
    return Range([(0, s - 1, 1) for s in an.desc(sdfg).shape])
```

(Implicit full-shape copies omit `other_subset`; the helper
reconstructs from the descriptor.) Every consumer of an AN -> AN
edge -- classifier, staging pass, lib-node emitter -- routes
through this helper.

### 3.8 Per-lane index materialisation (pre-pass)

A gather expression `idx[i, k]` with `i` a tile iter-var and `k` an
outer-constant is, before lowering, just a symbolic memlet. The
modern equivalent of the legacy 1D `_laneid_<i>` symbol fan-out is a
structural **pre-pass** `PreparePerLaneIndices` that materialises
every such expression into an integer index tile:

```
for each gather memlet with index expression I (per dim d):
  deps = sorted tile iter-var indices that I depends on
  shape = tuple(widths[p] for p in deps)             # see section 9.2
  mint a transient `_idx_<d>_<unique>` of shape + int dtype
  emit a constant-assignment tasklet that fills the tile per lane:
      for each lane offset l = (l_0, ..., l_{|deps|-1}):
          tile[l] = I  with  iter_var_p -> base_p + l_idx_of_p
  wire the tile to the lib node's _idx_<d> connector
```

This runs after staging (section 3.3) and before lib-node emission.
Every classifier read of a gather expression is replaced by a tile
the lib node can broadcast across the lane space at expansion time;
no per-lane symbol survives into the post-emit SDFG.

The corresponding post-emit audit (section 10.6) asserts the
absence of `_laneid_<i>`-style symbols, the same loud-failure
contract the legacy pipeline had.

### 3.4 Staged-transient scoping

Every transient introduced by the staging pipeline lives in **one
specific SDFG scope**. The rule:

| Transient origin | Owning SDFG | Lifetime |
|------------------------------------------|--------------------------------------|-----------------------------|
| Scalar bridge in section 3.3 case 1 | Body NestedSDFG | One outer Map iteration |
| Full-tile fallback in section 3.3 case 2 | Body NestedSDFG | One outer Map iteration |
| Widened body scalar (section 3.1) | Body NestedSDFG | One outer Map iteration |
| Mask transient (`_iter_mask`) | Body NestedSDFG | One outer Map iteration |
| `TileReduce` scalar output (section 10.3) | Body NestedSDFG (then drained out) | One outer Map iteration |

All inner transients are register-storage. Cross-iteration values
(reductions, recurrences) reach the outer scope through the **body's
output connector**, which writes to a non-transient outer array
exactly once per outer iteration. The body never owns a "live across
iterations" transient.

### 3.5 WCR (write-conflict-resolution) memlets

WCR memlets appear in **exactly one position** in the canonical shape:

```
        body NSDFG (out_conn)
              |
              v
         AccessNode (single-element output array)
              | -- memlet carries WCR --
              v
           MapExit
```

That is:

> A WCR memlet may only appear on the edge `AccessNode -> MapExit`,
> where the `AccessNode` is the destination of the body NSDFG's output
> connector and its descriptor is **single-element** (a `Scalar` or
> a length-1 `Array`).

Equivalently: the **body computes a single value** (via `TileReduce`
intra-tile, or by writing a single scalar element from a tasklet);
that value flows out of the body NSDFG to an outer-state AccessNode;
the AccessNode -> MapExit memlet carries the accumulation operator.
The outer Map's WCR codegen handles inter-iteration accumulation;
the body interface is unaffected.

**Hard constraints**:

- **No WCR inside the body.** The body NSDFG has no internal WCR
  memlets. A WCR memlet that appears on any edge other than
  `AccessNode -> MapExit` is rejected by `validate()` with a clear
  error naming the edge.
- **Single-element output only.** The destination AccessNode of a
  WCR edge has a descriptor whose total volume is 1 (`Scalar` or
  `Array(shape=(1,))` / `(1, 1)` / ...). Multi-element WCR
  destinations (e.g. `A[i_d:i_d+W_p] +=` across a tile dim) are
  rejected; the kernel must materialise such an accumulation as a
  pair of `(TileReduce -> scalar, scalar -> MapExit with WCR)` first.
- **Body lowering reasons about the output value, not the WCR.** The
  classifier and the tile lib-node emission see the body's output
  as a normal write. The WCR is a property of the outer-state edge
  only.

**Pattern in practice** (intra-tile sum into a scalar accumulator):

```
TileLoad -> TileBinop(+) -> TileReduce -> scalar_accum (in body)
                                          |
                                          v
                                 (body output connector)
                                          |
                                          v
                                  AccessNode 'acc' (Scalar)
                                          | wcr = lambda a,b: a+b
                                          v
                                       MapExit
                                          |
                                          v
                                  outer Array 'acc_out'
```

---

## 4. Per-Dimension Access Lattice

### 4.1 Kinds

Each tile dimension's access is classified as one of:

| Kind | Definition | Example |
|--------------|-----------------------------------------------------------------------------|------------------------------------------|
| **CONSTANT** | dim's index is loop-invariant (no tile iter-var anywhere in the expression) | `a[0]`, `a[N+1]` |
| **LINEAR** | dim's index is exactly `iter_var + c` (stride 1, constant offset) | `a[i]`, `a[i + 3]` |
| **AFFINE** | dim's index is `s * iter_var + c` with `s`, `c` outer-scope constants | `a[2*i + 1]` |
| **REPLICATE**| dim's index is `floor(c * iter_var + c0, k)` or `ceil(...)` (1 < k < W) | `a[i//2]`, `a[i//4 + 7]` |
| **MODULAR** | dim's index is `(c * iter_var + c0) % N` with `N` an outer-scope constant | `a[i % N]`, `a[(2*i + 1) % N]` |
| **GATHER** | none of the above | `a[idx[i]]`, `a[i*j]`, `a[2*sym + 1]` (sym loop-variant) |

### 4.2 Lattice order and join rule

```
CONSTANT <= LINEAR <= AFFINE <= GATHER
                      meet meet
                REPLICATE MODULAR
```

- `CONSTANT <= LINEAR <= AFFINE <= GATHER` is a strict chain.
- `REPLICATE` is **incomparable** with LINEAR / AFFINE -- it is a separate
  axis (group broadcast inside the dim with factor `k`). The join of
  REPLICATE with anything outside `{CONSTANT, REPLICATE}` is GATHER.
- `MODULAR` is **incomparable** with LINEAR / AFFINE -- it is a separate
  axis (cyclic wrap with period `N`). The join of MODULAR with anything
  outside `{CONSTANT, MODULAR}` is GATHER.

**Tile-aligned MODULAR -> LINEAR reduction**:

For `a[(c * i_d + c_0) % N]`, if the classifier can prove `N` divides
`c * W_d` (the per-tile stride covers an integer number of wrap
periods), the access reduces to **LINEAR with a per-tile constant
offset**: every tile starts at `(c * i_d + c_0) mod N` (loop-invariant
within the tile) and lane `l_d` reads element `base + l_d`. The
classifier should emit LINEAR in this case; only the generic MODULAR
falls back to GATHER.

The common K=2 instance that triggers this reduction: `a[i % W_d]`
where the outer Map step is `W_d` => tile start == 0 (mod `W_d`) =>
LINEAR with offset 0.

**Join rule (tile-dependent symbol -> GATHER)**:

> A symbol is **tile-dependent** iff materialising its value differs
> across lanes -- equivalently, the symbol (or any symbol it transitively
> depends on via interstate-edge assignments inside the body) is a tile
> iter-var. The implementation mechanism: a tile-dependent symbol is
> exactly the symbol kind that would need per-lane (laneid-style)
> expansion to materialise. The classifier reuses this notion as the
> ground truth.
>
> **Rule**: if any symbol in a dim's expression is tile-dependent and
> the surrounding form is not already CONSTANT / LINEAR / AFFINE /
> REPLICATE / MODULAR (over outer-scope constants), the dim's kind is
> **GATHER**.

In other words, the classifier asks one question per symbol: *is this
symbol tile-dependent or outer-scope?* -- using the same dependency walk
the codegen would use to decide whether a per-lane materialisation is
required. Outer-scope symbols and numeric literals are tile-independent;
tile iter-vars and any symbol whose definition transitively touches one
are tile-dependent.

Examples (with `i` a tile iter-var, `N` an outer-scope constant symbol,
`sym` an interstate-edge-assigned symbol inside the body):

| Expression | Symbol analysis | Kind |
|------------------|------------------------------------------|-------------------|
| `a[i]` | only `i` (tile iter-var) | LINEAR |
| `a[i + 3]` | only `i` | LINEAR |
| `a[N + 1]` | `N` outer-scope | CONSTANT |
| `a[2*i + 1]` | only `i` | AFFINE |
| `a[N*i + 1]` | `N` outer-scope, `i` tile iter-var | AFFINE (stride `N`) |
| `a[2*sym + 1]` (sym <- `i + 3`) | `sym` tile-dependent | **GATHER** |
| `a[2*sym + 1]` (sym <- outer) | `sym` outer-scope | AFFINE |
| `a[i*j]` | both `i`, `j` tile-dependent | **GATHER** |
| `a[idx[i]]` | `idx[i]` is a data-dependent read | **GATHER** |

This mirrors the FP-precision-type fallback: when in doubt, join up to
the strictest classification (GATHER) so correctness is preserved at
the cost of a slower path.

---

## 5. Cross-Dimension Composition

A tile access is a tuple `(d_0, ..., d_{K-1})` of per-dim kinds. The
composition decides which lib node lowers the access and how its
properties are populated.

### 5.1 Unified lowering -- one node per direction

Every read access lowers to a single `TileLoad`; every write access to
a single `TileStore`. There are no separate gather / scatter nodes.
The presence of an `_idx_<d>` connector on tile dim `d` marks that dim as GATHER; the lib node's
expansion picks the right intrinsic mix from `gather_dims` + the
source layout (section 6.4).

The `TileLoad` / `TileStore` constructor takes:

- `widths`: tile shape `(K_0, ..., K_{K-1})`.
- `dim_strides[d]`: per-dim stride coefficient for non-gather dims.
  `0` (CONSTANT), `1` (LINEAR), `c` (AFFINE).
- `replicate_factor_per_dim[d]`: `k` for REPLICATE dims, `1` otherwise.
- `src_dims[d]` / `dst_dims[d]`: source-array dim binding (Fortran /
  transposed handling).
- `gather_dims`: sorted subset of `range(src_ndim)` (resp. `dst_ndim` for `TileStore`) --
  **SOURCE-array dim indices** that gather. Lane geometry (`widths`) and source addressing
  (`gather_dims`) are orthogonal: `len(widths) = K_tile` is the lane count and may differ from
  `src_ndim`. The upper bound (`max(gather_dims) < src_ndim`) is checked at `validate()` time
  since `src_ndim` is read from the wired `_src` connector descriptor.
- `index_form`: `"none" | "per_dim" | "full"`. Mutually exclusive with
  the wired index connectors (section 9.4).
- `has_mask`, `pad_value` (gather OOB fill).

The lowered tile always has shape `(K_0, ..., K_{K-1})`. Per-dim
replication / splatting / gather happens *inside* the lib node's
expansion at codegen time.

### 5.2 Cross-dim broadcast resolution (sub-box -> full tile)

When some dims are narrower than "one element per lane" (CONSTANT,
REPLICATE, AFFINE-without-iter-var), the lib node replicates them to
fill the tile:

| Per-dim kind | Per-dim source range | Lane-to-source map |
|---|---|---|
| CONSTANT  | 1 element       | every lane reads the same element |
| LINEAR    | K_d elements    | lane `l` reads element `l`        |
| AFFINE    | K_d * s elem.   | lane `l` reads element `l*s`      |
| REPLICATE | K_d / k elem.   | lane `l` reads `floor(l/k)`       |
| MODULAR   | N elements      | lane `l` reads `(l*c + c_0) mod N`|
| GATHER    | data-dependent  | lane `l` reads `src[idx(l)]`      |

The Cartesian product of per-dim maps gives the K-dim load pattern;
the expansion emits it without a mid-pipeline reshape.

### 5.3 Diagonal and transpose -- gather encoding

- **Diagonal** (`a[i, i]` for K=2 tile vars `(i, j)`): two LINEAR dims
  share the same iter-var. Encoded as `gather_dims=(0, 1)` +
  `index_form="per_dim"` with `TileIota` 1-D index tiles per dim.
- **Transpose** (`a[j, i]` for K=2): K LINEAR dims in a non-canonical
  permutation. Same encoding -- permuted `TileIota` index tiles.

Both shapes reuse the unified `TileLoad` surface. The classifier
flags them structurally so the expansion can pick a dedicated
transpose intrinsic when one exists; otherwise it falls through to
the general gather path.

---

## 6. Library Nodes

### 6.1 Node set

| Node | Purpose |
|---|---|
| `TileLoad` | Unified tile read -- structured, gather, or mixed (section 5.1 + section 9). |
| `TileStore` | Unified tile write -- structured, scatter, or mixed. |
| `TileBinop` | Elementwise binary op. |
| `TileUnop` | Elementwise unary op. |
| `TileITE` | `where(mask, t, e)`. |
| `TileReduce` | Cross-lane reduction (tile -> scalar only; section 10.3). |
| `TileMaskGen` | ANY-dim-OOB conjunction -> bool tile (section 7.4). |
| `TileIota` | `arange`-style 1-D index tile for diagonal / transpose gather encoding (section 5.3). |

Per the design pivot, separate `TileGather` / `TileScatter` nodes are
**dropped**. Their behaviour is the gather / scatter path of
`TileLoad` / `TileStore`, selected by wiring an `_idx_<d>` or
`_idx_<d>` connectors (section 9).

### 6.2 Uniform operand contract

Every elementwise node (`TileBinop`, `TileUnop`, `TileITE`) and
every store-style node accepts each operand as one of:

| Kind | Connector | Source | Lowering |
|------------|-------------------|--------------------------------------------------------|------------------------------------------------|
| **Tile** | `_a` / `_b` ... | tile-shaped `Array` `AccessNode` | per-lane register read |
| **Scalar** | `_a` / `_b` ... | length-1 / `dace.data.Scalar` `AccessNode` | hardware splat (`_mm512_set1_pd`, `svdup_f64`) |
| **Symbol** | *no connector* | symbolic expression in `expr_a` / `expr_b` property | inline literal embedded in the per-lane body |

**Hardware rationale for Scalar as a first-class kind**: AVX-512, NEON
and SVE all provide single-cycle broadcast intrinsics from a scalar
register; cuTile auto-broadcasts in its DSL. Routing a length-1 read
through a per-lane Tile would waste both a register and an extra load.

#### Output kind rule (per user direction 2026-06-09)

Every elementwise op also produces a typed output (the `_c` connector).
The output kind is determined by the input kinds:

| Inputs | Output kind | Output descriptor |
|---|---|---|
| **Any** input is `Tile` | `Tile` | `Array(shape=widths)` |
| **All** inputs are `Scalar` / `Symbol` (no Tile) | `Scalar` (allowed) **or** `Tile` (also allowed for compositional flexibility) | `Scalar` / length-1 `Array` **or** `Array(shape=widths)` |

Verbatim from the user:

> Scalar-Scalar or Symbol-Scalar or Symbol-Symbol op: the output is allowed
> to be Scalar again.
>
> Scalar-Tile or Symbol-Tile -> full tile output.

Concretely:

* `TileBinop(kind_a=Scalar, kind_b=Symbol)` → may write a Scalar `_c`.
* `TileBinop(kind_a=Scalar, kind_b=Tile)` → must write a Tile `_c`.
* `TileBinop(kind_a=Tile, kind_b=Tile)` → must write a Tile `_c`.
* `TileUnop(kind_a=Scalar)` → may write a Scalar `_c`.
* `TileUnop(kind_a=Tile)` → must write a Tile `_c`.

Why this matters: a chain of pure scalar / symbolic ops (e.g. computing a
loop-invariant clamp threshold from outer-scope symbols) need not
materialise an intermediate tile-shape transient. The walker's Scalar
bridges can flow through Scalar-Scalar / Scalar-Symbol binops and
unops without paying the broadcast-to-tile cost until they actually
meet a tile-shape operand.

Validate enforces the contract: `_c` descriptor shape must equal
`widths` when any input is `Tile`; when all inputs are non-Tile, `_c`
may be either Scalar / length-1 Array (preferred for cheap chains) or
Tile-shape (allowed when downstream wants a broadcast register).

#### Forward-analysis pre-shape pass (single source of truth)

Per user direction 2026-06-09 ("If we perform widening before, the
replacement of tasklet to tile ops and inserting load/scatters should
become much easier"), all body-NSDFG widening (memlets + intermediate
transient descriptors) is done **proactively** by a single pre-pass.

* **`InferBodyTransientShapes(widths)`** runs AFTER `StrideMapByTileWidths`
  (so iter_vars mean "tile start") and BEFORE `GenerateTileIterationMask`,
  the walker, and the converter.

It owns TWO widening steps:

1. **Non-transient AN edge memlet widening**: classifies every
   non-transient AN's access pattern (CONSTANT vs non-CONSTANT). For
   accesses classified as non-CONSTANT, walks every edge whose memlet's
   data is that AN's name; for each single-element per-dim range whose
   begin references an iter_var, replaces `[beg]` with
   `[beg : beg + widths[k] - 1]`. So `A[ii]` becomes `A[ii:ii+W]`,
   `A[ii, jj]` becomes `A[ii:ii+W_0, jj:jj+W_1]`, etc.

   Leaves CONSTANT / Scalar / Symbol subsets alone (the begin doesn't
   reference an iter_var for those).

2. **Intermediate transient descriptor widening**: forward-propagates
   the kinds through every tasklet to fixed point. Each intermediate
   transient is mutated in place to either Scalar / length-1 (when
   the chain reads only CONSTANT non-transients) or
   `Array(shape=widths)` (when any producer reaches a non-CONSTANT
   non-transient). Memlets referencing widened transients are
   rewritten to span the full tile.

The user phrased the temporary-state invariant explicitly: after the
pre-pass, "we will have temporarily valid-looking SDFG that has
invalid tasklets but that is fine". The tasklets' bodies still
operate scalar-style (`_b = _a`), but their connectors now reference
tile-shape memlets / transients. The converter (next pass) fixes the
invalidity by rewriting the tasklets into Tile* lib nodes.

After this pass, every downstream pass is **shape-clean**:

* `StageInsideBody` mints bridges already aligned with the right
  shape; no special handling for transient mismatch.
* `ConvertTaskletsToTileOps` is a pure tasklet → lib-node rewriter,
  no widening / narrowing logic of its own. The lib-node's
  `_c` descriptor is whatever the pre-pass set; `validate()` checks
  consistency with the kind contract above.
* The lib-node pure expansion dispatches on the output descriptor
  shape (Scalar → single assignment; Tile → K-fold loop) without
  needing to re-analyse anything.

Conservative bias: when in doubt (e.g. classification fails, producer
unknown), the transient is widened to full tile. Over-widening is
benign (extra register pressure); under-widening is a correctness bug.

### 6.3 Symbolic broadcast

`Symbol` is not a degenerate `Scalar`. It is a *compile-time* constant
or symbolic expression embedded directly in the per-lane code string,
not materialised in any register. Use it for numeric literals
(`_out = _a + 1.0`) and outer-scope symbol broadcasts where no read
edge is needed.

### 6.4 Codegen dispatch for `TileLoad` / `TileStore`

The expansion makes one decision from the constructor properties + the
source array's layout:

```
let g = set(gather_dims)
let contig_tile_dim = src_dims.index(d) where src.strides[d] == 1, or None

if g is empty:
    -> structured load / store
       (per-dim ``dim_strides`` + ``replicate_factor`` drive the access pattern)
elif contig_tile_dim in g:
    -> real gather / scatter intrinsic
       (AVX-512 _mm512_i64gather_pd, SVE svld1_gather, cuTile ct.gather)
else:
    -> strip-mine -- outer scalar loop over the gather dim(s),
       inner contiguous / strided load per outer index
       (avoids the slow gather intrinsic when it isn't needed)
```

Why CPU codegen splits on contiguous-dim membership: gather intrinsics
on the contiguous dim hit a real per-lane scatter / gather; gather on
an outer (non-contiguous) dim collapses to an outer scalar loop where
each iteration does a cheap contiguous load of one row / column. The
lib node owns this dispatch -- the classifier just records facts.

### 6.5 Inner-dim-driven CPU emission strategy (formalised)

The CPU emission strategy for `TileLoad` (symmetric for `TileStore`)
is **driven by the per-dim kind of the contiguous source dim** -- the
dim whose stride is 1. Lane geometry on the contig dim dictates how
the inner SIMD register is materialised; outer (non-contig) dims
become explicit `for`-loops wrapping the SIMD inner.

Let `contig_src_dim` be the source dim with `src.strides[d] == 1`,
and `contig_tile_dim = src_dims.index(contig_src_dim)` if the contig
source dim is bound by a tile dim, else `None`. Let
`contig_kind = per_dim_kind[contig_tile_dim]` (or `CONSTANT` if no
tile dim is bound).

| `contig_kind` | Inner-SIMD strategy | Per-arch realisation |
|---|---|---|
| **CONSTANT** | Every lane reads the same element. Emit one scalar load, splat to the register. | AVX-512 `_mm512_set1_pd`; SVE `svdup_f64`; NEON `vdupq_n_f64`; cuTile `ct.full(shape, scalar)`; CUDA `__shfl_sync` broadcast. |
| **LINEAR** | Lane `l` reads element `base + l`. Emit one dense (aligned / unaligned) vector load. | AVX-512 `_mm512_loadu_pd`; SVE `svld1`; NEON `vld1q_f64`; cuTile `ct.load(view, ...)`; CUDA contiguous threads load consecutive elements. |
| **REPLICATE** (factor `k`) | Lane `l` reads element `base + floor(l/k)`. Emit one dense load of `W/k` elements followed by a broadcast / shuffle that duplicates each loaded value `k` times across consecutive lanes. | AVX-512 `_mm512_permutexvar_pd` on a stride-`k` index pattern; SVE `svdup_lane_f64`; NEON `vdupq_lane_f64`; cuTile broadcast via `ct.repeat`; CUDA per-thread index `floor(l/k)`. |
| **AFFINE** stride `s > 1` | Lane `l` reads element `base + l * s`. Try a strided-load intrinsic; fall back to GATHER. | AVX-512 `_mm512_i64gather_pd` with a precomputed index vector `[0, s, 2s, ...]`; SVE `svld1_gather` similarly; NEON has no strided load -> falls to scalar loop; cuTile relies on `ct.gather` with a strided index tile; CUDA permits the strided form via per-thread index computation. |
| **MODULAR** (`% N`) | Lane `l` reads element `base + ((l * c + c0) mod N)`. Materialise the mod index per lane and gather. | Same as AFFINE / gather path -- the modular index is precomputed via the per-lane index tile (G8). When `N | c * W_p` the classifier folds this back to LINEAR with a per-tile offset; the gather form only fires for irreducible cases. |
| **GATHER** | Lane `l` reads element `_idx_<contig_src_dim>[l] * 1`. Real gather intrinsic on the contig dim. | AVX-512 `_mm512_i64gather_pd`; SVE `svld1_gather_*_z`; NEON falls to scalar loop; cuTile `ct.gather(view, idx_tile)`; CUDA each thread issues `*ptr_base + idx[l]`. |

When the contig source dim is **NOT bound by any tile dim** (i.e. the
contig dim is a constant scope variable not iterated by the tile),
the inner load is a single scalar splat regardless of outer-dim kinds
-- the outer scalar `for`-loops walk the tile shape and the inner
register holds `W_{contig_tile_dim}` copies of the same scalar.

#### Outer-dim wrap

Every non-contig tile dim becomes an outer scalar `for`-loop wrapping
the inner SIMD body. The outer loop's per-iteration index contribution
follows §6.4: gather dims (`gather_dims` includes the outer source
dim) -> `_idx_<k>[<flat lane>] * src.strides[k]`; tile-mapped non-
gather dims -> affine; untouched dims -> handled by the outer `_src`
memlet base pointer.

K-fold nesting for `K = 3` (innermost-last contig):

```
for (l_0 = 0; l_0 < W_0; ++l_0)                  // outer dim 0
  for (l_1 = 0; l_1 < W_1; ++l_1)                // outer dim 1
    <inner-SIMD load over W_2 lanes>             // contig dim, per table above
```

#### Mask handling

For each strategy the mask gate composes as a separate decision on top
of the load itself:

* CONSTANT load + mask -> `mask[l] ? splat : dtype(0)` (the mask
  selects between the splat and the zero identity).
* LINEAR load + mask -> use the per-arch masked-load form
  (`_mm512_maskz_loadu_pd`, `svld1_z(pg, ptr)`).
* REPLICATE load + mask -> masked dense load of `W/k` followed by the
  same broadcast / shuffle as the unmasked path.
* AFFINE / MODULAR / GATHER load + mask -> per-arch masked-gather form;
  on architectures with no masked gather, fall to the strip-mined
  scalar loop with a per-lane `if (mask[l])` guard.

### 6.6 GPU emission strategy (formalised)

Two paths: raw CUDA codegen and cuTile.

#### CUDA codegen

The CUDA path maps **the contiguous tile dim to a warp** (or half-
warp / quarter-warp when `W_{contig} < 32`):

```
let W_c = widths[contig_tile_dim]
if W_c >= 32: assign full warp; each thread handles one lane
elif W_c >= 16: assign half-warp (16 threads)
elif W_c >= 8:  assign quarter-warp (8 threads)
else:           assign W_c threads + idle the rest
```

Outer (non-contig) tile dims become thread-block axes (`blockIdx.y`,
`blockIdx.z`) -- one thread block per outer-dim tile, threads within
the block share the inner SIMD register file.

Per-tile-dim inner-load strategy (mirror of §6.5):

* CONSTANT -> single global load + `__shfl_sync(0xffffffff, value, 0)`
  broadcast across the warp.
* LINEAR -> coalesced global load (each thread reads `src[base + tid]`).
* REPLICATE (`k`) -> coalesced load of `W_c/k` then `__shfl_sync` per
  thread to its broadcast lane.
* AFFINE / MODULAR / GATHER -> per-thread global load
  (`src[base + tid * s]` or `src[idx[tid]]`); coalescing depends on the
  pattern (LINEAR coalesces; GATHER is non-coalesced unless `idx[]`
  happens to be sorted).

Warp-wide load width (`W_{contig} > 32`) is split across multiple
loads; the compiler handles the loop unrolling.

**Tile-dim bounds < 32**: padded to a power of 2 and the unused
threads are masked out via the `_tile_iter_mask`. The contig tile dim
sets the warp lane count; outer dims set the grid shape.

#### cuTile

For cuTile we leverage the framework's own dispatch: every `TileLoad`
lowers to one `ct.load(view, shape=widths, padding_mode=...)` call;
cuTile's optimizer picks the per-load shape and the resulting
thread-block configuration based on architectural heuristics
(register pressure, SM count). We do NOT pre-bake the inner-dim
emission strategy on the cuTile path.

Gather / scatter on cuTile uses `ct.gather` / `ct.scatter` with
explicit index tiles; cuTile chooses the underlying intrinsic mix.
Mask handling delegates to `ct.where`.

### 6.7 Implementation phasing -- pure now, smarter strategies later

For the current phase **only the `pure` expansion is supported**:

* Every `TileLoad` / `TileStore` emits a K-fold nested `for`-loop
  with per-lane **scalar** loads / stores from `_src` / `_dst`,
  decoded by the unified per-source-dim offset (§9 + §6.4).
* The inner-dim-driven CPU strategy table (§6.5), CUDA warp mapping
  (§6.6), and cuTile dispatch (§6.6) are **planned but not yet
  implemented**. The lib node's `pure` expansion handles correctness
  uniformly; per-arch fast paths land as subsequent slices.

Why pure first: the walker (G7) + per-lane index materialisation (G8)
+ validators (§10) lock the IR shape end-to-end. The arch-specific
expansions are then a single-file slice each -- the IR contract is
already pinned by the lib-node `validate()` calls. Adding intrinsics
without the IR locked first would risk per-arch divergence on edge
cases (mask + replicate + strided combinations) that pure-expansion
testing surfaces cleanly.

---

## 7. Masking

### 7.1 Mask shape lock -- full-tile boolean Array

The mask is **only ever a full-tile boolean Array**. Specifically:

- shape exactly `(K_0, ..., K_{K-1})` -- same shape as the lib node's tile
  operands;
- dtype `dace.bool_`;
- storage `Register`, `transient=True`.

No other mask form is accepted. Per-dim masks, scalar masks, or
non-bool predicates are all rejected at constructor / validate time
(section 10). When only a single dim needs predication, the mask is still
generated at full `(K_0, ..., K_{K-1})` shape -- the unused dims contribute
the constant `true` to the conjunction and fold away at expansion
(section 7.4).

**Rationale**: a uniform shape is the only contract that makes the
lib-node surface closed. Hardware mask registers (AVX-512 `__mmask8`,
SVE predicate registers, NEON synthesised) are flat under the hood;
the K-dim layering is purely a software notion, and the codegen
collapses it to a single predicate at use.

### 7.2 `has_mask` toggle

Each node has a `has_mask: bool` constructor knob. When `False` the
`_mask` connector is omitted entirely (no spurious wiring, no
per-iteration mask load). When `True` every load / store / op inside
the expansion threads the predicate.

### 7.3 No-mask short-circuit

The interior region of a `masked_tail` split has `has_mask=False` on
every node -- the provably-divisible interior is the performance fast
path. Only the boundary regions carry masks.

### 7.4 `TileMaskGen` contract

For tile vars `(i_0, ..., i_{K-1})` with widths `(W_0, ..., W_{K-1})` and
global upper bounds `(N_0, ..., N_{K-1})`, `TileMaskGen` emits the
ANY-dim-OOB conjunction:

```
mask[l_0, ..., l_{K-1}] = (i_0 + l_0 < N_0) and ... and (i_{K-1} + l_{K-1} < N_{K-1})
```

When a particular dim doesn't need masking (the corner-absorbing peel
of section 8 covers it at full extent), its conjunct is the constant `true`
and folds away at expansion. The same lib node is used regardless of
which dim is being predicated.

#### Placement (2026-06-10): inside the body NSDFG

`GenerateTileIterationMask` emits the `TileMaskGen` lib node + the
`_tile_iter_mask` transient AccessNode **INSIDE the body NSDFG**, not
in the outer state. The walker (`StageInsideBody`) and converter
(`ConvertTaskletsToTileOps`) traverse the body NSDFG's arrays and
nodes to find the mask; placing it outside would mean the per-lib-node
`_mask` wiring couldn't see it.

The mask is a transient `Array(shape=widths, dtype=bool_,
storage=Register, transient=True)` registered on the body NSDFG's
inner SDFG. The TileMaskGen + its output AccessNode go into the body
NSDFG's start state.

#### Walker + converter wiring

Both `StageInsideBody` and `ConvertTaskletsToTileOps` look up the
inner mask name + the AccessNode that `TileMaskGen._o` writes to,
then for each lib node they mint:

* Set `has_mask=True` on the constructor.
* Wire `mask_producer_AN -> lib_node._mask` with memlet
  `_tile_iter_mask[0:W_0, ..., 0:W_{K-1}]`.

All consumer `_mask` edges read from the SAME AccessNode (the
producer's output AN). This is the SDFG-scheduling invariant of
section 3: fresh `add_access` calls per consumer would produce orphan
AccessNodes that DaCe's scheduler orders independently from the
producer.

Coverage (as of 2026-06-10):

| Pass | Lib nodes wired with `has_mask` + `_mask` |
|---|---|
| `StageInsideBody` (walker) | `TileLoad`, `TileStore` (structured + gather + scatter modes) |
| `ConvertTaskletsToTileOps` (converter) | `TileBinop` (plain + Symbol variant), `TileUnop`, `TileITE`, `TileReduce` |

### 7.5 Intra-tile (branch) masks

Branch normalisation produces `TileITE` with an explicit condition
tile. The iteration mask (`_iter_mask`) and the condition mask combine
inside the expansion to a single effective mask. No separate
`TileMaskAnd` / `Or` / `Not` lib nodes are needed for the common case;
they remain out of scope until a kernel demands them (section 10.5).

#### Cond-mask broadcasting (pending design slice, per user direction)

When a condition depends only on a SUBSET of tile dims -- e.g.
``if A[i] > 0.0:`` inside a K=2 widths `(W_0, W_1)` body, where the
condition depends on `i` (dim 0) but not `j` (dim 1) -- the
condition's natural shape is `(W_0,)`. The lib-node operand contract
requires the full `(W_0, W_1)` shape, so the condition must be
**broadcast** along the unused dim(s).

Two sub-cases for the cond-mask generation pass (NOT YET
IMPLEMENTED):

1. **Free-symbol-dependency analysis**: identify which tile dims the
   condition expression references (via the begin string + iter_var
   free symbols, same machinery the classifier uses).

2. **Per-dim shape + broadcast**: generate the partial-shape mask
   at the natural width (e.g. `bool[W_0]`), then broadcast along the
   dims it doesn't depend on (e.g. tile to `bool[W_0, W_1]`).

The mask wiring on `TileITE` already supports the full-shape `_cond`
operand; the gap is the pre-pass that generates the partial cond-mask
+ the broadcast lowering.

---

## 8. Remainder Loops

### 8.1 One remainder map per tiled dim

For K tiled dims the pipeline emits exactly **K + 1 regions**:

- 1 interior region (all tiled dims at main range, `has_mask=False`),
- K boundary regions (one per tiled dim, `has_mask=True`).

Untiled dims pass through at their full extent in every region.

### 8.2 Corner-absorbing peel (avoids 2^K - 1 corners)

**Algorithm (any K)**: pick an ordering `(d_{K-1}, d_{K-2}, ..., d_0)`
of the tiled dims -- innermost-first by convention. For each tiled dim
`d_p` (in order), emit a boundary region with:

- `d_p` at the **tail** range `[floor(N_{d_p} / W_{d_p}) * W_{d_p}, N_{d_p})`;
- **higher-priority dims** `d_q` with `q > p` (not yet peeled) at the
  **full** range `[0, N_{d_q})`;
- **lower-priority dims** `d_q` with `q < p` (already peeled) at the
  **main** range `[0, floor(N_{d_q} / W_{d_q}) * W_{d_q})`;
- untiled dims at their full extent in every region.

Each boundary region covers its own tail-strip and absorbs only those
corners that the not-yet-peeled dims still own. The lower-priority
"main" range guarantees disjointness with the regions that come later.

Define
- `M_d = floor(N_d / W_d) * W_d` -- the last main-tile boundary on dim `d`,
- `T_d = [M_d, N_d)` -- the tail strip on dim `d`,
- `F_d = [0, N_d)` -- the full range on dim `d`.

**K=2** (tiled dims `(i, j)` with `(i = d_1, j = d_0)`, innermost-first
peels `j` then `i`):

```
region 0 (interior): i in [0, M_i), j in [0, M_j)
region 1 (j boundary): i in F_i, j in T_j
region 2 (i boundary): i in T_i, j in [0, M_j)
```

The `(tail_i, tail_j)` corner is absorbed by region 1 (full i over j's
tail). Region 2 takes only `(tail_i, main_j)` -- no overlap.

**K=3** (tiled dims `(i, j, k)` with `(i = d_2, j = d_1, k = d_0)`,
innermost-first peels `k`, then `j`, then `i`):

```
region 0 (interior): i in [0, M_i), j in [0, M_j), k in [0, M_k)
region 1 (k boundary): i in F_i, j in F_j, k in T_k
region 2 (j boundary): i in F_i, j in T_j, k in [0, M_k)
region 3 (i boundary): i in T_i, j in [0, M_j), k in [0, M_k)
```

Disjointness:
- region 1 vs all others -- disjoint on `k` (only region 1 has `k in T_k`);
- region 2 vs region 0 -- disjoint on `j` (region 2 has `j in T_j`,
  region 0 has `j in [0, M_j)`);
- region 2 vs region 3 -- disjoint on `j` (region 2 has `j in T_j`,
  region 3 has `j in [0, M_j)`);
- region 3 vs region 0 -- disjoint on `i`.

Coverage: every `(i, j, k) in [0, N_i) x [0, N_j) x [0, N_k)` belongs to
exactly one region. **Result: K + 1 = 4 regions** (instead of
2^K - 1 = 7 corner cells + 1 interior = 8 a naive Cartesian split would
produce).

**Generalises to any K**: the pattern follows mechanically from the
algorithm above -- the `p`-th boundary region (counting from innermost)
has tail on `d_p`, full on `d_q` for `q > p`, main on `d_q` for `q < p`.

### 8.3 Mask threading

Each boundary region runs `TileMaskGen` over the conjunction of its
tiled dim's `(i_d + l_d < N_d)` predicate. Absorbed-full dims
contribute `true` and fold away (section 7.4). This way the lib-node
*interface* is uniform -- `has_mask=True` everywhere a region needs
predication -- even though the actual mask shape per region collapses
to a single non-trivial dim.

### 8.4 Strategies (orchestrator knob)

`remainder_strategy` selects:

- `masked_tail`: 1 interior + K boundary regions (8.1 + 8.2). Default
  for fixed-width back ends (AVX-512, NEON, scalar).
- `scalar_postamble`: boundary regions are Sequential scalar loops.
  No masks. Use when masked is measured slower.
- `full_mask`: 1 region, no split, has_mask=True everywhere. Default
  for runtime-VL back ends (SVE). No remainder needed.

### 8.5 Fixed-width vs runtime-VL (SVE) unification

The widths in section 2.1 are read-only inputs to the lib nodes; the
shape `(W_0, ..., W_{K-1})` is fixed at lib-node construction. The
back end interprets the shape:

| Property | Fixed-width (AVX-512/NEON/scalar/cuTile) | Runtime-VL (SVE) |
|-----------------|------------------------------------------|------------------|
| `widths[p]` | compile-time integer literal | symbolic; resolves to `svcntd()` at runtime |
| Lane count | exactly `widths[p]` | implementation-defined per VL |
| Iteration mask | optional (masked_tail) / always (full_mask) | always (svwhilelt) |
| Remainder | masked_tail or scalar_postamble | none (full_mask absorbs the tail) |

The lib-node *interface* is identical. Only two things change per
back-end:

1. `widths[p]` is an int literal vs a symbol.
2. The mask source: `TileMaskGen` over `i_d + l_d < N_d` for
   fixed-width; the predicate `svwhilelt(i_d, N_d)` for SVE
   (semantically the same conjunction, materialised as a runtime
   predicate register).

The classifier (section 4), composition (section 5), and operand
contract (section 6.2) are back-end agnostic.

### 8.6 K=1 SVE while-loop (no remainder)

For `K = 1`, runtime-VL SVE collapses the entire iteration into a
single while-loop:

```
i = 0
while i < N:
    pred = svwhilelt_b64(i, N)
    <body with has_mask=True, mask=pred>
    i += svcntd()
```

This is exactly the `full_mask` strategy at K=1: 1 region, mask
covers the tail on the last iteration, no remainder map. The outer
Map is replaced by a `LoopRegion` carrying the while form; the body
NestedSDFG and lib-node calls are unchanged.

For K >= 2, runtime-VL is applied only on the **innermost** tile
dim; outer tile dims remain fixed-width Map loops. The outer dims
follow section 8.2 (with full_mask on the innermost dim's regions);
the innermost dim's regions collapse to one as above.

---

## 9. Index Encoding On `TileLoad` / `TileStore`

`TileLoad` and `TileStore` carry the gather / scatter index as a set
of distinct, non-overlapping connectors. The connector name is the
**single source of truth** for the encoding form; `index_form`
(constructor property) is a sanity flag, not the discriminator.

### 9.1 Connector grammar

Each lib node has a fixed connector vocabulary, every name unique. The
`d` suffix on `_idx_<d>` is a **source-array dim index** (resp. dest
for `TileStore`), NOT a tile dim index -- lane geometry (`widths`) and
source addressing (`gather_dims`) are orthogonal.

| Connector | Direction | Shape | Role |
|---|---|---|---|
| `_src` | in (load) / in (store source tile) | full source / `(K_0, ..., K_{K-1})` | data |
| `_dst` | out (load) / out (store dest array) | `(K_0, ..., K_{K-1})` / full dest | data |
| `_mask` | in (optional) | `(K_0, ..., K_{K-1})` bool | predicate |
| `_idx_<k>` | in (optional, one per `k in gather_dims`) | **see section 9.2** | gather index for **source dim k** |

There is **no separate `_idx_full` connector**. The full N-D case is
just an `_idx_<k>` whose shape spans every tile dim.

### 9.2 Per-source-dim index shape encodes lane dependency

The shape of `_idx_<k>` is determined by **which tile lane indices the
gather expression for source dim `k` depends on**. Let `deps(k)` be
the sorted tuple of **tile dim** indices the expression touches. Then:

| `deps(k)` | `_idx_<k>` shape | Lib-node behaviour |
|---|---|---|
| `()` (no lane index) | `(1,)` -- scalar index | every lane reads the same source-dim-k index |
| `(p,)` one tile dim | `(W_p,)` | broadcast over the other tile lanes |
| `(p, q)` | `(W_p, W_q)` | broadcast over remaining tile lanes |
| `(0, ..., K_tile-1)` (full) | `(W_0, ..., W_{K_tile-1})` | full N-D (subsumes the legacy `_idx_full`) |

Worked examples (tile vars `(i, j)` with widths `(W_i, W_j)`,
optional outer-constant `k_o`). `K_tile = len(widths)` is the lane
count; `K_src = src.ndim` is the source rank. Indices in
`gather_dims` are SOURCE-dim indices:

| Access | `src.ndim` | `gather_dims` | `_idx_<k>` shapes |
|---|---|---|---|
| `a[idx[i]]` (K_tile=1, K_src=1) | 1 | `(0,)` | `_idx_0: (W_i,)` |
| `a[idx[i], j]` (K_tile=2, K_src=2) | 2 | `(0,)` | `_idx_0: (W_i,)` |
| `a[idx[i, j], j]` (K_tile=2, K_src=2) | 2 | `(0,)` | `_idx_0: (W_i, W_j)` |
| `a[idx[i], j, k_o]` (K_tile=2, K_src=3, k_o outer) | 3 | `(0,)` | `_idx_0: (W_i,)` |
| `a[idx[i], j, idb[i]]` (K_tile=2, K_src=3) | 3 | `(0, 2)` | `_idx_0: (W_i,)`, `_idx_2: (W_i,)` |
| `a[idx[i, k_o], j, idb[i, k_o]]` ICON (K_tile=2, K_src=3, k_o outer) | 3 | `(0, 2)` | `_idx_0: (W_i,)`, `_idx_2: (W_i,)` |
| `a[idx[i, j, k]]` (K_tile=3, K_src=1) | 1 | `(0,)` | `_idx_0: (W_i, W_j, W_k)` |

The lib node walks every source dim `k in range(src.ndim)` and emits
`_idx_<k>[<flat lane>] * src.strides[k]` for gather dims, the affine
`coeff[d] * src.strides[k] * __l<d>` (with `d = src_dims.index(k)`)
for tile-mapped non-gather dims, and nothing for source dims with no
tile-lane contribution (the outer `_src` memlet offset covers them).

The classifier records `deps(k)` per gather source dim; the staging
pass materialises the index tile at the right rank (see section 3.8).

### 9.3 Constructor signature

```python
TileLoad(name, widths,
         dim_strides,                # tuple of K_tile ints; 0 = CONSTANT, 1 = LINEAR, s = AFFINE
         replicate_factor_per_dim,   # tuple of K_tile ints; k for REPLICATE, 1 otherwise
         src_dims,                   # tuple of K_tile ints; per-tile-dim source-dim binding
         gather_dims=(),             # sorted subset of range(src_ndim); SOURCE-dim indices
         has_mask=False,
         pad_value=0)                # OOB fill for gather; ignored when gather_dims is empty
```

`TileStore` mirrors this with `dst_dims` in place of `src_dims` and
no `pad_value`. There is no `index_form` property -- the form is
inferred from the wired connectors' shapes. The `max(gather_dims) <
src_ndim` upper bound is checked at `validate()` time (not in the
constructor) since `src_ndim` is read from the wired `_src` edge.

### 9.4 Identifiability invariant

The validator (section 10) checks for each `TileLoad` / `TileStore`:

- `gather_dims` is sorted, unique, non-negative (constructor-time).
- Every `k in gather_dims` satisfies `k < src_ndim` (`dst_ndim` for `TileStore`); read from the wired `_src` / `_dst` edge at `validate()` time.
- An `_idx_<k>` connector is wired iff `k in gather_dims`.
- Each `_idx_<k>`'s descriptor shape is `tuple(widths[p] for p in deps_k)` for some sorted subset `deps_k` of `range(K_tile)` (or `(1,)` for scalar). The validator only checks the shape is a Cartesian product of widths.
- Each `_idx_<k>`'s descriptor dtype is signed integer (`int32` / `int64`); see section 10.4.

Any other combination is rejected at `validate()` with a message
naming the conflict.

---

## 10. Validation Rules

| Where | Checks |
|--------------|------------------------------------------------------------------------------|
| Constructor | `widths` length in `{1, 2, 3}`; operand kinds in `{Tile, Scalar, Symbol}`; `index_form` / `gather_dims` consistency; replicate factors divide widths. |
| `validate()` | All declared connectors are wired; tile-operand dtypes match the output dtype (uniform-dtype lock); mask present iff `has_mask`; **mask descriptor is `Array(shape=widths, dtype=bool_, storage=Register, transient=True)`** (section 7.1); each `_idx_<d>`'s shape is a Cartesian product of `widths` (section 9.4). |
| Expansion | Source array rank >= K; `src_dims` permutation valid; index tile shapes match `widths` (full / per-dim form); padding mode allowed. |

Loud failure at every layer. No silent fallbacks.

### 10.1 Operand-dtype uniformity (locked)

Tile operands across a single op share a dtype. Cross-dtype operations
(`f32 + f64`) raise `NotImplementedError` at `validate()` time. A mixed-
dtype kernel must materialise the cast through a `TileUnop` of kind
`cast_to_<dtype>` first. Rationale: AVX-512 / SVE lane widths differ per
dtype; supporting cross-dtype binops requires per-arch dispatch we'd
rather defer.

### 10.2 Mask descriptor lock

The `_mask` operand, when wired, is a `dace.data.Array` with shape
exactly `widths`, dtype `dace.bool_`, storage `Register`,
`transient=True`. Scalar masks, per-dim masks, non-bool predicates, or
non-Register storage are rejected at `validate()` time. The constructor
also rejects any `has_mask=True` request when the lib node's tile shape
is not declarable (e.g. an inconsistent operand width).

### 10.3 TileReduce shape lock -- tile -> scalar only

`TileReduce` is locked to **full-tile -> scalar** for the current
slice. A `(K_0, ..., K_{K-1})` tile input reduces to a single value
(`dace.data.Scalar`).

**Contract**:

> If `axis` is omitted (or equivalent to the full tile shape), the
> reduction lowers to the per-arch full horizontal reduce
> (`_mm512_reduce_*`, SVE `svaddv`, etc.) and writes a scalar output.
>
> If `axis` is given and does **not** reduce the full tile to a
> scalar -- i.e. any non-empty subset of dims is preserved
> (e.g. `axis=(0,)` on a K=2 tile producing a `(K_1,)` row, or
> `axis=(1,)` on a K=3 tile producing a `(K_0, K_2)` plane) -- the
> node raises **`NotImplementedError`** at construction time. The
> error message names the requested axis and the expected output
> shape so the caller can fall back to scalar code (the remainder
> map's scalar postamble) or refuse the kernel.

The classifier must not emit a partial-axis reduction. Axis-keep /
per-row / per-plane reductions are recognised as future work but
deliberately unimplemented.

Rationale: keeping reductions tile->scalar avoids the per-arch
horizontal-shuffle plumbing that axis-keep would require (SVE
`svaddv` is global; AVX-512 needs hand-rolled `_mm512_reduce_*`
plus per-row materialisation; cuTile needs a Tile-IR reduce with
an explicit axis). Lift when a kernel demands it.

### 10.4 Index tile dtype lock

`TileLoad` / `TileStore` `_idx_<d>` index connectors must carry a **signed integer** descriptor:

- dtype in `{dace.int32, dace.int64}` (int64 default);
- storage `Register`, `transient=True`;
- shape matches the form per section 9.1 / section 9.2 (full = `(K_0, ..., K_{K-1})`;
  per-dim = `(K_d,)`).

Unsigned, float, or bool indices are rejected at `validate()` time.
Mixing int32 and int64 across a single node's index connectors is
rejected for the same reason as section 10.1 (per-arch lane-width
divergence).

### 10.5 Error model -- raise vs refuse

Two distinct failure modes, with explicit rules for which fires when:

| Failure | Mode | Where |
|------------------------------------------------|-----------------------|----------------------|
| Operand shape / dtype / mask invariant broken | `NotImplementedError` | constructor / `validate()` |
| Pattern outside the per-dim lattice (section 4) | classifier returns `None` -> caller falls back to scalar code | classifier |
| Body NestedSDFG missing for `K >= 2` | `NotImplementedError` | G7 multi-dim entry hook |
| Non-transient AN survives mid-body after staging | `NotImplementedError` | G7 multi-dim entry hook |
| Cross-state WCR chain the linear-acc rule rejects | `NotImplementedError` | `StageGlobalArrayThroughScalars` |
| Remainder map produced > K + 1 regions | `AssertionError` | G5 invariant test |
| Indeterminate cross-dim composition | `NotImplementedError` | classifier emit step |
| `TileReduce(axis=...)` with non-scalar output | `NotImplementedError` | constructor |
| Array stride pattern is neither C nor Fortran | `NotImplementedError` | `validate()` (section 2.3) |
| WCR memlet appears anywhere other than `AccessNode -> MapExit` with single-element AccessNode | `NotImplementedError` | `validate()` (section 3.5) |
| `_idx_<d>` descriptor shape not a Cartesian product of `widths` | `NotImplementedError` | `validate()` (section 9.4) |
| Per-lane index symbol (`_laneid_<i>`-style) survives post-emit | `AssertionError` | post-emit audit (section 10.6) |
| Index tile dtype not in `{int32, int64}` | `NotImplementedError` | `validate()` (section 10.4) |

**Rule of thumb**:

- **Raise** when the input violates a *load-bearing* design
  invariant (section 2.4, section 3, section 7.1, section 10.\*). These are bugs in the staging
  / lowering pipeline that should never occur post-G1 / G7.
- **Refuse** (return `None` from the classifier; caller falls back to
  scalar code) when the input is syntactically valid but outside the
  lattice / unsupported pattern. This keeps the kernel functional via
  the scalar postamble.

The classifier never raises; it always returns a `TileAccess` or
`None`. All raises live in passes that consume the classifier.

### 10.6 Post-emit audit -- no per-lane index symbols

After lib-node expansion finishes, **no `_laneid_<i>`-style symbol
nor any per-lane index symbol that the staging pre-pass (section 3.8)
was meant to materialise** may appear in the SDFG. A post-emit pass
walks every tasklet body, every interstate-edge expression, and every
memlet subset; any residual lane-fanout symbol triggers
`AssertionError`. Mirrors the legacy `assert_no_laneid_in_tile_path`
audit but extended to the modern per-lane-index-tile contract.

---

## 11. Implementation Audit

### 11.1 What the codebase already provides

| Spec section | Existing implementation |
|--------------|----------------------------------------------------------------------------------------------------|
| section 4 Lattice | [utils/tile_access.py:85](utils/tile_access.py): `PerDimKind` with `BROADCAST` / `STRUCTURED_1` / `REPLICATE` / `AFFINE` / `GATHER`. |
| section 4.2 Join | [utils/tile_access.py](utils/tile_access.py): partial -- REPLICATE-detection via `int_floor`/`int_ceil`/`__int_floor`/`__int_ceil` ([line 340](utils/tile_access.py#L340)). **Missing**: explicit loop-variant-coefficient -> GATHER. |
| section 3 Staging | [stage_global_array_through_scalars.py](stage_global_array_through_scalars.py): per-subset scalar bridge. Full-tile fallback + multi-dim entry hook pending (G7). |
| section 3 Body lowering | [promote_nsdfg_body_to_tiles.py](promote_nsdfg_body_to_tiles.py): descent, canonical lowering for K=1 and K>=2. |
| section 6.1 Lib nodes | [libraries/tileops/nodes/](../../../libraries/tileops/nodes/): 10 lib nodes implemented (3,278 LoC). |
| section 6.2 Operand kinds | [tile_binop.py:62-65](../../../libraries/tileops/nodes/tile_binop.py#L62): `_TILE` / `_SCALAR` / `_SYMBOL`. `TileUnop` ditto. |
| section 7.1 Mask shape | [tile_mask_gen.py](../../../libraries/tileops/nodes/tile_mask_gen.py): emits ANY-OOB conjunction at tile shape. |
| section 7.2 `has_mask` | every lib node has the constructor knob. |
| section 8.2 Corner-absorbing peel | [split_map_for_tile_remainder.py](split_map_for_tile_remainder.py): comments at lines 7-34 describe the K boundary peel; impl appears to match. **Audit pending** (G5). |
| section 8.4 Strategies | [vectorize_cpu_multi_dim.py:133-228](vectorize_cpu_multi_dim.py#L133): `full_mask` / `masked_tail` / `scalar_postamble` knob. |
| section 5.1 `replicate_factor_per_dim` | [tile_load.py:310](../../../libraries/tileops/nodes/tile_load.py#L310). Wired through pure expansion. |
| section 9 Index encoding | Currently `tile_gather.py` + `tile_scatter.py` (separate nodes). G3 folds them into `TileLoad` / `TileStore` with `gather_dims` + variable-shape `_idx_<d>` connectors (section 9.2 lane-dependency rule). |

### 11.2 Vocabulary gap (cosmetic, mechanical commit)

The spec uses **CONSTANT / LINEAR / AFFINE / REPLICATE / MODULAR /
GATHER**. The codebase uses **BROADCAST / STRUCTURED_1 / REPLICATE /
AFFINE / GATHER** (no MODULAR yet). Rename + add MODULAR in
[utils/tile_access.py](utils/tile_access.py):

| Spec | Code today | Action |
|------------|-----------------|---------------------------------------|
| CONSTANT | `BROADCAST` | rename enum member |
| LINEAR | `STRUCTURED_1` | rename enum member |
| AFFINE | `AFFINE` | keep |
| REPLICATE | `REPLICATE` | keep |
| MODULAR | *(absent)* | **add new enum member** + detector |
| GATHER | `GATHER` | keep |

`TileAccessKind.BROADCAST` / `.STRUCTURED` (the whole-subset kind) also
get the analogous rename. Single mechanical commit, no behaviour
change. Update the compat shim, descent, and outer-state pass call
sites in the same commit.

### 11.3 Six gaps -- see section 12

---

## 12. Gaps And Action Items

Each item is a single self-contained slice. Ordering reflects
dependencies; landing them in order keeps every commit green.

### G1 -- DROPPED

The boundary full-subset convention is established by the existing
[`ExpandNestedSDFGInputs`](../../interstate/expand_nested_sdfg_inputs.py)
(invoked unconditionally by `VectorizeCPUMultiDim` before any tile
lowering). All analysis runs inside the body NSDFG; no dedicated
boundary-enforcement pass is needed.

### G2 -- Vocabulary rename + add MODULAR

Mechanical: `BROADCAST -> CONSTANT`, `STRUCTURED_1 -> LINEAR`, keep
`REPLICATE` / `AFFINE` / `GATHER`, add `MODULAR`. Update
[utils/tile_access.py](utils/tile_access.py) +
[utils/tile_access_compat.py](utils/tile_access_compat.py). Drop the
redundant `TileAccessKind` enum; the whole-subset kind is the max
of per-dim kinds.

### G3 -- Fuse gather/scatter into TileLoad/TileStore + identifiability

Fold `TileGather` / `TileScatter` into `TileLoad` / `TileStore` per
the design pivot (section 5.1 + section 6.1). Concrete work:

1. Add to `TileLoad` / `TileStore` constructor: `gather_dims: Tuple[int, ...]`
   and `index_form: Literal["none", "per_dim", "full"]`.
2. Add per-arch expansion dispatch (section 6.4): structured /
   real-gather / strip-mine based on `gather_dims` x contiguous-dim
   membership.
3. Connector grammar + validators (section 9): exactly one `_idx_<d>` per `d in gather_dims`; each connector's descriptor shape is a Cartesian product of widths matching the classifier's `deps(d)`.
4. Drop `TileGather` / `TileScatter` lib node classes (move existing
   call sites onto the unified API; no behaviour change for kernels
   that already work).
5. Unit tests for the three forms (none / per_dim / full) plus the
   contiguous-vs-non-contiguous CPU dispatch.

### G4 -- Tile-dependent symbol classifier

Add `_is_tile_dependent(symbol, iter_vars, inner_sdfg) -> bool`:
walks interstate-edge assignment definitions transitively (memoised).
Wire into the per-dim classifier per section 4.2: any tile-dependent
symbol in a dim's expression forces GATHER. K-agnostic.

### G5 -- Remainder region-count regression tests

Construct K=1 / K=2 / K=3 maps with non-divisible `N_d` and assert
exactly K+1 regions (2 / 3 / 4) per section 8.2. Verify each
region's ranges match the innermost-first peel.

### G6 -- AFFINE stride must be tile-invariant

Gate the AFFINE classification on `s`'s tile-independence (G4
helper). `a[N*i]` with `N` outer-constant -> AFFINE; with `N` tile-
dependent -> GATHER.

### G7 -- Stage-global full-tile fallback + multi-dim entry hook

`StageGlobalArrayThroughScalars` gains: (a) full-tile fallback per
section 3.3 case 2, (b) multi-dim entry hook that asserts the
call convention (section 2.4) and refuses when a non-transient
AccessNode survives mid-body after staging. `VectorizeCPUMultiDim`
wires `StageGlobalArrayThroughScalars` between G1 and the widening
passes. The descent
([`promote_nsdfg_body_to_tiles.py`](promote_nsdfg_body_to_tiles.py))
is the canonical lowering for K >= 2.

### G8 -- Per-lane index materialisation pre-pass + post-emit audit

New pass `PreparePerLaneIndices` (section 3.8) runs after staging
and before lib-node emission. For each gather memlet, computes the
lane-dependency set per gather dim, mints an integer transient of
shape `tuple(widths[p] for p in deps)`, emits a constant-assignment
tasklet that materialises the per-lane index, and wires the tile
into the corresponding `TileLoad` / `TileStore` `_idx_<d>` connector.

Post-emit audit (section 10.6) walks every tasklet / interstate
expression / memlet subset and asserts no `_laneid_<i>`-style
symbol survives. Modern equivalent of legacy
`assert_no_laneid_in_tile_path`, extended to per-lane index
materialisation contract.

Unit tests cover the lane-dependency patterns from section 9.2:
1-D, 2-D, scalar (constant index), full N-D, mixed (one per-dim
gather + one full-tile gather on the same node).

---

## 13. Delivery Plan

| Day | Goal |
|----------|--------------------------------------------------------------------------------------------|
| Mon | G2 (vocab rename + add MODULAR) + G7 stage-global multi-dim entry landed. (G1 dropped -- existing `ExpandNestedSDFGInputs` establishes the boundary.) |
| Tue AM | G3 (TileLoad/TileStore unification + variable-shape `_idx_<d>`) + G4 (tile-dep symbol classifier) + G8 (per-lane index materialisation pre-pass) landed. |
| Tue PM | Design freeze meeting. Lock lib-node interfaces. |
| Wed | G5 (K=1/K=2/K=3 remainder audit) + G6 (symbol-coefficient distinction) + G7 full-tile fallback. |
| Thu | Velocity-tendencies stencils parse + lower end-to-end through the pipeline. |

---

## 14. Out Of Scope (Explicit Non-Goals)

These are deliberately excluded from this design; revisit only when a
concrete kernel demands them:

- **Mask combinators as lib nodes** (`TileMaskAnd` / `Or` / `Not`).
  Branch normalisation produces `TileITE` which is sufficient for
  the current kernel corpus.
- **Cross-tile (across-Map-invocations) reductions**. `TileReduce`
  reduces within one tile; across-tile is a separate concern handled
  by the WCR / reduce-scope machinery in the rest of DaCe.
- **GPU scatter (`TileStore` with `gather_dims`) cross-block conflicts**. Single-thread-per-
  lane block layouts only; cross-block atomics deferred.
- **Mixed dtypes inside a single op** (section 10.1). Materialise via an
  explicit cast `TileUnop` first.
- **K >= 4**. The constructors refuse `len(widths) > 3` at construction
  time. Lift only when a kernel justifies the register pressure.
- **Diagonal / transpose as dedicated structured patterns**. Both fold
  into GATHER for now (section 5.4); a structured fast path can land later
  behind the same `TileLoad` surface.
- **Auto-classify load node** (a single node that defers
  classification to its expansion). The pipeline-time classifier
  already covers it cleanly; adding an umbrella node would duplicate
  the lattice. Marked NOT pursued.

---

## Appendix A -- Cross-Dimension Composition Examples

### A.1 K = 1

| Access | Per-dim kind | Lowering |
|------------|---------------|-------------------------------------------|
| `a[i]` | LINEAR | TileLoad, `dim_strides=(1,)` |
| `a[0]` | CONSTANT | TileLoad, broadcast |
| `a[i//2]` | REPLICATE k=2 | TileLoad, replicate factor 2 |
| `a[2*i+1]` | AFFINE s=2 | TileLoad, `dim_strides=(2,)` |
| `a[idx[i]]`| GATHER | TileLoad, `gather_dims=(0,)`, `index_form="per_dim"`, `_idx_0` |

### A.2 K = 2 (tile vars `(i, j)`)

| Access | Per-dim kinds (i, j) | Lowering |
|-----------------------|------------------------------|--------------------------------------------------|
| `a[i, j]` | (LINEAR, LINEAR) | TileLoad, `dim_strides=(1,1)` |
| `a[0, j]` | (CONSTANT, LINEAR) | TileLoad, broadcast dim 0, contiguous dim 1 |
| `a[i, 0]` | (LINEAR, CONSTANT) | TileLoad, contiguous dim 0, broadcast dim 1 |
| `a[i//2, j]` | (REPLICATE k=2, LINEAR) | TileLoad, dim 0 replicate factor 2 |
| `a[2*i + 1, j]` | (AFFINE s=2, LINEAR) | TileLoad, `dim_strides=(2,1)` |
| `a[i, i]` | (LINEAR i, LINEAR i) | TileLoad diagonal, `gather_dims=(0,1)`, `index_form="per_dim"` |
| `a[j, i]` | (LINEAR j, LINEAR i) | TileLoad transposed, `gather_dims=(0,1)`, `index_form="per_dim"` |
| `a[idx[i], j]` | (GATHER, LINEAR) | TileLoad, `index_form="partial"`, gather_dims=(0,) |
| `a[idx0[i], idx1[j]]` | (GATHER, GATHER) | TileLoad, `index_form="per_dim"` |
| `a[idx[i, j]]` | flat 1-D, idx is N-D | TileLoad, `index_form="full"` |
| `a[2*sym + 1, j]` | (AFFINE / GATHER on sym, LIN)| AFFINE if `sym` tile-independent else GATHER (section 4.2) |
| `a[i % N, j]` | (MODULAR N, LINEAR) | TileLoad (general MODULAR), `index_form="partial"` |
| `a[i % W_0, j]` | (MODULAR -> LINEAR, LINEAR) | TileLoad -- tile-aligned reduction (section 4.2) |
| `a[i % (2*W_0), j]` | (MODULAR -> LINEAR, LINEAR) | TileLoad with per-tile constant offset (section 4.2) |

### A.3 K = 3 (tile vars `(i, j, k)`)

| Access | Per-dim kinds (i, j, k) | Lowering |
|--------------------------|---------------------------------|-------------------------------------------------------|
| `a[i, j, k]` | (LINEAR, LINEAR, LINEAR) | TileLoad, `dim_strides=(1,1,1)` |
| `a[0, j, k]` | (CONSTANT, LINEAR, LINEAR) | TileLoad, broadcast dim 0 |
| `a[i, 0, k]` | (LINEAR, CONSTANT, LINEAR) | TileLoad, broadcast dim 1 |
| `a[i, j, 0]` | (LINEAR, LINEAR, CONSTANT) | TileLoad, broadcast dim 2 |
| `a[i//2, j, k]` | (REPLICATE k=2, LINEAR, LINEAR) | TileLoad, dim 0 replicate factor 2 |
| `a[i, j//4, k]` | (LINEAR, REPLICATE k=4, LINEAR) | TileLoad, dim 1 replicate factor 4 |
| `a[2*i + 1, j, k]` | (AFFINE s=2, LINEAR, LINEAR) | TileLoad, `dim_strides=(2,1,1)` |
| `a[i, i, k]` | (LINEAR i, LINEAR i, LINEAR) | TileLoad (diagonal on dims 0,1), `index_form="per_dim"` |
| `a[k, j, i]` | (LINEAR k, LINEAR j, LINEAR i) | TileLoad transposed, `index_form="per_dim"` (permuted) |
| `a[idx[i], j, k]` | (GATHER, LINEAR, LINEAR) | TileLoad, `index_form="partial"`, gather_dims=(0,) |
| `a[idx0[i], j, idx2[k]]` | (GATHER, LINEAR, GATHER) | TileLoad, `index_form="partial"`, gather_dims=(0,2) |
| `a[idx0[i], idx1[j], idx2[k]]` | (GATHER, GATHER, GATHER) | TileLoad, `index_form="per_dim"` |
| `a[idx[i, j, k]]` | flat 1-D, idx is K-D | TileLoad, `index_form="full"` |
| `a[i % N, j, k]` | (MODULAR N, LINEAR, LINEAR) | TileLoad (general MODULAR), `index_form="partial"` |
| `a[i % W_0, j, k]` | (MODULAR -> LINEAR, LINEAR, LINEAR) | TileLoad -- tile-aligned reduction (section 4.2) |

---

## Appendix B -- Symbol Definitions

| Symbol | Meaning |
|--------|------------------------------------------------------------------------|
| K | Number of tiled dimensions; K in {1, 2, 3}. |
| W_d | Tile width on dim `d`; `widths[d]`. |
| N_d | Global upper bound on dim `d`. |
| i_d | Tile iter-var on dim `d` (outer Map's parameter). |
| l_d | Lane index within the tile on dim `d`; `0 <= l_d < W_d`. |
| k | REPLICATE factor; the dim varies in groups of `k` lanes. |
| s | AFFINE stride coefficient. |
| c | AFFINE / LINEAR offset (loop-invariant constant). |

---

## Appendix C -- K capability matrix

| Feature | K=1 | K=2 | K=3 | Notes |
|------------------------------|-----|-----|-----|---------------------------------------------------------------|
| Full tile shape `(W_0,...)` | yes | yes | yes | constructors accept `len(widths) in {1, 2, 3}` |
| Per-dim lattice (section 4) | yes | yes | yes | K-agnostic; lattice rule applies per-dim |
| Cross-dim composition (section 5) | yes | yes | yes | sub-box expansion handles any K |
| Structured `TileLoad` / `TileStore` | yes | yes | yes | `dim_strides` + `replicate_factor_per_dim` per K |
| `TileLoad` full N-D gather | yes | yes | yes | `_idx_<d>` shape `(K_0, ..., K_{K-1})` |
| `TileLoad` per-dim gather (`_idx_<d>`) | yes (1)| yes | yes | `_idx_0, ..., _idx_{K-1}`; K=1 collapses to single `_idx_0` |
| `TileLoad` partial gather | n/a | yes | yes | needs >=2 dims to have a partial subset |
| Mask `(K_0, ..., K_{K-1})` | yes | yes | yes | always full-tile shape |
| Remainder regions | 2 | 3 | 4 | interior + K boundaries (section 8.2) |
| `TileReduce` (tile -> scalar) | yes | yes | yes | axis-keep refused (section 10.3) |
| `TileITE` | yes | yes | yes | shape-agnostic |
| Diagonal / transpose | n/a | yes | yes | fold to per-dim GATHER (section 5.4) |
| MODULAR -> LINEAR reduction | yes | yes | yes | when `N | c * W_p` (section 4.2) |
| NestedSDFG body (required) | opt | **req** | **req** | inlining permitted only for K=1 |
| Stage-global mandatory | opt | **req** | **req** | section 3.3 hard rule for `K >= 2` |

`opt` = optional / legacy-compatible; `req` = required by the design.

---

## Appendix D -- Backend <-> feature support matrix

| Lib node / feature | AVX-512 (`avx512`) | NEON (`neon`) | SVE (`sve`) | scalar (`pure`) | cuTile (`cutile_gpu`) |
|---------------------------|--------------------|-----------------|--------------------|-----------------|-----------------------|
| `TileLoad` CONTIGUOUS | yes `_loadu_pd` | yes `vld1q_f64` | yes `svld1_*` | yes | yes `ct.load` |
| `TileLoad` STRIDED | yes via gather | yes scalar loop | yes `svld1_gather_*` | yes | yes via gather |
| `TileLoad` REPLICATE | yes shuffle + bcast | yes `vdupq_lane` | yes `svdup` | yes | yes via indexing |
| `TileLoad` MODULAR (gen.) | gather | gather | `svld1_gather` | scalar loop | `ct.gather` |
| `TileLoad` masked | yes `_maskz_loadu_pd`| yes scalar guard | yes predicated load | yes if branch | yes `mask=` |
| `TileStore` masked | yes `_mask_storeu_pd`| yes scalar guard | yes predicated store | yes if branch | yes `mask=` |
| `TileLoad` gather (any form) | yes `_i64gather_pd` | scalar loop | yes `svld1_gather` | yes | yes `ct.gather` |
| `TileStore` scatter | yes `_i64scatter_pd` | scalar loop | yes `svst1_scatter` | yes | yes `ct.scatter` |
| `TileBinop` (Tile/Tile) | yes all `_OP_CPP` | yes | yes | yes | yes |
| `TileBinop` (Tile/Scalar) | yes `_set1` + op | yes `vdupq` + op | yes `svdup` + op | yes | yes scalar binop |
| `TileBinop` (Tile/Symbol) | yes inline literal | yes inline literal| yes inline literal | yes | yes inline literal |
| `TileUnop` all ops | yes | yes | yes | yes | yes |
| `TileITE` | yes `_mask_blend` | yes `vbslq_*` | yes `svsel_*` | yes | yes `ct.where` |
| `TileMaskGen` | yes -> `__mmask8` | yes -> bool vector | yes -> predicate reg | yes | yes -> bool tile |
| `TileReduce` (tile->scalar)| yes `_reduce_*` | yes `vaddvq_*` | yes `svaddv_*` | yes scalar loop | yes `ct.reduce` |
| `TileIota` | yes `_set_*` constant| yes constant init | yes `svindex_*` | yes | yes `ct.arange` |
| Runtime VL (`full_mask`) | n/a (fixed-length) | n/a | yes `svwhilelt` | n/a | n/a |

`scalar loop` = the lib node's `pure` expansion (correct on every host
but slower); used as the always-works fallback when no architectural
intrinsic exists. `n/a` = the feature is meaningless for that backend.

---

## Appendix E -- Open Questions (need a kernel to resolve)

These are explicitly flagged as undetermined. Each is currently
deferred behind a real kernel signal:

- **TileITE nested branches**. The branch-normalisation pipeline
  produces a single-level `merge(c, t, e)`. Nested merges (`merge(c0,
  merge(c1, t, e1), e0)`) compose naturally but the SVE expansion
  has not been measured for the predicate-stack cost. Lift when a
  kernel produces nested branches under the multi-dim path.
- **WCR on a strided destination**. `A[2*i + 1] +=` along a tiled dim
  fits section 3.5 syntactically but the per-arch scatter-with-accumulate
  intrinsic is missing on NEON and SVE. Decision: fall back to scalar
  postamble on those targets until a kernel demands the intrinsic
  path.
- **Mixed C / Fortran layout in a single binop**. section 2.3 permits it
  in principle (each operand carries its own `src_dims`) but the
  expansion has only been exercised on C-layout fixtures.
- **K > 3 tiles**. Constructors refuse `len(widths) > 3` (Appendix C);
  if a kernel demands K=4 (e.g. 4D conv), lift the constructor cap
  and add a K=4 row to Appendices A and C. The design machinery
  generalises mechanically.
- **Cross-tile reductions**. `TileReduce` is intra-tile only; the
  outer-Map's `wcr` carries inter-iteration accumulation. If a kernel
  wants a fused inter-tile reduction without going through the outer
  Map's WCR codegen, design a dedicated `TileGroupReduce` lib node.

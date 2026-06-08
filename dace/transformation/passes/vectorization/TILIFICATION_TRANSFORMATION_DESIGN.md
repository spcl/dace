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
- `TileGather` / `TileScatter` reads `arr.strides` to decode the
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
inlined-outer-state port both fought. The pre-pass `G1`
(`EnforceFullSubsetNSDFGBody`) guarantees the convention is
dataflow-safe before any lowering touches the body.

---

## 3. Inside The Body -- Tile Transients And The Scalar Exception

### 3.1 Staging rule

> Every non-transient read and write inside the tiled body is staged
> through a transient `Array(shape=widths, dtype=elem_dtype,
> storage=Register, transient=True)`.

The single exception: a **loop-invariant scalar broadcast** sourced
*before* the lane loop may stay at length 1 / `Scalar` shape. The lib
node that consumes it sees a `Scalar` operand (section 6.2). The broadcast is a
column splat, a kernel argument, or any value that is the same across
every lane of every dim.

### 3.2 Consequence

Each lib node sees only a full tile or a scalar. Partial-subset /
strided-lane complexity stays in the staging passes; the per-dim
lattice (section 4) is a closed surface.

### 3.3 Stage-global is mandatory for the multi-dim body

For `K >= 2` the body is always a NestedSDFG (section 2.4 invariant
4). `StageGlobalArrayThroughScalars` runs over the body and is
mandatory: no non-transient AccessNode may appear inside the body's
dataflow except as the producer / consumer of a staged scalar at the
body boundary.

Two-tier staging:

1. **Scalar bridge (default)**. Per `(producer, A, s1) x (consumer,
   A, s2)` pair, replace the global hop with a length-1 transient
   scalar (Case A / Case B in
   [STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md](STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md)).
   Section 3.1 widens the scalar to tile shape downstream.

2. **Full-tile fallback**. When the scalar bridge cannot fold a
   pair (subset mismatch, cross-state write breaks the linear-
   accumulation invariant), insert a `(K_0, ..., K_{K-1})` tile
   transient: `TileLoad` at body entry, in-body compute, `TileStore`
   at body exit. The global sits at the boundary only.

Refusal: if both tiers fail, the pass raises `NotImplementedError`
naming the array and state. Caller falls back to scalar code.

Pipeline order (K >= 2):
`EnforceFullSubsetNSDFGBody` (G1) -> `StageGlobalArrayThroughScalars`
(with G7 fallback) -> widening (`widen_in_map_nsdfg_inputs` /
`widen_body_scalars_to_tile`) -> `PromoteNSDFGBodyToTiles` (descent,
canonical lowering). The historical inlined-outer-state path was
dropped in tier S of [MULTI_DIM_CODE_AUDIT.md](MULTI_DIM_CODE_AUDIT.md).

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

### 5.1 No-GATHER case -- structured box

If `GATHER not in {d_0, ..., d_{K-1}}` the access lowers to a single
`TileLoad` / `TileStore` parameterised by:

- `dim_strides[d]`: per-dim stride coefficient.
  - `0` for CONSTANT,
  - `1` for LINEAR,
  - `c` for AFFINE,
  - `1` for REPLICATE (the replicate factor lives in its own property).
- `replicate_factor_per_dim[d]`: per-dim group-broadcast factor.
  - `k` for REPLICATE dims,
  - `1` otherwise.
- `src_dims[d]` / `dst_dims[d]`: which source-array dim each tile dim
  binds to (handles Fortran / transposed inputs).

The lowered tile always has shape `(K_0, ..., K_{K-1})`. Per-dim
replication / splatting happens *inside* the lib node's expansion, not
as a pre-pass.

### 5.2 Cross-dim broadcast resolution (sub-box -> full tile)

When some dims are narrower than "one element per lane" (CONSTANT,
REPLICATE, or AFFINE-without-iter-var), the lib node replicates them to
fill the tile:

| Per-dim kind | Per-dim source range | Per-dim lane-to-source map |
|--------------|----------------------|--------------------------------------|
| CONSTANT | 1 element | every lane reads the same element |
| LINEAR | K_d elements | lane `l` reads element `l` |
| AFFINE | K_d * s elements | lane `l` reads element `l*s` |
| REPLICATE | K_d / k elements | lane `l` reads element `floor(l/k)` |

The Cartesian product of these per-dim maps gives the K-dim load pattern
that the expansion emits. **No mid-pipeline reshape pass is needed**;
the lib node's expansion does the broadcast at codegen time.

### 5.3 GATHER case

If any dim is GATHER, the entire access lowers to `TileGather` /
`TileScatter` (section 9). Per-dim semantics survive: non-GATHER dims may still
be CONSTANT / LINEAR / AFFINE / REPLICATE and the gather node honours
their `dim_strides` / `replicate_factor` (i.e. those dims are addressed
affinely without a `_idx_<d>` connector).

### 5.4 Diagonal and transpose -- fall back to GATHER

- **Diagonal** (e.g. `a[i, i]` for K=2 tile vars `(i, j)`, or
  `a[i, i, i]` for K=3 `(i, j, k)`): two or more LINEAR dims sharing
  the *same* tile iter-var. **Lowers as GATHER**, per-dim form (section 9.2),
  with a 1-D `TileIota` index tile per dim.
- **Transpose** (e.g. `a[j, i]` for K=2, or `a[k, j, i]` for K=3): K
  LINEAR dims in a non-canonical permutation of the tile iter-vars.
  **Lowers as GATHER**, per-dim form with permuted `TileIota` index
  tiles.

Both shapes are recognisable structurally for any `K in {1, 2, 3}` but
the design explicitly falls back to GATHER for them -- the per-dim
gather form (section 9.2) with 1-D index tiles is cheap enough that a
dedicated structured load / store path is not justified for the kernel
corpus this pass targets. A future dedicated path can land later
behind the same lib-node surface without breaking the operand
contract.

---

## 6. Library Nodes

### 6.1 Node set (frozen for Tuesday)

| Node | Purpose |
|----------------|----------------------------------------------------|
| `TileLoad` | Structured tile load (section 5.1) |
| `TileStore` | Structured tile store |
| `TileGather` | Indirect / mixed tile load (section 9) |
| `TileScatter` | Indirect / mixed tile store |
| `TileBinop` | Elementwise binary op |
| `TileUnop` | Elementwise unary op |
| `TileMerge` | `where(mask, t, e)` |
| `TileReduce` | Cross-lane reduction (tile -> scalar only; section 10.3) |
| `TileMaskGen` | ANY-dim-OOB conjunction -> bool tile (section 7.4) |
| `TileIota` | `arange`-style index seed for gather/scatter |

### 6.2 Uniform operand contract

Every elementwise node (`TileBinop`, `TileUnop`, `TileMerge`) and
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

### 6.3 Symbolic broadcast

`Symbol` is not a degenerate `Scalar`. It is a *compile-time* constant
or symbolic expression embedded directly in the per-lane code string,
not materialised in any register. Use it for numeric literals
(`_out = _a + 1.0`) and outer-scope symbol broadcasts where no read
edge is needed.

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

### 7.5 Intra-tile (branch) masks

Branch normalisation produces `TileMerge` with an explicit condition
tile. The iteration mask (`_iter_mask`) and the condition mask combine
inside the expansion to a single effective mask. No separate
`TileMaskAnd` / `Or` / `Not` lib nodes are needed for the common case;
they remain out of scope until a kernel demands them (section 10.5).

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

## 9. Gather / Scatter Index Encoding

`TileGather` and `TileScatter` accept indices in three forms. Both the
full-tile form and the per-dim form are **first-class** -- neither is
preferred over the other; the classifier picks whichever matches the
source pattern (separable -> per-dim; cross-dim dependent -> full). All
three forms share the same lib node; the form is identified
unambiguously by the **connector names**, which are unique and
non-overlapping:

| Form | Connector(s) | Tile shape per connector |
|-------------------|-------------------------------------------|---------------------------|
| **Full N-D** | `_idx_full` | `(K_0, ..., K_{K-1})` |
| **Per-dim** | `_idx_0`, `_idx_1`, ..., `_idx_{K-1}` | `(K_d,)` each |
| **Partial** | a *subset* of `_idx_0`, ..., `_idx_{K-1}` | `(K_d,)` each |

`_idx_full` and `_idx_<d>` never co-occur; their names are disjoint so
the expansion identifies the form by inspecting which connectors are
wired (no additional discriminator needed).

### 9.1 Full N-D index tile (`_idx_full`)

One `(K_0, ..., K_{K-1})` integer tile. Per-lane:
`src[idx_full[l_0, ..., l_{K-1}]]` decoded against the source array's
flat-index layout. The classifier emits this form when the gather
indices are **not separable** across tile dims -- i.e. the index for
one dim depends on the lane in another dim (e.g. `src[idx[i, j]]`).

### 9.2 Per-dim index tiles (`_idx_0`, `_idx_1`, ...)

K integer tiles, each of shape `(K_d,)`. Per-lane:
`src[idx_0[l_0], idx_1[l_1], ..., idx_{K-1}[l_{K-1}]]`. The classifier
emits this form for **separable** patterns where each dim's index
depends only on that dim's lane index (diagonal, transpose, per-row
permutation, `src[idx0[i], idx1[j]]`). The 1-D per-dim tiles are much
cheaper to materialise than a full K-D index tile.

### 9.3 Partial index tiles

A *subset* of dims has a `_idx_<d>` connector wired; the rest fall
back to the affine default driven by `dim_strides[d]` /
`replicate_factor_per_dim[d]`. The classifier emits this form when a
multi-dim access has GATHER on a *subset* of dims; the non-GATHER dims
keep their per-dim kind and the GATHER dims get explicit indices. This
is structurally a special case of section 9.2 (per-dim with some indices
omitted), but the validation rule is looser: missing indices are
permitted iff the corresponding dim's per-dim kind is non-GATHER.

### 9.4 Wire-level rule

A `TileGather` constructor takes:

```python
TileGather(name, widths, src_ndim,
           gather_dims: Tuple[int, ...], # dims that carry an index (subset of [0..K))
           dim_strides, replicate_factor_per_dim, src_dims,
           index_form: Literal["full", "per_dim", "partial"],
           has_mask=False, pad_value=0)
```

- `index_form="full"`: connector `_idx_full` (shape
  `(K_0, ..., K_{K-1})`). `gather_dims == tuple(range(K))`. The connector
  name `_idx_full` is unique and does NOT clash with `_idx_<d>`, so the
  expansion identifies the form by name lookup.
- `index_form="per_dim"`: connectors `_idx_0, ..., _idx_{K-1}`,
  shape `(K_d,)` each. `gather_dims == tuple(range(K))`.
- `index_form="partial"`: connectors `_idx_<d>` only for `d in 
  gather_dims`. Other dims use `dim_strides` / `replicate_factor`.

**Identifiability invariant**: `_idx_full` and `_idx_<d>` are
**mutually exclusive** -- a `TileGather` with both wired is malformed
and rejected at `validate()`. The connector names are the source of
truth for the encoding form; `index_form` is a constructor-time
sanity flag and a downstream-pass hint.

The expansion handles all three with one switch on `index_form` at the
top of the codegen body; no separate node types.

---

## 10. Validation Rules

| Where | Checks |
|--------------|------------------------------------------------------------------------------|
| Constructor | `widths` length in `{1, 2, 3}`; operand kinds in `{Tile, Scalar, Symbol}`; `index_form` / `gather_dims` consistency; replicate factors divide widths. |
| `validate()` | All declared connectors are wired; tile-operand dtypes match the output dtype (uniform-dtype lock); mask present iff `has_mask`; **mask descriptor is `Array(shape=widths, dtype=bool_, storage=Register, transient=True)`** (section 7.1); `_idx_full` and `_idx_<d>` not both wired on the same node (section 9.4). |
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

`TileGather` / `TileScatter` index connectors (`_idx_full` /
`_idx_<d>`) must carry a **signed integer** descriptor:

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
| `_idx_full` and `_idx_<d>` both wired on the same gather node | `NotImplementedError` | `validate()` (section 9.4) |
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
| section 9.1 Single index tile | [tile_gather.py](../../../libraries/tileops/nodes/tile_gather.py): per-dim `_idx_<k>` connectors already exist for the K=2 / K=3 case (one connector per source-array dim). |

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

### G1 -- EnforceFullSubsetNSDFGBody

New standalone pass: for each tile-tagged Map whose body is a
NestedSDFG, widen every connector memlet to `array[0:N_0, ...,
0:N_{D-1}]`, assert inner body references the per-tile region via
`symbol_mapping`, refuse if the outer subset cannot be widened
safely. Unit tests on K=1 / K=2 / K=3 fixtures.

### G2 -- Vocabulary rename + add MODULAR

Mechanical: `BROADCAST -> CONSTANT`, `STRUCTURED_1 -> LINEAR`, keep
`REPLICATE` / `AFFINE` / `GATHER`, add `MODULAR`. Update
[utils/tile_access.py](utils/tile_access.py) +
[utils/tile_access_compat.py](utils/tile_access_compat.py). Drop the
redundant `TileAccessKind` enum; the whole-subset kind is the max
of per-dim kinds.

### G3 -- Gather index forms + identifiability

Extend `TileGather` / `TileScatter` constructors with
`gather_dims: Tuple[int, ...]` and `index_form: Literal["full",
"per_dim", "partial"]`. Validate that `_idx_full` and `_idx_<d>` are
mutually exclusive (section 9.4). Expansion switches on
`index_form`. Classifier emits `partial` when a subset has GATHER on
a strict subset of dims. Unit tests cover all three forms.

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

---

## 13. Delivery Plan

| Day | Goal |
|----------|--------------------------------------------------------------------------------------------|
| Mon | G1 (full-subset NSDFG body) + G2 (vocab rename + add MODULAR) + G7 stage-global multi-dim entry landed. |
| Tue AM | G3 (`_idx_full` + `_idx_<d>` + identifiability) + G4 (tile-dep symbol classifier) landed. |
| Tue PM | Design freeze meeting. Lock lib-node interfaces. |
| Wed | G5 (K=1/K=2/K=3 remainder audit) + G6 (symbol-coefficient distinction) + G7 full-tile fallback. |
| Thu | Velocity-tendencies stencils parse + lower end-to-end through the pipeline. |

---

## 14. Out Of Scope (Explicit Non-Goals)

These are deliberately excluded from this design; revisit only when a
concrete kernel demands them:

- **Mask combinators as lib nodes** (`TileMaskAnd` / `Or` / `Not`).
  Branch normalisation produces `TileMerge` which is sufficient for
  the current kernel corpus.
- **Cross-tile (across-Map-invocations) reductions**. `TileReduce`
  reduces within one tile; across-tile is a separate concern handled
  by the WCR / reduce-scope machinery in the rest of DaCe.
- **GPU `TileScatter` with cross-block conflicts**. Single-thread-per-
  lane block layouts only; cross-block atomics deferred.
- **Mixed dtypes inside a single op** (section 10.1). Materialise via an
  explicit cast `TileUnop` first.
- **K >= 4**. The constructors refuse `len(widths) > 3` at construction
  time. Lift only when a kernel justifies the register pressure.
- **Diagonal / transpose as dedicated structured patterns**. Both fold
  into GATHER for now (section 5.4); a structured fast path can land later
  behind the same `TileGather` surface.
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
| `a[idx[i]]`| GATHER | TileGather, `_idx_0` index tile |

### A.2 K = 2 (tile vars `(i, j)`)

| Access | Per-dim kinds (i, j) | Lowering |
|-----------------------|------------------------------|--------------------------------------------------|
| `a[i, j]` | (LINEAR, LINEAR) | TileLoad, `dim_strides=(1,1)` |
| `a[0, j]` | (CONSTANT, LINEAR) | TileLoad, broadcast dim 0, contiguous dim 1 |
| `a[i, 0]` | (LINEAR, CONSTANT) | TileLoad, contiguous dim 0, broadcast dim 1 |
| `a[i//2, j]` | (REPLICATE k=2, LINEAR) | TileLoad, dim 0 replicate factor 2 |
| `a[2*i + 1, j]` | (AFFINE s=2, LINEAR) | TileLoad, `dim_strides=(2,1)` |
| `a[i, i]` | (LINEAR i, LINEAR i) | TileGather diagonal, `index_form="per_dim"` |
| `a[j, i]` | (LINEAR j, LINEAR i) | TileGather transposed, `index_form="per_dim"` |
| `a[idx[i], j]` | (GATHER, LINEAR) | TileGather, `index_form="partial"`, gather_dims=(0,) |
| `a[idx0[i], idx1[j]]` | (GATHER, GATHER) | TileGather, `index_form="per_dim"` |
| `a[idx[i, j]]` | flat 1-D, idx is N-D | TileGather, `index_form="full"` |
| `a[2*sym + 1, j]` | (AFFINE / GATHER on sym, LIN)| AFFINE if `sym` tile-independent else GATHER (section 4.2) |
| `a[i % N, j]` | (MODULAR N, LINEAR) | TileGather (general MODULAR), `index_form="partial"` |
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
| `a[i, i, k]` | (LINEAR i, LINEAR i, LINEAR) | TileGather (diagonal on dims 0,1), `index_form="per_dim"` |
| `a[k, j, i]` | (LINEAR k, LINEAR j, LINEAR i) | TileGather transposed, `index_form="per_dim"` (permuted) |
| `a[idx[i], j, k]` | (GATHER, LINEAR, LINEAR) | TileGather, `index_form="partial"`, gather_dims=(0,) |
| `a[idx0[i], j, idx2[k]]` | (GATHER, LINEAR, GATHER) | TileGather, `index_form="partial"`, gather_dims=(0,2) |
| `a[idx0[i], idx1[j], idx2[k]]` | (GATHER, GATHER, GATHER) | TileGather, `index_form="per_dim"` |
| `a[idx[i, j, k]]` | flat 1-D, idx is K-D | TileGather, `index_form="full"` |
| `a[i % N, j, k]` | (MODULAR N, LINEAR, LINEAR) | TileGather (general MODULAR), `index_form="partial"` |
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
| `TileGather` full form | yes | yes | yes | `_idx_full` shape `(K_0, ..., K_{K-1})` |
| `TileGather` per-dim form | yes (1)| yes | yes | `_idx_0, ..., _idx_{K-1}`; K=1 collapses to single `_idx_0` |
| `TileGather` partial form | n/a | yes | yes | needs >=2 dims to have a partial subset |
| Mask `(K_0, ..., K_{K-1})` | yes | yes | yes | always full-tile shape |
| Remainder regions | 2 | 3 | 4 | interior + K boundaries (section 8.2) |
| `TileReduce` (tile -> scalar) | yes | yes | yes | axis-keep refused (section 10.3) |
| `TileMerge` | yes | yes | yes | shape-agnostic |
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
| `TileGather` (any form) | yes `_i64gather_pd` | scalar loop | yes `svld1_gather` | yes | yes `ct.gather` |
| `TileScatter` | yes `_i64scatter_pd` | scalar loop | yes `svst1_scatter` | yes | yes `ct.scatter` |
| `TileBinop` (Tile/Tile) | yes all `_OP_CPP` | yes | yes | yes | yes |
| `TileBinop` (Tile/Scalar) | yes `_set1` + op | yes `vdupq` + op | yes `svdup` + op | yes | yes scalar binop |
| `TileBinop` (Tile/Symbol) | yes inline literal | yes inline literal| yes inline literal | yes | yes inline literal |
| `TileUnop` all ops | yes | yes | yes | yes | yes |
| `TileMerge` | yes `_mask_blend` | yes `vbslq_*` | yes `svsel_*` | yes | yes `ct.where` |
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

- **TileMerge nested branches**. The branch-normalisation pipeline
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

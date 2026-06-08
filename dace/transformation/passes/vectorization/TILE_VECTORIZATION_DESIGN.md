# Tile Vectorization — Design Specification

**Status**: draft for Tuesday freeze. Sections 2-9 are the contract; Section 10
is the implementation audit and migration list.

## 1. Scope

Lower a K-dim register tile (K ∈ {1, 2, 3}) onto AVX-512, NEON, SVE, scalar,
and cuTile back ends. The pipeline that produces this representation is
otherwise out of scope; this document defines:

- the structural shape of the tiled body (Section 2),
- the data shape inside the body (Section 3),
- per-dimension access classification (Section 4),
- cross-dimension composition rules (Section 5),
- the library-node surface (Section 6),
- masking (Section 7),
- remainder loops (Section 8),
- gather/scatter index encoding (Section 9).

## 2. Tiled Body — NestedSDFG With Full-Array Subsets

The tiled body is a `NestedSDFG`. The outer Map iterates per-tile (step `widths`
in each tiled dim); the body is invoked once per tile.

**Call convention (load-bearing)**:

- Every input/output connector receives the **full array subset**.
- No pointer offsetting is performed at the call site.
- The body's inner memlets reference the full array; per-tile addressing is
  derived from the iter-vars passed through `NestedSDFG.symbol_mapping`.

Rationale: full-array subsets remove the only source of subset confusion that
the descent and the inlined-outer-state ports were both fighting (the
"connector array shape mismatch" family). A separate cleanup pass guarantees
that the resulting call convention is dataflow-safe (alias-free, no
overlapping writes the codegen would lose); see Section 10.

## 3. Data Shape Inside The Body — Tile Transients

Inside the tiled body **every non-transient read and write is staged through a
transient `Array`** of tile shape, with the following exceptions:

- A **column or scalar broadcast** sourced before the lane loop (a loop-
  invariant load) may remain at length 1 / scalar shape. The lib node consuming
  it sees a `Scalar` operand (Section 6.2).

All other transients are `Array(shape=widths, dtype=elem_dtype,
storage=Register, transient=True)`.

**Why**: every library node then sees either a *full tile* or a *scalar*
operand. No node has to reason about partial-tile or strided-into-tile
operands; that complexity moves to the staging passes that produce the
operand.

## 4. Per-Dimension Access Lattice

Each tile dimension's access is one of:

| Kind         | Definition                                                              | Example                                     |
|--------------|-------------------------------------------------------------------------|---------------------------------------------|
| **CONSTANT** | dim's index is loop-invariant (no tile iter-var)                        | `a[0]`, `a[N+1]`                            |
| **LINEAR**   | dim's index is exactly `iter_var + c` (stride 1, offset constant)       | `a[i]`, `a[i + 3]`                          |
| **AFFINE**   | dim's index is `s * iter_var + c` with `s` constant int                 | `a[2*i + 1]`                                |
| **REPLICATE**| dim's index is `floor(iter_var, k) + c` (1 < k < W, no other tile vars) | `a[i//2]`, `a[i//4 + 7]`                    |
| **GATHER**   | none of the above                                                       | `a[idx[i]]`, `a[i*j]`, `a[2*sym + 1]` (sym loop-variant) |

The lattice order on a single dim is

```
CONSTANT  ⊑  LINEAR  ⊑  AFFINE  ⊑  GATHER
                       ⊓
                  REPLICATE
```

`REPLICATE` is incomparable with LINEAR / AFFINE — it is a separate axis (group
broadcast within the dim with factor `k`); on the lattice it joins with any of
them to GATHER.

**Join rule (per-dim, for symbolic / non-constant coefficients)**:

> If a coefficient or offset is *itself* loop-variant — e.g. `2*sym + 1` where
> `sym` depends on a tile iter-var — the dim's classification rises to
> GATHER. Constants and pure outer-scope symbols are CONSTANT.

This mirrors the type-lattice fallback used for FP precision: when in doubt,
join up to GATHER.

## 5. Cross-Dimension Composition

A tile access is a tuple `(d_0, ..., d_{K-1})` of per-dim kinds. Composition
rules:

### 5.1 No-GATHER case

If `GATHER ∉ {d_0, ..., d_{K-1}}` the access is a **structured box** and
lowers to a single `TileLoad` / `TileStore` parameterised by:

- per-dim `dim_strides`: `0` for CONSTANT, `1` for LINEAR, `c` for `AFFINE`,
- per-dim `replicate_factor`: `k` for REPLICATE dims, `1` otherwise,
- `src_dims` / `dst_dims` permutation: the source-array dim each tile dim
  binds to (handles Fortran / transposed inputs).

**Sub-box expansion to full (K_0, K_1, ..., K_{K-1})**: dims whose kind is
narrower than "one element per lane" replicate to fill the tile:

- CONSTANT  →  splat the single value across all lanes of that dim,
- REPLICATE →  group-broadcast (`k` lanes share each loaded element),
- LINEAR    →  contiguous load,
- AFFINE    →  strided load with constant stride.

The lowered tile is always `(K_0, K_1, ..., K_{K-1})` — the per-dim
replication / broadcast happens inside the lib node, not as a separate
pre-pass.

### 5.2 GATHER case

If any dim is GATHER, the entire access is a `TileGather` / `TileScatter`. Per-
dim semantics survive: non-GATHER dims may still be CONSTANT / LINEAR / AFFINE
/ REPLICATE, encoded as `dim_strides` + `replicate_factor` + missing index for
that dim (Section 9).

### 5.3 Special-case shapes (recognised but lowered as GATHER for now)

- **Diagonal** (`a[i, i]` for K=2 tile vars `i, j`) — a LINEAR dim using a
  *non-canonical* iter-var. Classified as GATHER with a 1-D index tile.
- **Transpose** (`a[j, i]` where the canonical lane order is `(i, j)`) — same,
  with a permuted index tile.

These could be lowered to dedicated structured patterns later; the design
admits them as GATHER for now so a single index-array path covers them.

## 6. Library Nodes

### 6.1 Node set

| Node                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `TileLoad`            | Structured tile load (Section 5.1)                       |
| `TileStore`           | Structured tile store                                    |
| `TileGather`          | Indirect / mixed tile load (Section 5.2)                 |
| `TileScatter`         | Indirect / mixed tile store                              |
| `TileBinop`           | Elementwise binary op, Tile/Scalar/Symbol operands       |
| `TileUnop`            | Elementwise unary op                                     |
| `TileMerge`           | `where(mask, t, e)`                                      |
| `TileReduce`          | Cross-lane reduction                                     |
| `TileMaskGen`         | ANY-dim-OOB conjunction → `(K_0, ..., K_{K-1})` bool tile|
| `TileIota`            | `arange`-style index seed for gather/scatter             |

### 6.2 Operand kinds — uniform contract

Every elementwise node (`TileBinop`, `TileUnop`, `TileMerge`) accepts operands
classified as one of:

- **Tile** — a `(K_0, ..., K_{K-1})` transient AccessNode (wired through
  `_a` / `_b`).
- **Scalar** — a length-1 / `dace.data.Scalar` AccessNode; hardware
  scalar-to-tile broadcast (wired through `_a` / `_b`).
- **Symbol** — a symbolic expression embedded inline (no connector).

Per-arch expansions are responsible for emitting the hardware splat for Scalar
operands (e.g. AVX-512 `_mm512_set1_pd`). Scalar is a first-class kind, not
syntactic sugar for "Tile from a length-1 source."

### 6.3 Symbolic broadcast

Every node that takes operands accepts symbolic constants (`expr_a` / `expr_b`
properties on `TileBinop` / `TileUnop`). The expansion embeds the value inline
per lane; no per-lane materialisation in registers is required.

## 7. Masking

### 7.1 Uniform mask shape

Every tile node accepts an optional `_mask` input. The mask is **always shape
`(K_0, ..., K_{K-1})`**, even when only one dim needs predication. The
single-dim case lowers to a broadcast comparison inside `TileMaskGen`; the
rest of the pipeline sees a uniform K-dim mask. Two reasons:

- intrinsics (AVX-512 `__mmask8`, SVE predicate registers) only have a single
  K-dim predicate concept; a 1-D mask would need re-broadcast at every use,
- the lib-node `has_mask` toggle stays a single bool rather than per-dim.

### 7.2 has_mask toggle

Each tile node has a `has_mask: bool` constructor knob. When False the `_mask`
input connector is omitted. When True the expansion threads the predicate
through every load / store / op.

### 7.3 No-mask short-circuit

The main map of a `MASKED_TAIL` split has `has_mask=False` on every node — the
provably-divisible interior is the perf fast path. Only the remainder map
carries masks.

## 8. Remainder Loops

### 8.1 One remainder per tiled dim

For each tiled dim, the pipeline emits one remainder map. Untiled dims pass
through at their full extent in every remainder.

### 8.2 Subdivision rule (avoiding 3+ remainders for K=2)

For `K=2` with tiled dims `(i, j)`, naive cartesian splitting produces 4
regions: `(main_i, main_j)`, `(main_i, tail_j)`, `(tail_i, main_j)`,
`(tail_i, tail_j)`. To reduce this to **3 regions** (one per tiled dim, plus
the main), apply the corner-absorbing rule:

```
region 0:  (main_i,  main_j)        — interior, has_mask=False
region 1:  (FULL i,  tail_j)        — j remainder, FULL i absorbs the corner
region 2:  (tail_i,  main_j)        — i remainder, no overlap with region 1
```

i.e. the j-remainder map covers the *full* i extent (not just the i-main
range) so the `(tail_i, tail_j)` corner is handled there. The i-remainder
covers only the i-tail × j-main rectangle. Total: 1 interior + 2 boundary =
3 regions. Same shape generalises to K=3 with 1 interior + 3 boundary.

### 8.3 Mask threading on remainders

Each boundary region has `has_mask=True`; its `TileMaskGen` produces the
conjunction `((i + l_i) < ub_i) ∧ ((j + l_j) < ub_j) ∧ ...`. The same lib node
is used regardless of which dim is the remainder one — the absorbed-full dims
just contribute `True` to the conjunction.

### 8.4 Remainder strategies (existing knob, unchanged)

- `MASKED_TAIL` — Section 8.2 + 8.3 (default for AVX-512 CPU).
- `SCALAR_POSTAMBLE` — boundary regions left as `Sequential` scalar loops,
  no masks. Used for kernels where the masked variant is too expensive (rare
  on modern CPUs).
- `ALWAYS_ITER_MASK` — single map, no split, `has_mask=True` everywhere. SVE
  default (runtime VL).

## 9. Gather / Scatter Index Encoding

`TileGather` and `TileScatter` accept indices in three forms; partial forms
fill missing dims with the affine-default reading from `dim_strides` /
`replicate_factor`:

### 9.1 Full index tile

One `(K_0, K_1, ..., K_{K-1})` integer tile. Per-lane reads
`src[idx[l_0, l_1, ...]]` with the flat-index convention given by the source
array strides.

### 9.2 Per-dim index tiles

K integer tiles, each of length `K_d`. Per-lane reads
`src[idx_0[l_0], idx_1[l_1], ..., idx_{K-1}[l_{K-1}]]`. Cheaper than 9.1 when
each dim's indices are independent (the diagonal/transpose cases lower to this
form: a `TileIota` of length `K_d` per dim).

### 9.3 Partial index tiles

A *subset* of dims has an explicit index; the remaining dims fall back to the
affine default (i.e. behave as their per-dim kind from Section 4 dictates,
typically LINEAR or REPLICATE). The classifier emits this form when a
multi-dim access has GATHER on a subset of dims; the non-GATHER dims keep
their classification and the GATHER dims get explicit indices.

Wire-level: per-dim index tiles attach through `_idx_<k>` connectors. A
missing `_idx_<k>` means the kth tile dim uses its `dim_strides[k]` /
`replicate_factor_per_dim[k]`.

## 10. Implementation Audit

This section maps the design to the current codebase and lists the gaps to
close before Tuesday freeze.

### 10.1 What already aligns

| Design                                  | Existing code                                                           |
|-----------------------------------------|-------------------------------------------------------------------------|
| Per-dim access kinds (Section 4)        | `utils/tile_access.py::PerDimKind`: `BROADCAST`, `STRUCTURED_1`, `REPLICATE`, `AFFINE`, `GATHER` |
| Tile-shape transients (Section 3)       | `PromoteInlinedMapToTiles._widen_body_scalars` (slice 1)                |
| Uniform operand kinds (Section 6.2)     | `TileBinop` / `TileUnop` `kind_a` / `kind_b` ∈ `{Tile, Scalar, Symbol}` |
| `(K_0, ..., K_{K-1})` mask (Section 7.1)| `TileMaskGen` emits ANY-dim-OOB conjunction at tile shape               |
| `has_mask` toggle (Section 7.2)         | every lib node, optional `_mask` connector                              |
| Main + remainder split (Section 8.4)    | `SplitMapForTileRemainder` (MASKED_TAIL, SCALAR_POSTAMBLE)              |
| `replicate_factor_per_dim` (Section 5.1)| `TileLoad.replicate_factor_per_dim` (committed earlier this branch)     |
| Array<->scalar copy → load/store (Section 3) | `RewriteArrayScalarToTileOp`                                       |

### 10.2 Vocabulary alignment needed

The design uses **CONSTANT / LINEAR / AFFINE / REPLICATE / GATHER**. The
existing `PerDimKind` uses **BROADCAST / STRUCTURED_1 / REPLICATE / AFFINE /
GATHER**. Mapping:

| Spec name | Current enum   |
|-----------|----------------|
| CONSTANT  | `BROADCAST`    |
| LINEAR    | `STRUCTURED_1` |
| AFFINE    | `AFFINE`       |
| REPLICATE | `REPLICATE`    |
| GATHER    | `GATHER`       |

**Action**: rename the enum members. Spec names are clearer; current names
date from when `BROADCAST` was overloaded for the symbol-fanout case. Single
commit, mechanical.

### 10.3 Gaps to close before Tuesday

**G1 — NSDFG-of-tiled-body call convention.** The descent
(`PromoteNSDFGBodyToTiles`) currently widens connectors at a per-tile subset;
the design requires full-array subsets with no offsetting. Need a pre-pass
that (a) ensures the body NSDFG is invoked with `array_full[:]` on every
connector, (b) carries the offsetting into the body's inner memlets via
`symbol_mapping`.

*Existing pieces*: `expand_nested_sdfg_inputs.py` already widens connector
subsets; `insert_body_nsdfg_copies.py` inserts copy-in/copy-out states.
Combine into one explicit "FullSubsetCallConvention" pass.

**G2 — Generic "auto-classify" load node.** The design proposes a load node
that detects the access pattern at expansion time from its subset + connectors.
Currently the classifier runs at pipeline-time and lowers to one of `TileLoad`
/ `TileGather`. Either:

- (a) keep the current pipeline-time decision and document the classifier as
  the canonical entry point; or
- (b) add a `TileAutoLoad` umbrella node that defers classification to its
  expansion.

Recommendation: **(a)**. The classifier already covers the cases the
description lists. The umbrella node would duplicate the lattice
implementation. Mark `(b)` as not pursued.

**G3 — Per-dim partial gather indices (Section 9.3).** `TileGather` /
`TileScatter` currently take a single index tile. Need:

- `_idx_<k>` connectors (one per tile dim),
- per-dim shape `(K_d,)` integer tiles,
- expansion: missing `_idx_<k>` → fall back to
  `dim_strides[k]` / `replicate_factor_per_dim[k]` affine indexing.

This is the main lib-node interface change before Tuesday freeze.

**G4 — Lattice join when a coefficient is loop-variant.** The classifier
currently inspects symbolic expressions but does not formally implement the
"if coefficient is loop-variant → GATHER" rule (Section 4 join rule).
Implementation: when classifying a dim, walk the begin expression's free
symbols; if any is in the tile iter-var set OR depends on one transitively
(via interstate-edge assignments in the enclosing body), classify as GATHER.

**G5 — 3-region remainder split for K=2.** `SplitMapForTileRemainder` today
emits 1 interior + N (potentially 2^K - 1) boundary maps; the design pins this
to **K boundary maps** via the corner-absorbing rule (Section 8.2). Audit
SplitMapForTileRemainder against this rule; tighten if needed.

**G6 — Symbol-as-coefficient handling.** `a[N*i]` where `N` is an outer-scope
constant symbol is AFFINE with symbolic stride; `a[N*i]` where `N` is itself a
function of `i` is GATHER (join rule). The classifier needs to distinguish.
Current behavior: untested; surfaces as a corner case during velocity-
tendencies stencil parsing.

### 10.4 Out-of-scope / deferred

- Cross-tile reductions across map invocations (`TileReduce` already
  supports intra-tile; cross-tile is a separate concern).
- Mask combinators (`TileMaskAnd` / `Or` / `Not`) for branch-normalised
  if-else. Deferred to a follow-up; current branch path produces `TileMerge`
  which is sufficient.
- GPU `TileScatter` with arbitrary cross-block conflicts. Single-thread-per-
  lane block layouts only.
- Velocity-tendencies stencil parsing — that is the Thursday deliverable,
  not a design constraint.

## 11. Delivery Plan (informational)

| Day      | Goal                                                                        |
|----------|-----------------------------------------------------------------------------|
| Tuesday  | G1 + G3 + G4 + G5 landed. Lib-node interfaces frozen. Vocabulary renamed.   |
| Wed      | Baseline broadcast / gather / scatter end-to-end on hand-written K=2 kernel.|
| Thu      | Velocity-tendencies stencils parse + lower through the pipeline.            |

Sections 2-9 are the contract; Section 10 is the punchlist; Section 11 is the
schedule the contract serves.

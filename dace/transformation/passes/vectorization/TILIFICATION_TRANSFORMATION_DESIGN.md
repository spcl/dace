# Tilification Transformation — Design Specification

**Status**: Draft for Tuesday freeze.
**Scope**: Lower a K-dim register tile (K ∈ {1, 2, 3}) onto AVX-512 / NEON /
SVE / scalar / cuTile back ends.
**Layout**: §§ 2-10 are the contract; §11 audits the current codebase
against it; §12 lists the gaps and the migration order; §13 holds the
schedule.

---

## 1. Goals

1. **One canonical body shape**: the tiled loop body is a NestedSDFG with
   full-array subsets on every connector. No pointer offsetting at the
   call site.
2. **One operand contract per lib node**: each library-node input is either
   a *full tile* (shape `(K_0, …, K_{K-1})`) or a *scalar* (length-1 / true
   `Scalar`). Nothing in between.
3. **One classification surface**: a per-dimension access lattice
   (§4) plus a single composition rule (§5). All higher-level emission
   decisions read from the lattice; the lattice falls back to GATHER when
   it cannot decide (§4.2).
4. **One mask shape**: every lib node accepts an optional `(K_0, …,
   K_{K-1})` mask; per-dim masking is uniformly expressed by broadcasting
   into the K-dim mask (§7).
5. **One remainder rule**: K boundary regions (not 2^K - 1), produced by
   the corner-absorbing peel (§8).
6. **Three gather forms** (full / per-dim / partial) on the same node,
   with affine fallback on omitted dims (§9).

---

## 2. Tiled Body — NestedSDFG With Full-Array Subsets

The outer Map iterates per-tile in each tiled dim (step `W_d` per dim).
Its body is a single `NestedSDFG` invocation per iteration. Contract:

| Connector kind | Subset                     | Inner reference                   |
|----------------|----------------------------|-----------------------------------|
| Input          | full array `A[0:N_0, …]`   | `A_sub[i_0:i_0+W_0, …]`           |
| Output         | full array `A[0:N_0, …]`   | `A_sub[i_0:i_0+W_0, …]`           |
| Symbol         | passed via `symbol_mapping`| `i_d`, `j_d`, …                   |

**Invariants**:

- The call-site memlet is `A[0:N_0, …]` — the full extent on every dim.
- Per-tile addressing happens *inside* the body via `symbol_mapping`
  (the outer iter-vars enter as inner symbols).
- The body is responsible for staging its own reads/writes through
  transients (§3); the call convention itself never offsets.

**Rationale**: full-array subsets remove the only source of subset
confusion the descent and the inlined-outer-state port both fought (the
"connector array shape mismatch" family). A dedicated cleanup pass (G1
in §12) guarantees the convention is dataflow-safe.

---

## 3. Inside The Body — Tile Transients And The Scalar Exception

### 3.1 Staging rule

> Every non-transient read and write inside the tiled body is staged
> through a transient `Array(shape=widths, dtype=elem_dtype,
> storage=Register, transient=True)`.

The single exception: a **loop-invariant scalar broadcast** sourced
*before* the lane loop may stay at length 1 / `Scalar` shape. The lib
node that consumes it sees a `Scalar` operand (§6.2). The broadcast is a
column splat, a kernel argument, or any value that is the same across
every lane of every dim.

### 3.2 Why staging is load-bearing

Each lib node then sees either a full tile or a scalar — never a
partial-tile or strided-into-tile shape. All the "partial subset" /
"strided lane" complexity moves *out* of the lib nodes and *into* the
staging passes that produce the operand. That separation is what makes
the per-dim lattice (§4) implementable as a closed surface.

---

## 4. Per-Dimension Access Lattice

### 4.1 Kinds

Each tile dimension's access is classified as one of:

| Kind         | Definition                                                                  | Example                                  |
|--------------|-----------------------------------------------------------------------------|------------------------------------------|
| **CONSTANT** | dim's index is loop-invariant (no tile iter-var anywhere in the expression) | `a[0]`, `a[N+1]`                         |
| **LINEAR**   | dim's index is exactly `iter_var + c` (stride 1, constant offset)           | `a[i]`, `a[i + 3]`                       |
| **AFFINE**   | dim's index is `s · iter_var + c` with `s`, `c` outer-scope constants       | `a[2*i + 1]`                             |
| **REPLICATE**| dim's index is `floor(c · iter_var + c0, k)` or `ceil(…)` (1 < k < W)       | `a[i//2]`, `a[i//4 + 7]`                 |
| **MODULAR**  | dim's index is `(c · iter_var + c0) % N` with `N` an outer-scope constant   | `a[i % N]`, `a[(2*i + 1) % N]`           |
| **GATHER**   | none of the above                                                           | `a[idx[i]]`, `a[i*j]`, `a[2*sym + 1]` (sym loop-variant) |

### 4.2 Lattice order and join rule

```
CONSTANT  ⊑  LINEAR  ⊑  AFFINE  ⊑  GATHER
                     ⊓        ⊓
                REPLICATE   MODULAR
```

- `CONSTANT ⊑ LINEAR ⊑ AFFINE ⊑ GATHER` is a strict chain.
- `REPLICATE` is **incomparable** with LINEAR / AFFINE — it is a separate
  axis (group broadcast inside the dim with factor `k`). The join of
  REPLICATE with anything outside `{CONSTANT, REPLICATE}` is GATHER.
- `MODULAR` is **incomparable** with LINEAR / AFFINE — it is a separate
  axis (cyclic wrap with period `N`). The join of MODULAR with anything
  outside `{CONSTANT, MODULAR}` is GATHER.

**Tile-aligned MODULAR → LINEAR reduction**:

For `a[(c · i_d + c_0) % N]`, if the classifier can prove `N` divides
`c · W_d` (the per-tile stride covers an integer number of wrap
periods), the access reduces to **LINEAR with a per-tile constant
offset**: every tile starts at `(c · i_d + c_0) mod N` (loop-invariant
within the tile) and lane `l_d` reads element `base + l_d`. The
classifier should emit LINEAR in this case; only the generic MODULAR
falls back to GATHER.

The common K=2 instance that triggers this reduction: `a[i % W_d]`
where the outer Map step is `W_d` ⇒ tile start ≡ 0 (mod `W_d`) ⇒
LINEAR with offset 0.

**Join rule (tile-dependent symbol → GATHER)**:

> A symbol is **tile-dependent** iff materialising its value differs
> across lanes — equivalently, the symbol (or any symbol it transitively
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
symbol tile-dependent or outer-scope?* — using the same dependency walk
the codegen would use to decide whether a per-lane materialisation is
required. Outer-scope symbols and numeric literals are tile-independent;
tile iter-vars and any symbol whose definition transitively touches one
are tile-dependent.

Examples (with `i` a tile iter-var, `N` an outer-scope constant symbol,
`sym` an interstate-edge-assigned symbol inside the body):

| Expression       | Symbol analysis                          | Kind              |
|------------------|------------------------------------------|-------------------|
| `a[i]`           | only `i` (tile iter-var)                 | LINEAR            |
| `a[i + 3]`       | only `i`                                 | LINEAR            |
| `a[N + 1]`       | `N` outer-scope                          | CONSTANT          |
| `a[2*i + 1]`     | only `i`                                 | AFFINE            |
| `a[N*i + 1]`     | `N` outer-scope, `i` tile iter-var       | AFFINE (stride `N`) |
| `a[2*sym + 1]` (sym ← `i + 3`) | `sym` tile-dependent               | **GATHER**        |
| `a[2*sym + 1]` (sym ← outer)   | `sym` outer-scope                  | AFFINE            |
| `a[i*j]`         | both `i`, `j` tile-dependent             | **GATHER**        |
| `a[idx[i]]`      | `idx[i]` is a data-dependent read        | **GATHER**        |

This mirrors the FP-precision-type fallback: when in doubt, join up to
the strictest classification (GATHER) so correctness is preserved at
the cost of a slower path.

---

## 5. Cross-Dimension Composition

A tile access is a tuple `(d_0, …, d_{K-1})` of per-dim kinds. The
composition decides which lib node lowers the access and how its
properties are populated.

### 5.1 No-GATHER case — structured box

If `GATHER ∉ {d_0, …, d_{K-1}}` the access lowers to a single
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

The lowered tile always has shape `(K_0, …, K_{K-1})`. Per-dim
replication / splatting happens *inside* the lib node's expansion, not
as a pre-pass.

### 5.2 Cross-dim broadcast resolution (sub-box → full tile)

When some dims are narrower than "one element per lane" (CONSTANT,
REPLICATE, or AFFINE-without-iter-var), the lib node replicates them to
fill the tile:

| Per-dim kind | Per-dim source range | Per-dim lane-to-source map           |
|--------------|----------------------|--------------------------------------|
| CONSTANT     | 1 element            | every lane reads the same element    |
| LINEAR       | K_d elements         | lane `l` reads element `l`           |
| AFFINE       | K_d · s elements     | lane `l` reads element `l·s`         |
| REPLICATE    | K_d / k elements     | lane `l` reads element `floor(l/k)`  |

The Cartesian product of these per-dim maps gives the K-dim load pattern
that the expansion emits. **No mid-pipeline reshape pass is needed**;
the lib node's expansion does the broadcast at codegen time.

### 5.3 GATHER case

If any dim is GATHER, the entire access lowers to `TileGather` /
`TileScatter` (§9). Per-dim semantics survive: non-GATHER dims may still
be CONSTANT / LINEAR / AFFINE / REPLICATE and the gather node honours
their `dim_strides` / `replicate_factor` (i.e. those dims are addressed
affinely without a `_idx_<d>` connector).

### 5.4 Diagonal and transpose — fall back to GATHER

- **Diagonal** (e.g. `a[i, i]` for K=2 tile vars `(i, j)`, or
  `a[i, i, i]` for K=3 `(i, j, k)`): two or more LINEAR dims sharing
  the *same* tile iter-var. **Lowers as GATHER**, per-dim form (§9.2),
  with a 1-D `TileIota` index tile per dim.
- **Transpose** (e.g. `a[j, i]` for K=2, or `a[k, j, i]` for K=3): K
  LINEAR dims in a non-canonical permutation of the tile iter-vars.
  **Lowers as GATHER**, per-dim form with permuted `TileIota` index
  tiles.

Both shapes are recognisable structurally for any `K ∈ {1, 2, 3}` but
the design explicitly falls back to GATHER for them — the per-dim
gather form (§9.2) with 1-D index tiles is cheap enough that a
dedicated structured load / store path is not justified for the kernel
corpus this pass targets. A future dedicated path can land later
behind the same lib-node surface without breaking the operand
contract.

---

## 6. Library Nodes

### 6.1 Node set (frozen for Tuesday)

| Node           | Purpose                                            |
|----------------|----------------------------------------------------|
| `TileLoad`     | Structured tile load (§5.1)                        |
| `TileStore`    | Structured tile store                              |
| `TileGather`   | Indirect / mixed tile load (§9)                    |
| `TileScatter`  | Indirect / mixed tile store                        |
| `TileBinop`    | Elementwise binary op                              |
| `TileUnop`     | Elementwise unary op                               |
| `TileMerge`    | `where(mask, t, e)`                                |
| `TileReduce`   | Cross-lane reduction (tile → scalar only; §10.3)   |
| `TileMaskGen`  | ANY-dim-OOB conjunction → bool tile (§7.4)         |
| `TileIota`     | `arange`-style index seed for gather/scatter       |

### 6.2 Uniform operand contract

Every elementwise node (`TileBinop`, `TileUnop`, `TileMerge`) and
every store-style node accepts each operand as one of:

| Kind       | Connector         | Source                                                 | Lowering                                       |
|------------|-------------------|--------------------------------------------------------|------------------------------------------------|
| **Tile**   | `_a` / `_b` …     | tile-shaped `Array` `AccessNode`                       | per-lane register read                         |
| **Scalar** | `_a` / `_b` …     | length-1 / `dace.data.Scalar` `AccessNode`             | hardware splat (`_mm512_set1_pd`, `svdup_f64`) |
| **Symbol** | *no connector*    | symbolic expression in `expr_a` / `expr_b` property    | inline literal embedded in the per-lane body   |

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

### 7.1 Mask shape lock — full-tile boolean Array

The mask is **only ever a full-tile boolean Array**. Specifically:

- shape exactly `(K_0, …, K_{K-1})` — same shape as the lib node's tile
  operands;
- dtype `dace.bool_`;
- storage `Register`, `transient=True`.

No other mask form is accepted. Per-dim masks, scalar masks, or
non-bool predicates are all rejected at constructor / validate time
(§10). When only a single dim needs predication, the mask is still
generated at full `(K_0, …, K_{K-1})` shape — the unused dims contribute
the constant `true` to the conjunction and fold away at expansion
(§7.4).

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
every node — the provably-divisible interior is the performance fast
path. Only the boundary regions carry masks.

### 7.4 `TileMaskGen` contract

For tile vars `(i_0, …, i_{K-1})` with widths `(W_0, …, W_{K-1})` and
global upper bounds `(N_0, …, N_{K-1})`, `TileMaskGen` emits the
ANY-dim-OOB conjunction:

```
mask[l_0, …, l_{K-1}] = (i_0 + l_0 < N_0) ∧ … ∧ (i_{K-1} + l_{K-1} < N_{K-1})
```

When a particular dim doesn't need masking (the corner-absorbing peel
of §8 covers it at full extent), its conjunct is the constant `true`
and folds away at expansion. The same lib node is used regardless of
which dim is being predicated.

### 7.5 Intra-tile (branch) masks

Branch normalisation produces `TileMerge` with an explicit condition
tile. The iteration mask (`_iter_mask`) and the condition mask combine
inside the expansion to a single effective mask. No separate
`TileMaskAnd` / `Or` / `Not` lib nodes are needed for the common case;
they remain out of scope until a kernel demands them (§10.5).

---

## 8. Remainder Loops

### 8.1 One remainder map per tiled dim

For K tiled dims the pipeline emits exactly **K + 1 regions**:

- 1 interior region (all tiled dims at main range, `has_mask=False`),
- K boundary regions (one per tiled dim, `has_mask=True`).

Untiled dims pass through at their full extent in every region.

### 8.2 Corner-absorbing peel (avoids 2^K - 1 corners)

**Algorithm (any K)**: pick an ordering `(d_{K-1}, d_{K-2}, …, d_0)`
of the tiled dims — innermost-first by convention. For each tiled dim
`d_p` (in order), emit a boundary region with:

- `d_p` at the **tail** range `[⌊N_{d_p} / W_{d_p}⌋ · W_{d_p}, N_{d_p})`;
- **higher-priority dims** `d_q` with `q > p` (not yet peeled) at the
  **full** range `[0, N_{d_q})`;
- **lower-priority dims** `d_q` with `q < p` (already peeled) at the
  **main** range `[0, ⌊N_{d_q} / W_{d_q}⌋ · W_{d_q})`;
- untiled dims at their full extent in every region.

Each boundary region covers its own tail-strip and absorbs only those
corners that the not-yet-peeled dims still own. The lower-priority
"main" range guarantees disjointness with the regions that come later.

Define
- `M_d = ⌊N_d / W_d⌋ · W_d` — the last main-tile boundary on dim `d`,
- `T_d = [M_d, N_d)` — the tail strip on dim `d`,
- `F_d = [0, N_d)` — the full range on dim `d`.

**K=2** (tiled dims `(i, j)` with `(i = d_1, j = d_0)`, innermost-first
peels `j` then `i`):

```
region 0 (interior):    i ∈ [0, M_i),  j ∈ [0, M_j)
region 1 (j boundary):  i ∈ F_i,       j ∈ T_j
region 2 (i boundary):  i ∈ T_i,       j ∈ [0, M_j)
```

The `(tail_i, tail_j)` corner is absorbed by region 1 (full i over j's
tail). Region 2 takes only `(tail_i, main_j)` — no overlap.

**K=3** (tiled dims `(i, j, k)` with `(i = d_2, j = d_1, k = d_0)`,
innermost-first peels `k`, then `j`, then `i`):

```
region 0 (interior):    i ∈ [0, M_i),  j ∈ [0, M_j),  k ∈ [0, M_k)
region 1 (k boundary):  i ∈ F_i,       j ∈ F_j,       k ∈ T_k
region 2 (j boundary):  i ∈ F_i,       j ∈ T_j,       k ∈ [0, M_k)
region 3 (i boundary):  i ∈ T_i,       j ∈ [0, M_j),  k ∈ [0, M_k)
```

Disjointness:
- region 1 vs all others — disjoint on `k` (only region 1 has `k ∈ T_k`);
- region 2 vs region 0 — disjoint on `j` (region 2 has `j ∈ T_j`,
  region 0 has `j ∈ [0, M_j)`);
- region 2 vs region 3 — disjoint on `j` (region 2 has `j ∈ T_j`,
  region 3 has `j ∈ [0, M_j)`);
- region 3 vs region 0 — disjoint on `i`.

Coverage: every `(i, j, k) ∈ [0, N_i) × [0, N_j) × [0, N_k)` belongs to
exactly one region. **Result: K + 1 = 4 regions** (instead of
2^K - 1 = 7 corner cells + 1 interior = 8 a naïve Cartesian split would
produce).

**Generalises to any K**: the pattern follows mechanically from the
algorithm above — the `p`-th boundary region (counting from innermost)
has tail on `d_p`, full on `d_q` for `q > p`, main on `d_q` for `q < p`.

### 8.3 Mask threading

Each boundary region runs `TileMaskGen` over the conjunction of its
tiled dim's `(i_d + l_d < N_d)` predicate. Absorbed-full dims
contribute `true` and fold away (§7.4). This way the lib-node
*interface* is uniform — `has_mask=True` everywhere a region needs
predication — even though the actual mask shape per region collapses
to a single non-trivial dim.

### 8.4 Strategies (existing knob)

The orchestrator's `remainder_strategy` selects:

- `masked_tail` (default for AVX-512 CPU): §8.1 + §8.2.
- `scalar_postamble`: boundary regions left as `Sequential` scalar
  loops, no masks. Used when the masked path measures slower than the
  scalar fallback.
- `full_mask`: single map, no split, `has_mask=True` everywhere
  (SVE default; runtime VL via `svwhilelt`).

---

## 9. Gather / Scatter Index Encoding

`TileGather` and `TileScatter` accept indices in three forms. Both the
full-tile form and the per-dim form are **first-class** — neither is
preferred over the other; the classifier picks whichever matches the
source pattern (separable → per-dim; cross-dim dependent → full). All
three forms share the same lib node; the form is identified
unambiguously by the **connector names**, which are unique and
non-overlapping:

| Form              | Connector(s)                              | Tile shape per connector  |
|-------------------|-------------------------------------------|---------------------------|
| **Full N-D**      | `_idx_full`                               | `(K_0, …, K_{K-1})`       |
| **Per-dim**       | `_idx_0`, `_idx_1`, …, `_idx_{K-1}`       | `(K_d,)` each             |
| **Partial**       | a *subset* of `_idx_0`, …, `_idx_{K-1}`   | `(K_d,)` each             |

`_idx_full` and `_idx_<d>` never co-occur; their names are disjoint so
the expansion identifies the form by inspecting which connectors are
wired (no additional discriminator needed).

### 9.1 Full N-D index tile (`_idx_full`)

One `(K_0, …, K_{K-1})` integer tile. Per-lane:
`src[idx_full[l_0, …, l_{K-1}]]` decoded against the source array's
flat-index layout. The classifier emits this form when the gather
indices are **not separable** across tile dims — i.e. the index for
one dim depends on the lane in another dim (e.g. `src[idx[i, j]]`).

### 9.2 Per-dim index tiles (`_idx_0`, `_idx_1`, …)

K integer tiles, each of shape `(K_d,)`. Per-lane:
`src[idx_0[l_0], idx_1[l_1], …, idx_{K-1}[l_{K-1}]]`. The classifier
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
is structurally a special case of §9.2 (per-dim with some indices
omitted), but the validation rule is looser: missing indices are
permitted iff the corresponding dim's per-dim kind is non-GATHER.

### 9.4 Wire-level rule

A `TileGather` constructor takes:

```python
TileGather(name, widths, src_ndim,
           gather_dims: Tuple[int, ...],  # dims that carry an index (subset of [0..K))
           dim_strides, replicate_factor_per_dim, src_dims,
           index_form: Literal["full", "per_dim", "partial"],
           has_mask=False, pad_value=0)
```

- `index_form="full"`: connector `_idx_full` (shape
  `(K_0, …, K_{K-1})`). `gather_dims == tuple(range(K))`. The connector
  name `_idx_full` is unique and does NOT clash with `_idx_<d>`, so the
  expansion identifies the form by name lookup.
- `index_form="per_dim"`: connectors `_idx_0, …, _idx_{K-1}`,
  shape `(K_d,)` each. `gather_dims == tuple(range(K))`.
- `index_form="partial"`: connectors `_idx_<d>` only for `d ∈
  gather_dims`. Other dims use `dim_strides` / `replicate_factor`.

**Identifiability invariant**: `_idx_full` and `_idx_<d>` are
**mutually exclusive** — a `TileGather` with both wired is malformed
and rejected at `validate()`. The connector names are the source of
truth for the encoding form; `index_form` is a constructor-time
sanity flag and a downstream-pass hint.

The expansion handles all three with one switch on `index_form` at the
top of the codegen body; no separate node types.

---

## 10. Validation Rules

| Where        | Checks                                                                       |
|--------------|------------------------------------------------------------------------------|
| Constructor  | `widths` length in `{1, 2, 3}`; operand kinds in `{Tile, Scalar, Symbol}`; `index_form` / `gather_dims` consistency; replicate factors divide widths. |
| `validate()` | All declared connectors are wired; tile-operand dtypes match the output dtype (uniform-dtype lock); mask present iff `has_mask`; **mask descriptor is `Array(shape=widths, dtype=bool_, storage=Register, transient=True)`** (§7.1); `_idx_full` and `_idx_<d>` not both wired on the same node (§9.4). |
| Expansion    | Source array rank ≥ K; `src_dims` permutation valid; index tile shapes match `widths` (full / per-dim form); padding mode allowed. |

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

### 10.3 TileReduce shape lock — tile → scalar only

`TileReduce` is locked to **full-tile → scalar** for the current
slice. A `(K_0, …, K_{K-1})` tile input reduces to a single value
(`dace.data.Scalar`).

**Contract**:

> If `axis` is omitted (or equivalent to the full tile shape), the
> reduction lowers to the per-arch full horizontal reduce
> (`_mm512_reduce_*`, SVE `svaddv`, etc.) and writes a scalar output.
>
> If `axis` is given and does **not** reduce the full tile to a
> scalar — i.e. any non-empty subset of dims is preserved
> (e.g. `axis=(0,)` on a K=2 tile producing a `(K_1,)` row, or
> `axis=(1,)` on a K=3 tile producing a `(K_0, K_2)` plane) — the
> node raises **`NotImplementedError`** at construction time. The
> error message names the requested axis and the expected output
> shape so the caller can fall back to scalar code (the remainder
> map's scalar postamble) or refuse the kernel.

The classifier must not emit a partial-axis reduction. Axis-keep /
per-row / per-plane reductions are recognised as future work but
deliberately unimplemented.

Rationale: keeping reductions tile→scalar avoids the per-arch
horizontal-shuffle plumbing that axis-keep would require (SVE
`svaddv` is global; AVX-512 needs hand-rolled `_mm512_reduce_*`
plus per-row materialisation; cuTile needs a Tile-IR reduce with
an explicit axis). Lift when a kernel demands it.

---

## 11. Implementation Audit

### 11.1 What the codebase already provides

| Spec section | Existing implementation                                                                            |
|--------------|----------------------------------------------------------------------------------------------------|
| §4 Lattice   | [utils/tile_access.py:85](utils/tile_access.py): `PerDimKind` with `BROADCAST` / `STRUCTURED_1` / `REPLICATE` / `AFFINE` / `GATHER`. |
| §4.2 Join    | [utils/tile_access.py](utils/tile_access.py): partial — REPLICATE-detection via `int_floor`/`int_ceil`/`__int_floor`/`__int_ceil` ([line 340](utils/tile_access.py#L340)). **Missing**: explicit loop-variant-coefficient → GATHER. |
| §3 Staging   | [promote_inlined_map_to_tiles.py](promote_inlined_map_to_tiles.py): slice 1 widens body scalars; slice 2 rewrites binop/unop tasklets to lib nodes. |
| §3 Array↔scalar copies | [rewrite_array_scalar_to_tile_op.py](rewrite_array_scalar_to_tile_op.py): direct `AN(Array) ↔ AN(tile-transient)` edges → `TileLoad` / `TileStore`. |
| §6.1 Lib nodes | [libraries/tileops/nodes/](../../../libraries/tileops/nodes/): 10 lib nodes implemented (3,278 LoC).                |
| §6.2 Operand kinds | [tile_binop.py:62-65](../../../libraries/tileops/nodes/tile_binop.py#L62): `_TILE` / `_SCALAR` / `_SYMBOL`. `TileUnop` ditto. |
| §7.1 Mask shape | [tile_mask_gen.py](../../../libraries/tileops/nodes/tile_mask_gen.py): emits ANY-OOB conjunction at tile shape. |
| §7.2 `has_mask` | every lib node has the constructor knob.                                                       |
| §8.2 Corner-absorbing peel | [split_map_for_tile_remainder.py](split_map_for_tile_remainder.py): comments at lines 7-34 describe the K boundary peel; impl appears to match. **Audit pending** (G5). |
| §8.4 Strategies | [vectorize_cpu_multi_dim.py:133-228](vectorize_cpu_multi_dim.py#L133): `full_mask` / `masked_tail` / `scalar_postamble` knob. |
| §5.1 `replicate_factor_per_dim` | [tile_load.py:310](../../../libraries/tileops/nodes/tile_load.py#L310). Wired through pure expansion. |
| §9.1 Single index tile | [tile_gather.py](../../../libraries/tileops/nodes/tile_gather.py): per-dim `_idx_<k>` connectors already exist for the K=2 / K=3 case (one connector per source-array dim). |

### 11.2 Vocabulary gap (cosmetic, mechanical commit)

The spec uses **CONSTANT / LINEAR / AFFINE / REPLICATE / MODULAR /
GATHER**. The codebase uses **BROADCAST / STRUCTURED_1 / REPLICATE /
AFFINE / GATHER** (no MODULAR yet). Rename + add MODULAR in
[utils/tile_access.py](utils/tile_access.py):

| Spec       | Code today      | Action                                |
|------------|-----------------|---------------------------------------|
| CONSTANT   | `BROADCAST`     | rename enum member                    |
| LINEAR     | `STRUCTURED_1`  | rename enum member                    |
| AFFINE     | `AFFINE`        | keep                                  |
| REPLICATE  | `REPLICATE`     | keep                                  |
| MODULAR    | *(absent)*      | **add new enum member** + detector    |
| GATHER     | `GATHER`        | keep                                  |

`TileAccessKind.BROADCAST` / `.STRUCTURED` (the whole-subset kind) also
get the analogous rename. Single mechanical commit, no behaviour
change. Update the compat shim, descent, and outer-state pass call
sites in the same commit.

### 11.3 Six gaps — see §12

---

## 12. Gaps And Action Items

Each item is a single self-contained slice. Ordering reflects
dependencies; landing them in order keeps every commit green.

### G1 — NSDFG body call convention: full subsets, no offsetting

**Why**: Section 2 says "no pointer offsetting at the call site". The
descent's existing wider machinery
([expand_nested_sdfg_inputs.py](expand_nested_sdfg_inputs.py),
[insert_body_nsdfg_copies.py](insert_body_nsdfg_copies.py)) already
covers part of this, but the *convention* — full-array memlets on every
connector — is not yet enforced as an invariant.

**What lands**: a single pass `EnforceFullSubsetNSDFGBody` that, for each
tile-tagged Map whose body is a NestedSDFG:

1. Widens every connector memlet on the outer NSDFG node to
   `array[0:N_0, …, 0:N_{D-1}]`.
2. Asserts the inner body still references the correct per-tile region
   via `symbol_mapping`.
3. Refuses (NotImplementedError) when the outer subset cannot be widened
   safely (e.g. the connector has aliased non-disjoint writes upstream).

Unit tests on hand-built K=1 and K=2 fixtures.

**Owner**: standalone pass; lives next to
[promote_nsdfg_body_to_tiles.py](promote_nsdfg_body_to_tiles.py).

### G2 — Vocabulary rename (mechanical)

See §11.2. Single commit. Update both
[utils/tile_access.py](utils/tile_access.py) and
[utils/tile_access_compat.py](utils/tile_access_compat.py).
Bonus: drop the now-redundant `TileAccessKind` enum and use
`PerDimKind` everywhere with a top-level "whole-subset kind = max of
per-dim kinds" helper.

### G3 — Per-dim partial gather indices (`_idx_<d>` semantics)

**Why**: §9.3. The classifier needs to be able to emit a `TileGather`
that takes indices for only the GATHER dims and leaves the others on
the affine path.

**What lands**:

1. Extend `TileGather` constructor with `gather_dims: Tuple[int, ...]`
   and `index_form: Literal["full", "per_dim", "partial"]` (§9.4).
2. Make the expansion read `index_form` at the top and switch.
3. Update the classifier to emit the right form: when a subset has
   mixed kinds with some GATHER, prefer `index_form="partial"` and
   set `gather_dims` to the GATHER-dim indices.
4. Mirror the same surface on `TileScatter`.

Unit tests on three fixtures: full, per-dim, partial.

### G4 — Tile-dependent symbol classification in the classifier

**Why**: §4.2 says a tile-dependent symbol in a dim's expression
forces GATHER. The current classifier
([utils/tile_access.py:316](utils/tile_access.py#L316)) detects
REPLICATE but doesn't have the explicit symbol-dependency walk.

**Mechanism**: a symbol is tile-dependent iff it (or any symbol it
transitively depends on via interstate-edge assignments inside the
body) is a tile iter-var. This is the same dependency relation the
codegen uses to decide whether a symbol requires per-lane
(laneid-style) materialisation; G4 reuses it as the ground truth.

**What lands**:

1. A helper `_is_tile_dependent(symbol, iter_vars, inner_sdfg) -> bool`
   that walks the SDFG's interstate-edge assignment definitions
   transitively (memoised on `symbol`).
2. A helper `_classify_symbols(expr, iter_vars, inner_sdfg) -> Dict[str, bool]`
   returning per-symbol tile-dependence.
3. Wire into the per-dim classifier: if any symbol in the expression
   is tile-dependent and the form is not already
   CONSTANT / LINEAR / AFFINE / REPLICATE / MODULAR over the *outer-
   scope* symbols, return GATHER.

The mechanism is **K-agnostic**: it scales to K=1, K=2, K=3 without
change (the iter-var set passed in determines the dependency walk).

### G5 — Audit `SplitMapForTileRemainder` for §8.2 (all K)

**Why**: the docstring at
[split_map_for_tile_remainder.py:7-34](split_map_for_tile_remainder.py#L7)
describes the K-boundary peel, but the spec requires it as a
*correctness* invariant for **any** `K ∈ {1, 2, 3}`. Need regression
tests that construct:

- A **K=1** map with `N_0 % W_0 ≠ 0` → assert **2 regions** (interior + 1 boundary).
- A **K=2** map with `N_d % W_d ≠ 0` on both dims → assert **3 regions** (interior + 2 boundary). The failure mode to catch is a Cartesian split producing 4 regions (1 interior + 2^K - 1 = 3 corner cells).
- A **K=3** map with `N_d % W_d ≠ 0` on all three dims → assert **4 regions** (interior + 3 boundary). The failure mode is a Cartesian split producing 8 regions.

Each test should verify the ranges per region match the §8.2 algorithm
(innermost-first peel; later boundaries take "main" on already-peeled
dims, "full" on not-yet-peeled).

### G6 — Symbol-coefficient distinction

**Why**: §4.2 split: `a[N*i]` where `N` is an outer constant → AFFINE;
where `N` depends on `i` → GATHER. This is the velocity-tendencies
trigger pattern. The classifier currently treats `N*i` as AFFINE with
symbolic stride; need to gate that on `N`'s loop-invariance (G4 covers
the helper; G6 wires it into the AFFINE path).

### G2-also — drop `_operand_kind` partial in slice 2

The slice-2 outer-state operand classifier
([promote_inlined_map_to_tiles.py](promote_inlined_map_to_tiles.py)
`_operand_kind`) currently handles Tile + Symbol only. Once G3 +
G4 land, extend it to:

- broadcast Scalar (length-1 source in scope),
- gather index (NDTile walk-back: a transient hides an N-D source
  per-lane read behind a length-1 operand).

This is the slice-2 deliverable's GATHER extension.

---

## 13. Delivery Plan

| Day      | Goal                                                                                 |
|----------|--------------------------------------------------------------------------------------|
| Mon      | G1 (full-subset NSDFG body) + G2 (vocab rename) landed.                              |
| Tue AM   | G3 (`_idx_<d>` partial indices) + G4 (lattice join in classifier) landed.            |
| Tue PM   | Design freeze meeting. Lock lib-node interfaces.                                     |
| Wed      | G5 (K=2 remainder audit) + G6 (symbol-coefficient distinction) + slice-2 GATHER ext. |
| Thu      | Velocity-tendencies stencils parse + lower end-to-end through the pipeline.          |

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
- **Mixed dtypes inside a single op** (§10.1). Materialise via an
  explicit cast `TileUnop` first.
- **K ≥ 4**. The constructors refuse `len(widths) > 3` at construction
  time. Lift only when a kernel justifies the register pressure.
- **Diagonal / transpose as dedicated structured patterns**. Both fold
  into GATHER for now (§5.4); a structured fast path can land later
  behind the same `TileGather` surface.
- **Auto-classify load node** (a single node that defers
  classification to its expansion). The pipeline-time classifier
  already covers it cleanly; adding an umbrella node would duplicate
  the lattice. Marked NOT pursued.

---

## Appendix A — Cross-Dimension Composition Examples

### A.1 K = 1

| Access     | Per-dim kind  | Lowering                                  |
|------------|---------------|-------------------------------------------|
| `a[i]`     | LINEAR        | TileLoad, `dim_strides=(1,)`              |
| `a[0]`     | CONSTANT      | TileLoad, broadcast                       |
| `a[i//2]`  | REPLICATE k=2 | TileLoad, replicate factor 2              |
| `a[2*i+1]` | AFFINE s=2    | TileLoad, `dim_strides=(2,)`              |
| `a[idx[i]]`| GATHER        | TileGather, `_idx_0` index tile           |

### A.2 K = 2 (tile vars `(i, j)`)

| Access                | Per-dim kinds (i, j)         | Lowering                                         |
|-----------------------|------------------------------|--------------------------------------------------|
| `a[i, j]`             | (LINEAR, LINEAR)             | TileLoad, `dim_strides=(1,1)`                    |
| `a[0, j]`             | (CONSTANT, LINEAR)           | TileLoad, broadcast dim 0, contiguous dim 1      |
| `a[i, 0]`             | (LINEAR, CONSTANT)           | TileLoad, contiguous dim 0, broadcast dim 1      |
| `a[i//2, j]`          | (REPLICATE k=2, LINEAR)      | TileLoad, dim 0 replicate factor 2               |
| `a[2*i + 1, j]`       | (AFFINE s=2, LINEAR)         | TileLoad, `dim_strides=(2,1)`                    |
| `a[i, i]`             | (LINEAR i, LINEAR i)         | TileGather diagonal, `index_form="per_dim"`      |
| `a[j, i]`             | (LINEAR j, LINEAR i)         | TileGather transposed, `index_form="per_dim"`    |
| `a[idx[i], j]`        | (GATHER, LINEAR)             | TileGather, `index_form="partial"`, gather_dims=(0,) |
| `a[idx0[i], idx1[j]]` | (GATHER, GATHER)             | TileGather, `index_form="per_dim"`               |
| `a[idx[i, j]]`        | flat 1-D, idx is N-D         | TileGather, `index_form="full"`                  |
| `a[2*sym + 1, j]`     | (AFFINE / GATHER on sym, LIN)| AFFINE if `sym` tile-independent else GATHER (§4.2) |
| `a[i % N, j]`         | (MODULAR N, LINEAR)          | TileGather (general MODULAR), `index_form="partial"` |
| `a[i % W_0, j]`       | (MODULAR → LINEAR, LINEAR)   | TileLoad — tile-aligned reduction (§4.2)         |
| `a[i % (2*W_0), j]`   | (MODULAR → LINEAR, LINEAR)   | TileLoad with per-tile constant offset (§4.2)    |

### A.3 K = 3 (tile vars `(i, j, k)`)

| Access                   | Per-dim kinds (i, j, k)         | Lowering                                              |
|--------------------------|---------------------------------|-------------------------------------------------------|
| `a[i, j, k]`             | (LINEAR, LINEAR, LINEAR)        | TileLoad, `dim_strides=(1,1,1)`                       |
| `a[0, j, k]`             | (CONSTANT, LINEAR, LINEAR)      | TileLoad, broadcast dim 0                             |
| `a[i, 0, k]`             | (LINEAR, CONSTANT, LINEAR)      | TileLoad, broadcast dim 1                             |
| `a[i, j, 0]`             | (LINEAR, LINEAR, CONSTANT)      | TileLoad, broadcast dim 2                             |
| `a[i//2, j, k]`          | (REPLICATE k=2, LINEAR, LINEAR) | TileLoad, dim 0 replicate factor 2                    |
| `a[i, j//4, k]`          | (LINEAR, REPLICATE k=4, LINEAR) | TileLoad, dim 1 replicate factor 4                    |
| `a[2*i + 1, j, k]`       | (AFFINE s=2, LINEAR, LINEAR)    | TileLoad, `dim_strides=(2,1,1)`                       |
| `a[i, i, k]`             | (LINEAR i, LINEAR i, LINEAR)    | TileGather (diagonal on dims 0,1), `index_form="per_dim"` |
| `a[k, j, i]`             | (LINEAR k, LINEAR j, LINEAR i)  | TileGather transposed, `index_form="per_dim"` (permuted) |
| `a[idx[i], j, k]`        | (GATHER, LINEAR, LINEAR)        | TileGather, `index_form="partial"`, gather_dims=(0,)  |
| `a[idx0[i], j, idx2[k]]` | (GATHER, LINEAR, GATHER)        | TileGather, `index_form="partial"`, gather_dims=(0,2) |
| `a[idx0[i], idx1[j], idx2[k]]` | (GATHER, GATHER, GATHER)  | TileGather, `index_form="per_dim"`                    |
| `a[idx[i, j, k]]`        | flat 1-D, idx is K-D            | TileGather, `index_form="full"`                       |
| `a[i % N, j, k]`         | (MODULAR N, LINEAR, LINEAR)     | TileGather (general MODULAR), `index_form="partial"`  |
| `a[i % W_0, j, k]`       | (MODULAR → LINEAR, LINEAR, LINEAR) | TileLoad — tile-aligned reduction (§4.2)           |

---

## Appendix B — Symbol Definitions

| Symbol | Meaning                                                                |
|--------|------------------------------------------------------------------------|
| K      | Number of tiled dimensions; K ∈ {1, 2, 3}.                             |
| W_d    | Tile width on dim `d`; `widths[d]`.                                    |
| N_d    | Global upper bound on dim `d`.                                         |
| i_d    | Tile iter-var on dim `d` (outer Map's parameter).                      |
| l_d    | Lane index within the tile on dim `d`; `0 ≤ l_d < W_d`.                |
| k      | REPLICATE factor; the dim varies in groups of `k` lanes.               |
| s      | AFFINE stride coefficient.                                             |
| c      | AFFINE / LINEAR offset (loop-invariant constant).                      |

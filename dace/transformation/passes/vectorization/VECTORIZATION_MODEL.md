# Vectorization — Mathematical Model

High-level model behind the K-dim CPU vectorizer. The detailed IR contract lives
in [TILE_LOWERING.md](TILE_LOWERING.md) (lib-node lowering) and in the lib-node
`validate()` methods; this document is the *theory* — what a tile is, what the
body computes, and how a loop is brought to that form.

## 1. Iteration domain

A parallel map `M` carries parameters `i = (i_0, …, i_{n-1})` over an iteration
domain

```
D = { i : lo_d ≤ i_d < ub_d,  d = 0..n-1 }
```

(an integer box; affine symbolic bounds allowed). `M` asserts the body is
independent across points of `D` — i.e. `D` is fully parallel. Producing such a
map from arbitrary loop nests is the job of *canonicalization* (§5).

## 2. Tiling

Pick the `K` innermost dims (`1 ≤ K ≤ 3`) and per-dim **widths**
`W = (W_0, …, W_{K-1})`. Tiling factors each tiled dim `d` into an **outer index**
`t_d` (stepping by `W_d`) and a **lane index** `l_d ∈ [0, W_d)`:

```
i_d = t_d + l_d           (tiled dim d, t_d a multiple of W_d)
```

The **tile** is the lane hypercube `L = ∏_d [0, W_d)` (a register tile of
`|L| = ∏ W_d` lanes). The outer tile indices `t_d` and all non-tiled dims form the
enclosing scope; the **body runs once per tile**, evaluating every lane of `L`.
`K = 1` is the common 1-D SIMD case (`L = [0, W)`); `K = 2` tiles a 2-D nested map.

## 3. Remainder and the iteration mask

When `ub_d − lo_d` is not divisible by `W_d` the last tile along `d` is *partial*:
some lanes map to `i_d ≥ ub_d` (out of bounds). The **iteration mask** is the lane
predicate

```
m(l) = ⋀_d ( t_d + l_d < ub_d )           m : L → {0,1}
```

A lane is *active* iff `m(l) = 1`. Three remainder strategies realise it:

| strategy | behaviour |
|---|---|
| `masked_tail` | full tiles run unmasked; only the partial tail tile carries `m` |
| `full_mask` | every tile carries `m` (uniform, no separate tail) |
| `scalar_postamble` | tiled interior runs unmasked; the remainder is a scalar loop |

Mask **semantics** (the correctness contract, enforced at lowering): a *producer*
(load / compute) ZERO-FILLs inactive lanes and never dereferences their address; a
*writer* (store) RMW-skips inactive lanes. The mask is branch-independent, so its
producer must **dominate** every consumer (see `tile_mask_gen_dominates_consumers`).

## 4. Access expansion

Every array reference `A[φ(i)]` has an index map `φ`. Restricted to a tile and
projected onto `A`'s strides `σ_A`, the **per-lane element offset** is

```
off_A(l) = φ(t) · σ_A  +  Σ_d  J_{A,d} · l_d
```

where `J_{A,d}` is the (byte/element) coefficient of lane `l_d` in `A` — the
"expanded access subset". Each `(A, d)` pair is classified into the **access
lattice**, which selects how the lane vector is materialised:

| kind | per-lane offset | materialisation |
|---|---|---|
| `CONSTANT` | `J = 0` | one scalar, splat to all lanes |
| `LINEAR` | `J = 1` (unit) | one dense vector load |
| `REPLICATE(k)` | `⌊l/k⌋` | dense load of `W/k` + broadcast ×k |
| `AFFINE(s)` | `s·l`, `s>1` | strided load / gather |
| `MODULAR(N)` | `(c·l + c₀) mod N` | per-lane index + gather |
| `GATHER` | `idx[l]` | data-dependent gather |

A *diagonal* access (one iter-var indexing several dims of `A`, e.g. `A[i,i]`)
folds the per-dim coefficients into one combined stride on a unit-stride basis;
absent a unit-stride dim it is *refused* rather than mis-strided.

## 5. The tile function — read set / write set

Inside a tile the body is a pure function

```
f : (read tiles)  ⟶  (write tiles)
```

with a **read set** `R = { (A, off_A) : A read }` and **write set**
`W = { (A, off_A) : A written }`. A value is a **tile** (shape `W`, one element per
lane) when it depends on a lane index, and a **scalar / broadcast** otherwise.
`R`/`W` drive three decisions: which transients must be *widened* to tile shape
(lane-dependent ones), where the mask gates (every active-lane read/write), and
which accesses are gathers vs affine (the lattice of §4). Reductions collapse a
tile to a scalar (`TileReduce`, tile → scalar only).

## 6. Bringing a loop to this form (pipeline)

```
canonicalize            loop nest ⟶ parallel map(s)         (§5 of the suite)
  ├ LoopToMap / fission / peel / break-antidep / wavefront-skew
  └ reductions ⟶ Reduce/Scan lib-nodes
MarkTileDims            choose K innermost dims + widths; classify each access (§4)
SplitMapForTileRemainder  carve the remainder per strategy (§3)
NestInnermostMapBody    body ⟶ NestedSDFG (the tile function f)
WidenAccesses           lane-dependent transients ⟶ tile shape W
GenerateTileIterationMask  add m in a dominating start state (§3)
ConvertTaskletsToTileOps   scalar tasklets ⟶ tile lib-nodes (§ TILE_LOWERING)
EmitTileOps             expand lib-nodes: pure loop OR per-ISA intrinsic
```

The lib-node `validate()` calls lock the IR shape (operand-dtype uniformity, mask
descriptor, index dtype, reduce shape), so each per-arch backend is a leaf slice
that cannot diverge on the contract.

## 7. Invariants (correctness anchors)

- **Parallelism**: the tiled map's domain `D` is independent across points.
- **Mask domination**: the `TileMaskGen` producer dominates all masked consumers.
- **Producer zero-fill / writer skip-inactive**: inactive lanes are never read out
  of bounds nor written.
- **No value-truncating casts**: an op whose operand and output dtypes differ
  defers to the `pure` expansion rather than emitting a C-style cast.
- **Refuse, don't miscompile**: an access the affine model cannot express (e.g. a
  no-unit-stride diagonal) raises rather than silently mis-striding.

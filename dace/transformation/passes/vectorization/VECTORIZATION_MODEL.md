# Vectorization — Model

Theory behind the K-dim CPU tile vectorizer: what a tile is, what the body
computes, how a loop reaches that form. The IR contract is enforced by the
lib-node `validate()` methods.

## 1. Domain

Parallel map `M`, params `i = (i_0, …, i_{n-1})`, domain
`D = { i : lo_d ≤ i_d < ub_d }` (integer box, affine symbolic bounds allowed).
`M` asserts the body is independent across `D`. Canonicalization (§6) produces
such maps.

## 2. Tiling

Pick the `K` innermost dims (`1 ≤ K ≤ 3`) and per-dim widths `W = (W_0, …, W_{K-1})`.
Each tiled dim `d` factors into an outer index `t_d` (step `W_d`) and a lane
`l_d ∈ [0, W_d)`: `i_d = t_d + l_d`. The tile is the lane hypercube
`L = ∏_d [0, W_d)` (`∏ W_d` lanes); the body runs once per tile over all lanes.
`K = 1` is 1-D SIMD; `K = 2` tiles a 2-D nested map.

## 3. Remainder / iteration mask

When `ub_d − lo_d` is not divisible by `W_d` the last tile is partial. Mask
`m(l) = ⋀_d ( t_d + l_d < ub_d )`; a lane is active iff `m(l) = 1`.

| strategy | behaviour |
|---|---|
| `masked_tail` | full tiles unmasked; only the tail tile carries `m` |
| `full_mask` | every tile carries `m` |
| `scalar_postamble` | tiled interior unmasked; remainder is a scalar loop |

Contract (enforced at lowering): a producer zero-fills inactive lanes and never
dereferences their address; a writer RMW-skips inactive lanes. The mask producer
must dominate every consumer.

## 4. Access lattice

Reference `A[φ(i)]` has per-lane offset `off_A(l) = φ(t)·σ_A + Σ_d J_{A,d}·l_d`.
Each `(A, d)` is classified:

| kind | per-lane | materialisation |
|---|---|---|
| `CONSTANT` | `J = 0` | scalar splat |
| `LINEAR` | `J = 1` | dense vector load |
| `REPLICATE(k)` | `⌊l/k⌋` | dense load `W/k` + broadcast ×k |
| `AFFINE(s)` | `s·l, s>1` | strided load / gather |
| `MODULAR(N)` | `(c·l + c₀) mod N` | per-lane index + gather |
| `GATHER` | `idx[l]` | data-dependent gather |

A diagonal `A[i,i]` folds the per-dim coefficients onto a unit-stride basis; with
no unit-stride dim it is refused, not mis-strided.

## 5. Tile function

The body is pure `f : read tiles → write tiles`, with read set `R` / write set `W`
of `(A, off_A)` pairs. A value is a tile (one element per lane) iff it depends on a
lane index, else a scalar/broadcast. `R`/`W` drive which transients widen to tile
shape, where the mask gates, and gather-vs-affine. Reductions collapse a tile to a
scalar (`TileReduce`).

## 6. Pipeline (loop → tiles)

```
canonicalize               loop nest → parallel map(s); reductions → Reduce/Scan
MarkTileDims               K innermost dims + widths; classify accesses (§4)
SplitMapForTileRemainder   remainder per strategy (§3)
NestInnermostMapBody       body → NestedSDFG (f)
WidenAccesses              lane-dependent transients → tile shape
GenerateTileIterationMask  mask m in a dominating start state (§3)
ConvertTaskletsToTileOps   scalar tasklets → tile lib-nodes
EmitTileOps                expand lib-nodes: pure loop OR per-ISA intrinsic
```

## 7. Invariants

- Parallelism: the tiled map domain is independent across points.
- Mask domination: the `TileMaskGen` producer dominates all masked consumers.
- Producer zero-fill / writer skip-inactive: inactive lanes are never read out of
  bounds nor written.
- No value-truncating casts: differing operand/output dtypes defer to the `pure`
  expansion, never a C-style cast.
- Refuse, don't miscompile: an access the affine model cannot express raises.

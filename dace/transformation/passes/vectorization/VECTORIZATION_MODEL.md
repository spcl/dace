# Multi-Dim Vectorization — Model

Theory behind the K-dim CPU tile vectorizer. The IR contract itself is enforced by the
lib-node `validate()` methods; the pass order lives in the package. This doc holds the
model and the invariants.

## Domain and tiling

Parallel map `M` over integer box `D = { i : lo_d ≤ i_d < ub_d }` (affine symbolic bounds
allowed); `M` asserts the body is independent across `D`. Canonicalization produces such
maps.

Take the `K` innermost dims (`1 ≤ K ≤ 3`) with widths `W = (W_0, …, W_{K-1})`. Dim `d`
factors into outer index `t_d` (step `W_d`) and lane `l_d ∈ [0, W_d)`: `i_d = t_d + l_d`.
The tile is the lane hypercube `L = ∏_d [0, W_d)`, `∏ W_d` lanes; the body runs once per
tile, over all lanes. `K = 1` is plain SIMD; `K = 2` tiles a 2-D nested map.

## Remainder / iteration mask

`ub_d − lo_d` not divisible by `W_d` ⇒ partial last tile. Mask
`m(l) = ⋀_d ( t_d + l_d < ub_d )`; lane active iff `m(l) = 1`.

| strategy | behaviour |
|---|---|
| `masked_tail` | full tiles unmasked; only the tail tile carries `m` |
| `full_mask` | every tile carries `m` |
| `scalar_postamble` | tiled interior unmasked; remainder is a scalar loop |
| `branched_masked_tail` | GPU K=1 default: ONE map, `if t + W − 1 ≤ ub` unmasked tile `else` masked tile — no scalar loop |
| `branched_tail` | GPU K=1: ONE map, `if t + W − 1 ≤ ub` unmasked tile `else` scalar lane loop |

## Access lattice

Reference `A[φ(i)]` has per-lane offset `off_A(l) = φ(t)·σ_A + Σ_d J_{A,d}·l_d`. Each
`(A, d)` classifies as:

| kind | per-lane | materialisation |
|---|---|---|
| `CONSTANT` | `J = 0` | scalar splat |
| `LINEAR` | `J = 1` | dense vector load |
| `REPLICATE(k)` | `⌊l/k⌋` | dense load `W/k` + broadcast ×k |
| `AFFINE(s)` | `s·l, s>1` | strided load / gather |
| `MODULAR(N)` | `(c·l + c₀) mod N` | per-lane index + gather |
| `GATHER` | `idx[l]` | data-dependent gather |

A diagonal `A[i,i]` folds its per-dim coefficients onto a unit-stride basis. With no
unit-stride dim it is refused, not mis-strided.

## Tile function

The body is pure `f : read tiles → write tiles`, read set `R` / write set `W` of
`(A, off_A)` pairs. A value is a tile (one element per lane) iff it depends on a lane
index, else a scalar/broadcast. `R`/`W` decide which transients widen to tile shape, where
the mask gates, and gather-vs-affine. Reductions collapse a tile to a scalar
(`TileReduce`).

## Invariants

- Parallelism: the tiled map domain is independent across points.
- Mask domination: the `TileMaskGen` producer dominates every masked consumer.
- Producer zero-fills inactive lanes and never dereferences their address; a writer
  RMW-skips them. Inactive lanes are never read out of bounds nor written.
- No value-truncating casts: differing operand/output dtypes defer to the `pure`
  expansion, never a C-style cast.
- Refuse, don't miscompile: an access the affine model cannot express raises.

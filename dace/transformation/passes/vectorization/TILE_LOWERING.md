# Tile Lib-Node Lowering

How the tile function (see [VECTORIZATION_MODEL.md](VECTORIZATION_MODEL.md)) is
expressed as library nodes and lowered to code. Covers the node set, the connector
grammar, broadcasting, and the per-ISA dispatch.

## 1. The register-tile abstraction

A **tile** is a `dace.data.Array` of shape `widths = (W_0, …, W_{K-1})`,
`storage = Register`, `transient = True` — one element per lane of the tile (§2 of
the model). Every tile lib-node consumes/produces tiles plus an optional mask tile;
its expansion turns one node into either a portable loop (`pure`) or a per-ISA SIMD
sequence. The node set:

| node | role |
|---|---|
| `TileLoad` | gather a (strided / indexed) source region into a dense lane tile |
| `TileStore` | scatter a dense lane tile into a (strided / indexed) dest region |
| `TileBinop` / `TileUnop` | element-wise arithmetic / logical op over tiles |
| `TileITE` | per-lane select `cond ? then : else` |
| `TileReduce` | collapse a tile to a scalar (sum / min / max / …) |
| `TileMaskGen` | produce the iteration mask `m` (`base + l < ub`) |

## 2. Why `TileLoad` and `TileStore` are distinct nodes

They are **not** a symmetric copy — read and write differ on three axes:

- **Addressing.** Load gathers a strided/indexed source into a dense tile; store
  scatters a dense tile into a strided/indexed dest. Gather and scatter are
  distinct intrinsics with distinct hazards (scatter aliases/write-conflicts).
- **Masking.** A masked load ZERO-FILLs inactive lanes (value semantics); a masked
  store SKIPs them (side-effect semantics) — one never reads OOB, the other never
  writes.
- **Broadcast.** Load has a read-side broadcast (`CONSTANT` → splat, §4); store has
  none, but carries write-conflict resolution (WCR) for reductions into memory.

So the two directions need separate node types, validators, and expansions — a
single "copy" node would have to branch on direction everywhere anyway.

## 3. Connector grammar

Every connector name is unique and is the **single source of truth** for its role.

| connector | nodes | shape | role |
|---|---|---|---|
| `_src` | load (in), store (in, the tile) | full source / `widths` | data in |
| `_dst` | load (out, the tile), store (out) | `widths` / full dest | data out |
| `_a`, `_b`, `_c` | binop | `widths` (or scalar/symbol) | operands `_a`,`_b` → result `_c` |
| `_a`, `_c` | unop | `widths` | operand → result |
| `_mask`, `_t`, `_e`, `_o` | ITE | `widths` | predicate, then, else → out |
| `_o` | reduce, mask-gen | `widths` (in) / scalar (reduce out) | output tile / mask |
| `_idx_<k>` | load / store (optional) | see below | gather index for **source dim `k`** |

The `<k>` on `_idx_<k>` is a **source-array dim index** (resp. dest for store), not
a tile-dim index — lane geometry (`widths`) and source addressing (`gather_dims`)
are orthogonal. The connector's **shape encodes lane dependency**: an index that
depends on tile lanes `deps ⊆ {0..K-1}` is wired at shape `∏_{p∈deps} W_p`
(`(1,)` for a lane-independent index). There is no `_idx_full` — the full N-D case
is just an `_idx_<k>` spanning every tile dim. An `_idx_<k>` is present **iff**
`k ∈ gather_dims`, and is a signed-integer tile (the index-dtype lock).

## 4. Operand kinds and broadcasting

A tile operand carries a **kind** that selects how it is materialised per lane:

| kind | source | per-lane realisation |
|---|---|---|
| `Tile` | a `widths`-shaped tile | read lane `l` directly (`Broadcast = false`) |
| `Scalar` | a length-1 array | read `[0]`, splat to all lanes (`Broadcast = true`) |
| `Symbol` | an inline symbolic expr | evaluate once, splat (`Broadcast = true`) |

At least one operand of a binop must be `Tile`. The **replicate factor** `k`
(lane `l` reads element `⌊l/k⌋`) handles `REPLICATE` accesses (§4 of the model);
factor `> 1` defers to `pure` (the intrinsic has no per-lane replicate divisor).
Per-arch broadcast realisation: `CONSTANT` → `_mm512_set1_pd` / `vdupq_n` / `svdup`;
`LINEAR` → `_mm512_loadu_pd` / `vld1q` / `svld1`; `AFFINE`/`GATHER` →
`_mm512_i64gather_pd` / `svld1_gather`, falling to a scalar loop where no gather
intrinsic exists.

## 5. Lowering tiers and dispatch

Each node has several **implementations**, selected per node:

- `pure` — a K-fold nested CPP loop; correct for any `K`, dtype, and access shape.
  The universal fallback.
- ISA backends `scalar` / `avx512` / `avx2` / `neon` / `sve` — **K = 1 only**; each
  emits the same `dace::tileops::tile_<op>` call and differs only in the header its
  *environment* pulls in. The five near-identical expansion classes per node are
  built by one factory (`_isa_codegen.make_isa_expansions`).
- `cutile` — `cuda.tile` Python (GPU; opt-in).

Selection: `K ≥ 2` → `pure`; `K = 1` → the target ISA (`AUTO` detects the host),
falling back to `pure` when an ISA impl is absent. Within an ISA op the dispatch
refines by access shape:

```
tile_load:   stride == 1  →  dense SIMD load  (_mm512_loadu_pd / maskz_loadu)
             stride != 1  →  gather over [0,s,2s,…]  (_mm512_i64gather_pd)
tile_gather: data-dependent idx  →  gather over the index tile (_mm512_i64gather_pd)
             (fp64+int64 fast path; other dtype/width → scalar loop)
```

The clean 1-D unit-stride gather (`a[idx[i]]`) is emitted as `tile_gather`; richer
gathers (multi-dim source, non-unit stride, replicate) stay on `pure`.

## 6. Mask and validation contract

The mask is **only** a full-tile boolean Array (`shape = widths`, `dtype = bool_`,
`storage = Register`, `transient = True`) — the §10.2 descriptor lock. Masked
intrinsics compose on top of each op: masked dense load → `_mm512_maskz_loadu_pd`;
masked gather → `_mm512_mask_i64gather_pd` (reads active lanes only — inactive OOB
indices are never dereferenced). Other locks the validators enforce:

- **Operand-dtype uniformity** — a binop/unop with mixed operand/output dtype
  defers to `pure` (no value-truncating cast).
- **Index dtype** — every `_idx_<k>` is signed integer.
- **Reduce shape** — `TileReduce` is tile → scalar only (no `axis` / `keepdims`).
- **Logical binops** — `&&` / `||` operate on bool tiles only.

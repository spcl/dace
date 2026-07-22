# Broadcast in tile lib nodes — Option A (in-expansion)

**Status**: implemented (current state of `multi-dim-tileops`).

Broadcast is NOT a dedicated lib node. Every broadcast pattern is
expressed as a property on the consuming / producing tile lib node, and
the node's per-target `expansion()` lowers the broadcast inline. There
is no separate `TileBroadcast` IR node.

## Two orthogonal broadcast axes

| Axis | What it broadcasts | Property | Affects |
|------|--------------------|----------|---------|
| 0-D inline | A single scalar / literal to every lane | `src_kind ∈ {Tile, Scalar, Symbol}` (+ `src_expr` for Symbol) | `TileLoad`, `TileStore`, `TileBinop` (per-operand `kind_a/kind_b`) |
| Per-lane | A smaller-rank tile to a higher-rank tile by zeroing the unbound lane offsets | `dim_strides[p] = 0` for the unbound lane p | `TileLoad`, `TileStore` |

The two axes compose: a `TileLoad(src_kind="Scalar")` is per-definition
broadcast on every lane (no `dim_strides` needed); a `TileLoad(
src_kind="Tile", dim_strides=(1, 0))` is a 1-D tile read with the inner
lane broadcast.

## Property contracts

### `src_kind` — 0-D inline

```python
src_kind = Property(dtype=str, default="Tile")   # "Tile" | "Scalar" | "Symbol"
src_expr = Property(dtype=str, allow_none=True)  # required iff src_kind == "Symbol"
```

- **Tile** (default): `_src` connector is a tile-shape transient / strided view;
  per-lane indexed read via the lane offset formula below.
- **Scalar**: `_src` connector is a length-1 ``Array`` (pointer, accessed as
  `_src[0]`) or a `dace.data.Scalar` (passed by value, accessed as bare `_src`).
  The expansion picks the form based on the source descriptor.
- **Symbol**: NO `_src` connector. The literal / symbolic expression is embedded
  inline at expansion time. Constructor enforces non-empty `src_expr`.

### `dim_strides` — per-lane

Per-tile-dim integer coefficient. The lane offset into the source array is

```
src_off = sum_{p in 0..K-1}  dim_strides[p] * src_strides[match_dims[p]] * __l<p>
```

A lane with `dim_strides[p] = 0` contributes zero to the offset → every value
of `__l<p>` reads the same source element → broadcast along that lane.

When the source array's rank is smaller than K, multiple `match_dims[p]` map
to the same source dim. The zero strides make this safe (the duplicated
source-dim references are scaled by 0).

## Recording broadcast intent during the descent

`PromoteNSDFGBodyToTiles._widen_boundary_connectors` records per-connector
broadcast info in two dicts on the pass instance:

```python
self._conn_match_dims:  Dict[str, Tuple[int, ...]]   # lane p -> source dim
self._conn_dim_strides: Dict[str, Tuple[int, ...]]   # lane p coefficient
```

`_box_classification(subset, arr, iter_vars, conn_name)` consults these
when `arr.shape != widths` (the connector is a proper rank-prefix of the
tile) and returns a `TileAccessClassification` with the recorded
`match_dims` + `dim_strides`. `_promote_loads` / `_promote_stores` then
construct the lib node with those properties.

## Validation invariants

Each lib node's `validate()` checks:

1. `src_kind == "Symbol"` ⇒ `_src` connector is NOT declared and NOT
   connected (asserted at construction time too).
2. `src_kind != "Symbol"` ⇒ `_src` is required and connected.
3. `len(dim_strides) == K == len(widths)`.
4. `len(match_dims) == K` when present; each entry is a valid source-dim
   index.

`dim_strides[p] = 0` is a legal value (broadcast lane). The codegen does
NOT reject zero strides.

## Expansion dispatch (`ExpandTileLoadPure` is the reference)

The pure expansion emits a K-fold nested CPP loop with the body:

```cpp
_dst[<dst_off>] = <src_ref>;
```

`<src_ref>` is chosen at expansion time based on `src_kind`:

| `src_kind` | `<src_ref>` |
|------------|-------------|
| Symbol | `(dtype)(<src_expr>)` |
| Scalar (length-1 Array source) | `(dtype)(_src[0])` |
| Scalar (`dace.data.Scalar` source) | `(dtype)(_src)` |
| Tile | `_src[<src_off>]` |

`<src_off>` is computed via `offset_via_strides(dim_strides, src_strides)`.
Stride-0 lanes drop out of the sum; the resulting offset depends only on
the bound lanes. The cuTile expansion mirrors the dispatch (`ct.scalar`
for Scalar / Symbol, `ct.load` for Tile; broadcast-across-lane via
`ct.broadcast_to`).

## Composability with non-load / non-store nodes

| Consumer | Broadcast support today |
|----------|------------------------|
| `TileBinop` | `kind_a/kind_b ∈ {Tile, Scalar, Symbol}` per-operand. Same dispatch. |
| `TileStore` | `src_kind` symmetric to TileLoad. |
| `TileMerge` | `_cond`, `_t`, `_e` must all be Tile shape today. A 0-D condition is materialised by an upstream TileLoad(Scalar) → tile. |
| `TileUnop` | Tile-only operand. Broadcast inputs must be staged through TileLoad(Scalar/Symbol). |
| `TileGather` | Tile-only inputs (data + indices). No `src_kind`. Per-lane broadcast across an unbound lane is NOT supported. |
| `TileScatter` | Symmetric to `TileGather`. |
| `TileReduce` | Tile-only operand. |
| `TileMaskGen` | No inputs; produces a tile. Broadcast-irrelevant. |
| `TileIota` | No `src_kind`; per-lane expression. Equivalent broadcast achieved by a tile-var-free `expr`. |

## Sites that emit broadcasts in the descent

| Site | What it emits |
|------|---------------|
| `_promote_const_stores` (literal RHS) | `TileStore(src_kind="Symbol", src_expr=literal)` |
| `_emit_const_tile` (EmitTileOps) | same shape |
| `_promote_stores` (Scalar src via `_box_classification` rank-mismatch path, outer subset tile-var-free) | `TileLoad(src_kind="Scalar")` → tile transient → `TileStore` |
| `_promote_stores` / `_promote_loads` (partial-binding, outer subset has tile-var dep) | `TileLoad(src_kind="Tile", dim_strides=<partial>)` with stride 0 on unbound lanes |
| `_promote_binops` `_operand_ref` (literal / Scalar binop operand) | `TileBinop(kind_<x>="Scalar" \| "Symbol")` |

## Pros (why this design was chosen)

- **No IR overhead**: a broadcast does not add a node. The SDFG is smaller
  and inspection sees one node per logical operation.
- **Codegen fuses naturally**: the broadcast is part of the same nested
  loop as the load / store / binop. No intermediate transient buffer.
- **Existing TileBinop pattern**: `kind_a/kind_b` already worked this
  way; extending the same shape to TileLoad / TileStore keeps the design
  consistent across the lib node family.
- **Compact literals**: `TileStore(src_kind="Symbol", src_expr="0.0")`
  is one node + zero connectors — the minimal SDFG shape for a
  constant store.

## Cons (limitations of the in-expansion approach)

- **Property surface grows on every consumer**: every node that should
  accept a broadcast input needs its own `src_kind` (or `kind_a/kind_b`)
  property and its expansion has to dispatch on it. `TileUnop`,
  `TileGather`, `TileScatter`, `TileMerge`, `TileReduce` do NOT have
  this today — a broadcast input to any of them must be materialised
  through a TileLoad(Scalar/Symbol) first.
- **`dim_strides=0` is opaque**: the lane-broadcast intent is encoded
  by the value `0` inside an integer tuple. A reader of the SDFG cannot
  distinguish "broadcast lane" from "stride 0 step" without chasing the
  lane offset formula.
- **Partial broadcast scope is limited to {Load, Store}**: TileGather
  cannot today say "broadcast along the inner lane" — would need its
  own `dim_strides` field and matching expansion changes. The K=2
  gather + broadcast composition (zekin and 4 unit-test xfails) needs
  either (a) extending TileGather to accept `dim_strides` with zeros
  or (b) routing through a TileLoad-shape post-gather.
- **Two broadcast axes, two mechanisms**: `src_kind` and `dim_strides=0`
  are conceptually the same operation (broadcast a smaller-rank source
  to a larger tile) but live as different properties on different
  fields. A caller has to pick the right mechanism based on whether the
  source is 0-D or partial-rank.
- **No CSE across broadcasts**: a scalar broadcast consumed by two
  consumers is materialised separately in each consumer's expansion.
  Codegen common-subexpression-elimination may catch it; the SDFG does
  not.

## When you'd reconsider

Add a dedicated `TileBroadcast` node if any of these become true:

1. More than ~2 additional Tile* nodes need broadcast input support
   and adding `src_kind` to each is repetitive enough to be a bug
   surface.
2. The K=2 gather + broadcast composition cannot be expressed cleanly
   via `dim_strides=0` on TileGather (e.g. when the gather indices
   themselves are partial-rank).
3. cuTile lowering would benefit from a `ct.broadcast_to(...)` call
   that is visible at the IR level (today's `cutile` expansion inlines
   the broadcast inside each consumer).

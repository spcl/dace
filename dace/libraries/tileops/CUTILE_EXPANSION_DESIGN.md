# cuTile (`cuda.tile`) Expansion Design for Tile-Op Library Nodes

Design for lowering the DaCe `tileops` library nodes to NVIDIA cuTile
(`import cuda.tile as ct`). This document assesses every existing `cutile`
expansion against cuTile's actual capabilities, specifies the corrected
emitted body shape per node (K=1 and K>=2), and states where each node's
mask is applied. It is a design only — no code is changed by this file.

## cuTile capability summary (authoritative, from prior research)

The following constraints are treated as ground truth and are referenced
throughout by short name:

- **L-load-nomask**: `ct.load(array, index, shape, *, padding_mode=...)`
  has **no `mask=` parameter**. Partial/edge tiles are handled only via
  `padding_mode ∈ {UNDETERMINED(default), ZERO, NEGATIVE_ZERO, NAN,
  POSITIVE_INFINITY}`. No arbitrary padding scalar, no per-lane bool mask.
- **L-store-nomask**: `ct.store(array, index, tile)` has **no per-lane
  `mask=`**. Lane-masked writes must use `ct.scatter`.
- **L-gs-mask**: `ct.gather(array, indices, *, padding_value=0,
  check_bounds=True[, mask=])` and `ct.scatter(array, indices, value, *,
  check_bounds=True[, mask=])` have a Tile-IR mask operand; the Python
  `mask=` kwarg is **strongly indicated but NOT 100% confirmed** on the
  per-function page. `indices` is a single 1-D index tile or a tuple of
  per-dim index tiles (broadcastable). OOB gather → `padding_value`
  (undefined if not supplied); OOB/masked scatter → no store.
- **L-elt-nomask**: elementwise binops/unops (`+ - * /`, `ct.minimum`,
  `ct.maximum`, comparisons, `ct.abs`, `ct.exp`, …) have **no mask
  operand**. Masking must be relocated to the store (scatter) or applied
  via a select.
- **L-reduce-nomask**: `ct.reduce(x, axis, func, identity, keepdims)` and
  `ct.sum/prod/max/min(x, axis, keep_dims)` have **no mask / valid-region
  arg**. A masked reduction must pre-select the **identity** element into
  masked lanes before reducing (identity: `0` sum, `1` prod, `+inf` min,
  `-inf` max).
- **L-where-unconfirmed**: a `ct.where` / select primitive's existence is
  **UNCONFIRMED** — no doc page found. The design must not depend on it;
  every use of `ct.where` is paired with an arithmetic-blend fallback.
- **L-pad-identity**: `ct.load(padding_mode=ZERO)` fills OOB lanes with
  `0`, which is the **wrong identity** for a downstream `min`/`max`/`prod`
  reduction. Padding mode must be chosen per downstream consumer.
- **L-pow2**: tile dims must be powers of two; shapes are compile-time
  constant; NumPy broadcasting; rank 1..3.

## Connector-name convention

The orchestrator renames node connectors to the `__`-prefixed names the
cutile tasklet bodies use (the cutile expansions never read the node's `_a`
etc. directly). The established map, taken from the existing bodies, is:

| node connector | cutile body name |
| --- | --- |
| `_src` (array)      | `__src` |
| `_dst` (array out)  | `__output` |
| `_a` Tile / `_b` Tile | `__rhs1` / `__rhs2` |
| `_a` Scalar / `_b` Scalar | `__const1` / `__const2` |
| `_idx_<k>`          | `__idx_<k>` (gather/scatter), `__idx<k>` (store arange) |
| `_mask`             | `__mask` |
| `_cond` / `_t` / `_e` (merge) | `__cond` / `__then` / `__else` |
| output tile         | `__output` |

Note the existing inconsistency: `TileStore`/`TileMaskGen` arange uses
`__idx<k>`/`__mask<k>` (no underscore before the digit) while
`TileGather`/`TileScatter` use `__idx_<k>`. This document **keeps the
existing names per node** to avoid churn and keep the current tests green;
a future cleanup may unify them, but that is out of scope here.

---

## Per-node assessment and design

### 1. TileLoad — `ExpandTileLoadCutile` EXISTS

**Correctness today**: structurally correct but has two issues against the
limitations.

- It correctly does **not** pass a `mask=` to `ct.load` (respects
  **L-load-nomask**); mask is deferred to the store. Good.
- **Issue A (L-pad-identity)**: it hardcodes `padding_mode=ct.PaddingMode.ZERO`
  unconditionally. For a load whose tile feeds a `min`/`max`/`prod`
  reduction, `0` is the wrong neutral element and corrupts the reduction
  of partial tiles. The padding mode must be selectable.
- **Issue B (unused `__mask` input)**: when `has_mask=True` the expansion
  declares a `__mask` input connector but the body never references it.
  A dangling input connector is at best dead and at worst a validation
  error downstream. The load body must not declare `__mask`.

**Designed body** (K = number of `widths`):

K=1, default padding:
```python
__pid0 = ct.bid(0)
__output = ct.load(__src, index=(__pid0,), shape=(8,), padding_mode=ct.PaddingMode.ZERO)
```

K>=2:
```python
__pid0 = ct.bid(0)
__pid1 = ct.bid(1)
__output = ct.load(__src, index=(__pid0, __pid1), shape=(4, 8), padding_mode=ct.PaddingMode.ZERO)
```

**Mask handling**: NONE on the load (L-load-nomask). `has_mask` must NOT
add a `__mask` input to the load tasklet (fix Issue B). Edge handling is
entirely via `padding_mode`.

**Padding-mode selection (fix Issue A)**: add a node property
`pad_mode: str` defaulting to `"ZERO"`, with allowed values mapping to the
enum: `ZERO → ct.PaddingMode.ZERO`, `NAN → ct.PaddingMode.NAN`,
`POS_INF → ct.PaddingMode.POSITIVE_INFINITY`, `NEG_ZERO →
ct.PaddingMode.NEGATIVE_ZERO`, `UNDETERMINED → ct.PaddingMode.UNDETERMINED`.
The orchestrator that fuses a TileLoad into a reduction chain sets it:
`+` → ZERO, `max` → NEG_INF (use `NEGATIVE_ZERO` is **not** -inf; cuTile
offers no `-inf` padding enum — see *cannot-faithfully-lower* note below),
`min` → `POS_INF`, `prod` → there is **no `1` padding enum**.

> **Limitation — partial-tile reduction of `max`/`prod`/`min` via load
> padding is NOT fully expressible.** cuTile's padding enum has ZERO,
> NEGATIVE_ZERO, NAN, POSITIVE_INFINITY — it has `+inf` (good for `min`)
> but **no `-inf` and no `1`** (needed for `max` and `prod`). Therefore a
> masked-load → reduce chain for `max`/`prod` **cannot** be handled by
> padding alone. The correct path is to load with `UNDETERMINED`/`ZERO`
> and let the **reduction's pre-select** (see TileReduce) install the
> right identity using the explicit `_mask`. So TileLoad's padding mode is
> a *fast path* for `+` (ZERO) and `min` (POSITIVE_INFINITY); all other
> reductions rely on the mask reaching TileReduce.

This means TileLoad's cutile expansion is **safe to emit now** (it is just a
contiguous block load); the only required fixes are dropping the dead
`__mask` input and making `pad_mode` configurable.

---

### 2. TileStore — `ExpandTileStoreCutile` EXISTS, CORRECT

**Correctness today**: correct against the limitations.

- Unmasked → `ct.store` (no mask; respects **L-store-nomask**).
- Masked → `ct.scatter` with per-lane `arange`-based indices and
  `mask=__mask`. This is exactly the prescribed way to do a lane-masked
  write (L-store-nomask + L-gs-mask).

**Designed body** (unchanged from current):

K=1 unmasked:
```python
__pid0 = ct.bid(0)
ct.store(__output, index=(__pid0,), tile=__src)
```

K=1 masked:
```python
__pid0 = ct.bid(0)
__idx0 = ct.arange(8, dtype=ct.int32) + __pid0 * 8
ct.scatter(__output, (__idx0,), __src, mask=__mask)
```

K=2 masked:
```python
__pid0 = ct.bid(0)
__pid1 = ct.bid(1)
__idx0 = ct.arange(4, dtype=ct.int32) + __pid0 * 4
__idx1 = ct.arange(8, dtype=ct.int32) + __pid1 * 8
ct.scatter(__output, (__idx0, __idx1), __src, mask=__mask)
```

**Mask handling**: at the store, via `ct.scatter(..., mask=__mask)`. This
is the single sink where the iteration mask is materialised into actual
suppressed writes — every other node leaves the mask alone and relies on
this.

**MUST VERIFY**: the `mask=` kwarg on `ct.scatter` (L-gs-mask is not 100%
confirmed). Fallback if absent: see *Gather/Scatter mask fallback* below.

---

### 3. TileBinop — `ExpandTileBinopCutile` EXISTS, CORRECT

**Correctness today**: correct against **L-elt-nomask**. The body is the
bare elementwise expression with no mask wrap; the mask flows to the store
(scatter). Symbol operands inline; `min`/`max` route to `ct.minimum` /
`ct.maximum`. The `has_mask=True` case correctly **drops** the `__mask`
input because the binop never reads it.

One subtlety to preserve: `&&`/`||` map to `&`/`|` (bitwise on bool
tiles), which is the correct cuTile spelling for elementwise logical ops
on tiles.

**Designed body**:

```python
__output = __rhs1 + __rhs2          # Tile op Tile
__output = __rhs1 * __const2         # Tile op Scalar (NumPy broadcast)
__output = __rhs1 + alpha            # Tile op Symbol (inlined)
__output = ct.minimum(__rhs1, __rhs2)
__output = ct.maximum(__rhs1, __rhs2)
```

**Mask handling**: NONE (L-elt-nomask). The mask is applied downstream at
the store. No change needed.

> Note: the binop cannot itself "select identity into masked lanes" for a
> reduction — that responsibility is centralised in TileReduce's
> pre-select step (see node 9). The binop stays mask-free.

---

### 4. TileUnop — `ExpandTileUnopCutile` EXISTS, CORRECT

**Correctness today**: correct against **L-elt-nomask** — bare elementwise
unary call, no mask. The doc statement "cutile may be MISSING" is **stale**:
`ExpandTileUnopCutile` is present (tile_unop.py) and maps every op to its
`ct.*` form. It does not declare a `__mask` input (the unop body has no
mask handling), which is correct.

**Designed body**:

```python
__output = -__rhs1
__output = ct.abs(__rhs1)
__output = ct.exp(__rhs1)
__output = ct.log(__const1)        # Scalar operand
__output = ct.sqrt(alpha)          # Symbol operand inlined
# also: sin cos floor ceil tanh -> ct.sin/ct.cos/ct.floor/ct.ceil/ct.tanh
```

**Mask handling**: NONE (L-elt-nomask); deferred to the store. No change
needed.

**MUST VERIFY**: that `ct.exp/ct.log/ct.sqrt/ct.sin/ct.cos/ct.floor/
ct.ceil/ct.tanh/ct.abs` all exist by those names in the installed package.
If a transcendental is missing, that specific op must raise
`NotImplementedError` in the cutile expansion rather than emit a call to a
nonexistent symbol (see *Raise policy*).

---

### 5. TileMerge — `ExpandTileMergeCutile` EXISTS, RISKY (depends on `ct.where`)

**Correctness today**: emits `__output = ct.where(__cond, __then,
__else)`. This is the **single most fragile** expansion because
`ct.where`'s existence is UNCONFIRMED (**L-where-unconfirmed**). If
`ct.where` is absent the emitted body references an undefined symbol and
fails at cuTile compile time, not at DaCe expansion time.

**Designed body — primary (`ct.where` present)**:

```python
__output = ct.where(__cond, __then, __else)
```

**Designed body — fallback (`ct.where` absent), arithmetic blend**:

`_cond` is stored as `0.0`/`1.0` (branch normalization typed it after a
float operand) or as `bool`. Cast the condition to the output dtype and
blend:

```python
__m = __cond.astype(__then.dtype)            # 1.0 where then, 0.0 where else
__output = __m * __then + (__m * (-1.0) + 1.0) * __else
```

or, equivalently and more readable (only if scalar-broadcast subtraction
from a Python literal is supported — verify):

```python
__m = __cond.astype(__then.dtype)
__output = __m * __then + (1.0 - __m) * __else
```

The arithmetic blend is exact for the `0.0/1.0` encoding and for `bool`
(`True→1`, `False→0`) on integer and float output dtypes. It is **not**
valid if `_then`/`_else` can contain `NaN`/`inf` in the *unselected* lane
(because `0.0 * inf = NaN` would leak). For float tiles where the
unselected branch may be non-finite, the fallback must instead use a
gather/scatter-style masked select — but that requires materialised
indices and is heavyweight; therefore:

**Decision**: TileMerge's cutile expansion chooses at expansion time:

1. Try `ct.where` (primary) — gated behind a `_CT_HAS_WHERE` capability
   flag resolved by probing the installed package once.
2. If `ct.where` is absent AND the output dtype is integer OR both
   branches are known finite → emit the arithmetic blend.
3. If `ct.where` is absent AND the output is float with possibly
   non-finite branches → **raise `NotImplementedError`** ("cuTile select
   without `ct.where` cannot safely blend possibly-non-finite branches;
   verify `ct.where` in the installed cuda-tile package").

Because the capability flag cannot be resolved on CI (no cuTile install),
the **CI-time behavior** is: emit the `ct.where` form as the documented
default (matching the current expansion and current docstring), and treat
the fallback as a runtime-selected alternative. The test asserts the
`ct.where` shape (primary) and separately asserts the fallback string
shape when the fallback code path is requested via an injected flag (see
test plan).

**Mask handling**: the surrounding iteration `_mask` is NOT applied at the
merge (matches the existing docstring) — it flows to the store/scatter.
The merge only consumes `_cond`/`_t`/`_e`.

---

### 6. TileMaskGen — `ExpandTileMaskGenCutile` EXISTS, CORRECT

**Correctness today**: correct and self-contained — it builds a boolean
tile from `ct.arange + __pid*W < ub` per dim, broadcasting and `&`-ing for
K>=2. It uses no masked primitive (it *produces* the mask), so no
limitation is violated.

**Designed body**:

K=1:
```python
__pid0 = ct.bid(0)
__offsets0 = ct.arange(8, dtype=ct.int32)
__mask0 = __offsets0 + __pid0 * 8 < (N_ub)
__output = __mask0
```

K=2:
```python
__pid0 = ct.bid(0)
__pid1 = ct.bid(1)
__offsets0 = ct.arange(4, dtype=ct.int32)
__mask0 = __offsets0 + __pid0 * 4 < (M_ub)
__offsets1 = ct.arange(8, dtype=ct.int32)
__mask1 = __offsets1 + __pid1 * 8 < (N_ub)
__output = ct.broadcast_to(__mask0[:, None], (4, 8)) & ct.broadcast_to(__mask1[None, :], (4, 8))
```

**Mask handling**: N/A (this node is the mask source). No change.

**MUST VERIFY**: `ct.arange(n, dtype=...)` and `ct.broadcast_to(tile,
shape)` signatures, and that comparison of an int tile against a scalar
upper-bound expression yields a bool tile.

---

### 7. TileGather — `ExpandTileGatherCutile` EXISTS, MOSTLY CORRECT

**Correctness today**: correct shape. 1D source → single index tile;
multi-dim → tuple of index tiles. Masked → `mask=__mask, padding_value=0`.

Two points:

- **L-pad-identity for gather**: `padding_value=0` is the right neutral
  for a downstream `+` reduction but wrong for `min`/`max`/`prod`. The
  gather's `padding_value` should be selectable the same way TileLoad's
  `pad_mode` is. Since `ct.gather`'s `padding_value` is an arbitrary
  scalar (not an enum), gather is actually **more** flexible than load —
  it *can* express `1`, `+inf`, `-inf`. So gather should accept a
  `pad_value` property (default `0`) and emit it.
- **L-gs-mask**: the `mask=` kwarg is unconfirmed (see MUST VERIFY).

**Designed body**:

1D source, unmasked:
```python
__output = ct.gather(__src, __idx_0, padding_value=0)
```

1D source, masked:
```python
__output = ct.gather(__src, __idx_0, mask=__mask, padding_value=0)
```

2D source, masked:
```python
__output = ct.gather(__src, (__idx_0, __idx_1), mask=__mask, padding_value=0)
```

**Mask handling**: at the gather, via `mask=` (lanes whose mask is false
read `padding_value`), AND via the automatic bounds mask. This is the one
read-side node where the mask is applied locally (gather is itself the
masked-read primitive), unlike TileLoad.

**Stride caveat**: the pure expansion supports `index_strides` (lane `l`
reads `_idx[c*l]`). cuTile `ct.gather` indexes the index tile directly and
has no per-lane stride concept. A non-unit `index_strides` therefore
**cannot** be expressed by `ct.gather` over the index tile as-is; the
strided index window must be pre-gathered/sliced into a contiguous index
tile before this node, or the cutile expansion must **raise
`NotImplementedError`** when `any(s != 1 for s in index_strides)`. The
safe-now behavior: emit `ct.gather` for unit strides; raise for non-unit.

---

### 8. TileScatter — `ExpandTileScatterCutile` EXISTS, CORRECT

**Correctness today**: correct — `ct.scatter(__output, idx, __src,
mask=__mask)`, 1D single index / multi-dim tuple. Matches L-gs-mask.

**Designed body**:

1D, unmasked:
```python
ct.scatter(__output, __idx_0, __src)
```

1D, masked:
```python
ct.scatter(__output, __idx_0, __src, mask=__mask)
```

2D, masked:
```python
ct.scatter(__output, (__idx_0, __idx_1), __src, mask=__mask)
```

**Mask handling**: at the scatter via `mask=` (masked/OOB lanes do not
store). This is correct per L-gs-mask.

**MUST VERIFY**: `mask=` kwarg (L-gs-mask). Fallback if absent below.

---

### 9. TileReduce — `ExpandTileReduceCutile` EXISTS, INCORRECT for `has_mask`

**Correctness today**: the unmasked path is correct (`ct.sum/prod/min/max`
with optional `axis`). The **masked path is broken**: when
`has_mask=True` the expansion declares a `__mask` input but the body does
**not** apply it — it just reduces `__src` directly. cuTile reductions
take no mask (**L-reduce-nomask**), so the masked lanes are wrongly
included in the reduction. The current docstring even says "the upstream
TileBinop must pre-zero inactive lanes", but (a) nothing enforces that, (b)
zero is only the correct identity for `+`, and (c) the dead `__mask` input
is emitted regardless.

**Designed strategy — pre-select identity into masked lanes (L-reduce-nomask
+ L-pad-identity)**:

The reduce expansion must, when `has_mask=True`, first build a
masked-source tile where inactive lanes hold the op's identity, then
reduce that. Identities: `+ → 0`, `* → 1`, `min → +inf`, `max → -inf`.

Primary (if `ct.where` confirmed):
```python
__masked_src = ct.where(__mask, __src, IDENT)
__output = ct.sum(__masked_src, axis=AX)        # or ct.prod/ct.min/ct.max
```

Fallback (no `ct.where`) — only valid for `+` and `*` via arithmetic, and
for `min`/`max` only via a finite-blend that is unsafe for non-finite
inputs:

```python
# op == '+' : zero out inactive lanes (0 is + identity)
__m = __mask.astype(__src.dtype)
__output = ct.sum(__m * __src, axis=AX)

# op == '*' : set inactive lanes to 1 (mult identity)
__m = __mask.astype(__src.dtype)
__output = ct.prod(__m * __src + (1.0 - __m), axis=AX)
```

For `min`/`max` the fallback needs `IDENT = +inf/-inf` injected only into
masked lanes; without `ct.where` this requires
`__src * __m + IDENT * (1 - __m)`, which produces `NaN` when `__src`
contains `inf` in a masked lane (`inf * 0`). Therefore:

**Decision for TileReduce**:

1. `op in {+, *}` and `has_mask` → safe to emit the arithmetic fallback
   even without `ct.where` (no non-finite hazard for the identity, modulo
   pre-existing inf in *active* lanes which is the user's data, not ours).
2. `op in {min, max}` and `has_mask` → require `ct.where`. If `ct.where`
   is unavailable, **raise `NotImplementedError`** ("masked min/max tile
   reduction needs `ct.where` to inject ±inf into masked lanes; cuTile
   reductions take no mask (L-reduce-nomask) and the arithmetic blend is
   unsafe for non-finite data").
3. `has_mask=False` → emit the plain `ct.sum/prod/min/max` (current,
   correct).

In all `has_mask` cases the dead `__mask` input must actually be consumed
by the pre-select (fixing the current dead-connector bug). The `__output`
remains the reduced tile.

**Designed body** (has_mask=False, unchanged):
```python
__output = ct.sum(__src)              # full reduction
__output = ct.max(__src, axis=1)      # single-axis
```

**Designed body** (has_mask=True, `op=+`, primary with where):
```python
__masked_src = ct.where(__mask, __src, 0)
__output = ct.sum(__masked_src, axis=1)
```

**MUST VERIFY**: `ct.where`; the `axis`/`keep_dims` kwarg spellings; whether
`ct.sum(x, axis=k)` drops the axis (the pure path produces a kept-dim
tile, so the cutile output shape must match — if cuTile keeps dims, pass
`keep_dims=False`).

---

## Cross-cutting: gather/scatter `mask=` fallback (L-gs-mask unconfirmed)

If the installed package's `ct.gather`/`ct.scatter` do **not** accept a
`mask=` kwarg, the mask must be folded into the indices:

- **scatter**: redirect masked-out lanes to an OOB index so the automatic
  bounds mask drops them (relies on `check_bounds=True`):
  ```python
  __safe_idx = ct.where(__mask, __idx_0, -1)   # -1 is OOB -> not stored
  ct.scatter(__output, __safe_idx, __src)
  ```
  (itself needs `ct.where`; if neither `mask=` nor `ct.where` exists,
  **raise `NotImplementedError`**.)
- **gather**: same redirect to an OOB index gives `padding_value` in
  masked lanes:
  ```python
  __safe_idx = ct.where(__mask, __idx_0, -1)
  __output = ct.gather(__src, __safe_idx, padding_value=0)
  ```

These fallbacks are documented but **not** the default emission; the
default keeps the `mask=` kwarg per the strong indication in the release
notes.

---

## Raise policy: emit-now vs raise `NotImplementedError`

Reconciling the user's "cuTile implementations should raise a
NotImplementedError for now" with the fact that most expansions already
exist and are correct:

**Safe to emit now (no raise):**

- TileLoad (block load; only fix dead `__mask` + add `pad_mode`).
- TileStore (store / masked-scatter).
- TileBinop (bare elementwise; mask deferred).
- TileUnop (bare elementwise; mask deferred) — *except* any op whose `ct.*`
  symbol is unverified, which raises.
- TileMaskGen (pure mask construction).
- TileGather / TileScatter with unit `index_strides` and `has_mask`
  expressed via the `mask=` kwarg (default).
- TileReduce with `has_mask=False`, and `has_mask=True` for `+`/`*`.

**Must raise `NotImplementedError` (cannot be faithfully lowered with
current confirmed primitives):**

- TileMerge when `ct.where` is absent AND output is float with possibly
  non-finite branches.
- TileReduce `has_mask=True` with `op in {min, max}` when `ct.where` is
  absent.
- TileGather/TileScatter with non-unit `index_strides` (no per-lane stride
  in `ct.gather`).
- TileGather/TileScatter `has_mask=True` when neither the `mask=` kwarg
  nor `ct.where` (for the OOB-index fallback) is available.
- TileLoad masked-tail feeding a `max`/`prod` reduction relying on load
  padding alone (no `-inf`/`1` padding enum) — but this is handled by
  routing the mask to TileReduce's pre-select instead, so TileLoad itself
  does not raise; it just must not be asked to provide the wrong identity.

The raise must carry a precise message naming the missing primitive and
the limitation (e.g. `"TileReduce: masked min/max needs ct.where
(L-reduce-nomask); verify ct.where in the installed cuda-tile package"`).

Because CI has no cuTile install, capability probing returns "assume
present" so the default (richest) form is emitted and string-asserted; the
raise paths are exercised by explicitly forcing the capability flag off in
a unit test (see below).

---

## Test plan (consistent with `test_cutile_expansions.py`)

The existing file tests TileLoad, TileStore, TileBinop, TileMaskGen by
calling `cls.expansion(node, state, sdfg)` and asserting substrings of
`tasklet.code.as_string` plus `ast.parse(body)` validity, with
`language == Python`. New/changed tests follow the same `_expand_cutile`
helper and post-round-trip assertions (trailing tuple commas dropped,
binop rhs paren-wrapped). The cuTile runtime is never executed.

**TileLoad (changed)**
- `pad_mode="ZERO"` default → body contains `padding_mode=ct.PaddingMode.ZERO`.
- `pad_mode="POS_INF"` → `padding_mode=ct.PaddingMode.POSITIVE_INFINITY`.
- `has_mask=True` → body has **no** `__mask` reference and the tasklet has
  no `__mask` input connector (assert `"__mask" not in tasklet.in_connectors`).

**TileStore (unchanged, keep current 3 tests)**
- unmasked → `ct.store(...)`, no `ct.scatter`.
- masked K=1 → `ct.scatter(__output, (__idx0,), __src, mask=__mask)`.
- masked K=2 → two `arange` index tiles + 2-tuple scatter.

**TileBinop (unchanged, keep current 4 tests)**
- bare op, no `ct.where`; masked still bare; symbol inlines; `min` →
  `ct.minimum`.

**TileUnop (new)**
- `op="abs"` → `ct.abs(__rhs1)`; `op="neg"` → `-__rhs1`.
- `op="exp"` → `ct.exp(__rhs1)`; scalar operand → `ct.log(__const1)`;
  symbol operand → expression inlined.
- assert no `__mask` reference for `has_mask=True`.

**TileMerge (new)**
- primary: body == `ct.where(__cond, __then, __else)`; parses; Python.
- fallback (force capability flag off via a kwarg/monkeypatch): body
  contains `__then`, `__else`, a multiply and an add (`* __then`,
  `* __else`), and no `ct.where`; parses.
- raise path: float output + non-finite-possible + no `ct.where` →
  `with pytest.raises(NotImplementedError)`.

**TileMaskGen (unchanged, keep current 2 tests)**
- K=1 single mask, no `&`; K=2 `broadcast_to` + `&`.

**TileGather (new)**
- 1D unmasked → `ct.gather(__src, __idx_0, padding_value=0)`.
- 1D masked → `... mask=__mask, padding_value=0`.
- 2D masked → `(__idx_0, __idx_1)` tuple form.
- `pad_value=1` → `padding_value=1`.
- non-unit `index_strides=(2,)` → `pytest.raises(NotImplementedError)`.

**TileScatter (new)**
- 1D unmasked → `ct.scatter(__output, __idx_0, __src)` (no `mask=`).
- 1D masked → `... __src, mask=__mask`.
- 2D masked → tuple form.

**TileReduce (new + fix)**
- `has_mask=False`, full → `ct.sum(__src)`; axis → `ct.sum(__src, axis=1)`.
- `has_mask=False`, op=max → `ct.max(...)`.
- `has_mask=True`, op=`+`, primary → contains `ct.where(__mask, __src, 0)`
  then `ct.sum(...)`; `__mask` IS referenced (assert it is consumed).
- `has_mask=True`, op=`+`, fallback (flag off) → `__mask.astype` + `* __src`
  + `ct.sum`.
- `has_mask=True`, op=`min`, no-`where` → `pytest.raises(NotImplementedError)`.

Each new test asserts `_assert_parses_as_python(body)` and
`lang == dace.dtypes.Language.Python`, mirroring the existing helpers.

---

## MUST VERIFY against the installed `cuda-tile` package

Before any of these expansions are trusted at runtime, confirm in the
actually installed package (the design assumes presence on CI but must be
checked once a GPU/cuTile environment exists):

1. **`ct.where(cond, a, b)` exists** and its arg order. (Blocks: TileMerge
   primary, TileReduce masked min/max, gather/scatter OOB-index fallback.)
   Fallback if absent: arithmetic blend (TileMerge int/finite, TileReduce
   `+`/`*`); raise for min/max and non-finite float merge.
2. **`ct.gather(..., mask=...)` and `ct.scatter(..., mask=...)` accept the
   `mask=` kwarg** (L-gs-mask: strongly indicated, not confirmed).
   Fallback if absent: fold mask into indices via OOB redirect (needs
   `ct.where`).
3. **`ct.gather` `padding_value` is an arbitrary scalar** (assumed; lets
   gather express `1`/`±inf` identities that load's enum cannot).
4. **`ct.load` `padding_mode` enum members**: ZERO, NEGATIVE_ZERO, NAN,
   POSITIVE_INFINITY, UNDETERMINED — and that there is **no** `-inf`/`1`
   member (forces max/prod identity to TileReduce, not load padding).
5. **Reduction kwargs**: `ct.sum/prod/min/max(x, axis=..., keep_dims=...)`
   spelling and whether the reduced axis is dropped or kept (the pure path
   keeps non-reduced dims; cutile output shape must match — set `keep_dims`
   accordingly).
6. **`ct.arange(n, dtype=ct.int32)` and `ct.broadcast_to(tile, shape)`**
   signatures (TileMaskGen, TileStore masked indices).
7. **Transcendental/unary names**: `ct.abs/exp/log/sqrt/sin/cos/floor/
   ceil/tanh` and `ct.minimum/ct.maximum` all exist by those names; any
   missing one must raise in the corresponding cutile expansion.
8. **`ct.bid(k)` block-id** signature and that `index=` to `ct.load`/
   `ct.store` is a tuple of block ids (not element offsets).
9. **`.astype(dtype)` on a tile** (used by the `ct.where`-free fallbacks).

## REFINEMENT (2026-05-27, user): implicit-padding is unsafe -> scalar width-1 remainder

**Problem:** cuTile `ct.load`'s implicit `padding_mode` pads OOB lanes; that
padded value can corrupt downstream compute (it is not a true mask), and cuTile
does not support per-lane masking on most ops. Relying on padding/masking for the
boundary can produce INCORRECT programs.

**Fix:** the cuTile lowering must NOT use a padded/masked W-wide tile for the
remainder. Split like `scalar_postamble`:
- **Main (divisible) region:** full W-wide cuTile tiles, no boundary mask (fast path).
- **Remainder region:** a SCALAR loop with **width-1 (single-element) tile nodes**
  — each tail element is a `widths=(1,)` cuTile op (no padding, no mask). This is
  the "scalar variant of cutile": a cutile expansion path for width-1 tiles that
  emits single-element scalar ops.

**cuTile test matrix (structural; cuda.tile not installed -> not executed):**
1. **no-remainder** (trip % W == 0): full tiles; test branch lowering via
   `fp_factor` AND `masked`; the masked tile NODE is still exercised (mask on the
   COMPUTE / merge select, not on the load padding).
2. **remainder** (trip % W != 0): main W-tiles + **scalar width-1 remainder lib
   nodes** for the tail; AND a masked-remainder variant.

So cuTile's masking lives only on the compute/merge node (where a select can be
synthesized), never on the load/store boundary — the boundary is handled by the
width-1 scalar tail. NEXT cuTile slice: add the width-1 scalar cutile expansion +
the test matrix above.

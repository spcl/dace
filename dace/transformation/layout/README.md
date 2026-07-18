# Layout transformations

Change an array's physical layout (shape, dim order, element type); propagate the change through every memlet, descriptor, nested SDFG, interstate edge, and library node; pick the fastest layout by measured brute-force sweep.

SC26 paper: *"Why Schedule Transformations Are Not Enough: Layout Optimizations for Block-Granular I/O."* Five primitives: **Pad, Permute, Block, Shuffle, Zip/Unzip.**

## Quick Start

**(a) One transform.** `add_permute_maps=True` wraps — relayouts the input, its own layout untouched; `False` rewrites in-place:

```python
from dace.transformation.layout import PermuteDimensions, prepare_for_layout

prepare_for_layout(sdfg)                                    # normal form first
PermuteDimensions(permute_map={"A": [1, 0]},
                  add_permute_maps=True).apply_pass(sdfg, {})
```

**(b) Brute-force sweep over two kernels, static layouts only:**

```python
import numpy
import dace
from dace.transformation.layout import sweep, best, permutation_candidates, block_candidates

N = dace.symbol("N")

@dace.program
def add2d(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]

@dace.program
def scale2d(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] * 2.0

def eval_shape(desc, n):
    return tuple(int(dace.symbolic.evaluate(s, {N: n})) for s in desc.shape)

def bench(prog, inputs, n, oracle):
    def make(apply):
        def m():
            sdfg = prog.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg
        return m
    # static layouts only (id + permutations + block factors); shuffle/indirection excluded (dynamic-data)
    layouts = {**dict(permutation_candidates("A", 2)), **dict(block_candidates("A", 2, factors=(8, 16)))}
    candidates = {name: make(apply) for name, apply in layouts.items()}
    arrays = {name: numpy.random.rand(n, n) for name in inputs}
    def run(sdfg):
        kwargs = {name: arrays[name].reshape(eval_shape(sdfg.arrays[name], n)).copy() for name in inputs}
        kwargs["C"], kwargs["N"] = numpy.zeros(eval_shape(sdfg.arrays["C"], n)), n
        sdfg(**kwargs)
        return {"C": numpy.asarray(kwargs["C"]).reshape(n, n).copy()}
    return best(sweep(candidates, run, {"C": oracle(arrays)}, reps=5, warmup=1, isolate=True))

bench(add2d, ["A", "B"], 16, lambda a: a["A"] + a["B"])
bench(scale2d, ["A"], 16, lambda a: a["A"] * 2.0)
```

## Design

- **Normal form: packed C-contiguous.** Every array is packed C-contiguous before layout work starts, so a layout change is a real shape / dim-order / dtype change, never a stride trick: Pad grows a dim, Permute reorders, Block adds dims, Zip changes element type.
- **Layout algebra.** A layout is a tuple of mixed-radix digits `(dim, stride, extent)`; the 7 ops compose and cancel structurally (`Block ∘ Unblock = id`, `Permute ∘ Permute⁻¹ = id`, `Pad + Pad`, `Unzip ∘ Zip = id`) — sympy only touches the final index, never the algebra; identity is a tuple compare.
- **Two apply modes** (generalizes Permute's `add_permute_maps`): in-place rewrites descriptor, memlets, nested SDFGs, and interstate edges; wrap keeps the input's layout and drops a `LayoutChange` node at the boundary — chained changes to one array fold into a single node (`simplify_ops`).
- **Optimizer = brute force, global only.** No cost model in the loop: enumerate candidates, apply, compile, verify against a numpy oracle, time, and keep the fastest correct one. Transforms, algebra, and `LayoutChange` are the deliverable — the picker is just a sweep.
- **Multi-nest: one trajectory per array.** Different nests can want different layouts for the same array, so global assignment picks one layout per array — or a trajectory with paid relayouts at state boundaries — via a per-array Viterbi DP checked against a brute-force oracle. Sizes must be concrete (`specialize` first); v1 is flat line graphs, Permute/Block, CPU only — branches, LoopRegions, and symbolic extents refuse loudly.

Pipeline: `prepare_for_layout` → apply primitive(s) → `normalize_schedule_for_layout` → sweep.

## Modules
### DSL — `dace/libraries/layout/`

| file | role |
|---|---|
| `algebra.py` | `Digit`+`LayoutMap`; 7 op dataclasses (`Permute`, `Block`, `Unblock`, `Pad`, `Shuffle`, `Zip`, `Unzip`); `compose_ops`, `simplify_ops`, `is_identity`, `physical_index_exprs`. Pure Python, no compilation. |
| `layout_change.py` | `LayoutChange` libnode: carries a fused op sequence, `expand()`s one relayout. Impls: pure copy map, cuTENSOR, HPTT. |
| `lowering.py`, `shuffle.py` | expansion helpers + shuffle-bijection registry (`register_shuffle`). |

### Primitives — `dace/transformation/layout/`

| file | primitive |
|---|---|
| `permute_dimensions.py` | **Permute** — reorder dims. Template for the rest. |
| `split_dimensions.py` / `unblock_dimensions.py` | **Block** / **Unblock** — split a dim into outer/inner (`[i] → [i//b, i%b]`) + inverse. |
| `untile_loops_and_blocks.py` | collapse a manually-tiled loop nest and unblock its materialized dim. |
| `pad_dimensions.py` | **Pad** — grow extent, keep packed strides (pad cells never accessed). |
| `zip_arrays.py` / `unzip_arrays.py` | **Zip / Unzip** — SoA ⇄ AoS. Homogeneous fields → plain `[*S, F]` array (tasklets untouched); heterogeneous → `dtypes.struct` AoS. |
| `split_array.py` | **Unzip by symbol** — split a symbol-extent dim into named arrays. |
| `shuffle_elements.py` | **Shuffle** — renumber a dim by a bijection σ (RCM / Eytzinger / argsort). |
| `indirect_access.py` | detect data-dependent `A[idx[i]]` gather / `y[idx[i]] += …` scatter (sparse-layout carrier). |

### Preprocessing + schedule alignment + libnodes

| file | role |
|---|---|
| `prepare.py` | `prepare_for_layout` — normal-form contract (canonicalize, parallelize, expand nested inputs, drop views, pack strides). |
| `block_aware_map_tiling.py` / `normalize_schedule.py` | tile the schedule to a block factor before Block (avoids spurious `%`/`int_floor`); the dual re-tiles to the layout's block width after applying it. |
| `rewrite_libnodes.py` | expose an operand's layout to a libnode: `GemmToTensorDot`, `transform_einsum`, copy/memset passthrough. |
| `select_lowering.py` | device-driven choice of relayout expansion (`pure` / cuTENSOR), right before compile. |

### Global assignment (multi-nest)

| file | role |
|---|---|
| `externalize.py` | one nest → standalone runnable SDFG (deterministic inputs, per-nest oracle). |
| `line_graph.py` | one kernel per state; flat line graph; refuses branches / LoopRegions loudly. |
| `relayout_boundary.py` | insert relayout states (parallel `LayoutChange` nodes) on line-graph boundaries. |
| `apply_assignment.py` | apply a chosen trajectory end to end: segment clones, entry/exit conversions, bit-exact. |
| `nest_eval.py` | externalize → run reference → time candidates → rank. |
| `global_assign.py` | per-array Viterbi DP + brute-force oracle + greedy baseline + conflict report. |
| `assignment_costs.py` | cost table: LogGP model (`model_costs`) or measurement (`eval_costs`); tested but parked, sweep still decides. |
| `timing.py` | time the COMPUTE region, excluding the bracketing relayout copies. |
| `brute_force.py` | sweep engine — compile → verify → time → rank; `best()` picks fastest correct. |
| `isolation.py` | run each candidate in a forked child (segfault-safe); OpenMP pool torn down first (CPU only). |

## Docs

- [GLOBAL_LAYOUT_DESIGN.md](GLOBAL_LAYOUT_DESIGN.md) — multi-nest global assignment.
- [cost_model/LOGP_DESIGN.md](cost_model/LOGP_DESIGN.md) — LogGP cost model design.
- [cost_model/ACCESS_PATTERNS.md](cost_model/ACCESS_PATTERNS.md) — access-pattern classification.

# Global layout over multiple loop nests

Choose ONE layout per array across a whole program of several loop nests — or a trajectory of
layouts with paid relayouts between nests — plus the canonical schedule per nest that exploits it.
The k17 experiment (per-op greedy vs global vs oracle) promoted from a numpy toy into the SDFG
pipeline. Top-level suite overview: [README.md](README.md).

## High-level design

- **Program shape: a FLAT line graph of kernels.** One kernel per state; relayouts only on state
  boundaries; the per-array DP walks the states in order. Branches, DAGs, and LoopRegions refuse
  loudly — the audit confirmed nested loops are not needed for the paper claim (greedy vs global vs
  oracle on 2–3 nest programs). The nested-loop design is kept below as a documented, deferred
  extension.
- **Layouts tried: Permute and Block, alone or composed.** Shuffle is indirect-access only (replay
  path, not v1). Zip is multi-nest by definition (fields co-accessed across nests) — v1 detects
  candidates, does not apply them.
- **Concrete sizes only.** Free size symbols are specialized from a config (`sdfg.specialize`)
  BEFORE candidate generation; any remaining symbolic extent refuses loudly. Both the schedule
  passes (which no-op on symbolic strides) and the DP comparisons need numbers.
- **Two ranking modes over the SAME machinery:** cost-model mode (tier-0 counts → tier-2
  `nest_memory_time` on undecided pairs) and eval mode (externalize, compile, time). The cost
  PROVIDER is pluggable; the assignment algorithm does not care where the numbers came from.
- **Output = the k17 triad:** per-op greedy / global DP / brute-force oracle, plus a conflict
  report naming every disagreeing array, its per-nest preferences, and the chosen resolution.

Preprocessing contract: canonicalize + maximal map fusion already ran (so splitting destroys no
fusion opportunity), THEN one kernel per state, THEN relayouts only on boundaries. After the split
the pipeline must never call full `canonicalize()`/`simplify` again — simplify re-fuses the split
states (the schedule stage runs standalone on externalized copies instead).

## Pipeline

```
canonicalize + max fusion          (exists)
  -> kernel-per-state split         (A2)
  -> line-graph extraction          (A3)
  -> per-nest candidate layouts     (B1)  permutations from access strides; block factors {8,16,32}
  -> canonical schedule per (nest, layout)  (B2)  loop-permute to min stride + tile to match blocks
  -> per-nest cost                  (B2 model / B3 eval)
  -> global assignment              (C1 DP / C2 brute force)
  -> apply: layout + relayout states + schedules   (A4 + A5 + existing passes)
  -> verify bit-exact vs the untransformed program
```

## The assignment problem

Arrays `a`, nests `k_1..k_n` in line order, candidate layouts `L(a)`. An assignment gives each array
a layout PER SEGMENT; changes are allowed only on state boundaries; each change costs a relayout
(`2*S*G` streaming bound, or its measured/expanded cost).

**Per-array DP (Viterbi).** Node weight = the array's access cost in nest `k` under layout `l`; edge
weight between consecutive nests = 0 if the layout is kept, relayout cost if changed. Shortest path
= the array's optimal layout trajectory, `O(n·|L|²)` per array. "No intermediate changes" is the
same DP with infinite edge cost on changes — both regimes come free (`per_array_dp(allow_changes=)`).

**Scoring convention (schedules are SHARED; weights must be jointly realizable).** A nest has ONE
schedule, but the DP scores arrays independently, so per-array canonical schedules can be jointly
unrealizable. Convention: when scoring `(array a, layout l)` in nest `k`, all OTHER arrays hold
their baseline layouts and the schedule is derived JOINTLY from all the nest's accesses. After the
DP composes an assignment, the schedule is RE-DERIVED from the composed layouts and the program
re-scored (and, in eval mode, re-timed) — the DP's number is a proposal, the composed re-score is
the result. Candidate identity stays `{nest}_{tag}`: the schedule is a deterministic FUNCTION of the
layout tuple, never a free axis. The identity/unblocked candidate is enumerated FIRST and wins ties
(enumeration order is load-bearing — law, not accident).

**Separability caveat (KNOWN approximation).** The DP treats arrays independently, but the tier-2
nest cost couples them through `max(B·G, M·L/C)` — the max of sums is not the sum of per-array
maxes. Per-array DP is exact when one term binds for the whole nest (the usual case), an
approximation otherwise. Mitigation (mandatory): after the DP picks an assignment, re-score the
composed program with the full nest model and against the brute-force oracle; if composition ever
loses to a brute-force winner, the coupling matters and the DP needs a coupled refinement (OPEN).

**Conflict resolution = the DP doing its job.** An array whose nests disagree gets either the
globally cheaper single layout (access-cost difference beats relayout) or a trajectory with paid
changes (relayout beats carrying the wrong layout) — the `uses·delta ≥ 2S/rate` break-even decided
per boundary by the shortest path.

**Loops (nested line graphs — DEFERRED).** A boundary inside a loop body pays its relayout every
iteration (weight × trip count), and layout at loop exit must equal layout at loop entry (the back
edge is a consistency constraint). Solve by enumerating the loop-carried layout (|L| options) and
running the inner DP for each, recursively. Kept as documented extension, not built.

## Sub-design (per component)

Tasks A (infrastructure) / B (candidates + scoring) / C (assignment) / D (validation). Status: ✅
landed with fixture-driven tests, ⏸ deferred.

| task | module | what it does | status |
|---|---|---|---|
| A1 | `externalize.py` | state → fresh runnable SDFG via `SDFGCutout.singlestate_cutout`; transients → deterministic random, non-transients → config. | ✅ |
| A2/A3/A6 | `line_graph.py` | iterated `state_fission` to one kernel per state (to fixpoint); flat `line_graph` with loud refusals (branches, LoopRegions, non-map work); `check_kernel_per_state` guard. | ✅ |
| A4 | `relayout_boundary.py` | boundary state of parallel `LayoutChange` nodes; `add_layout_change(create_output=False)` so conversions into existing descriptors never clobber them. | ✅ |
| A5 | `apply_assignment.py` | apply a trajectory end to end — segment clones (`B__seg1_perm10`), per-segment rewrite via the extracted `rewrite_state_for_permute`, entry/exit conversions. PERMUTE trajectories; Block refused. | ✅ |
| B1 | (in `assignment_costs`) | per array/nest, permutations that make an observed access contiguous; block factors {8,16,32}, ≤2 dims (cap 3), only extent-dividing power-of-two factors ≥4× the factor. v1-lite enumerates d! for d≤3. | ⏸ prune |
| B2 | `assignment_costs.py` | model provider: externalize → in-place permute on the copy → `count_loop_nest` → per-array tier-2 `nest_memory_time`. | ✅ |
| B3 | `nest_eval.py` + `assignment_costs.py` | eval provider: `evaluate_nest` medians via `sweep`; identity-first tie-break enforced. | ✅ |
| C1/C2/C3 | `global_assign.py` | provider-agnostic `AssignmentCosts`; Viterbi DP (both edge regimes), capped brute-force oracle, greedy baseline, conflict report. DP == oracle on randomized tables. | ✅ |
| D1 | `tests/.../multinest_programs.py` | conflict2 / conflict3 / agree2 fixtures. | ✅ |

Candidate space pruning "no silent caps": every prune is logged. Cross-family pruning (blocked vs
unblocked) is FORBIDDEN — the continuous-fraction tier-0 counts overcount sub-block tiles (~2×),
biasing counts against exactly the blocked+tiled candidates, and tier-2 shares those counts, so
escalation is no escape; cross-family pairs survive to eval. Brute force is guarded: compute
`∏_a |L(a)|^segments` up front and refuse above ~1e6 assignments (a stated cap, not a vibe).

## Schedule canonicalization per (nest, layout)

Layout first, then schedule, then count — all on the EXTERNALIZED single-nest copy with concrete
sizes baked into descriptor strides (the whole-SDFG passes must never run on the multi-nest program):

1. **Permute candidates**: rewrite the copy's descriptor strides to the permutation, then reorder
   the single multi-param map via `MapDimShuffle` for minimal innermost stride. (`MinimizeStride-
   Permutation` as-is cannot: it refuses symbolic strides and breaks on multi-param maps — extending
   it is future work; MapDimShuffle on the copy does it meanwhile.)
2. **Block candidates**: `BlockAwareMapTiling` with the factors ON THE COPY, then `SplitDimensions` —
   its perfect-block-match path emits clean tile/offset indices with no residual `%`/`int_floor`.
   ("Virtual blocked strides on the 2-D access" is incoherent — a blocked layout is higher-rank.)
3. **THEN `count_loop_nest`** on the result — the schedule changes the counts.

Exactly one canonical schedule per (nest, layout) — blocked → matched tiled schedule, unblocked →
min-stride permutation — so `{nest}_{tag}` uniquely identifies a candidate; no schedule axis exists.

## Eval discipline

Externalization: a kernel state is self-contained after the split, so deepcopy it into a fresh SDFG
+ its descriptors. No nest-forge dependency (its emit/arena machinery solves cross-compiler lanes, a
different problem). Used for eval-mode timing and per-nest correctness oracles.

Box rules: builds sequential (`-n1`, 8 GB cap), each variant in its OWN `DACE_default_build_folder`,
compiled kernels run FORKED, timing strictly serial per device on a quiet box. Cache keyed by (nest
content hash, layout, schedule).

The sampling loop must not be malloc-bound:

- **Buffer arena, allocated ONCE per array, with a PRISTINE SNAPSHOT.** Every nest writes something,
  so naive reuse verifies candidate N against buffers candidate N−1 clobbered. Two allocations: the
  pristine initialized snapshot (read-only after init) and the working set; before each candidate,
  restore working from pristine (one streaming memcpy — no malloc, and itself a live
  bandwidth/relayout-shaped measurement), relaid to the candidate's layout by the `LayoutChange`
  expansion (timed: it DOUBLES as the DP's relayout cost). Oracle outputs computed once from pristine.
- **`sweep` is the one engine** (B3/C2 extend it) — compile-verify-time-rank; the arena staging hook
  and external timer are additions, not a second engine.
- **Python stays ONE process; the KERNEL is what gets spawned.** The kernel under test runs fully
  multithreaded (saturated-multicore is the model's premise), so during its slot it owns every core;
  any concurrent process would steal cores from the measurement. No pools, no workers, no IPC.

**Fork crash guard (`sweep(isolate=True)`, `isolation.py`).** The parent compiles each candidate,
then runs+verifies+times it in a forked child — a segfault/runaway is a non-viable result, not a
dead campaign. `os.fork` shares the compiled `.so` and the numpy arrays copy-on-write (no pickling);
the child returns a JSON verdict over a pipe, timeout → SIGKILL. The OpenMP-fork deadlock (a live
parent pool blocks the child's first parallel region; libgomp — gcc's default — installs no
`pthread_atfork` handler) is solved by `pause_openmp_pools()` calling OpenMP-5.0
`omp_pause_resource_all(soft)` on every already-loaded runtime BEFORE the fork, tearing the pool down
(measured on libgomp as a genuine 16→1 teardown; the pool is a cache and respins). Nest-forge's
proven pattern, PORTED not imported (dace must not depend on nest-forge — wrong direction; the probe
+ fork are ~40 lines of stdlib). CPU only — a CUDA context cannot survive fork, so `isolate` + GPU
is refused and the GPU campaign relies on the attempt log.

**Measurement protocol.** ≥2 warmup runs (excluded), then ~10 timed reps; report the MEDIAN plus the
spread `(max−min)/min`. Median, not min, for whole kernels (a kernel's distribution is two-sided).
Spread above a threshold (default 10%) flags the sample CONTENDED — kept but marked untrusted, the
refuse-don't-guess doctrine.

**Externalization validity caveat.** Timing a nest in isolation destroys inter-nest cache reuse.
Valid when working sets ≫ LLC (reuse across nests negligible); detect (sum of footprints vs LLC) and
warn otherwise.

## Invariants (each a confirmed bug — do not regress)

`apply_assignment` liveness planner walks segments carrying the LIVE holder:

- Entry conversions are decided at the segment's first TOUCHING kernel, NOT `kernels[start]` — an
  untouched first kernel must not suppress them.
- An untouched segment stays unmaterialized (no clone, no conversion); a later same-physical-layout
  segment ALIASES onto the live holder (no conversion, no clone).
- The entry skip needs PROOF of full production — `writes_cover_array` reads the INNER write memlet
  (distinct-param permutation over full map ranges). The propagated OUTER memlet is over-approximated
  to the full array and is UNSOUND for coverage — never consult it.
- A WCR write counts as a READ: `reads_before_write` scans the state's edges for `wcr != None`.
- The exit conversion fires only when the original does not already hold the post-last-write value —
  a later restoring entry conversion suppresses it (the double-charge case).
- A copy with ONE relaid operand becomes transposing → `TensorTranspose`, via the shared
  `permute_dimensions` bookkeeping (`note_copy_side` / `retranspose_copies`, extracted to module
  level); a sub-region copy refuses. Else a silent `C = Aᵀ` on square arrays.
- A NestedSDFG receiving a reassigned array through a SPANNING memlet (`spanned_dims > 0`) refuses
  loudly; a unit-element (scalar-slice) edge is layout-transparent and passes.

Objective (`trajectory_cost`, `per_array_dp`) is liveness-aware: `AssignmentCosts` carries
`entry_conversion_needed` + `last_write_kernel` (filled by both providers via `liveness_facts`, the
same predicates as apply). The cost prices the entry edge, an exit surcharge at the last write, and
ONE FREE switch back to identity after it (that switch IS the exit conversion moved onto the
boundary — so `[perm10, identity]` prices one conversion, not two, and oscillation is not free). The
DP tracks a `(layout, restored)` state and still equals the oracle. The single-layout regime pays
entry/exit too (fair single-vs-trajectory comparison). Residual approximation (docstring): every tag
switch is priced even where apply would alias/skip — optima are unaffected, D3 re-scores.

Timing: the relayout-boundary classification is the SYMMETRIC DIFFERENCE `copy_in ^ copy_out` — a
copy state on BOTH sides IS the compute (an externalized transpose nest) and must be timed, else the
identity candidate ranks last at `inf`. Every rep must produce a FRESH instrumentation report and a
full sample set (stale/missing reports are hard errors, not silent shrinkage). `best()` widens ties
to a noise window (default = the contended threshold) resolved by enumeration order, so identity-
first is not dead code for measured sweeps.

Structure: a bare access-node→access-node copy state COUNTS as work (`state_does_work` scans edges)
and is refused as non-map — it was invisible to liveness. `kernel_per_state` runs to fixpoint over
the states `state_fission` creates and raises honestly on an undraggable shared sink (fission drags
the second nest along) instead of mis-reporting a split. `relayout_on_boundary` preserves
start-block status when inserting before the start state (the boundary was unreachable dead code).

Costs: `eval_costs` keeps contended medians but MARKS them — `AssignmentCosts.untrusted` carries the
`(array, kernel, tag)` set, a loud warning names them, and the conflict report flags rows whose
chosen trajectory consumed one.

Fixture design (the load-bearing lesson): nests survive maximal fusion only via NON-pointwise
consumers (transposed / stencil / reversed reads); a conflict survives schedule canonicalization
only if each conflicting schedule is pinned by a MAJORITY of straight accesses — a lone transposed
read is dissolved by loop permutation, and the fixture stops testing a layout conflict.

## Status

Full suite `tests/transformations/layout/` green (790 passed, `-n2`, 2026-07-17), including the
review-hardening regressions. Headline: the model table finds conflict3's `B: [identity, perm10,
perm10]` trajectory, DP == brute-force oracle, the trajectory strictly beats every single layout,
and the composed assignment applies bit-exactly end to end; the eval table drives the same pipeline
from measured medians. Branch `extended`.

## Deferred / open

- **C4** — the default cost model / escalation policy, and the coupled (non-separable) DP
  refinement, and when to trust model vs eval. The v1 deliverable is brute force + DP with both
  providers reporting disagreements; the default comes after the eval-vs-model data exists (D3).
- Block/blocked TRAJECTORY application (per-nest blocked candidates still flow through
  `evaluate_nest`); B1 stride-driven permutation pruning; MapDimShuffle-based schedule re-derivation
  (v1 scores under the nest's pinned schedule); measured relayout edge costs (streaming bound
  meanwhile); interstate-edge references to reassigned arrays (refused).
- Zip application (v1 detects candidates only), Shuffle / indirect, non-line CFGs AND LoopRegions
  (nested line graphs), the GPU eval path + GPU arena, windowed/exact tile counts (would lift the
  cross-family pruning exemption).

# Global layout over multiple loop nests — design + tasks

Goal: choose ONE layout per array across a whole program of several loop nests — or a trajectory of
layouts with paid relayouts between nests — and the canonical schedule per nest that exploits it.
This is the k17 experiment (per-op greedy vs global vs oracle) promoted from a numpy toy into the
SDFG pipeline.

## Scope (v1)

- **Layouts tried: Permute and Block, alone or composed.** Shuffle only matters for indirect access
  (replay path; not in v1). Zip is a MULTI-nest pattern by definition (fields co-accessed across
  nests) — v1 detects candidates, does not apply them.
- **Program shape: a FLAT line graph of kernels.** Branches/DAGs AND LoopRegions: refused loudly in
  v1 — the audit confirmed nested loops (loop-carried-layout enumeration, recursive DP, symbolic
  trip counts) are not needed for the paper claim (greedy vs global vs oracle on 2-3 nest
  programs). The nested design below is kept as the documented extension, deferred.
- **Preprocessing contract**: canonicalize + maximal map fusion already ran (so no fusion
  opportunity is destroyed by splitting), then ONE KERNEL PER STATE, then relayouts are only ever
  inserted ON STATE BOUNDARIES.
- **Concrete shapes only (v1)**: free size symbols are specialized from a provided config
  (`sdfg.specialize`) BEFORE candidate generation; any remaining symbolic extent is refused loudly.
  Both the schedule passes (which no-op on symbolic strides) and the DP comparisons need numbers.
- Two ranking modes over the SAME composition machinery: **cost-model mode** (tier 0 counts →
  tier 2 `nest_memory_time` on `undecided`) and **eval mode** (externalize, compile, time). The
  cost PROVIDER is pluggable; the assignment algorithm does not care where costs came from.
- **Choice of the default cost model / escalation policy: OPEN (explicitly deferred).** v1 ships
  brute force + DP with both providers and reports disagreements; picking the default comes after
  the eval-vs-model comparison data exists.

## The pipeline

```
canonicalize + max fusion          (exists)
  -> kernel-per-state split        (A2)
  -> line-graph extraction         (A3)   nested via LoopRegions, trip counts kept symbolic
  -> per-nest candidate layouts    (B1)   permutations from access strides; block factors {8,16,32}
  -> canonical schedule per (nest, layout)  (B2)  loop-permute to min stride + tile to match blocks
  -> per-nest cost                 (B2 model / B3 eval)  tier-0 counts -> tier-2; or measured
  -> global assignment             (C1 DP / C2 brute force)
  -> apply: global layout + relayout states + canonical schedules   (A4 + existing passes)
  -> verify bit-exact vs the untransformed program
```

## The assignment problem, precisely

Arrays `a`, nests `k_1..k_n` in line order, candidate layouts `L(a)`. An assignment gives each
array a layout PER SEGMENT, changes allowed only at state boundaries, each change costing a
relayout (`2*S*G` streaming bound — or its measured/expanded cost).

**Per-array DP (Viterbi).** Node weight = the array's access cost in nest `k` under layout `l`;
edge weight between consecutive nests = 0 if the layout is kept, relayout cost if changed. Shortest
path = the array's optimal layout trajectory, `O(n * |L|^2)` per array. "No intermediate changes"
is the same DP with infinite edge cost on changes — both variants come free.

**Scoring convention (audit A1 — schedules are SHARED, weights must be realizable):** a nest has
ONE schedule, but the DP scores arrays independently — so per-array canonical schedules can be
jointly unrealizable. Convention: when scoring `(array a, layout l)` in nest `k`, all OTHER arrays
hold their baseline layouts and the canonical schedule is derived JOINTLY from all of the nest's
accesses (the stride-minimization pass already weighs all arrays). After the DP composes an
assignment, the schedule is RE-DERIVED from the composed layouts and the program re-scored (and,
in eval mode, re-timed) — the DP's number is a proposal, the composed re-score is the result.
Candidate identity stays `{nest}_{layout_tag}`: the schedule is a deterministic FUNCTION of the
layout tuple, never a free axis. Tie-breaks are an explicit invariant: the identity/unblocked
candidate is enumerated FIRST and wins ties (enumeration order is load-bearing; make it law, not
accident).

**Loops (nested line graphs).** A boundary INSIDE a loop body pays its relayout EVERY iteration
(weight x trip count), and the layout at loop exit must equal the layout at loop entry (the back
edge is a consistency constraint). Solve by enumerating the loop-carried layout (|L| options) and
running the inner DP for each — recursively for nested loops. Trip counts may be symbolic; the DP
compares costs, so symbolic trip counts need concrete sizes (the tier-0 lesson: sizes are not
parameters).

**The separability caveat (KNOWN approximation).** The DP treats arrays independently, but the
tier-2 nest cost couples them through `max(B*G, M*L/C)` — the max of SUMS is not the sum of
per-array maxes. Per-array DP is exact when one term binds for the whole nest (the usual case) and
an approximation otherwise. Mitigation, mandatory: AFTER the DP picks an assignment, re-score the
composed program with the full nest model and against the brute-force oracle on the test programs;
if composition ever loses to a brute-force winner, that is the signal the coupling matters and the
DP needs a coupled refinement (OPEN, later).

**Conflict resolution = the DP doing its job**: an array whose nests disagree gets either the
globally cheaper single layout (access-cost difference beats relayout) or a trajectory with paid
changes (relayout beats carrying the wrong layout) — exactly the `uses * delta >= 2S/rate`
break-even, decided per boundary by the shortest path. The report surfaces every disagreeing array
with its per-nest preferences and the chosen resolution (the k17 triad: greedy / global / oracle).

## Externalization (replaces any nest-forge dependency)

A kernel state is self-contained after the split, so: deepcopy the state into a fresh SDFG, add the
descriptors it references, done. Inputs per the standing rule: transients -> deterministic random,
non-transients -> config-provided. Used for (a) eval-mode per-nest timing, (b) per-nest
correctness oracles. nest-forge is NOT needed — its emit/arena machinery solves a different problem
(cross-compiler lanes); one map scope to a runnable SDFG is a deepcopy plus descriptor bookkeeping.

Eval discipline (standing box rules): builds sequential (`-n1`, 8 GB cap), each variant in its OWN
`DACE_default_build_folder`, compiled kernels run FORKED, timing STRICTLY serial per device on a
quiet box. Cache results keyed by (nest content hash, layout, schedule).

**Evaluation efficiency — the sampling loop must not be malloc-bound:**

- **Buffer arena, allocated ONCE per array — with a PRISTINE SNAPSHOT (audit C2)**: every nest
  WRITES something, so a naive reuse verifies candidate N against buffers candidate N-1 clobbered.
  Two allocations per array: the pristine initialized snapshot (read-only after init) and the
  working set; before each candidate, restore working from pristine (one streaming memcpy — no
  malloc, and the copy itself is a live bandwidth/relayout-shaped measurement), relaid to the
  candidate's layout by the LayoutChange expansion (timed: **it doubles as the DP's relayout
  cost**). Oracle outputs computed once from pristine.
- **B3/C2 EXTEND `brute_force.sweep`** — it already owns compile-verify-time-rank for one nest;
  the additions are the arena staging hook and the external timer. No second engine.
- **GPU-resident init and staging (DEFERRED with the GPU eval path — audit A2)**: the design is
  recorded (buffers/init/relayout/oracles on device via cupy+cuTENSOR, H2D once at arena creation)
  but v1 is CPU-only; the existing `sweep(device="gpu")`/GPU_Events bits stay untouched.
- **Python stays ONE simple process; the KERNEL is what gets spawned.** The kernel under test runs
  FULLY MULTITHREADED (the saturated-multicore scope is the model's premise), so during its slot it
  owns every core; any concurrent process — another candidate, a gcc — would steal cores from the
  measurement itself. No pools, no workers, no IPC machinery. Each kernel invocation is SPAWNED as
  a child (the standing `run_isolated` pattern: fork, run full-threaded, timings/verdict back over
  a pipe, timeout -> SIGKILL) — that is a crash guard, not parallelism: a segfaulting candidate
  must not kill the campaign. GPU runs cannot fork; the campaign resumes from the results log
  (attempt logged BEFORE it runs; arena init deterministic, so state reproduces).
  *(2026-07-17 amendment: the child-spawn crash guard is DEFERRED — the parent process has an
  initialized OpenMP runtime by the time candidates run, and fork() with live OpenMP threads can
  deadlock the child; a safe design needs spawn + picklable candidate builders. The batch phases
  and the attempt log ARE implemented in `sweep`; the log is the crash forensics meanwhile.)*
  Sequential batch phases:
    1. **COLLECT**: generate all candidate SDFGs up front. **Candidate identity = `{nest}_{layout_tag}`
       — nothing else.** The canonical schedule is a FUNCTION of the layout (blocked -> the matched
       tiled schedule; unblocked -> the min-stride loop permutation), so there is no schedule axis
       in the candidate space. The unique name gives a disjoint build folder per candidate (the
       same-name build-cache hazard the test suite already hit once).
    2. **COMPILE + VERIFY**: build each candidate (`-n1`, 8 GB cap) and check it against the
       oracle. All compiles finish BEFORE any timing starts — no gcc noise inside the timing phase.
    3. **TIME**: back-to-back, one candidate at a time, arena-staged (the relayout between
       candidates is itself timed — it doubles as the DP's relayout cost).
    4. **REDUCE**: winner per NEST; the per-nest tables feed the DP / brute-force composition.
  Distribution, when wanted, is ACROSS MACHINES: shard the candidate pool, each machine runs the
  whole sequential pipeline on its shard, merge result tables.

- **Measurement protocol**: warmup runs (>= 2, excluded), then **~10 timed reps; report the MEDIAN
  plus the spread `(max - min) / min`**. Median, not min, for whole kernels (min is for the
  microbenchmarks, where noise is one-sided; a kernel's distribution is not). A spread above a
  threshold (default 10%) flags the sample as contended -- kept, but marked untrusted, mirroring
  `membench.check_environment`'s refuse-don't-guess doctrine.

**Externalization validity caveat**: timing a nest in isolation destroys inter-nest cache reuse.
Valid in the suite's regime (working sets >> LLC, so reuse across nests is negligible); a program
whose nests share LLC-resident data violates it — detect (sum of footprints vs LLC) and warn.

## Schedule canonicalization per (nest, layout)

Layout first, then schedule, then count — but the audit refuted the "exists" claims as-is, so the
concrete recipe is (all on the EXTERNALIZED single-nest copy from A1, with concrete sizes baked into
descriptor strides — the whole-SDFG passes must never run on the multi-nest program):

1. **Permute candidates**: rewrite the copy's descriptor strides to the candidate permutation, then
   reorder the (post-canonicalize, single multi-param) map via `MapDimShuffle` for minimal innermost
   stride. `MinimizeStridePermutation` as-is CANNOT do this — it refuses symbolic strides
   (`minimize_stride_permutation.py:220`) and breaks on multi-param maps (`:191`), which is exactly
   the canonical post-collapse shape; extending it is part of the task, not a given.
2. **Block candidates**: run `BlockAwareMapTiling` with the candidate's factors ON THE COPY (it has
   no per-nest targeting — another reason externalize-first is mandatory), then APPLY
   `SplitDimensions` — its perfect-block-match path then emits clean tile/offset indices with no
   residual `%`/`int_floor`, i.e. exactly the 4-D fixture form the cost model can score. The
   "virtual blocked strides on the original 2-D access" idea is incoherent for blocking (a blocked
   layout is higher-rank) and `normalize_schedule_for_layout` cannot fire on it (it detects
   `Mod(param, b)` accesses that only exist after the split).
3. **THEN `count_loop_nest`** on the resulting copy — the schedule changes the counts.

Exactly ONE canonical schedule per (nest, layout) — blocked layouts get the matched tiled schedule,
unblocked ones the min-stride permutation — so `{nest}_{layout_tag}` uniquely identifies a
candidate and no schedule axis exists in the search space.

## Tasks

Phase A — infrastructure
- **A1 externalize_nest**: state -> fresh runnable SDFG (deepcopy state, copy descriptors + symbols
  the nest reads, transients -> inputs with deterministic random init). Test: 2-nest program -> 2
  SDFGs, each bit-exact vs a per-nest numpy oracle.
- **A2 kernel_per_state**: split multi-kernel states so each top-level map scope stands alone;
  producer/consumer chains through in-state transients must split into state-crossing transients.
  Audit what DaCe state-fission utilities already give before writing anything.
- **A3 line_graph**: extract the (nested) kernel sequence; LoopRegion -> nested segment with trip
  count; REFUSE branches/DAGs loudly.
- **A4 relayout_on_boundary**: insert a state holding LayoutChange node(s) between two kernel
  states (multiple arrays = parallel nodes in one state); descriptor changes (blocking changes
  shape) handled by LayoutChange's in/out descriptors. Correctness test round-trip.
- **A5 apply_assignment (audit C4 — was MISSING, and it is the blocker)**: apply a chosen
  TRAJECTORY end-to-end. Per layout segment: clone the descriptor under a segment name
  (`A__seg2`), rewrite memlets/access nodes ONLY in that segment's states (refactor the per-array
  rewrite cores of PermuteDimensions/SplitDimensions to take a state subset), wire the A4
  LayoutChange between old and new descriptor on the boundary. Without A5 the greedy-vs-global
  figure cannot be measured end-to-end — nothing else in the plan produces it.
- **A6 invariant ownership (audit C7)**: DaCe `simplify` FUSES split states back — after A2, the
  pipeline must never call full `canonicalize()`/simplify again (the schedule stage runs
  STANDALONE on externalized copies); A3 re-checks one-map-scope-per-state and refuses if the
  invariant broke.

Phase B — candidates + per-nest scoring
- **B1 candidate_layouts**: per array per nest, permutations that make some observed access
  contiguous (from access strides — NOT all d!). Blocking space PRUNED hard, and every prune
  logged ("no silent caps"):
    * factors: powers of two only, {8, 16, 32} default (vector width .. line elems);
    * at most `max_blocked_dims` dims blocked (default 2, hard cap 3);
    * only dims whose extent the factor DIVIDES (no remainder tiles in v1 — power-of-two and
      perfect-square extents pass trivially; ragged extents are Pad-then-Block, deferred);
    * only dims with extent >= 4x the factor (a tile that is most of the dim buys nothing);
    * dominance pruning (`pareto_front` on tier-0 counts) BEFORE any compile or timing — with one
      EXEMPTION (audit A0, blocker): the continuous-fraction counts OVERCOUNT sub-block tiles
      (documented ~2x), which biases counts against exactly the blocked+tiled candidates, and
      tier-2 shares the same counts, so escalation is no escape. Therefore count-based pruning is
      only allowed WITHIN a family (blocked vs blocked, unblocked vs unblocked); CROSS-family
      pairs (any blocked/tiled candidate vs an untiled one) are never pruned by counts — they
      survive to eval, or to a windowed/exact tile count when that lands.
- **B2 model scoring**: virtual strides -> canonical schedule -> `count_loop_nest` ->
  `dominance_verdict`/`pareto_front` prune -> tier-2 `nest_memory_time` only for survivors/
  undecided pairs.
- **B3 eval scoring**: externalized nest x surviving candidates, serial timing per the discipline
  above, medians + spread recorded.

Phase C — global assignment
- **C1 per_array_dp**: the Viterbi over the line graph with loop consistency; both edge regimes
  (changes allowed / forbidden).
- **C2 brute_force_global**: enumerate full assignments — the oracle; emits the greedy/global/
  oracle triad report (k17 generalized). Guarded: compute `prod_a |L(a)|^segments` up front,
  refuse with a logged message above ~1e6 assignments (scoring is table arithmetic, so the cap is
  generous — but it is a cap, stated, not a vibe about "small programs").
- **C3 conflict_report**: disagreeing arrays, per-nest preferences, resolution, break-even numbers.
- **C4 OPEN — cost-model default + coupled DP refinement + when to trust model vs eval.**

Phase D — validation (D1 comes FIRST — audit A3: the fixtures drive every phase, test-first)
- **D1 test programs**: 2-nest and 3-nest SDFGs with a genuine conflict (nest 1 wants A row-major,
  nest 2 wants A transposed — the k17 pattern), plus an agreeing pair (dominance settles it).
  Loop-nested variant: DEFERRED with LoopRegions. Bit-exact end-to-end after apply. Written
  BEFORE A1 — every A/B/C task lands against these fixtures.
- **D2 zip_candidates (analysis only)**: arrays co-accessed with identical index functions in >= 2
  nests -> report as Zip candidates. Application: LATER.
- **D3 model-vs-eval**: rankings compared on D1; disagreements explained (C_flip or model gap) —
  the data C4 needs.

Explicitly OPEN / deferred: C4 (cost-model choice), Zip application, Shuffle/indirect, non-line
CFGs AND LoopRegions (nested line graphs), GPU eval path + GPU arena, coupled (non-separable) DP,
windowed/exact tile counts (lifts the A0 cross-family pruning exemption), multi-param extension of
MinimizeStridePermutation (B2 does it via MapDimShuffle on the copy meanwhile).

## Implementation record (2026-07-17)

Landed, in audit order (each with fixture-driven tests, all green):

- **D1** `tests/.../multinest_programs.py` -- conflict2 / conflict3 / agree2. Two engineered
  properties: nests survive maximal fusion (non-pointwise consumers), conflicts survive schedule
  canonicalization (each conflicting nest pins its schedule with a majority of straight accesses,
  so the conflict array's remedy is a layout, not a loop permutation).
- **A1** `externalize.py` -- `SDFGCutout.singlestate_cutout` wrapper + `nest_arguments`
  (deterministic fills). **A2/A3/A6** `line_graph.py` -- iterated `state_fission`,
  `check_kernel_per_state` guard, flat `line_graph` with loud refusals (branches, LoopRegions,
  non-map work; symbol-assumption states and relayout states pass through).
- **A4** `relayout_boundary.py` -- boundary states of parallel `LayoutChange` nodes
  (`add_layout_change` grew `create_output=False` so conversions into EXISTING descriptors never
  clobber them). **A5** `apply_assignment.py` -- trajectory application: segment clones
  (`B__seg1_perm10`), per-segment state rewrite via the extracted shared core
  (`permute_dimensions.rewrite_state_for_permute`), entry conversions only when live-in
  (write-first segments are never fed from garbage), exit conversion when the LAST write happened
  under a non-identity layout. v1 applies PERMUTE trajectories; Block trajectories are refused
  loudly.
- **Eval wrapper** `nest_eval.py` (user request; B3's core) -- `evaluate_nest`: externalize ->
  reference run -> wrap-mode candidates -> `sweep` with `compute_region_stats_timer` (median +
  spread/contended metadata via the new tuple-timer protocol in `brute_force.sweep`); candidate
  identity `{nest}__{tag}`; identity-first tie-break enforced.
- **C1/C2/C3** `global_assign.py` -- provider-agnostic `AssignmentCosts` table; Viterbi DP (both
  edge regimes), capped brute-force oracle, per-op greedy baseline, conflict report with the
  greedy/global/single triad. DP == oracle on randomized tables.
- **B2/B3-lite** `assignment_costs.py` -- model provider (externalize -> in-place permute ->
  `count_loop_nest` -> per-array tier-2 `nest_memory_time`) and eval provider (`evaluate_nest`
  medians); both price relayout edges with `streaming_relayout_time`. End-to-end: the model table
  finds conflict3's `B: [identity, perm10, perm10]` trajectory, DP == oracle, the trajectory
  strictly beats every single layout, and the composed assignment applies bit-exactly.

Deferred beyond the earlier list (stated in code, refused loudly where reachable): Block/blocked
TRAJECTORY application (per-nest blocked candidates still flow through `evaluate_nest`'s custom
candidates); B1 stride-driven permutation pruning (v1-lite enumerates d! for d <= 3, refuses
above); MapDimShuffle-based schedule re-derivation (v1 scores under the nest's existing pinned
schedule); measured relayout edge costs (streaming bound under one LogGP default meanwhile);
interstate-edge references to reassigned arrays (refused).

## Review-response record (2026-07-17, second pass)

A 5-lens adversarial review (correctness-rewrite / -structure / -costs, timing-validity,
test-integrity; 31 findings, 14 confirmed with repros) drove a hardening pass. All confirmed
findings are fixed with regression tests:

- **apply_assignment liveness planner rewritten** (3 confirmed criticals + 2 majors, one root):
  the planner now walks segments carrying the LIVE holder. Entry conversions are decided at the
  segment's first TOUCHING kernel (an untouched first kernel no longer suppresses them);
  untouched segments stay unmaterialized and later same-layout segments ALIAS onto the live
  holder (no conversion, no clone); the entry skip needs PROOF of full production
  (`writes_cover_array`: inner memlet = distinct-param permutation over full map ranges -- the
  propagated outer memlet is unsound for this and deliberately not consulted); a WCR write counts
  as a read (`reads_before_write` scans the state's edges for `wcr`); the exit conversion fires
  only when the original does not hold the post-last-write value (a later restoring entry
  conversion suppresses it -- the double-charge case); copies with one relaid operand are
  converted to `TensorTranspose` via the shared `permute_dimensions` bookkeeping (extracted to
  module level: `note_copy_side` / `retranspose_copies`), sub-region copies refuse; NestedSDFGs
  receiving a reassigned array through a SPANNING memlet refuse loudly (unit-element edges are
  layout-transparent and pass). Regressions: `apply_assignment_regressions_test.py` (gap3 /
  partial2 / wcr2 / copy3 fixtures + hand-built nested refusal).
- **Objective made liveness-aware** (confirmed major): `AssignmentCosts` carries
  `entry_conversion_needed` + `last_write_kernel` (filled by both providers via
  `liveness_facts`, same predicates as apply); `trajectory_cost` prices the entry edge, an exit
  surcharge at the last write, and ONE free switch back to identity after it (that switch IS the
  exit conversion moved onto the boundary -- so `[perm10, identity]` prices one conversion, not
  two, and oscillation is not free). The DP tracks a `(layout, restored)` state and still equals
  the enumeration oracle (randomized-facts test); the single-layout regime pays entry/exit too,
  keeping the single-vs-trajectory comparison fair. Known residual approximation, stated in the
  docstring: every tag switch is priced even where apply would alias/skip -- optima are
  unaffected (cost-equal constant twins exist), D3 re-scores the composed result.
- **Timing**: the relayout-boundary classification is now the symmetric difference `copy_in ^
  copy_out` -- a copy state on both sides IS the compute (externalized transpose/copy nest), so
  the identity candidate gets timed instead of ranking last at `inf`. Every rep must produce a
  FRESH instrumentation report and a full sample set (stale/missing reports are hard errors, not
  silent shrinkage). `best()` widens ties to a noise window (default = the contended threshold)
  resolved by enumeration order, so the identity-first law is no longer dead for measured sweeps.
- **sweep** runs in the design's batch order (compile+verify ALL candidates, then time the
  verified back-to-back) and takes an `attempt_log`. The child-spawn crash guard below is
  AMENDED, not implemented: the timed kernels are OpenMP-parallel, and forking a process whose
  OpenMP runtime is already initialized can deadlock in the child; a spawn-based design needs
  picklable candidate builders. Until then a segfaulting candidate still kills the campaign --
  the attempt log names it. (This is the one review finding deferred rather than fixed.)
- **line_graph / kernel_per_state / relayout_boundary**: a bare access-node copy state counts as
  work and is refused (it was invisible to liveness); `kernel_per_state` runs to fixpoint over
  the states `state_fission` creates and raises honestly when fission drags the second nest along
  (shared-sink case) instead of mis-reporting success; `relayout_on_boundary` preserves
  start-block status when inserting before the start state (the boundary was unreachable dead
  code there).
- **eval_costs** keeps contended medians but marks them: `AssignmentCosts.untrusted` carries the
  `(array, kernel, tag)` set, a loud warning names them, and the conflict report flags rows whose
  chosen trajectory consumed one.
- **Test integrity**: boundary tests assert the structural witness (kernel states reference the
  segment CLONE -- the suite previously passed with the rewrite loop deleted); the eval-pipeline
  test forces a non-identity trajectory when measured noise picks all-identity (the apply leg can
  no longer degrade to a no-op); a tautological assertion was deleted.

Refuted findings (9) stayed refuted for reasons worth keeping: tag<->ops aliasing cannot arise
from `permutation_layouts` (injective by construction); order-only empty-memlet edges do not
occur in the kernel-per-state normal form; `run_and_check`'s allclose-vs-bit-exact gap is closed
by dtype refusals in `LayoutChange` plus `array_equal` lowering tests elsewhere.

## Audit record (2026-07-17)

5-lens adversarial audit, 8 CONFIRMED + 4 ADJUSTED, 0 refuted (3 verify agents lost to a session
limit; their lenses' primary findings still landed). Blockers found before a line of code: the
missing apply-assignment task (now A5), tier-0 pruning bias against blocked+tiled candidates (now
the cross-family exemption), jointly-unrealizable DP node weights (now the scoring convention).
Also: B2's "exists" citations refuted in detail (MinimizeStridePermutation refuses symbolic strides
and multi-param maps; normalize_schedule_for_layout cannot fire pre-split; BlockAwareMapTiling has
no targeting), arena clobbering by writes (pristine-snapshot restore), simplify re-fusing split
states, missing symbol-specialization contract, unbounded brute force, LoopRegions cut from v1.

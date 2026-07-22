# Is our LogGP-based cost model good? An evaluation against the literature

Verdict up front: **good and well-scoped for the job it does — rank layouts and price a mid-flight
relayout on a fixed schedule for a memory-bound kernel — and methodologically strong (SDFG-derivable +
validatable + symbolic dominance). It is NOT a general performance predictor.** Its gaps (no cache-
capacity/reuse, single memory level, no compute, no schedule axis) are mostly ORTHOGONAL to the layout
decision, so the model is narrow rather than wrong. The one gap that genuinely bites layout is reuse/
capacity (blocked layouts, the √S regime).

## What our model is

Per memory level `LogGP(L, o, g, G, line, sector)`; nest time

    T = max( bytes_moved * G , messages * L / concurrency ) + o * messages

with `(messages, sectors, bytes)` derived STATICALLY from the SDFG access subsets by counting UNIQUE
blocks touched (`count_loop_nest`), a coalescing-aware sector granularity, an MLP `concurrency` term, a
regime split (bandwidth vs latency), `break_even_uses` for relayout, tier-0 symbolic dominance/pareto
for provable relative ranking, and a microbenchmark `fit` + `validate` cross-check.

## Against the literature

**Roofline** (Williams et al. 2009). Single roof, perfect overlap, aggregates the hierarchy. Our
`max(bytes*G, messages*L/conc)` is a TWO-roof roofline: it adds a latency roof roofline lacks — which is
exactly what caught the GPU coalescing penalty (a latency-term effect). Better than plain roofline for
the latency-bound regime; like it in collapsing the hierarchy to one level.

**ECM** (Hager/Treibig/Wellein; Kerncraft). Models NON-overlapping, serialized data transfer across each
cache level plus in-core execution. This is strictly richer than ours on the hierarchy: ECM predicts
in-L2/L3-bound kernels and multicore saturation that roofline (and we) mispredict. We model ONE level and
no in-core compute. But ECM needs per-level traffic, which needs a cache/reuse model — the thing we (and
roofline) omit. So the gap is real but comes bundled with a dependency we also lack.

**LogGP** (Alexandrov et al. 1995). Originally a NETWORK model (L, o, g, G over point-to-point messages).
We repurpose it for on-node memory: a memory request is "a message to the memory controller." Defensible
analogy, but honest framing matters — the `o` (=0) and `g` (issue gap) terms are underused, so the model
is really a roofline/MWP hybrid wearing LogGP parameters. Call it that; a reviewer will note memory is not
message-passing.

**Hong & Kim MWP-CWP** (ISCA 2009), the gold-standard analytic GPU model (13.3% geomean error). MWP =
warps accessing memory simultaneously = our `concurrency`/MLP. Their coalesced/uncoalesced weighted
latency = our messages counted at sector granularity. Our model is essentially a STATIC, memory-only
MWP-CWP: we have the MWP side (memory-level parallelism) but not CWP (compute-warp overlap), and we derive
counts from the SDFG instead of a trace. Their number is an ABSOLUTE-time bar we do not claim — we validate
relative ranking, not 13% absolute error.

**Reuse / stack distance** (Ding & Zhong; Kerncraft; PPT-Multicore). THE gap. Counting unique blocks over
the whole iteration space = compulsory misses under an INFINITE cache; no capacity/conflict misses, no
temporal-locality or working-set model, no tile-size selection. Reuse-distance models exist precisely to
compute optimal tile sizes and predict capacity misses. For a pure layout swap this is often fine (layout
moves compulsory traffic; reuse is the schedule's job) — but for BLOCKED layouts and "does this fit in L2"
the model is blind.

**I/O complexity / red-blue pebble** (Hong & Kung 1981; Deinsum, Olivry et al. — same lab). The Ω(n³/√S)
matmul bound and automated parametric data-movement lower bounds. We are an UPPER-bound estimator with no
cache size `S`, so we cannot express the √S tiling regime or prove "no layout beats Y." Tier-0 dominance
gives relative provability only. Deinsum/Olivry are the lower-bound complement worth citing.

## Scorecard

| Property                         | Ours | Roofline | ECM | MWP-CWP | Reuse-distance |
|----------------------------------|:----:|:--------:|:---:|:-------:|:--------------:|
| Latency (not just bandwidth) roof|  yes |    no    | yes |   yes   |     n/a        |
| Coalescing / access-pattern aware|  yes |    no    | partial | yes |     yes        |
| Multi-level hierarchy            |  no  |    no    | yes |   no    |     yes        |
| Cache capacity / reuse           |  no  |    no    | via input | no  |    YES        |
| Compute / overlap                |  no  |    via AI| yes |   yes   |      no        |
| Static (no profiling)            |  yes |  partial | partial | no  |   often no     |
| Symbolic / parametric ranking    |  YES |    no    | no  |   no    |     some       |
| Validatable parameters           |  yes |   yes    | yes |   yes   |     n/a        |

## Where it is genuinely good

1. It answers the RIGHT question for a layout pass: layout A vs B on a fixed schedule, memory-bound, where
   the differentiator is compulsory traffic + transaction count. On that question it is sound and the GPU
   finding shows the two-roof form is necessary.
2. Static + symbolic: rankings come from the access subsets with no profiling, and tier-0 dominance gives
   compiler-usable provable comparisons — rare among these models (all the accurate ones need traces).
3. Validatable: the `fit`/`validate` path cross-checks each parameter two ways. Most analytic models are
   used with hand-set numbers; ours refuses a bad fit.

## Where it is not, and what to do

1. **Reuse/capacity (worst gap).** Add a coarse working-set-vs-`S` term: compare a nest's footprint to
   cache size, flag reuse that fits vs spills. Closes the blocked-layout and √S blind spot; a simple
   reuse-distance histogram or polyhedral footprint suffices. Until then, scope claims to compulsory
   traffic.
2. **Single level + no compute.** Fine for memory-bound layout decisions (the target; compute-bound is
   exactly where the 85/5 says layout is irrelevant). For full kernel time, compose levels ECM-style.
3. **Schedule axis uncosted.** Cannot reproduce the 85/5 schedule-vs-layout split; only ranks layouts
   under the canonical schedule. A schedule x layout sweep is the missing measurement.
4. **MLP is estimated, not derived.** `exposed_concurrency` returns `inf`/`n_cores*mlp` heuristically; the
   GPU finding showed this must be finite and careful. MWP-CWP derives it from occupancy — a good target.
5. **Frame as roofline/MWP hybrid with LogGP parameters**, not "a LogGP model." And validate absolute
   error on a few kernels, or state plainly it is relative-ranking-only and lean on tier-0 dominance.

## Bottom line

For choosing layouts and placing the mid-flight transpose, it is a good model — arguably better than plain
roofline (latency roof) and more compiler-friendly than MWP-CWP or reuse-distance (static + symbolic +
validatable). It is not a rival to ECM/MWP-CWP as a general time predictor, and it should not claim to be.
The single upgrade with the most leverage is a cache-capacity/reuse term, because that is the one place a
layout decision actually touches the memory hierarchy the model currently cannot see.

## Decision: stay LogP-based (a LogGP-parameterized two-roof), not the I/O model

We evaluated the pure I/O model (count transfers, price at bandwidth) as the layout cost and rejected it —
it fails for two reasons that are precisely a layout model's job:

1. **Bandwidth-only.** No latency term. Layout's biggest signal is on LATENCY-bound kernels (strided /
   uncoalesced = a transaction explosion, not a byte explosion), which a volume model cannot see. Our
   probe: the GPU penalty is a latency-term effect (15x under finite MLP, 3.94x on bandwidth).
2. **Asymptotically layout-blind.** For reuse-heavy kernels (GEMM, Theta(n^3/sqrt(S))) the I/O-optimal
   strategy just inserts one repack (Theta(n^2)), asymptotically dwarfed — so the I/O model's own
   optimality says layout is free to fix and the inflation goes to 0. But the repack cost is real at
   finite n, and without it the per-pass spatial/latency penalty is real; both are sub-asymptotic, so the
   volume model drops exactly the layout signal.

Layout's whole signal is two sub-asymptotic terms — spatial efficiency (bytes per useful byte) and
transaction count — which map to the two roofs LogGP carries: the G/bandwidth roof at SECTOR granularity,
and the L/latency roof (messages * L / concurrency). The I/O model is our bandwidth roof, volume-only —
the strict subset that already failed. Roofline lacks the latency roof; ECM and reuse-distance need traces
and carry no transaction term. Only the LogGP two-roof carries L AND per-message granularity, the two
things layout perturbs — so LogP-based is the family whose shape matches the phenomenon, not a compromise.

3. **Coverage / frontend.** The I/O model's frontend (SOAP, polyhedral I/O bounds, red-blue pebble) is
   built on the per-element CDAG plus AFFINE (Presburger) access functions. Many real programs use ``//``
   and ``%`` in their indices (blocking, wraparound, packing); those are non-affine, so the polyhedral
   machinery REJECTS them and the per-element CDAG (n^3 nodes) has no closed form to reason over. LogP
   block-counting works at subset/stride granularity — aggregate over the nest, not per-element — deriving
   cost from the descriptor stride (plus a dynamic-replay path for data-dependent gathers), so on ``//``/
   ``%`` it degrades to a bound instead of refusing. (Caveat: ``%`` is a sawtooth stride, so the estimate
   is a reasonable bound, not exact — but it is produced, not rejected.) Right shape AND right frontend.

Name it honestly: a **LogGP-parameterized two-roof memory model** (o=0, g underused, "a memory request is a
message to the controller" by analogy), not literal network LogP.

Two additions complete it WITHOUT leaving the LogP core:
- a **compute roof** (flops x peak) so "compute-bound => layout irrelevant" is a model OUTPUT, not an
  assumption (three roofs: compute / bandwidth / latency);
- **SOAP/IOLB** (Kwasniewski et al.) as an EXTERNAL audit — not part of the cost — proving IOUB(schedule)
  / IOLB -> 1, i.e. the asymptotic volume lever belongs to the SCHEDULE (the 85%), isolating layout to the
  LogP-priced spatial+latency residual (the 5%, regime-dependent: CPU real, GPU small).

LogP two-roof is the ranking cost; the compute roof is the compute-bound cutoff; SOAP is the
lower-order proof. Three tools, LogP at the centre.

Sources: Williams et al., Roofline (CACM 2009); Hager/Treibig/Wellein, ECM (arXiv:1509.03118, Kerncraft
arXiv:1702.04653); Alexandrov et al., LogGP (1995); Hong & Kim, MWP-CWP GPU model (ISCA 2009); Ding &
Zhong, reuse distance (PLDI/TOPLAS 2009; PPT-Multicore arXiv:2104.05102); Hong & Kung, red-blue pebble
(1981); Kwasniewski et al., SOAP I/O lower bounds (SPAA 2021, arXiv:2105.07203); Deinsum (arXiv:2206.08301);
Olivry et al., automated data-movement lower bounds (arXiv:1911.06664).

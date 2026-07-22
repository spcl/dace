# LogGP cost analysis of a loop nest

## Purpose

Score a layout by what it costs the memory system, reading the loop nest off the SDFG. A layout
transform does not change what a nest computes or how many bytes it *needs* — it changes how those
bytes pack into transfer blocks, i.e. **how many blocks the same computation must move and how many
requests it must issue**. That is the entire lever, and it is why a block/message-granular model is
the right abstraction for layout.

The model keeps LogP's latency term — deliberately. Latency is where load/store performance is
*explained*: a nest that cannot keep enough requests in flight runs at its concurrency limit, not at
the machine's bandwidth, and which of the two binds is itself a function of the layout (a scattered
access needs `line/sector` times more outstanding requests to saturate than a coalesced one, see
§Saturation). A bandwidth-only model answers "how fast at best"; the latency term answers "and can
this nest actually get there".

## Derivation: memory access as point-to-point communication

The formula is NOT posited -- it is LogP's own message-stream law with the stages instantiated for
the memory system. Setting, per LogP's original framing: the core and the memory controller are two
"processors" exchanging point-to-point messages. A line fill is one exchange: a small REQUEST
message (an address) and a long REPLY message (the line, ``s`` bytes).

**One exchange.** LogGP prices a single long-message exchange ``2o + L + (s-1)G`` (single-message
form, verified against Hoefler's dissertation Eq. II.1 and the SPCL/Illinois LogGP lecture notes).
Our pointer-chase ``L`` is the measured ROUND TRIP, so one fill is ``T_fill = o' + L + (s-1)G ~= L``
(for a 64-byte line, ``line*G << L``). This is ``loggp.message_time``.

**A stream of M independent fills.** The LogP k-message law (Hoefler Eq. II.1, on file verbatim):

```
RTT(n) = 2L + 2o + (n-1) * max{o, g}
```

Two structural facts, both LogP's, neither ours:

1. **Latency appears ONCE** -- it is the pipeline depth ("the network is a pipeline of depth L with
   initiation rate g", Culler et al.). The message count multiplies the GAP, not L. The Illinois
   notes state the consequence: "there is no latency term like (s/k)*alpha, so the protocol achieves
   noticeable overlap". Charging L per message (the old serialized branch) contradicts the original
   model.
2. **The per-message cost is already a MAX over the pipeline's stages** -- ``max{o, g}`` in II.1 is
   the bottleneck law: a pipeline runs at its slowest stage's rate.

Instantiate the stages for a stream of line fills:

* **Issue/tracking stage**: LogP's capacity constraint -- "at most ceil(L/g) messages can be in
  transit ... it stalls" -- IS Little's Law. With ``C`` the in-flight budget the requester actually
  has, the effective inter-message gap is ``g_eff = L / C``.
* **Channel stage**: the reply occupies the shared channel for ``s*G`` seconds (LogGP's long-message
  gap). Over ``M`` messages moving ``B = M*s`` bytes: ``B*G`` total.

Pipeline-bottleneck composition (the same ``max`` II.1 already applies per message):

```
T(M) = L + M * max( L/C , s*G ) + o*M  =  L + max( M*L/C , B*G ) + o*M
```

The implementation drops the single leading ``L`` (the one-time pipeline fill; relative error
``<= C/M``, negligible for any nest with ``M >> C``) and keeps the rest verbatim. Every endpoint is
inherited rather than assumed: ``C=1`` gives the dependent chain ``M*L`` (each fill IS the pipeline,
depth un-overlapped), large ``C`` gives LogGP's "gaps dominate, latency may be disregarded" stream.

**Sharing among cores** is LogP contention: the channel stage's ``G`` is one shared resource (its
rate is the ALL-core saturated bandwidth), while each core brings its own issue stage -- which is
exactly why ``C`` composes as ``n_units * unit_mlp`` (SSConcurrency) while ``G`` does not scale
with cores.

## The formula

One continuous expression per nest and memory level (`loggp.nest_memory_time`, used by
`LoopNestLogP.total_time`):

```
T  =  max( B · G ,   M · L / C )  +  o · M            (o = 0 today)

M  =  Σ_arrays  new blocks touched at REQUEST granularity  (line_bytes)   × iterations
B  =  Σ_arrays  new blocks touched at TRANSFER granularity (sector_bytes) × sector_bytes × iterations
C  =  independent requests the nest keeps in flight (§Concurrency)
```

There is **no regime switch**. The endpoints of the same formula:

| C | T reduces to | what it is |
|---|---|---|
| 1 | `M · L` | the dependent chain — every miss waits out the full round trip (pointer chase) |
| `1 < C < C*` | `M · L / C` | Little's Law: sustained request rate `C / L`; latency partially hidden |
| `≥ C* = M·L/(B·G)` | `B · G` | the channels saturate; bandwidth is the only limit |

The `max` is **continuous at the crossover** (both terms equal `M·L/C*` there). The previous design
switched formulas at the bandwidth-delay product and returned the *serialized sum* `M·L + B·G` below
it — a ~26x discontinuity for a one-request change in `C`, and refuted by measurement (§Measurements:
rate scales linearly with independent chains, so charging full `L` per miss is wrong by the measured
MLP, ~8x on one core). The serialized sum survives only as `total_time_serialized()`, a diagnostic
ceiling: a *measurement* above it means the parameters are wrong.

### Why line granularity, not element granularity

`A[:] += B[:]` analyzed per element would report one message latency plus a few bytes per iteration
— and low line utilization. Wrong twice: the hardware requests a **full line**, and the next 7
iterations (fp64) hit that same line. Counting **new blocks per iteration**
(`blocks_touched.average_blocks_touched`) gets both right at once: `1/8` messages per element, and
every fetched byte used. The element view double-counts latency events 8x and misreports utilization;
the line view is what the coalescing/streaming hardware actually does.

### The two granularities

`M` and `B` are counted at **different** granularities, and this split is load-bearing:

| granularity | counts | pays | x86 | NVIDIA |
|---|---|---|---|---|
| `line_bytes` (request) | `M` — latency events | `L` | 64 | 128 |
| `sector_bytes` (transfer) | `B` — bytes on the channels | `G` | 64 (= line) | 32 |

Collapsing them is a 4x GPU error in whichever direction is chosen: one 128-byte number overcounts a
scattered access's bytes 4x; one 32-byte number overcounts a coalesced access's messages 4x. On x86
they coincide, which is why the conflation went unnoticed on CPU.

### Saturation — where the latency term explains layout

The crossover is `C* = M·L/(B·G)`. Write `k = B/(M·sector_bytes)` for the sectors actually used per
request (`1 ≤ k ≤ line/sector`); then

```
C*  =  L / (k · sector_bytes · G)
```

A **coalesced** access (`k = line/sector`) needs the fewest outstanding requests to saturate; a
**scattered** one (`k = 1`) needs `line/sector` times more (4x on NVIDIA). This is a prediction the
bandwidth-only view cannot make: two access patterns with identical byte traffic differ in how much
concurrency they demand before the channels fill — and whether the nest *has* that concurrency is a
property of its schedule and its device (§Concurrency). Pinned by
`test_scattered_access_needs_line_over_sector_more_concurrency_to_saturate`.

## Concurrency: C

`C` composes as **units × per-unit MLP**, and the two factors mean different things per device:

```
C  =  n_units · unit_mlp
```

| | unit | n_units | unit_mlp (affine/streaming) | unit_mlp (scattered/indirect) |
|---|---|---|---|---|
| CPU | core | populated cores (threads pinned) | `core_stream_mlp = bw_core·L/line` (~59) — prefetch-INCLUSIVE, Little's Law on the single-core triad | `core_mlp` ≈ 8 (chase knee — prefetch-DEFEATED by design) |
| GPU | **warp** | resident warps = grid × occupancy | `core_mlp` = outstanding loads per warp — no prefetcher; latency hiding IS warp switching | same |

**Concurrency is PATTERN-dependent, and that is itself a layout statement** (the "second penalty",
formerly a noted-unmodelled refinement, now first-class): a layout that scatters an access does not
only touch more blocks — it demotes the nest from the prefetch-extended `core_stream_mlp` to the
demand-miss `core_mlp`, because the prefetcher only runs ahead of predictable streams. The chase
knee deliberately defeats the prefetcher (random Hamiltonian cycle), so applying it to a streaming
nest under-credits concurrency severalfold — the box's own single-core triad (≈40 GB/s) implies ~50
outstanding lines, not 8. The affine analysis scores prefetch-friendly patterns and uses
`core_stream_mlp`; scattered/replayed nests are costed with `concurrency = n_units · core_mlp`
passed explicitly.

Estimation (`exposed_concurrency`), schedule-driven:

- **Parallel map, `n_cores` given** → `n_cores · core_mlp`. Iterations distribute in contiguous
  chunks (OpenMP static): each core streams its own chunk, so per-core block counts are the nest's
  divided by the core count, plus one shared boundary line per chunk edge — `O(cores)`, negligible.
  An *interleaved* distribution would instead multiply `M` by up to `line/elem` (every core touches
  every line) — distribution quality is expressible as a factor on `M`; DaCe's CPU maps chunk
  contiguously, so the default carries no penalty.
- **Parallel map, no core count** → `inf`, the saturated assumption. With `n_cores`, the verdict
  splits by pattern — and now agrees with both measurements at once: a STREAMING nest saturates
  easily (`16 × core_stream_mlp ≫ BDP`; the triad indeed saturates at ~2 cores), while a SCATTERED
  one at `16 × core_mlp = 132 ≈ BDP ≈ 103–148` is marginal — a scattered parallel nest does not
  comfortably saturate, which blanket `inf` hides.
- **All maps `Sequential`** → `core_stream_mlp`, NOT 1 and not the chase knee. Sequential does not
  mean one request at a time: affine addresses are computable ahead of the loads, and the prefetcher
  keeps fills in flight beyond the demand-miss queue. `C = 1` is **data-dependent addressing** —
  which the affine analysis never scores (dynamic memlets are refused, see §Indirect).
- **Caller knows better** → `concurrency=` overrides everything (a measured device MLP, a modelled
  chase).

### GPU, one thread per cell

Thread-per-cell means no thread streams anything — spatial adjacency moves from *consecutive
iterations on one core* to *consecutive lanes of one warp*, and the coalescer's merge of the 32 lane
addresses per instruction **is** the new-blocks count over the innermost (lane-mapped) axis.
Contiguous fp64: 32 lanes × 8 B = 256 B = 2 requests of 128 B per warp = `1/16` per element — the
same continuous fraction the CPU analysis computes. A layout that makes lane-adjacent accesses
strided costs one sector per lane: 16x the messages, 4x the bytes (32 of 128 useful per request).

Consequence: on GPU, `M` is determined **jointly** by the layout and by which axis the schedule maps
to `threadIdx.x`. The layout algebra's Permute moves the array; the block-map assignment moves the
lane axis; the model prices both through one term.

Occupancy enters through `n_units`: an under-occupied kernel (register pressure, tiny grid) has few
resident warps, misses the ~3900-request GPU BDP (500 ns × 1 TB/s / 128 B), and lands in the latency
regime — a shortfall the blanket saturated assumption hides. Pinned by
`test_gpu_occupancy_is_the_unit_count`.

## The cache hierarchy: a chain of point-to-point links

Caching does not break the point-to-point picture -- it repeats it. The hierarchy is a chain of
links, each with its own parameters and granularity:

```
core <-(L1)-> L1 <-(L2)-> L2 <-(LLC)-> LLC <-(DRAM link)-> memory controller
     (L_i, G_i, C_i, s_i) per link i
```

Two model families supply the two halves, and they compose cleanly:

* **Traffic per link -- the I/O model** (Aggarwal-Vitter external memory): the block transfers
  ``Q_i`` crossing link ``i`` are the I/O complexity at capacity ``M_i`` and block size ``s_i``.
  This is where CACHING enters: an inner level FILTERS the traffic that reaches the next link
  (reuse), and "latency similar but bandwidth less" is simply each link having its own ``G_i`` --
  bytes that hit in L2 pay ``G_L2``, only the filtered ``Q_mem`` pays the DRAM link's ``G``. The
  layout sensitivity is in ``Q_i`` via the block size (the I/O model's ``B``), which is exactly the
  pebble-game/Q* division of labour report1 already assigns.
* **Time per link -- this model**: ``T_i = max( M_i * L_i / C_i , B_i * G_i )``.

**Cross-link composition** is the honest open choice, and we BRACKET it instead of asserting it:

```
max_i T_i   <=   T   <=   sum_i T_i
```

full inter-link overlap on the left, none on the right. The published models are the two limits:
ECM is the SUM instance with every latency term dropped (its own words: "assuming there is no
access latency", perfect-prefetch streaming; it sums T_L1L2 + T_L2L3 + T_L3Mem and validates well
on Intel), Roofline is the MAX instance with only the DRAM term. Our single-level model is the
DRAM link alone -- valid precisely in the layout regime (working set >> LLC, inner links faster),
which is the regime every kernel in the suite is sized for. A multi-level extension is additive:
per-link counts from `blocks_touched` at ``s_i``, per-link ``Q_i`` filtering from reuse -- the
streaming "new blocks" count is the no-reuse instance of ``Q_i``, and the replay bounds below are
its data-dependent counterpart.

## Indirect accesses: static replay

`A[idx[i]]` has no affine subset — statically unknowable. When the index array is **materialized**
(the static-indirection case: `idx` known before the nest runs), replay it:
`blocks_touched.replayed_blocks_touched(idx, block_elems)` returns two bounds per access,

- **streaming** — block changes between consecutive accesses (cache of one block; upper bound), the
  same convention the affine metric's brute-force oracle uses;
- **distinct** — unique blocks (infinite cache; lower bound),

and the truth sits between, decided by whether the reuse distance fits the cache. For **fully
scattered** indices the bounds coincide — the answer is exact precisely where layout matters most.
For clustered-but-shuffled indices they diverge (distinct `1/8`, streaming ~1) and an honest model
reports the pair rather than picking one.

Wiring: `analyze_loop_nest(..., replayed_counts={'A': (messages_per_iter, sectors_per_iter)})`. An
array reached through a **dynamic memlet** without a replayed count is **refused loudly** — its
declared subset is a whole-array over-approximation, and pricing it would score a gather as one
invariant read, silently and enormously wrong. Any other unscoreable subset also raises instead of
being silently dropped from the nest's cost. (This is the quantitative bridge to the Shuffle
primitive and the static-replace analysis in `relayout.py`.)

## Cache efficiency: the bandwidth term's layout lever

```
eps = distinct useful bytes / bytes moved  ∈ (0, 1]        B = useful_bytes / eps
```

`useful_bytes` is the algorithm and layout-invariant; `eps` is the layout. `eps = eps_spatial ·
eps_write`, with `eps_write = 1/2` for a partial write (read-for-ownership) and `1` for a read or a
block-covering write. For fp64/64 B: `eps ∈ [1/8, 1]` reading, `[1/16, 1]` writing — layout matters
roughly twice as much where the nest writes. `eps` **cannot replace the latency term**: four sectors
in one request and four spread over four requests have identical `eps` and 1-vs-4 messages
(`test_efficiency_cannot_determine_the_latency_term`).

Bounds this makes non-negotiable: `eps` saturates once the stride reaches one block, so
`sector/elem` (8x for fp64) is a **hard ceiling** on any layout win. A kernel reporting more is
confounded, not fast.

## Relayout: cost and break-even

A tiled relayout reads the array once and writes it once, both at `eps = 1`, at the **saturated**
rate (a streaming copy over all cores — its break-evens are stated in that scope):

```
T_relayout = 2·S·G          break-even:   passes · (1/eps0 − 1/eps1)  ≥  2 + overhead_passes
```

`G` and `S` cancel: whether a relayout pays is a pure traffic ratio. Against a perfect target,
**`eps0 ≤ 1/3` means a single pass already pays**. The `pure` expansion does not stream (one flat
mapped-tasklet copy, one side strided) — cost it by expanding and analyzing the copy nest like any
other nest. `overhead_passes > 0` models inspector-executor (not pursued; a threat-to-validity
quantifier — see report1 "Threats to state").

## Parameters

| symbol | meaning | unit | source |
|---|---|---|---|
| `L` | unloaded round-trip latency of one line | s | pointer chase, MLP=1 |
| `o` | per-message issue overhead | s | 0 (≈0.25 ns vs `L≈95 ns`; deferred with SRAM) |
| `g` | min interval between one core's requests | s | diagnostic; `L/g` is LogP's capacity cap |
| `G` | per-byte gap = `1/BW_saturated` | s/byte | STREAM triad, **all cores** |
| `line_bytes` | request granularity → `M`, `L` | bytes | 64 x86 / 128 NVIDIA |
| `sector_bytes` | transfer granularity → `B`, `G` | bytes | 64 x86 (= line) / 32 NVIDIA |
| `c_core` | **measured** per-unit MLP (knee of the sweep) | count | `membench.fit_core_mlp`; falls back to `L/g` |

`core_mlp = c_core if measured else L/g`. The two differ on purpose: `L/g` is the issue-rate cap,
the knee is `min(L/g, miss-queue depth)` — and the miss queue usually binds first (measured ~8 vs
`L/g` = 23.75 from the default `g`). `validate()` cross-checks them; a large disagreement means `g`
was defaulted, not measured.

## Benchmarking methodology

One benchmark per parameter; each has a pitfall that silently invalidates it. `membench.c` is C so
the chase compiles to a dependent `mov (%rax),%rax` — vectorized it becomes a gather at MLP=width
and measures throughput instead.

| param | benchmark | pitfall |
|---|---|---|
| `L` | pointer chase, MLP=1, arena ≫ LLC, random Hamiltonian cycle (`chase_verify` proves it) | **prefetch** (linear chase measures the prefetcher); **TLB** — at stride 4096 every element is a fresh page: measured 1.03 dTLB misses/load, so the number is `L + page-walk`. Hugepages or sub-page stride. |
| `G` | STREAM triad, **all cores**, arena ≫ LLC | one core yields `BW_core` not `BW_saturated`; without non-temporal stores the triad's write adds a read-for-ownership stream (4 streams, not 3) |
| `c_core` | **concurrency sweep**: ns/load vs number of independent chains; `fit_core_mlp` = `L(1)/min` (Little's Law on the floor — no curve model needed) | a defaulted `g` invents `c_core`; the sweep must reach the flat region |
| GPU `unit_mlp` | multi-warp P-chase (the GPU analog of the chain sweep) | not yet run — see status |
| peak BW | `dmidecode -t 17` (the accepted sudo case); CUDA props | denominator only, never a model input |

**Quiet box required** (`check_environment` refuses otherwise): a plausible-but-wrong parameter is
worse than none.

**Validation protocol** — the falsifiable check: fit `L` (MLP=1), `G` (triad), `c_core` (knee).
Nothing else is fitted. Then *predict* the whole latency-vs-MLP curve `t(C) = max(B·G, M·L/C)` and
compare it to the measured sweep. Three parameters, one curve, no free knobs.

## Lineage and attribution (verified against primary sources, 2026-07-17)

The assembly is ours; the ingredients are not, and each must be cited:

| idea | prior art (verified) |
|---|---|
| memory access as point-to-point LogP | **Cameron & Sun, "Memory logP", IPDPS 2003**; generalized in **Cameron, Ge, Sun, lognP/log3P, IEEE TC 56(3) 2007** — verbatim `T = Σᵢ (max(oᵢ,gᵢ) + lᵢ)` over a succession of buffer-to-buffer transfers. The originating frame for what this model does. |
| capacity / in-flight constraint | **LogP** (Culler et al., PPoPP 1993): "at most ⌈L/g⌉ messages … in transit … it stalls"; "the network is treated as a pipeline of depth L with initiation rate g" |
| per-byte long-message gap | **LogGP** (Alexandrov et al., SPAA 1995): `T = 2o + L + (s−1)G`; k pipelined transfers pay `L` once |
| per-level parameter tuples | **Valiant, Multi-BSP, JCSS 2011**: depth-d machine as `(pᵢ, gᵢ, Lᵢ, mᵢ)` per level; older: **HMM** (Aggarwal et al., STOC 1987), **UMH** (Alpern et al., Algorithmica 1994) |
| per-link block traffic | **Aggarwal & Vitter, CACM 1988** (the I/O model: `N, M, B`, cost = block transfers) |
| bandwidth-vs-compute max | **Roofline** (Williams et al., CACM 2009): `Min(peak, BW × intensity)` |
| concurrency-limited bandwidth | **McCalpin**: "Effective Concurrency = memory latency × measured bandwidth" (Little's Law) |
| contention, sync, size-dependent params | **LoGPC** (Moritz & Frank, SIGMETRICS 1998), **LogGPS** (Ino et al., PPoPP 2001), **pLogP** (Kielmann et al., IPDPS-W 2000) |
| multi-level composition endpoints | **ECM** (Treibig/Hager; Stengel et al., ICS 2015): `max(T_OL, T_nOL + ΣT_level)` — the SUM end; Roofline — the MAX end |

Genuinely ours, and only this: (1) the per-line-fill reduction `T = L + M·max(L/C, s·G)` with the
max between a latency/concurrency leg and a per-byte leg **on a memory link** (no source writes this
exact form); (2) threading Aggarwal–Vitter per-link traffic through a chain of LogGP links; (3)
stating the cross-level composition as an explicit max/sum **bracket**; (4) the pattern-dependent
`C` (stream vs demand-miss) as a layout lever, and the ranking-crossover theorem `C_flip = M₂L/(B₁G)`.

## Example parameters (2026-07-17, Ryzen 7 8845HS, **contended box — illustrative only, NOT calibration**)

The model is pure theory; these one-off numbers exist only to ground the examples and to show the
parameter-extraction path works. Nothing in the model depends on them.

| what | measured | note |
|---|---|---|
| chase, MLP=1 | 146 ns/load | includes a page walk (1.03 dTLB miss/load) |
| MLP=2 / 4 / 16 | 74.0 / 36.6 / 17.7 ns/load | **1.97x / 3.99x / ~8.2x** — linear in C, then flat |
| `c_core` (fit) | **~8.2** (= L(1)/floor) | `L/g` default said 23.75 — `g` was never measured; and this is the DEMAND-MISS budget, not the streaming concurrency (`core_stream_mlp` ≈ 50 from the same box's triad) |
| triad 2→16 threads | 44.6 → 39.5 GB/s | flat: ~2 cores already saturate the triad |
| cache-misses/load | 1.03 (perf, differential) | every hop a real miss |

The sweep is the empirical refutation of the serialized branch: rate scales linearly with
independent chains *inside the latency regime*, exactly where that branch charged full `L` per miss.

## Minimality: which tier of the model a claim needs

The formula degrades gracefully into three nested tiers; use the lowest tier that carries the claim,
because every parameter you do not use is a parameter you cannot get wrong.

| tier | model | parameters | settles | cannot settle |
|---|---|---|---|---|
| 0 | counts `(M, B)` + dominance lemma | **none** | layout ranking whenever one layout wins both counts (the common Permute case; what the sweep's ranking rests on) | ties where counts disagree |

Tier 0 is implemented: `count_loop_nest` (the parameter-free counting core `analyze_loop_nest`
itself delegates to -- one core, two tiers), `dominance_verdict` (`first`/`second`/`tie`/
`undecided`; refuses when the counts disagree or a symbolic sign is open -- concrete SIZES resolve
the `int_floor` signs and are still zero measured parameters), and `pareto_front` (sweep pruning:
every dropped layout is at least as slow as a survivor for ALL `L, G, C`, so it is dropped before
compiling or timing anything; `undecided` never prunes). `cost_model_tier0_test.py` pins the lemma
across tiers: a tier-0 `first` is checked against tier-2 times over three decades of each parameter.
| 1 | `T = B·G` with `eps`, write factor | `G`, `sector_bytes` | absolute times at saturation; relayout break-even (`G` cancels: `passes·(1/eps0−1/eps1) ≥ 2`) | anything latency-side |
| 2 | `T = max(B·G, M·L/C)` | + `L`, `line_bytes`, `bw_core` (→ `C`) | device-dependent verdicts (`C_flip`), GPU occupancy, the chase, the scattered second penalty | multi-level traffic, depth floors, loaded latency |

Dropping any tier-2 ingredient loses a demonstrated capability: no `L/C` → no `C_flip` (the
cross-device check the validation plan requires); one granularity → 4x GPU error; no write factor →
2x on written arrays; latency priced by bytes → refuted by the same-eps/different-messages
construction. Everything above tier 2 (lognP/ECM level chains, `D·L`, loaded latency, `o`) is
deliberately out — see §What is deliberately NOT modelled.

## Theoretical properties

All pure theory — no measurement enters any of these; parameters are symbols.

**Continuity & shape.** `T(C) = max(B·G, M·L/C)` is continuous everywhere (both terms equal
`M·L/C*` at the crossover), non-increasing and convex in `C` (max of a constant and a convex
function). The old regime switch violated continuity at its own threshold by ~26x.

**Dominance lemma.** If layout 1 has `M₁ ≤ M₂` AND `B₁ ≤ B₂`, then `T₁(C) ≤ T₂(C)` for EVERY
`L, G, C > 0` — the ranking is parameter-free (each term is monotone in its count, and max preserves
the order). This is why ranking survives bad parameters whenever one layout dominates on both
counts — the common case: a Permute that makes the inner axis contiguous lowers `M` and `B`
together.

**Ranking-crossover theorem.** When the counts DISAGREE — `M₁ < M₂` but `B₁ > B₂` (fewer requests
vs fewer bytes; reachable under sectoring, padding, or mixed access patterns) — the winner is a
function of concurrency, and the flip is unique with a closed form:

```
layout 1 wins  ⟺  C < C_flip = M₂ · L / (B₁ · G)
```

Proof sketch: for `C ≤ M₁L/(B₁G)` both are latency-bound and `T₁−T₂ = (M₁−M₂)L/C < 0`; for
`C ≥ M₂L/(B₂G)` both are bandwidth-bound and `T₁−T₂ = (B₁−B₂)G > 0`; in the mixed interval
`T₁ = B₁G` (constant) and `T₂ = M₂L/C` (strictly decreasing), so their difference is strictly
increasing with exactly one zero, at `C_flip` — which lies inside the interval since
`C_flip/C₁* = M₂/M₁ > 1` and `C_flip/C₂* = B₂/B₁ < 1`. ∎

Consequence: **device-dependent layout verdicts are a theorem, not an anomaly.** A layout that wins
on a low-concurrency device (single-threaded CPU, under-occupied GPU kernel) and loses on a
saturated one flips at exactly `C_flip` — the closed-form mechanism behind the "k06/k08 win on GPU
but not CPU" class of observations, and a per-kernel prediction the sweep harness can test.

**Ranking within one device.** For one kernel on one device, `iters`, granularities, `G`, `L`, `C`
are identical across layouts, so `T` is monotone in `(M, B)`, which are structural. Parameters set
the absolute time and the regime; the ranking needs only the counts (plus `C_flip` when they
disagree).

## What is deliberately NOT modelled

- **The dependency-depth floor `D·L`** (EDAN, arXiv 2512.13176, Eq. 1 — the span-law bound): real
  for dependent recurrences (k11's K-long Thomas chain, k14's log-depth descent), but `B·G` grows
  with the problem while `D·L` is fixed, so it binds only at sizes where nothing is bandwidth-bound.
  A three-term `max(B·G, D·L, M·L/C)` would fuse two literatures that do not cite each other (EDAN
  has no bandwidth term; LogP has no depth — checked: 0 cross-citations) — an invented model inside
  a layout paper. `C=1` recovers the practically relevant dependent case.
- **Loaded latency**: `L` is unloaded; queueing inflates it ~20% near saturation (McCalpin), so the
  latency term is optimistic near the knee — conservative in the safe direction for layout ranking,
  since it under-rewards extra concurrency there.
- **False sharing / boundary lines** in the multicore distribution: `O(cores)` lines per array,
  negligible per chunk; the pathological interleaved distribution is documented, not priced.
- **`o`** and the SRAM/local level: local memory is free until the SRAM task lands.

## Implementation status

Fixed this pass (2026-07-17): the serialized-branch regime switch (→ one continuous `max`), the
nest-concurrency bug, the line/sector conflation, `nest_time` duplication, sequential ≠ dependent,
`c_core` as a measured parameter with `fit_core_mlp` (with a truncated-sweep guard), static replay
for indirection (dynamic memlets refused without `replayed_counts`; unscoreable subsets raise
instead of silently dropping), `n_units` core/warp distribution, pattern-dependent per-core
concurrency (`core_stream_mlp` vs `core_mlp` — the adversarial review's chief finding: the chase
knee is prefetch-defeated and under-credits streams ~6x), write-allocate factor 2 on written
arrays' bytes, element-wider-than-granularity spans, one-sided `validate()` knee check (knee =
min(L/g, queue) can sit far below the cap legitimately). 117 cost-model tests green; full layout
suite 722 green pre-review-fixes.

Open:
- **`c_core`, `L`, `G` on a quiet box** — every number above is contended and `L` carries a page
  walk; the validation-protocol run (predict the sweep from 3 fitted params) is pending on it.
- **GPU `unit_mlp` / P-chase** — the GPU column of the concurrency table is a structure, not yet a
  measurement.
- **Warp-window block counting** — the continuous fraction averages over an unbounded stream; a warp
  cuts adjacency at 32 lanes. Coincides for contiguous/strided; sub-warp tiles need the windowed
  count (same next-refinement as the sub-block tile overcount).
- **Loaded-latency curve**, **`D·L` if a dependent-recurrence kernel ever needs pricing**, **SRAM**.

## API

```python
cost = analyze_loop_nest(state, map_entry, p,            # granularities from p (line + sector)
                         n_cores=16)                      # or concurrency=... to override outright
cost.total_time()            # max(B*G, M*L/C) -- THE answer
cost.total_time_bandwidth()  # B*G term          cost.total_time_latency()   # M*L/C term
cost.total_messages()        # M                 cost.total_sectors(), cost.total_bytes()
cost.regime()                # diagnostic: which term is expected to bind
cost.total_time_serialized() # diagnostic ceiling (zero overlap); NOT a predictor
```

## Module map

| file | role |
|---|---|
| `loggp.py` | parameters (`LogGP`, `c_core`), **`nest_memory_time` — the formula**, `achievable_rate(p, C)`, BDP, `validate` |
| `logp_analysis.py` | the nest analysis: counts at both granularities, concurrency estimate, `LoopNestLogP` |
| `blocks_touched.py` | affine new-blocks metric + `replayed_blocks_touched` (static replay for indirection) |
| `membench.c/.py` | chase (`L`), triad (`G`), `fit_core_mlp` (knee), environment gate |
| `access_subsets.py` | per-array access subsets from the map scope |
| `relayout.py` | relayout cost, `eps`, break-evens (saturated scope) |
| `hardware.py` | theoretical peak BW (dmidecode / CUDA props) — the denominator |

## Accuracy and limitations

- **`blocks_touched` is asymptotically exact**: continuous fractions overcount sub-block tiles (a
  contiguous 4x4 fp64 tile is 2 blocks, reported ~4); ranking preserved, absolutes not; brute-force
  oracle in the tests.
- **Split-index accesses** (`A[i, int_floor(j,8), j%8]` after `SplitDimensions`) are not affine —
  score blocked layouts by giving the analysis the blocked strides on the original access.
- **`validate()` currently fails on this box's defaults** — by design: `L/g` (23.75) vs measured
  knee (~8) means `g` was never measured here. The check is doing its job; measure, don't loosen it.

## Worked example

`C[i,j] = A[i,j] + B[i,j]`, fp64, 64 B blocks, `G = 1/100 GB/s`, N=4096. Only A's layout changes:

| A layout | A msgs/iter | eps(A) | bytes/iter (A+B+C, C written ×2) | T = B·G at C=inf |
|---|---|---|---|---|
| row-major `(N,1)` | 1/8 | 1 | 8+8+16 = 32 | 5.4 ms |
| transposed `(1,N)` | 1 | 1/8 | 64+8+16 = 88 | 14.8 ms |

The 2.75x (not 8x — B stays contiguous and the written C, at its RFO-doubled 16 B/iter, dilutes it
further) is attributed entirely to A's block count.
With a real transform the same signal appears: `PermuteDimensions` on a transposed-read operand takes
its messages from ~1.0 to ~0.125 (the 8x the relayout was for); `PadDimensions` leaves a contiguous
access at ~0.125 — the model does not credit a pad that moved nothing relevant. Pinned in
`cost_model_transformations_test.py`.

## Extending to distributed MPI: a two-layer LogGP (design note, not implemented)

The model prices one node's core↔memory-controller traffic. The OMEN transpose (SC19 Gordon Bell,
arXiv:1912.10024 — the `tests/library/mpi/mpi_omen_transpose_test.py` / k10 pattern) needs a *second*
layer: the inter-node network. Structure it exactly as the intra-node model already treats core and
memory-controller as two LogP processors — add a network `LogGP(L_net, o_net, g_net, G_net)` layer
above the memory layer. A remote access pays the network layer (`o_net + L_net + bytes·G_net`), lands,
then pays the existing memory layer; a local access pays only the memory layer. Same
`max(bytes·G, messages·L/concurrency)` two-regime form one level up — `nest_memory_time` is reused
unchanged; only the parameters and a "which layer" tag differ.

**PGAS uniform-latency simplification.** Model every remote location at a single `L_net`, no topology,
no distance. Verdict for the OMEN transpose: **sound as a first cut** — the transpose is a bytes-bound
all-to-all in the `T ≈ bytes·G_net` bandwidth regime (BDP > 1), where per-message `L_net` is
second-order (amortized over the transfer). It correctly ranks *which* layout minimizes exchanged bytes
— the ~2.58 PiB → ~1.8 TiB decision the transpose exists to make. Two blind spots it must not hide
(fail-loud on the ranking, never silently misorder):

- **Latency-bound patterns** — halo `Sendrecv`, small messages, the `mpi_pack_unpack_exchange` column
  swap — sit in the *other* regime, where uniform-`L` with no topology under-orders.
- **All-to-all congestion / bisection saturation**, the dominant real cost of the transpose at scale,
  is invisible to an unloaded per-link `G_net`.

The three intra-node gaps above transpose cleanly onto the network layer:

| intra-node gap (What is NOT modelled) | network-layer analogue |
|---|---|
| `D·L` span floor | collective latency floor: `log P·L_net` (tree) / the `P`-term (all-to-all) |
| unloaded `L` (~20% queueing near knee) | congestion-loaded `G_eff = all_to_all_bytes / bisection_BW` |
| no fast-memory-size `S` (no √S tiling bound) | pack granularity — max contiguous buffer before send; *this is what the transpose optimizes* |

Not implemented (design-note-only) pending a measured comm trace of the transpose (bytes, message
count, bisection). The all-to-all bytes and the pack granularity are already readable off the SDFG (the
`Alltoall` buffer subset), so the *ranking* term is derivable without new microbenchmarks; the
`L_net`/`G_net` absolutes are not — same status as the intra-node `c_core, L, G on a quiet box` Open
item.

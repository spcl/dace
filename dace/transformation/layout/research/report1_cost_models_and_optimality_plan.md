# Cost Models for Layout–Schedule Analysis & an Optimality-Proof Action Plan

Consolidates the LogP-family analysis, the audited GPU cost model, and the deep-sweep additions (accelerator-mapping models, learned models, sparse-format α models) into one reference, then gives a concrete plan for proving layouts optimal. Companion to the layout survey (`report2_layout_optimization_survey.md`) and the kernel suite.

---

## Part I — Cost model families

### Family A: LogP and message-based models (latency-aware)

| Model | Params | Predicts | Layout-parametric? |
|---|---|---|---|
| LogP (Culler '93) | L, o, g, P; capacity ⌈L/g⌉ | msg time, in-flight bound | no (msg count is input) |
| LogGP (Alexandrov '95) | + G (per-byte) | long-message bandwidth | no |
| LogGPS / LogGOPS (Ino; Hoefler) | + S (sync), O (per-byte overhead) | rendezvous, o(size) | no |
| pLogP (Kielmann) | L,o,g as f(size) | size-dependent transfer | via effective msg size |
| memory-logP (Cameron & Sun) | L,o,g,G on memory hierarchy | single-node memory transfer | closest existing home |
| LogCA (Altaf & Wood ISCA'17) | L,o,g(=data size),C,A | offload break-even g₁ | **no** (g is size, no access pattern) |

Message time (LogGP): a k-byte message costs `o + (k−1)·G + L`, consecutive messages separated by `max(g,o)`; at most `⌈L/g⌉` in flight (= Little's Law concurrency).

**Layout enters as message fragmentation** (the SC26/audit contribution): for access set A under layout Φ at hierarchy level ℓ with granularity mℓ, `frag_ℓ(A,Φ)` = number of level-ℓ messages. Contiguous load = 1 long message; scattered = N short messages. Per-level counting rule:

| Level | granularity | rule | worst case (fp32) |
|---|---|---|---|
| L2←DRAM coalescer | 32 B sector | distinct sectors touched | ×8 |
| shared memory | 4 B bank ×32 | max bank multiplicity (serialization) | ×32 |
| DRAM channel | channel hash | max channel multiplicity (camping) | ×#channels |
| TMA/DMA | descriptor tile | contiguous runs | per-element |

Decisive limitation: LogP has **no fast-memory-size S**, so it cannot generate the √S reuse of a tiling bound. Correct architecture: pebble game supplies Q\* (layout-sensitive via B); LogGP converts Q\* to time. LogCA is rejected for layout (its g is data size; a scattered and contiguous tile of equal size are identical to it).

### Family B: memory/cache-hierarchy analytical models

- **ECM (Execution-Cache-Memory; Hager/Wellein/Stengel/Hofmann)** — predicts single-core cycles as max/overlap of in-core compute and data transfers through each cache level: `T_ECM = max(T_core, T_data)` with `T_data = Σ_levels bytes_ℓ / BW_ℓ`. Bandwidth-centric; **no MLP/latency term** — a gap your latency roof fills. Layout enters only through bytes_ℓ (volume), so ECM sees the bandwidth-waste half but not scatter-at-constant-volume.
- **Roofline / Cache-Aware Roofline (CARM; Ilic et al.)** — performance vs arithmetic intensity ceilings; CARM adds per-cache-level bandwidth ceilings. No native latency ceiling (the gap your bound's third roof addresses).
- **HAYSTACK (Gysi et al. PLDI'19)** — fast analytical fully-associative cache model for affine programs (Ehrhart-polynomial cache-miss counts); Bao et al. POPL'18 similar. Layout-parametric in principle (miss count depends on layout), volume-only (no latency).
- **MWP-CWP (Hong & Kim ISCA'09)** — GPU model: memory-warp-parallelism vs compute-warp-parallelism; the closest existing latency+bandwidth+MLP GPU model. Takes the access pattern as *input* rather than as a function of layout — exactly the seam frag_ℓ(Φ) fills. Successors (GPUMech, MDM, GCoM) are trace/profile-driven.
- **Volkov (latency hiding on GPUs)** — Little's Law concurrency = latency × throughput; the mechanistic basis for the S_flight term.

### Family C: I/O-complexity / pebble-game models (volume lower bounds)

Hong–Kung red-blue pebble game (PSPACE-complete in general, Demaine–Liu SPAA'18); Aggarwal–Vitter external memory with block B; Savage red-blue-white; blocked-CDAG layout-aware bound (SC26: layout inflates the block-granular I/O lower bound up to B×); IOLB/IOOpt (Olivry PLDI'20/'21, automated affine bounds via sub-CDAG selection); COSMA/SOAP (Kwasniewski); multiprocessor red-blue pebbling (Böhnlein–Papp–Yzelman SIROCCO'25); disaggregated-memory pebble games (MEMSYS'25). These are the only family that yields a **lower-bound certificate**; all are volume-only (latency-blind), which is why they must be *combined* with Family A/B for time.

### Family D: layout-parametric traffic models (the exemplars)

- **SELL-C-σ α-model (Kreutzer et al. SISM'14)** — SpMV traffic per row includes an α term for the RHS gather `x[col]`: effective bytes = matrix bytes + α·(RHS bytes), where α∈[1/B, 1] captures how many distinct cache lines the gather touches. This is the cleanest published *layout-parametric* cost term and the direct model-side justification for frag counting; it is exactly what LayoutBench N1/k05 measures via the σ-sweep.
- **OSKI / SPARSITY (Vuduc–Demmel–Yelick, SC'02, SciDAC'05)** — register-blocked SpMV with **upper and lower Mflop/s bounds** and a fill-ratio (Pad-waste) model; hybrid offline-profile × online-fill selection of block size. The twenty-year-old ancestor of both Kreutzer's α and the optimality-certificate route.

### Family E: accelerator mapping / dataplacement models (added this sweep)

Timeloop (ISPASS'19), MAESTRO (MICRO'19), ZigZag (TC'21): analytical energy/latency over the full mapspace (tilings × loop orders × spatial partitionings). TCM (arXiv 2602.15172): the *dataplacement* concept prunes 10³⁷→10⁵ mappings while preserving optimality — full optimal search in 17 s. CMDS (arXiv 2406.14574): cross-layer **data-layout** optimization with a fine-grained inter-layer cost model. These are the closest existing "analytical cost model + search for optimal mapping/layout" systems; Timeloop's finding that 6,582 minimal-DRAM mappings vary 11× in energy is the volume-optimal-≠-optimal thesis at accelerator scale.

### Family F: learned cost models (added this sweep)

WACO (ASPLOS'23, sparse-CNN over the sparsity pattern, format+schedule co-opt); XLA learned TPU cost model (Kaufman et al. MLSys'21) + TpuGraphs dataset (NeurIPS'23, HLO graph × layout config × runtime); Halide/TVM learned autoschedulers. The learned alternative to analytic frag; position analytic-vs-learned explicitly (analytic = interpretable, certifiable, no training data; learned = captures unmodeled hardware, needs data, no certificate).

### The combination (three-roof) model

The unifying time bound (yours, now grounded across families):

    T ≥ max( N³/W_c ,        [compute roof]
             Q*·B / BW ,     [bandwidth roof — ECM/roofline/ Kreutzer-α agree]
             Q*·L / S_flight )[latency roof — LogP capacity/Little's Law; MWP]

with the coupling `S = S_work + S_flight`, `Q* = Q(S_work)` from the pebble game. Multi-buffering (double/triple) makes `S_eff = S/k − BW·L`; the volume cost of this is only ~√k (why the pebble game looked "flat"), while the latency term it pays for is what dominates — a term no volume model expresses. Fragmentation consumes the in-flight budget *per transaction not per byte*, so scattered access drops useful bandwidth by B even when the wires are idle (this is what MWP formalizes per-warp).

**Model selection (which roof binds):** compute-bound → roofline/W_c; bandwidth-saturated with contiguous access → ECM/roofline (volume ≈ real); scattered at constant volume → the o/g message-count terms; low-MLP / near-ridge → the L/S_flight latency roof. Validation by counter attribution: DRAM-bytes counter ↔ Q*·B (bandwidth), sectors-per-request ↔ frag_sec, shared-memory wavefronts actual/ideal ↔ bank conflicts, long-scoreboard stall cycles ↔ latency roof. A right answer for the right reason requires the attribution to match, not just the ranking.

---

## Part II — Action plan: proving we find optimal layouts

"Optimal" is bounded by hardness: cache-conscious data placement is NP-hard (Petrank–Rawitz), GPU data repositioning is NP-hard (Wu et al.), dynamic remapping is NP-complete (Kremer), and minimum-linear-arrangement (the ordering core) is NP-hard. So "prove optimal" can only mean one of three well-defined claims. The plan pursues all three at different scopes.

### Route 1 — Optimal within a declared candidate space (ILP / exhaustive), with an optimality-preserving pruning certificate
- Declare the layout space per array as compositions of the five primitives (finite, small per array).
- Score each candidate with the three-roof analytic model (frag via Barvinok for affine, replay for indirect).
- **Certificate**: prove the pruning steps (μ/Δ filters; per-array argmin; the binding-roof enumeration) discard only dominated candidates, so the exhaustive residual search is optimal *within the declared space*. This is exactly TCM's move (dataplacement pruning 10³⁷→10⁵ while preserving optimality) and Kennedy–Kremer's 0-1 ILP — cite both as precedent that this claim is respectable and achievable.
- Deliverable: for each LayoutBench kernel, enumerate the full (Φ,θ,repack) grid, prove the pruned set contains the grid optimum, and report top-1 regret = 0.

### Route 2 — Lower-bound certificate (match a modeled/measured cost to an information-theoretic bound)
- Use IOLB/IOOpt (affine) or the blocked-CDAG bound to compute the layout-aware I/O lower bound Q\*_LB.
- A layout whose measured DRAM traffic matches Q\*_LB·B is **certifiably bandwidth-optimal** (no layout moves fewer bytes). OSKI's SC'02 upper/lower Mflop/s bounds are the precedent for this exact certificate on a layout decision.
- For the latency roof, the certificate is weaker (no tight latency lower bound for general CDAGs); state this honestly and use the bandwidth certificate where the kernel is bandwidth-bound (GEMM, dense), the model-ranking validation where it is latency-bound.

### Route 3 — Restricted-algebra exact optimum (closed-form for a sub-problem)
- For sub-problems where the algebra is small and the objective convex, solve exactly and prove it: e.g., Triton Linear Layouts computes the *optimal* swizzle over F₂ (vectorization vs bank conflicts); the per-array "make the innermost streamed dimension contiguous" choice is a closed-form argmin; AoS/SoA/AoSoA field partition is exact by enumeration over the small field-affinity graph.
- Deliverable: identify which LayoutBench levels admit a closed-form optimum (N2 field partition; single-reference permutation) and prove it there; use Route 1 elsewhere.

### Validation methodology (all routes)
- **Oracle**: LayoutBench kernels are sized so the full (Φ,θ) grid is exhaustively measurable → the true optimum is known, and the model's top-1 regret and Kendall-τ against the oracle are directly computable.
- **Baselines to beat**: volume-only ranking (pebble/ECM — must fail on scatter-at-constant-volume and bank conflicts, which is the motivation figure) and per-nest greedy (must fail on the global-conflict kernels, the "layouts are global" figure).
- **Counter attribution**: as above — the model must be right for the right reason.
- **Cross-device check**: k06/k08 (sorted-shuffle, padded-gather) win on GPU but not CPU; a correct model must predict the device-dependent verdict (the Kokkos LayoutLeft/Right flip is the production analog).

### Threats to state
- **Inspector-executor subsumption (2026-07-17).** The first reviewer objection to static replace of a
  static indirection will be "this is inspector-executor, and it is from 1999" (Ding & Kennedy PLDI'99;
  Mellor-Crummey, Whalley, Kennedy IJPP'01). **We are not pursuing I/E** — no I/E benchmark, no runtime
  inspector — but the answer must be on file, because the distinction is narrow and it is quantitative,
  not conceptual. Both remove `A[sigma[i]]` by physically reordering `A`; they differ only in what they
  pay to learn `sigma`. In the break-even
  `passes * (1/eps0 - 1/eps1) >= 2 + overhead_passes`, static replace is exactly the
  `overhead_passes -> 0` limit of I/E:

  | | overhead | eps0=1/8 (scattered) | eps0=1/2 (mild) |
  |---|---|---|---|
  | static replace (`sigma` known without running) | 0 | 1 pass | 2 passes |
  | inspector-executor (`sigma` at runtime: read + bucket the index array) | ~3 | 1 pass | 5 passes |

  So the honest claim is NOT "we are faster than I/E" — at a badly scattered gather (eps0=1/8) I/E's
  first executor pass already wins too, and the gap is small. The defensible claims are: (i) static
  replace needs **no reuse argument at all**, so it applies where I/E cannot amortize (a single pass,
  a `sigma` that changes every step); (ii) it pays **nothing** at runtime; and (iii) it composes with
  the other four primitives inside one algebra, whereas I/E is a standalone runtime technique. If a
  reviewer shows a case where I/E amortizes just as well, we lose nothing but the novelty of (i)-(ii)
  — so do not overclaim speed. Quantified by `relayout_pays_by_efficiency(..., overhead_passes=...)`
  and pinned in `tests/transformations/layout/cost_model_access_patterns_test.py`.
- Restricted transform space (only bijective five primitives; im2col/CSR/Dremel excluded — see survey gap analysis).
- Surrogate-vs-hardware gap (analytic frag vs coalescer/prefetcher/replacement-policy reality); mitigated by counter attribution.
- Phased-vs-joint search (the single latency back-edge handles the one region where the decoupling fails; provably volume-optimal in the bandwidth regime).
- No tight latency lower bound → Route 2 certifies bandwidth-optimality only.

---

## Part III — One-paragraph positioning for the paper
Existing models split cleanly: volume-only lower bounds (pebble/IOLB, layout-aware but latency-blind), bandwidth-centric analytical models (ECM/roofline/CARM, no MLP), latency+MLP GPU models (MWP-CWP/Volkov, but access pattern is an input not a function of layout), and learned or mapspace models (WACO/TpuGraphs/Timeloop-TCM, powerful but either uncertifiable or accelerator-scoped). None couples a *layout-parametric message count* to a latency+bandwidth+MLP time model *and* an I/O lower bound. The three-roof model does exactly that composition — Kreutzer's α and frag counting supply the layout-parametric traffic, MWP/LogGP supply the latency conversion, the pebble game supplies the certifiable lower bound — and the S_work/S_flight coupling (absent from all prior models) is what explains why volume inflation looks negligible at real cache sizes while measured layout penalties are large.

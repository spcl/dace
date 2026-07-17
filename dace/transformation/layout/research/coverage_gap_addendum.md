# Coverage-gap addendum (post-internalization web sweep)

Cross-checked the two research reports (cost models & kernels; layout-transformation survey), the v2 audited model report, the LayoutBench design, and the kernel suite against a targeted second sweep. Six bodies of work are genuinely missing. Each entry: what it is, why it matters, and where it slots.

## 1. Accelerator mapping cost models: Timeloop / MAESTRO / ZigZag (+ TCM, CMDS)

- Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator Evaluation," ISPASS 2019 — constructs the complete mapspace (tilings × loop orders × spatial partitionings) for an architecture and searches it for optimal mappings under a detailed analytical energy/latency model; their Fig. 1 point that 6,582 mappings share minimal DRAM traffic yet vary 11× in energy is precisely the "volume-optimal ≠ optimal" argument of the SC26 line, made at accelerator scale.
- Kwon et al., MAESTRO (MICRO 2019 / IEEE Micro Top Picks 2020) — analytical dataflow cost model (reuse-based, spatio-temporal occupancy).
- Mei et al., ZigZag (IEEE Trans. Computers 70(8), 2021) — enlarged (uneven) mapping space, memory-hierarchy-aware.
- **TCM, "The Turbo-Charged Mapper" (arXiv 2602.15172)** — introduces *dataplacement* as the analysis concept and prunes the mapspace 10³⁷→10⁵ **while preserving optimality**, making full-mapspace optimal search feasible (17 s). This is the strongest existing precedent for the "prove layout optimal within a declared space by optimality-preserving pruning + exhaustive residual search" route in the action plan.
- **CMDS (arXiv 2406.14574)** — cross-layer **data-layout** optimization for DNN accelerators with multi-bank memories: per-layer optimal mapping from ZigZag/Timeloop, then a fine-grained cost model over inter-layer layout dependencies. This is the layout-state-graph idea, shipped in the accelerator-compiler world.

Slots: cost-model report gains a fourth model family ("accelerator mapping models: analytical, layout/dataflow-parametric"); the optimality plan gains TCM as the pruning-with-certificate precedent; the Maia path-(i) internship proposal should cite Timeloop/MAESTRO/ZigZag as the modeling baseline a kernel-level spatial dataflow language must relate to.

## 2. OSKI / SPARSITY: the original layout-selection-by-cost-model system

- Vuduc, Demmel, Yelick, "OSKI: A library of automatically tuned sparse matrix kernels," SciDAC/J. Phys. Conf. 2005; Im, Yelick, Vuduc, SPARSITY, IJHPCA 18(1), 2004.
- **Vuduc et al., "Performance optimizations and bounds for sparse matrix-vector multiply," SC 2002** — develops upper and lower Mflop/s bounds for register-blocked SpMV and a hybrid offline/online heuristic (offline machine profile × online fill-ratio estimate) to select block size. This is (a) the direct ancestor of Kreutzer's SELL-C-σ α model, (b) a twenty-year-old **bound-certificate methodology for a layout decision** — measured performance compared against a per-(matrix, machine) bound — that the optimality action plan should cite as prior art for its route (b), and (c) a fill-ratio (Pad-waste) cost model matching the ELL/SELL padding term already in LayoutBench N1/k05.
- Choi, Singh, Vuduc, "Model-driven autotuning of sparse matrix-vector multiply on GPUs," PPoPP 2010 — BCSR/BELLPACK on GPUs, model-driven parameter choice.

Slots: cost-model report (layout-parametric traffic models, before Kreutzer); optimality plan route (b); k05 docstring attribution could add OSKI/SPARSITY.

## 3. YASK vector folding: a missing survey entry (Block+Permute for stencils)

- Yount, "Vector Folding: Improving Stencil Performance via Multi-dimensional SIMD-vector Representation," HPCC 2015; Yount et al., YASK, WOLFHPC@SC 2016 (framework, genetic-algorithm tuner); production at Intel.
- Mechanism (survey-precision): instead of the traditional 1D in-line SIMD layout along the unit-stride dimension, elements are packed into small multi-dimensional tiles (e.g., 8 doubles as 1×8, 2×4, or 2×2×2 per 512-bit register), and these folded tiles may themselves be ordered in memory along a dimension independent of the vectorized dimensions. Before: addr(i,j,k)=((i·Nj)+j)·Nk+k, vector = 8 consecutive k. After (4×2 fold, k-major tile order): A_folded[i, j//4, k//2, j%4, k%2] — a Block of (j,k) into (4,2) tiles + Permute of tile order. Reduces redundant loads for multi-dimensional stencils (each neighbor access reuses more of the loaded vector).
- This is the stencil-world production counterpart of the DLT entry (Henretty CC'11) already in the survey, and it belongs in the same blocking family; it also connects to k12.

Slots: survey family 3 (blocking/nonlinear layouts); kernel suite k12 docstring cross-reference.

## 4. GraphIt: layout as a first-class scheduling-language directive

- Zhang et al., "GraphIt: A High-Performance Graph DSL," OOPSLA 2018 — the scheduling language includes **vertex-data layout** directives: vector tags configured with the `fuseFields` scheduling function choose AoS vs SoA per vertex-data group, alongside traversal direction, parallelization, and an autotuner over the schedule space.
- Why it matters: it is the clearest existing system where **Zip/Unzip is a schedule-space dimension searched by an autotuner** — the graph-domain precedent for treating the layout algebra as a searchable space. For the SC26 Table V "no existing framework jointly supports all five primitives" claim, GraphIt doesn't threaten the claim (no shuffle/pad/block of field data) but should be cited next to Zip coverage, and its autotuner is a baseline for the N1/N3 ordering-and-layout search.

Slots: survey family 5 (AoS/SoA) + family 7 (joint frameworks); LayoutBench N1 baseline list.

## 5. WACO and sparse format-schedule co-optimization with learned cost models

- Won, Mendis, Emer, Amarasinghe, "WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program," ASPLOS 2023 — a sparse-CNN cost model over (sparsity pattern, format, schedule) plus ANNS search over the joint space; 1.14–1.43× average over MKL/BestFormat/TACO/ASpT across 726 sparsity patterns on SpMV/SpMM/SDDMM/MTTKRP.
- Ahrens, Kjolstad et al. (co-optimizing computation and format via asymptotic ranking) — the analytical counterpart in the TACO line.
- Why it matters: (a) the strongest current *learned* alternative to the analytic frag-based cost model — the model-selection section should position analytic-vs-learned explicitly; (b) 726-pattern evaluation is the methodology precedent for N1's "sweep graph families, predict the crossover" protocol; (c) it validates the thesis that format (layout) and schedule must be co-optimized, from the sparse side.

Slots: cost-model report (learned family); LayoutBench N1 evaluation protocol + baselines.

## 6. XLA layout assignment, the graph-level production system + its datasets

- OpenXLA docs (XLA:GPU architecture): the Layout Assignment pass chooses per-op physical layouts (minor-to-major dimension orders) under constraints, propagates them through the graph, and materializes conflicts as inserted copy/transpose ops — Kennedy–Kremer's structure running in production at Google scale, alongside the ALT/TVM/SmartMem entries already in the survey.
- TpuGraphs (Phothilimthana et al., NeurIPS 2023 D&B, arXiv 2308.13490): a public dataset of HLO graphs × **layout configurations** × measured TPU runtimes, released because the XLA graph-level autotuner (which tunes layout assignment among other passes and has delivered 10–20% production speedups) takes hours to converge; layout tuning is singled out as offering the most speedup in general. Kaufman et al. (MLSys 2021) is the learned TPU kernel cost model in the same stack.
- A current production pain point worth citing in motivation: the Ragged Paged Attention TPU kernel report (arXiv 2604.15464) documents XLA's layout assignment overriding user-specified layout constraints and silently transposing intermediates — evidence that even mature layout-assignment systems lack a principled, per-kernel-accurate layout cost model.

Slots: survey family 7 (production layout propagation, next to ALT); cost-model report (learned family + the "graph-level vs kernel-level" boundary); LayoutBench positioning (TpuGraphs is the DNN-graph-level layout benchmark; LayoutBench is the HPC kernel-level one — complementary, not overlapping).

## Verified as already covered (no action)

Padding lineage (Rivera–Tseng, Hong PLDI'16), permutation matrices (Kandemir, O'Boyle), Morton/4D (Chatterjee), swizzles (CuTe, Linear Layouts), AoS/SoA (ASTA, Cabana, Zhong, Chilimbi, PAX), SELL-C-σ, oneDNN nChw16c, NCHW/NHWC, Dymaxion, G-Streamline, Gorder/Rabbit/RCM, Hilbert/Peano, Shirako–Sarkar, ALT, DaCe-OMEN, Kennedy–Kremer/Bixby ILP, im2col/CSR scope exclusions, LogP family incl. memory-logP and LogCA, ECM/CARM/HAYSTACK/Bao-POPL18, MWP-CWP/Volkov/GPUMech/MDM, pebble-game lineage incl. IOLB/IOOpt/SOAP and SIROCCO'25/MEMSYS'25, hardness results (Petrank–Rawitz, Wu, Kremer, MinLA), oracle methodology (ATLAS/FFTW), LLAMA/Kokkos/GAP benchmark positioning.

## Flagged, unverified (cite only after checking)

Interleaved batched small-GEMM layouts in MAGMA (Abdelfattah et al.) as a k10/k11-adjacent production example — plausible, not verified this sweep. Data Calculator (Idreos et al., SIGMOD 2018) as cost-model-driven data-structure/layout synthesis in databases — adjacent to the optimality plan, not verified this sweep.

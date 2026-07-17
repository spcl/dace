# Report 3 — Attributions, real-world layout bugs, and what to add to the suite

Compiled 2026-07-17. Sources gathered by web search + direct fetch. **Verification status is recorded
per item and must be preserved**: ✅ = full text retrieved and the quoted text extracted from it;
⚠️ = located with a real URL, full text NOT verified (claims limited to abstract/search text);
❌ = searched for and NOT found — do not cite.

Nothing in this file is cited from memory. Items that could not be verified are listed as such rather
than dropped silently, because the gap is itself information.

---

## 0. The headline finding — layout is missing from the performance-bug literature

**This is the strongest motivation available for the SC26 paper, and it is a measured claim, not a
rhetorical one.**

The canonical empirical study of performance bugs — Jin et al., PLDI 2012 — names data layout exactly
once, in order to rule it out of scope (§4.1, verbatim ✅):

> "There are a large variety of potential root causes for inefficient code, such as poorly designed
> algorithms, non-optimal data structures, **cache-unfriendly data layouts**, etc. **Our goal here is
> not to discover previously unheard-of root causes**, but to check whether there are common
> root-cause patterns among real-world performance bugs that bug detection can focus on."

Its conclusion fixes the field's centre of gravity elsewhere:

> "Common patterns do exist and **performance is mostly lost at call sites and function boundaries**."

Its taxonomy is *Uncoordinated Functions / Skippable Function / Synchronization Issues / **Others***.
The entire layout class lives in "Others" (23 of 110 bugs), described only as "Some use wrong data
structures. Some are related to hardware architecture issues."

Every successor inherited the call-site/API framing. Full-text grep for
`data layout|memory layout|AoS|SoA|field order|cache line|spatial locality|coalesc` over each
downloaded paper:

| Study | Venue | Corpus | layout hits |
|---|---|---|---|
| Azad et al. ✅ | **MSR 2023** | 186 HPC perf issues / 23 projects | **17 — real, first-class** |
| Jin et al. ✅ | PLDI 2012 | 110 bugs / 5 suites | 2 (1 = the scoping sentence above) |
| Zhao et al. ✅ | IEEE TSE | 570 issues / 13 projects, 3 languages | 2 — **both false positives** ("Soares", "soap") |
| Liao et al. ✅ | TOSEM 2025 | 60k reviews + 749k SO threads | 1 — **false positive** (author name) |
| Cao et al. ✅ | FSE 2022 | 224 DL perf problems | 1 — **false positive** (bibliography) |
| Liu ICSE'14 · Makkouk ICSME'22 · Selakovic ICSE'16 · Mazuera EMSE'20 · Han ESEM'16 · Yi OOPSLA'26 · Nistor MSR'13 · Zaman MSR'12 ✅ | — | — | **0** |

**The one exception proves the rule.** Azad et al., MSR 2023 — the only study that mined *commits* in
*scientific/HPC* code — has layout as a first-class category: "Microarchitectural (58)" → **"Data
locality (36)"** → **"Cache locality (18)"** + "GPU memory (18)". That is ~19% of its corpus. Concrete
fixes it documents: a Kokkos false-sharing bug fixed "by padding the array so that elements in the
array reside in the separate cache line" (**200× on 20 threads**); cp2k's `Gamma_P_ia` with "poor
spatial locality due to its access pattern".

**Why the absence is structural, not a search artifact:** layout bugs are found by *profilers*, not
filed in *bug trackers*. A bug-tracker-mining methodology is blind to them by construction. The moment
someone mined commits in HPC code, layout appeared immediately at 19%.

### Three traps to avoid when citing this literature

1. **"Data structure" ≠ "data layout."** Zhao et al.'s "Inefficient Data Structure" category is
   container/ADT selection — its own Table 5 lists the fixes: "Array, List → Set, Map (23)",
   "HashMap → ConcurrentHashMap (8)", "String, StringBuffer → StringBuilder (7)",
   "LinkedList → ArrayList (2)". Asymptotics and API, not physical arrangement. Exactly one of 54 is
   layout-flavoured (MESOS-2126, Vector over Queue to "reduce CPU cache usage"). **Citing IDS as
   layout coverage is a misreading.**
2. **"Layout" in mobile papers means Android XML UI layout.** Mazuera-Rozo EMSE'20 ("Layout files
   optimization") and Liao TOSEM'25 ("Frequent layout inflations during scrolling", "Improper layout
   stacking") — keyword-searching those corpora for "layout" produces false hits.
3. **DL taxonomies omit tensor layout entirely.** Neither Makkouk ICSME'22 (12 categories / 19
   sub-categories) nor Cao FSE'22 (5 categories / 15 inner) has an NCHW-vs-NHWC category, despite it
   being a first-order DL performance concern. Nearest neighbours: "Inefficiency for specific sizes
   and/or shapes", "Inefficient caching", "Inefficient Data Transmission".

**Corrected misattribution (recorded so it does not resurface):** a search engine attributed to Han &
Yu (ESEM'16) the claim "over half of performance bugs arise from inefficient data processing … within
data structures". **That sentence is not in the paper.** Its actual Finding 10: "A majority of
performance bugs involves inefficient memory usage (up to 35%)".

**Highest-value open lead:** Bi et al., "Understanding Performance Problems in CUDA Programs",
PACMSE/FSE 2026 — 216 CUDA perf problems, 69 reproduced. The one domain where a coalescing/layout
category is likely. Taxonomy **not accessible** (ACM DL 403, no preprint). Worth institutional access.

---

## 1. Performance-bug studies — attributions

| # | Paper | Venue | Corpus | Dataset? | ver |
|---|---|---|---|---|---|
| 1 | Jin, Song, Shi, Scherpelz, Lu, *Understanding and Detecting Real-World Performance Bugs* | PLDI 2012 | 110 bugs (Apache, Chrome, GCC, Mozilla, MySQL) | rule-checkers, [perfevo](https://github.com/songlh/perfevo) | ✅ |
| 2 | Zaman, Adams, Hassan, *A Qualitative Study on Performance Bugs* | MSR 2012 | 400 reports (Firefox, Chrome) | no | ✅ |
| 3 | Nistor, Jiang, Tan, *Discovering, Reporting, and Fixing Performance Bugs* | MSR 2013 | 210+210 (Eclipse JDT/SWT, Mozilla) | no | ✅ |
| 4 | Han, Yu, *Perf Bugs for Highly Configurable Software* | ESEM 2016 | 113 config-related | no | ✅ |
| 5 | Song, Lu, *Statistical Debugging for Real-World Performance Problems* | OOPSLA 2014 | — | — | ⚠️ |
| 6 | Selakovic, Pradel, *Performance Issues and Optimizations in JavaScript* | ICSE 2016 | 98 issues / 16 projects | [yes](https://github.com/marijaselakovic/JavaScriptIssuesStudy) | ✅ |
| 7 | Liu, Xu, Cheung, *Characterizing and Detecting Perf Bugs for Smartphone Apps* | ICSE 2014 | 70 bugs / 8 Android apps | PerfChecker | ✅ |
| 8 | Mazuera-Rozo et al., *Types and Survivability of Perf Bugs in Mobile Apps* | EMSE 25(3) 2020 | **500** bugs / 78 apps | [yes](https://github.com/amazuerar/perf-bugs-mobile/) | ✅ |
| 9 | Zhao, Xiao, Bondi, Chen, Liu, *Large-Scale Empirical Study of Real-Life Perf Issues* | IEEE TSE 2022 | **570** issues, Java/C++/Python | — | ✅ |
| 10 | **Azad, Iqbal, Hassan, Roy, *An Empirical Study of HPC Performance Bugs*** | **MSR 2023** | 1,729 commits → **186** issues / 23 HPC projects | [figshare](https://figshare.com/s/00c24aae3177e45db7ab) | ✅ |
| 11 | Makkouk, Kim, Chen, *Perf Bugs in Deep Learning Frameworks* | ICSME 2022 | 141 fixes (TF, PyTorch) | [yes](https://github.com/dlframeworkperfbugs/) | ✅ |
| 12 | Cao et al., *Understanding Performance Problems in Deep Learning Systems* | ESEC/FSE 2022 | 224 (TF/Keras) | DeepPerf | ✅ |
| 13 | Liao et al., *Android Perf Issues in Real-world Apps vs Literature* | TOSEM 2025 | 60,684 reviews + 749,067 SO threads | [yes](https://github.com/Dianshu-Liao/Android-Performance-Analysis) | ✅ |
| 14 | Bi et al., *Understanding Performance Problems in CUDA Programs* | PACMSE/FSE 2026 | 216 problems, 69 reproduced | claimed | ⚠️ |
| 15 | Muse et al., *Data-Access Performance Anti-Patterns* | EMSE 2024 | 526,672 issues | — | ⚠️ (arXiv v1 = registered report only; cite the EMSE version) |
| 16 | Yi, Ding, Shi, Gligoric, *Understanding and Finding JIT Compiler Performance Bugs* | PACMPL OOPSLA1 2026 | 191 bugs / 4 JITs | [jittery](https://github.com/EngineeringSoftware/jittery) | ✅ (0 layout hits) |
| 17 | Rathnasuriya et al., *Real-World Bugs in Tile Programs* | 2026 | 301 codegen bugs / 8 frameworks | yes | ✅ ("Indexing and Stride … layout transformations", 35/301 — but a **correctness** taxonomy) |
| 18–20 | Zhou et al. (GCC/LLVM opt bugs, JSS'20) · Theodoridis, Grosser, Su (ASPLOS'22) · MOD (FSE'23) | — | compiler, correctness-oriented | — | ⚠️ |
| 21 | Garg et al., *PerfBench: Can Agents Resolve Real-World Performance Bugs?* | arXiv 2509.24091 | 81 .NET tasks | claimed | ⚠️ |
| 22 | Yi, Gay, Leitner, *Do AI Models Dream of Faster Code?* | arXiv 2510.15494 | 65 Java tasks | — | ⚠️ |

---

## 2. Layout-transformation papers — attributions

### Foundational (12)

| Paper | Venue | Contribution |
|---|---|---|
| Lam, Rothberg, Wolf, *The Cache Performance and Optimizations of Blocked Algorithms* | ASPLOS 1991 | Blocked-algorithm cache interference is acutely stride-sensitive — made layout×tiling a first-class concern |
| Cierniak, Li, *Unifying Data and Control Transformations for DSM Machines* | PLDI 1995 | Loop transformation + array layout as ONE search problem |
| Anderson, Amarasinghe, Lam, *Data and Computation Transformations for Multiprocessors* | PPoPP 1995 | First automatic parallelize + relayout system (SUIF) |
| Coleman, McKinley, *Tile Size Selection Using Cache Organization and Data Layout* | PLDI 1995 | Tile size from cache size + line size + layout |
| **Rivera, Tseng, *Data Transformations for Eliminating Conflict Misses*** | **PLDI 1998** | **The canonical inter-/intra-variable padding treatment — our Pad primitive's origin** |
| Kandemir, Choudhary, Ramanujam, Banerjee, *Improving Cache Locality by a Combination of Loop and Data Transformations* | IEEE TC 48(2) 1999 | Unified loop+layout; handles **fixed** layouts for some arrays |
| Chilimbi, Hill, Larus, *Cache-Conscious Structure Layout* | PLDI 1999 | Structure splitting + field reordering |
| Ding, Kennedy, *Improving Cache Performance in Dynamic Applications …* | PLDI 1999 | Runtime data + computation reordering |
| Chatterjee, Lebeck, Patnala, Thottethodi, *Recursive Array Layouts and Fast Parallel Matrix Multiplication* | SPAA 1999 / TPDS 2002 | Z/Morton + recursive block layouts, 1.2–2.5× |
| Mellor-Crummey, Whalley, Kennedy, *Improving Memory Hierarchy Performance for Irregular Applications* | IJPP 29(3) 2001 | Space-filling-curve data+computation reordering, 2–4× |
| **Henretty, Stock, Pouchet, Franchetti, Ramanujam, Sadayappan, *Data Layout Transformation for Stencil Computations on Short-Vector SIMD*** | **CC 2011** | **Stream alignment conflict; dimension-lifting-transpose (DLT) — k12's direct ancestor** |
| Che, Sheaffer, Skadron, *Dymaxion: Optimizing Memory Access Patterns for Heterogeneous Systems* | SC 2011 | Programmer-directed layout remap to restore GPU coalescing |
| *(sparse pair)* Kjolstad et al. *taco* OOPSLA'17 · Chou, Kjolstad, Amarasinghe, *Format Abstraction* OOPSLA'18 | | Arbitrary dense/sparse format composition |

### Recent (2023–2026) — **our layout algebra's actual competition; must be cited**

| Paper | Venue | Why it matters to us |
|---|---|---|
| **Zhang et al., *Hexcute: Automating Layout Synthesis in GPU Programs*** | arXiv 2504.16214 (2025) | GPU layout selection as **constraint programming solved by type inference** |
| **Zhou et al., *Linear Layouts: Robust Code Generation … Using 𝔽₂*** | arXiv 2505.23819 (2025, rev. 2026) | **Binary matrix algebra over 𝔽₂ as a uniform layout model — shipped in Triton.** The closest published thing to our digit algebra |
| **Carlisle, Shah, Stern, VanKoughnett, *Categorical Foundations for CuTe Layouts*** | arXiv 2601.05972 (Jan 2026) | Category-theoretic semantics for CUTLASS/CuTe layout algebra; complete characterization of representable layouts |
| Xu et al., *ALT: Breaking the Wall between Data Layout and Loop Optimizations* | EuroSys 2023 | Argues the graph-layout / operator-loop split is artificial — joint tuning |
| Ben-Nun, Ates, Calotoiu, Hoefler, *Bridging Control-Centric and Data-Centric Optimization* | CGO 2023 | DCIR / DaCe |
| Swatman et al., *Evolutionary Algorithms to Find Cache-Friendly Generalized Morton Layouts* | arXiv 2309.07002 | GA over generalized Morton space, up to 10× |
| Singhal et al., *Marmoset* ECOOP'24 · *SoCal* arXiv 2605.01140 (2026) | | Whole-program ADT layout; SoA for tree-shaped data |
| Radtke, Weinzierl (PPAM'24; C&C:P&E 2025; arXiv 2512.05516) | | Annotation-guided AoS→SoA + GPU offload; 2.6× on GH200 |
| Liu et al., *UniSparse* | OOPSLA 2024 | Custom sparse format + **target-specific memory layouts**, MLIR |
| Hong et al., *Effective Padding of Multi-Dimensional Arrays to Avoid Cache Conflict Misses* | PLDI 2016 | **Optimal** padding for set-associative caches, nested tiles; tool **PAdvisor**. *"conflict misses occur even if the working set is much smaller than cache capacity"* |

### ❌ Searched for and NOT found — do not cite

- **ASX ("Array of Structures eXtended")** — no locatable paper, venue, or URL. Do not cite.
- **Intel SDLT** — real but **vendor documentation**, not a paper. Closest citable artifact: the
  "Vectorization with SDLT" chapter in *Intel Xeon Phi Processor High Performance Programming*, 2nd
  ed., Elsevier 2016.
- **"DLT framework"** as a named system — "DLT" denotes two unrelated things: Henretty's
  *dimension-lifting-transpose* (CC'11) and Sung's *DL* system (InPar'12). No single framework.
- **XLA layout assignment** — no academic paper found; design docs only.
- **Frens & Wise**, **original Cuthill–McKee** — referenced inside other work but not independently
  verified here; omitted rather than reconstructed.

---

## 3. Real-world layout bugs — the three buckets

### 3a. REPOS — verified issues/PRs/commits (every URL fetched; numbers from the page)

| # | Project | What was wrong | Measured effect | Status |
|---|---|---|---|---|
| **R1** | [pytorch/vision#6619](https://github.com/pytorch/vision/issues/6619) | No channels-last `roi_align` kernel: *"on NCHW or channels first memory format, it can only use scalar logic"*. Separately, `Conv2d` reordered between plain and mkldnn blocked formats every call | **RoIAlign 82.6 s → 2.3 s (≈36×)**; share of CPU time 38.33% → 1.61%. Conv2d **88.3 → 63.1 s** from removed reorders. Overall +126.87% 1-core, +206.84% 20-core | fixed |
| **R2** | [apache/tvm#1585](https://github.com/apache/incubator-tvm/issues/1585) | Per-layer schedule tuning picks each layer's own best layout ⇒ **layout transforms injected between every pair of layers** | ResNet50 v1 **8.07 → 5.73 ms (+29%)**; InceptionV3 13.88 → 10.68 (+23%); DenseNet201 (+23%); VGG19-BN (+11%); **SSD-ResNet50 44.56 → 29.05 ms (+35%)** | fixed (graph tuner) |
| **R3** | [torvalds/linux 3a6358c0dbe6](https://github.com/torvalds/linux/commit/3a6358c0dbe6) | `pcpu_chunk`: hot-read `base_addr` shares a line with hot-write `free_bytes`/`chunk_md` | **+24%** at 160-parallel, for ~2 KB total | merged v6.5-rc1 |
| **R4** | [torvalds/linux 802f1d522d5f](https://github.com/torvalds/linux/commit/802f1d522d5fdaefc2b935141bc8fe03d43a99ab) | `page_counter`: hot-write `usage` adjacent to hot-read `parent` | will-it-scale/malloc1 **+8.9%**, pagefault2 **+9.9%**, fio write **+4.5%** | merged |
| **R5** | [facebook/folly#378](https://github.com/facebook/folly/pull/378) | SPSC `readIndex_`/`writeIndex_` co-located ⇒ line ping-pong. (The benchmark was itself contaminated: `iters_` false-shared with `writeIndex_`) | 25,477 → 28,019 ops/ms (**+10%**); RTT 282 → 247 ns; **L1 store misses 6,310,580 → 284,599 (−95%)** | merged |
| **R6** | [scipy#13211](https://github.com/scipy/scipy/issues/13211) | `__rmatmul__` transposes internally and delegates; "the optimal access pattern for untransposed data gives a highly inefficient access pattern for transposed data" | **846 ms vs 70.5 µs** (~10⁵× spread) | **open** |
| **R7** | [numpy#19650](https://github.com/numpy/numpy/issues/19650) | `np.dot` on strided views misses the BLAS fast path and won't copy-to-contiguous | **1.59 s → 8.06 ms (≈197×)** via `ascontiguousarray` | **open** |
| **R8** | [pytorch#37142](https://github.com/pytorch/pytorch/issues/37142) | channels-last propagates through autograd ⇒ backward on slow paths | CPU **5.07 → 21.5 s**; GPU **0.665 → 3.31 s** | open |
| **R9** | [pytorch#50036](https://github.com/pytorch/pytorch/issues/50036) | Incomplete channels-last op coverage: NHWC input treated as non-contiguous NCHW | NCHW **0.97 s** vs NHWC **2.35 s** (layout that should win, loses) | open |
| **R10** | [onnxruntime#18128](https://github.com/microsoft/onnxruntime/issues/18128) | SD 2.1 VAE decoder degenerates to Transpose→Conv→Transpose→Reshape→Transpose→… | **no numbers given** | closed, not planned |
| **R11** | [NVIDIA/cutlass#281](https://github.com/NVIDIA/cutlass/discussions/281) | Epilogue shared-mem pad `[0,16]` where the config needed `[0,8]`; 128-bit access compiled to `st.shared.u32` not `.v4.u32` | **no numbers in thread** | partly |
| **R12** | [rust-lang/rust#37429](https://github.com/rust-lang/rust/pull/37429) | Field reordering (least→most aligned) to kill padding holes | **no numbers**; merged then **disabled by #38523** — broke Servo, `repr(C,packed)` size bug (21 B computed as 24 B) | reverted |

⚠️ Referenced by the [kernel false-sharing doc](https://docs.kernel.org/kernel-hacking/false-sharing.html)
but **not individually fetched, no numbers**: `91b6d3256356` (tcp_memory_allocated), `7b1002f7cfe5`
(bcache), `292648ac5cf1` (gup FOLL_PIN), `520f897a3554` (ext4 percpu_counters), `56f3547bfa4d`
(vm_committed_as_batch).

### 3b. BENCHMARKS — suites that exercise layout

| Suite | URL | Layout effect |
|---|---|---|
| STREAM | https://www.cs.virginia.edu/stream/ | Bandwidth ceiling; arrays sized ≫ cache |
| BabelStream | https://github.com/UoB-HPC/BabelStream | STREAM across every parallel model, CPU+GPU |
| PolyBench/C | https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1 | 30 SCoP kernels; **dimension sizes and padding directly controllable** — the substrate Hong et al. PLDI'16 exploits |
| TSVC | [RISC-V analysis, SC'25](https://dl.acm.org/doi/10.1145/3731599.3767535) | **151 loop nests** built around non-unit strides, reverse accesses, indirect/gather |
| Rodinia | https://github.com/yuhc/gpu-rodinia | [IISWC'10](https://www.cs.virginia.edu/~skadron/Papers/rodinia_iiswc10.pdf) credits it with surfacing "the consequent importance of **data layout**" |
| SuiteSparse | https://sparse.tamu.edu/ | 4,822 matrices — CSR/CSC/blocked format studies |
| **hardware-effects-gpu / bank-conflicts** | https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md | **Verified numbers**: offset=1 → 0 conflicts, 95.74% efficiency, 1 txn/req; offset=32 → ~310,000 conflicts, **3.14% efficiency**, 32 txn/req. Ships a sweep script |
| PerfCurator | https://arxiv.org/html/2406.11731v1 | 408.5K mined perf commits / 322K repos — a mining substrate, not a runnable suite |

### 3c. Honest gaps in the bug evidence

- **No verified AoS→SoA repo fix with measured numbers.** Abundant in papers and blogs; no
  project issue was promoted without verification.
- **No verified uncoalesced-GPU-access project issue.** Same reason.
- Neither gap is evidence the classes are unreal — only that we do not have a citable *repo* instance.

---

## 4. What to add to the suite

Current state (report: assessment 2026-07-17): **2 STRONG (k07, k11), 2 OK, 6 WEAK, 5 CONFOUNDED**,
and every kernel whose speedup the README leads with is confounded (k13 = einsum-vs-BLAS; k15 = 26×
against its own 16× ceiling; k03 = 17× against an 8× bound; k14 = batching, not BFS layout).

### Added this session

| # | File | From | Primitive | Why it is worth having |
|---|---|---|---|---|
| **k16** | `k16_roi_align_channel_gather.py` | **R1** (torchvision#6619, 36×) | Permute | NCHW puts channels at stride H·W ⇒ a per-sample C-vector touches **C distinct lines** vs **C/8**. **No schedule transformation can make the channel axis contiguous** — the paper's claim in its sharpest form. Both variants gather via the *same* `np.take` on a flat buffer, so only index arithmetic differs ⇒ bit-exact, no dispatch confound. **Asserts the 8× line-traffic bound is not exceeded** (the k15/k03 lesson made mechanical). |
| **k17** | `k17_resnet_layout_thrashing.py` | **R2** (tvm#1585, +29–35%) | Permute | The **global assignment** instance the suite lacks. Every other kernel is ≤2 nests / 1 array, so greedy is trivially optimal. Here conv (wants C contiguous) and channel-stats (wants M contiguous) **disagree**, so greedy pays a relayout at every seam. Reports greedy vs single-global vs **exhaustive oracle**, all measured. This is the "per-nest greedy must fail" figure `report1:93` rests on. |

### Recommended next — ranked

1. **Pad / set-conflict kernel (Rivera–Tseng PLDI'98; Hong et al. PLDI'16 + PAdvisor).**
   **The single highest-value addition.** A conflict miss **cannot be tiled away** — so it is the one
   case where *no schedule* can ever close the gap, which is exactly the paper's thesis. Today: Pad
   has **no winning witness anywhere** (k05's SELL is Python-bound; k08's padded gather *loses*;
   the DaCe k05 pads lanes "never read" = a perf no-op), and the cost model has **zero** hits for
   `conflict|cache.?set|assoc|bank.?conflict`. Ironically k02 already uses N=1<<12 — a power of two —
   and never pads it. Hong et al.'s own premise is the design: *conflict misses occur even when the
   working set is far smaller than the cache*.
2. **GPU shared-memory bank-conflict kernel**, calibrated against
   [hardware-effects-gpu](https://github.com/Kobzol/hardware-effects-gpu/blob/master/bank-conflicts/README.md)
   (95.74% → 3.14% efficiency is a ready-made ground truth) and CUTLASS#281's `[0,16]`-vs-`[0,8]` pad.
   Also the only route to `report1:95`'s named validation experiment (device-dependent verdict flip).
3. **False-sharing / multi-threaded layout kernel** from **R3/R4/R5** — Linux `pcpu_chunk` (+24% for
   a two-field move) and folly#378 (−95% L1 store misses) are unusually clean, tiny, and cited.
   All 15 current kernels pin threads to 1, so false sharing, NUMA first-touch, and partition camping
   are entirely absent — and a reviewer *will* ask why every number is single-threaded.
4. **Fix the confounded four before showing them**: k13 needs a BLAS baseline (not `einsum`), k14 a
   vectorized binary-search descent baseline (not scalar `searchsorted`), k15 a `scan_col`-comparable
   row scan (not advanced indexing that materializes a copy), k03 an explanation for exceeding 8×.
   Each currently reports a non-layout effect as a layout win.
5. **Sparse CSR/CSC orientation** from **R6** (scipy#13211, 846 ms vs 70.5 µs) — a ~10⁵× real,
   *still-open* bug, and the cleanest possible motivation for format-as-layout. Pairs with SuiteSparse.
6. **Repack break-even / layout-state graph** — only k10 reports it and k02 hides it *outside* the
   timer. `report2:134` calls this the central amortization question; R1's conv2d reorder saving
   (88.3 → 63.1 s) is the real-world instance.

### Deliberately NOT added

- **numpy#19650 (strided-view `dot`, 197×)** — a numpy *dispatch* artifact: it declines the BLAS fast
  path on non-contiguous input. There is no analogue in our pipeline (we have no numpy views), so it
  would measure numpy's dispatcher, not a layout effect. Recorded as evidence, not as a kernel.

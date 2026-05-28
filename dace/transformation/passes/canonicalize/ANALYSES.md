# Canonicalize analyses — a concise reference

Quick map of the *what* and *why* of every analysis in the canonicalize pipeline plus the loop-to-map / loop-to-scan / break-anti-dep passes, with links to the classical algorithm each is built on.

---

## Dependence & disjoint-access analyses

### Per-dimension linear Diophantine (GCD test)
[`loop_to_map.py::_cross_iter_disjoint`](../../interstate/loop_to_map.py)
For each array dimension, the cross-iteration alias condition `a1·i + b1 = a2·j + b2` (where `i, j` both run in a strided iteration set) has integer solutions iff `gcd(a1, a2)` divides `(b2 − b1)`. Negative → disjoint on that dim. Unblocks TSVC s111 (odd-write/even-read).
- Banerjee, *"Dependence Analysis for Supercomputing"*, Kluwer 1988.
- Wolfe, *"High Performance Compilers for Parallel Computing"*, Addison-Wesley 1996, Ch. 9.

### Joint multi-dim Diophantine (2-equation Cramer)
[`loop_to_map.py::_joint_disjoint_2d`](../../interstate/loop_to_map.py)
The per-dim check sees each axis in isolation. The wavefront after `(i, j) → (i+j, j)` skewing has dim 0 saying `p1 = p2` and dim 1 saying `p1 = p2 − 1`: each consistent alone, joint system inconsistent. Solves the 2×2 system Cramer-style; `det != 0` with non-integer solution OR `det == 0` with augmented system inconsistent ⇒ disjoint.
- Equivalent to the **Omega test** for 2 dimensions.
- Pugh, *"The Omega test: a fast and practical integer programming algorithm for dependence analysis"*, SC '91.

### Stride-aware range disjointness
[`loop_to_map.py::_ranges_disjoint_by_stride`](../../interstate/loop_to_map.py)
`subsets.Range.intersects` is stride-blind (`[1:N:2]` vs `[0:N-1:2]` looks like overlapping `[0:N]` boxes). When two propagated ranges have the same stride `s` but starts in different residue classes mod `gcd`, they cannot overlap.

### Constant-axis disjointness
[`loop_to_map.py::_constant_dim_disjoint`](../../interstate/loop_to_map.py)
When some iteration-independent dimension carries a different numeric constant on the read vs the write side (e.g. `aa[0, i]` vs `aa[1, i-1]`), `_sanitize_by_index` would discard that dim. The check runs *before* sanitisation. Unblocks TSVC s132.

### Interstate-edge symbol substitution in subsets
[`loop_to_map.py::_collect_iedge_bindings` + `_subset_with_iedge_subs`](../../interstate/loop_to_map.py)
Frontend stages compound write indices as `a_slice := i + M` on an interstate edge, leaving the final write subset as `a[a_slice]` — opaque to the affine matcher. The helper transitively resolves these bindings into the memlet subset so the matcher sees the original `a[i + M]`. Unblocks TSVC s174.

### Anti-dependence classification under symbolic offsets
[`break_anti_dependence.py::_dep_class`](../break_anti_dependence.py)
For each carry dimension classifies the read–write offset as `WAR` (numeric `> 0`), `WAR_symbolic` (symbolic offset assumed positive — caller emits a runtime `offset > 0` guard), `WAR_indirected` (offset reduces to `arr[i]`, runtime per-element guard), `RAW` (`< 0` — true recurrence, refused), or `complex`. The "guard at runtime" trick is what makes DaCe go beyond classical polyhedral schedulers that require oracle-known symbol signs.

### Forward-stride safe-stride extension
[`break_anti_dependence.py::_safe_stride`](../break_anti_dependence.py)
Original required `stride == 1`; now accepts any **forward** stride (numeric `> 0` or symbolic, positivity rolled into the `WAR_symbolic` guard). Unblocks TSVC s175 (`a[i] = a[i + inc] + b[i]`).

---

## Loop normalisation

### Negative-stride normalisation (rebinding-style)
[`normalize_negative_stride.py::NormalizeNegativeStride`](normalize_negative_stride.py)
`for i in range(start, end, −k)` becomes `for _loop_pos_N in range(0, trip): i = start + (−k)·_loop_pos_N`. The original `i` is rebound on every iteration via an interstate-edge assignment, **preserving iteration order**, so recurrences keep working but every downstream matcher (LoopToMap's affine subset, LoopToScan's `stride != 1`, RerollUnrolledLoops) only ever sees a positive-stride loop.

### 2-D wavefront skewing
[`wavefront_skew.py::WavefrontSkew`](wavefront_skew.py)
Detects perfect 2-D loop nests whose body has wavefront dependences (`(0, −1)` + `(−1, 0)` and symbolic variants), then applies the classical polyhedral skew `(i, j) → (t = i+j, p = j)`. Numeric *and* symbolic offsets (`(0, −sym1)`, `(−sym2, 0)`) both accepted — soundness on symbolic offsets is deferred to a follow-up runtime `sym > 0` guard.
- Wolfe, *"More Iteration Space Tiling"*, SC '89.
- Wolf & Lam, *"A loop transformation theory and an algorithm to maximize parallelism"*, IEEE TPDS '91.
- Bondhugula et al., *"A practical automatic polyhedral parallelizer and locality optimizer"* (Pluto), PLDI '08.

### Induction-variable substitution (symbolic + typecast-unwrap)
[`induction_variable_substitution.py::InductionVariableSubstitution`](induction_variable_substitution.py)
Recognises `acc = acc OP c` loops and replaces with the closed form `acc_N = acc_init + c·N` (Add) / `acc_init · c^N` (Mult). `c` may be a numeric literal or a loop-invariant SDFG symbol; frontend casts `dace.<typeclass>(c)` are stripped via `_UnwrapTypecasts`. Symbolic strides supported.
- Aho/Lam/Sethi/Ullman *"Compilers: Principles, Techniques, and Tools"* (Red Dragon Book), Ch. 9.6 — induction variables and scalar evolution.
- LLVM `IndVarSimplify` pass.

### Hoist IV updates out of compound bodies
[`hoist_iv_updates.py::HoistInductionVariableUpdates`](hoist_iv_updates.py)
For a compound-body loop containing an IV-eligible statement that's *independent* of the rest (BFS-isolated component), fissions it into a sibling single-statement loop the IV-substitution pass then collapses. The "independence" criterion allows pure-copy `__out = __inp` tasklets inside the component.

### Materialise loop-exit symbols
[`materialize_loop_exit_symbols.py::MaterializeLoopExitSymbols`](materialize_loop_exit_symbols.py)
Affine-IV symbol read *after* a loop blocks `LoopToMap` ("loop-defined symbol used after the loop"). The pass synthesises a fresh `_loop_exit_<sym>_<N>` symbol whose value is the closed-form exit on a post-loop interstate edge, then rewrites every post-loop reader. Handles both the loop iterator itself and body interstate-edge IV updates.

---

## Reduction-shape lifts

### Loop → Reduce
[`loop_to_reduce.py::LoopToReduce`](../loop_to_reduce.py)
Detects `acc OP= arr[i]` reductions and lifts to a `Reduce` libnode. Tasklet matcher refuses any loop whose *write subset* depends on the loop var (the scan shape), which is exactly the complementary domain to `LoopToScan`.

### Loop → Scan (with scan-stride detection)
[`loop_to_scan.py::LoopToScan`](../loop_to_scan.py)
Prefix-scan recognition: body shape `out[i + k_w] = out[i + k_r] OP delta` with `k_w − k_r = 1` (contiguous scan) lifts to a `Scan` libnode. The matcher *also* classifies `k_w − k_r > 1` (residue-class scans) — captured in `_Scan.scan_stride` and threaded to the libnode's `stride` property; rewrite gated until per-residue-class seeding lands.

The `Scan` libnode internally uses:
- **CPU**: OpenMP 5.0 `#pragma omp scan` directive (chunked sequential + tree-reduce over per-chunk totals + offset adjustment).
- **CUDA**: `cub::DeviceScan::InclusiveScan` / `ExclusiveScan` (single-pass decoupled look-back algorithm by Merrill & Garland).
- **Pure / strided**: hand-rolled outer-loop-over-residue-classes + inner sequential scan.

Scan-with-init shortcut (`_scan_init` connector): when the LoopToScan rewrite can read the seed `out[start + k_r]` as a single scalar, it wires it as the libnode's init. CPU uses C++17 `std::inclusive_scan` 5-arg overload; CUDA uses `cub::DeviceScan::InclusiveScanInit` (CUB ≥ 2.0 / CUDA 12+). The seed-add Map is then skipped for 1-D outputs.

### Scan algorithms (background)
- Blelloch, *"Prefix Sums and Their Applications"*, 1990 — the upsweep/downsweep canonical paper.
- Mark Harris, *"Parallel Prefix Sum (Scan) with CUDA"*, GPU Gems 3 Ch. 39 — practical reference.
- Merrill & Garland, *"Single-pass Parallel Prefix Scan with Decoupled Look-back"*, NVIDIA TR-2016-002 — the algorithm CUB uses.

---

## Scatter & write-conflict resolution

### Scatter-to-guarded-maps (runtime permutation check)
[`scatter_to_guarded_maps.py::ScatterToGuardedMaps`](../scatter_to_guarded_maps.py)
For loops `a[idx[i]] = …` over an indirection array `idx`, emits a sort + adjacent-equal scan over `idx` at runtime: traps if any pair of indices collides; otherwise permissively lifts the loop to a parallel map.
- Tomescu, *"Parallel Scatter via Permutation-Check"*, contemporary literature on irregular-access parallelism.

### Scatter-conflict guard (libnode-level reduction-count)
[`scatter_conflict_guard.py`](../scatter_conflict_guard.py)
The sort + adjacent-equal compare is implemented as a parallel Map with WCR-`+` reduction into an `int64` counter, then a sequential `__builtin_trap()` state reads the count via interstate-edge symbol binding and traps if positive. Per-thread accumulation + tree-merge end-of-region; no false sharing.

### Reroll-unrolled-loops (merge-tree generalisation)
[`reroll_unrolled_loops.py::RerollUnrolledLoops`](reroll_unrolled_loops.py)
Detects manually-unrolled lane chains (step `S`, `m` equally-spaced lanes at offsets `{0, g, …, (m−1)g}`) and re-rolls to a step-`g` loop. Allows lane components to overlap at associative-merge tasklets (`+`, `·`, `min`, `max`) classified via `dace.symbolic.pystr_to_symbolic` + `type(expr).__name__`. Generalisation covers TSVC s352 (single-expression `m`-term dot product).

---

## Library-node persistent scratch

### Per-class, per-stream CUB scratch pool
[`runtime/include/dace/cub_scratch.cuh`](../../runtime/include/dace/cub_scratch.cuh)
`template<typename Tag> get_scratch(bytes, stream) / release_scratch<Tag>()` — function-static `unordered_map<cudaStream_t, ScratchEntry>` per tag, 128 MB pre-allocated on stream 0 via `SortScratch` / `ScanScratch` / `ReduceScratch` envs, grow-on-demand for other streams, release-all at SDFG finalize. Each `(Tag, stream)` is independent — concurrent libnode launches across streams cannot race.

---

## Polyhedral-school references (background reading order)

1. Aho/Lam/Sethi/Ullman, *Compilers* (Red Dragon Book), Ch. 9, 11 — dependence analysis vocabulary, IV detection, loop transformations.
2. Banerjee, *Dependence Analysis for Supercomputing* (1988) — the GCD test as a baseline.
3. Pugh, *"The Omega test"*, SC '91 — exact integer-programming-based dependence test, generalises GCD.
4. Wolfe, *High Performance Compilers for Parallel Computing* (Addison-Wesley 1996) — the practical compendium; Ch. 9 covers loop skewing.
5. Bondhugula et al., *"A practical automatic polyhedral parallelizer and locality optimizer"* (Pluto), PLDI '08 — the affine-schedule ILP formulation; modern compilers (Polly, Pluto, PPCG) all descend from this.
6. Grosser, Groesslinger, Lengauer, *"Polly — Performing Polyhedral Optimizations on a Low-Level Intermediate Representation"*, PPL '12 — LLVM's polyhedral framework; useful for comparing what an analyser without runtime-guard capability does and doesn't catch.

---

*Updated:* 2026-05-28. When you add an analysis, append it here with the file pointer and the one-line classical-reference link — the goal is one-screen orientation, not a textbook.

# Readable CPU/GPU code generator — merge design (`readable-codegen` → `extended`)

Adds an experimental, **toggleable** "readable" C++/CUDA code generator to DaCe:
human-readable output that compiles and runs **bit-exact** to the classic generator,
plus a correctness harness and daint.alps perf jobs. Off by default — zero effect on
existing users until the flag is set.

```
compiler.cpu.implementation: legacy | experimental        # DACE_compiler_cpu_implementation=experimental
```

The flag also governs GPU-kernel tasklets: the legacy CUDA generator emits its device
tasklets through the shared CPU generator instance, so the readable form flows into
`__global__` kernels without touching the CUDA target.

---

## 1. What it produces

Classic elementwise tasklet (copy-in / compute / copy-out, offset math inlined at every access):

```cpp
{
  ///////////////////
  double __in1 = A[((__i0 * N) + __i1)];
  double __in2 = B[((__i0 * N) + __i1)];
  double __out;
  __out = (__in1 + __in2);
  C[((__i0 * N) + __i1)] = __out;
  ///////////////////
}
```

Readable:

```cpp
static DACE_HDFI constexpr long long A_idx(long long __d0, long long __d1, long long N) { return ((N * __d0) + __d1); }
...
C[C_idx(__i0, __i1, N)] = (A[A_idx(__i0, __i1, N)] + B[B_idx(__i0, __i1, N)]);  // _Add_
```

### Features (each is bit-exact vs legacy on CPU + GPU)

| feature | what |
|---|---|
| **Per-array `<arr>_idx(...)` index functions** | `static DACE_HDFI constexpr` (host+device), offset math emitted **once** per array; symbolic dims/strides passed as trailing params; layout falls out of `desc.strides`. Deduplicated by a line-level post-pass (`deduplicate_functions`) — **no `#ifndef` macro guards**. |
| **Connector-free single-line tasklets** | a connector-free single-`Assign` tasklet is one brace-free line `stmt;  // <label>`. WCR / library / multi-statement tasklets keep their `{ }` block. |
| **`const T x = expr;` for single-write scalars** | a scope-local scalar written exactly once (by a fuseable `out = expr` tasklet whose scope encloses all reads) is emitted as a fused const binding instead of `T x; x = expr;`. |
| **len-1 → scalar** | transient single-element arrays — rank-1 `(1,)` and higher-rank `(1,1)` MapFusion scratch — are scalarized (`ConvertLengthOneArraysToScalars(single_element=True, transient_only=True)`), so they read + emit as `T x` (and become const bindings) instead of `DACE_ALIGN(64)` length-1 buffers. |
| **`<arr>_size(...)` + heap wrappers** | allocation extents through a `<arr>_size(...)` helper; `dace::aligned_alloc<T>` / `dace::calloc<T>` / `dace::free` (in `dace/runtime/include/dace/alloc.h`, `#pragma once`) instead of aligned `new[]` / `delete[]`. |
| **Views are first-class** | an `ArrayView` inlines like any array — `V[V_idx(...)]` from the view's own strides; never a Pure fallback (reduce / gemm / batched_matmul / transpose / copy / memset all handle views). |
| **clang-tidy + clang-format** | always-on for experimental (if `clang-tidy` is found): standalone `-fix-errors`, headers stripped (external calls treated as black boxes — no vendor include paths). **`readability-non-const-parameter` excluded** — its const inference is unsound on generated code (it const-qualifies a written scatter accumulator forwarded to a `DACE_DFI`, breaking the compile). |

### Effect (npbench `cavity_flow`, CPU, `simplify + LoopToMap + MapFusion`)

| codegen | non-blank lines |
|---|---:|
| legacy | 1541 |
| experimental | **365** (**−76 %**) |

Bit-exact on all outputs.

---

## 2. Architecture

Two SDFG passes run once in `codegen.generate_code`, gated on the flag, so CPU and
legacy-GPU both see the transformed SDFG:

- **`InlineTaskletConnectors`** (`dace/transformation/passes/inline_tasklet_connectors.py`) — rewrites a tasklet body to reference arrays/views directly (`__in1` → `A[__i0,__i1]`); skips WCR outs (atomic path), non-single-element, code→code, references/container-arrays, and library-pointer CPP args.
- **`MarkConstInit`** (`dace/transformation/passes/mark_const_init.py`) — classifies write-once data. `constexpr_static` (compile-time value → SDFG constant) and `const_runtime` (single fuseable `out = expr` tasklet whose scope encloses all reads → `const T x = expr`).

plus, in the experimental preprocessing: NSDFG inlining (`InlineSDFG` / `ExpandNestedSDFGInputs(top_level_only=True)` / `InlineMultistateSDFG`), `ConvertLengthOneArraysToScalars(single_element=True, transient_only=True)` + `simplify`, `infer_types`.

Emission lives in **`ExperimentalCPUCodeGen`** (`dace/codegen/targets/experimental_cpu.py`, subclass of `CPUCodeGen`, selected explicitly in `codegen.py`), which overrides only the emission hooks. The base `CPUCodeGen` gained small overridable hooks so legacy output is **byte-identical**: `tasklet_body_comment`, `tasklet_body_open/close_marker`, `emit_tasklet_body_block`.

`_get_const_params` (cuda.py) now delegates to the shared `sdutil.get_constant_data` — the same read/write-set analysis `new-gpu-codegen-dev`'s `experimental_cuda.py` uses (kernel-arg const-qualification unified across legacy + new GPU codegen).

---

## 3. Files

**New:**
- `dace/codegen/targets/experimental_cpu.py` — the readable generator.
- `dace/transformation/passes/inline_tasklet_connectors.py`, `.../mark_const_init.py` — the two passes.
- `dace/runtime/include/dace/alloc.h` — heap wrappers.
- `tests/codegen/readable/` — the correctness suite (below).
- `cpu_codegen_perf_jobs/` — daint.alps perf jobs (4 corpora, cavity compare, plots).

**Modified** (all flag-gated or additive; legacy path unchanged):
`dace/codegen/{codegen,compiler}.py`, `dace/codegen/targets/{cpu,cpp,cuda}.py`,
`dace/config_schema.yml`, `dace/data/core.py` (`const_init` props), `dace/dtypes.py`,
`dace/runtime/include/dace/dace.h` (include alloc.h),
`dace/transformation/interstate/expand_nested_sdfg_inputs.py` (`top_level_only`),
`dace/transformation/passes/length_one_array_scalar_conversion.py` (`single_element`),
`dace/libraries/{standard/nodes/reduce, standard/reduction_planner, linalg/nodes/{cholesky,inv,solve}}.py` (view-robust reductions + cuSOLVER).

---

## 4. Tests — `tests/codegen/readable/`

Gated by `experimental_available()` (skips cleanly until the generator is active), so
the suite is green on any branch and flips on automatically.

- `test_corpus_equivalence.py` — **17 npbench/polybench/tsvc kernels × {cpu, gpu} × {legacy, experimental}**, bit-exact on CPU / tight `allclose` on GPU.
- `test_single_line_tasklet.py`, `test_view_access.py`, `test_const_init_codegen.py`, `test_readable_smoke.py`, `test_layouts.py`, `test_array_size.py`, `test_cpp_tasklets.py`, `test_library_lowering.py`, `test_clang_tidy.py` (config guard + a functional GPU scatter that fails without the `readability-non-const-parameter` exclusion).
- `tests/library/test_reduce_view_gpu.py` — reduce over strided/sliced views (cpu+gpu).
- `tests/passes/` — `InlineTaskletConnectors` + `MarkConstInit` unit tests.

Run: `PYTHONPATH=<repo> OMP_NUM_THREADS=1 pytest tests/codegen/readable -n4`.

---

## 5. Perf jobs — `cpu_codegen_perf_jobs/`

daint.alps `sbatch` jobs (legacy vs experimental, `dace + simplify + LoopToMap + MapFusion`), S single-core + paper multi-core presets:
- `submit_daint_cpu_4rank.sbatch` — 1 node, 4 ranks (one per Grace socket), kernels split round-robin, per-rank TSVs merged.
- `submit_daint_cpu.sbatch` — single-rank corpus sweep.
- `submit_daint_cavity.sbatch` — `cavity_flow` g++ vs clang++ codegen/compile/runtime comparison.

Reuse the sibling `performance_regression_jobs` engine. All `#SBATCH` account/partition/chdir + `REPO_ROOT` are `TODO`-marked.

---

## 6. Merge plan

Worktree `Work/dace-readable`, branch `readable-codegen`, off `extended`. No merge with
`new-gpu-codegen-dev`.

```bash
# from Work/dace (on extended):
git merge readable-codegen          # or: git rebase readable-codegen onto extended, then ff-merge
```

Expected conflicts: none with `extended` (branched off it). If cherry-picking onto a
moved `extended`, watch `dace/codegen/targets/{cpu,cuda,cpp}.py` (the added hooks +
`_get_const_params` delegation) and `dace/config_schema.yml` (the `implementation` key).

**Pre-merge gate:** `pytest tests/codegen/readable -n4` green (corpus equivalence + units),
and the legacy path unchanged (the base-class hooks default to the exact prior output).

---

## 7. Risks / follow-ups

- clang-tidy time counts as codegen time; it is best-effort (a missing binary only warns).
- `const_runtime` emission is conservative by design (fuseable `out = expr` + scope-encloses-reads); anything else keeps the mutable declaration.
- Open: const-emit fallback for the rare non-scalarizable len-1 array; symbol const/constexpr; wider GPU-kernel const-init.

# Codegen style levers — standalone reproducers

Each file compiles two or more **semantically identical** forms of the same computation and times
them. They back the measurements in [`CODEGEN_STYLE_PERFORMANCE.md`](../../../CODEGEN_STYLE_PERFORMANCE.md)
§6 and exist so the claims can be re-checked — the results are microarchitecture- and
compiler-specific, and at least one of them does not reproduce across boxes.

Build and run one (no build system, deliberately):

```sh
g++     -O3 -march=native -o /tmp/l3_gcc   lever3_div_strength.cpp && /tmp/l3_gcc
clang++ -O3 -march=native -o /tmp/l3_clang lever3_div_strength.cpp && /tmp/l3_clang
# lever6 is a two-TU case -- the boundary IS the lever, so do not merge the files or add LTO:
g++ -O3 -march=native -o /tmp/l6 lever6_call_barrier.cpp lever6_helper.cpp && /tmp/l6
```

`bench.h` enforces the methodology: median of ≥15 reps, a **twin** of the fast form under a second
symbol to establish the intra-binary noise+layout floor, and a bit-identical-output assertion (a
"fast" form that changes results is a bug, not a lever). **Reportable threshold on the reference box
was ~1.2×**; anything under that is an ordinary draw — see §1 (clang's mean CoV across
semantically-equivalent forms is ~16.9%) and §3 (layout confound).

| File | What it contrasts | Zen 4 result |
|---|---|---|
| `lever3_div_strength.cpp` | divide/modulo by a runtime variable vs a constant | **12.0–12.6× g++, 9.2–11.7× clang** — a constant divisor becomes a Granlund–Montgomery magic-number multiply and vectorizes; a runtime one is a scalar `div` |
| `lever6_call_barrier.cpp` (+ `lever6_helper.cpp`) | elementwise op behind an out-of-line call vs inlined | **6.3–6.5× g++, 7.3–8.0× clang** — a TU boundary blocks the inliner, so no LoopVectorize/LICM across it |
| `lever1_accum.cpp` | reducing into a memory location vs a local accumulator | **4.3× g++**; clang recovers it with a runtime alias check (1.13×) — aliasing blocks LICM scalar promotion, so no reduction PHI, so no vectorization |
| `lever2_memcpy_idiom.cpp` | per-row `memcpy` idiom vs an inlined copy | **REFUTED here** (0.86–1.08×, inside the floor) though the idiom demonstrably fires. It measured 2.6× on another box, so it is microarch/glibc-specific — kept precisely so it can be re-run elsewhere. The `=`-vs-`memcpy` choice is the *only* codegen (pure-syntax) component of `InsertExplicitCopies` (`dace::CopyND<…>::Copy` bottoms out in `memcpy`, `runtime/include/dace/copy.h`). **NOTE — do not attribute the heat_3d `InsertExplicitCopies` win to this lever:** measured heat_3d = 0.790× single-core but **0.442× (2.26× faster) at 72 threads**, and seidel_2d 0.657× single-core — that large, thread-scaling win is the *semantic* half (the explicit copy becomes its own `#pragma omp parallel for`, parallelizing a previously serial `CopyND` copy that was an Amdahl bottleneck), which is a **pass, not codegen**. This syntax lever is only the small microarch-dependent residue |
| `divisor_spellings.cpp` | seven spellings of the same `b % 4` | `b % 4`, `const a = 4`, and `constexpr a = 4` all fold identically (magic/mask). `volatile`, a `const&` parameter, and an `asm`-laundered constant each **force the scalar `div`** — on g++ *and* clang. Inspect with `objdump -d`, no timing needed |
| `heat3d_layout_confound.cpp` (+ `layout_probe.cpp`) | the full heat_3d stencil body in the legacy vs experimental-readable source forms, plus every one-axis hybrid, **all in one binary** | **REFUTED as a lever.** heat_3d runs ~1.30× slower under the experimental generator at *single core* only, but that is the layout confound (§3): the two forms have identical FP instruction selection (SROA/mem2reg/GVN erase the `[1]`-temp, brace, index-fn and restrict axes before the vectorizer), so in one binary every form lands 0.97–1.00× (Neoverse-V2 g++/clang). `layout_probe.cpp` shows five **byte-identical** twins span the same 1.28× (→1.17× with `-falign`). No `codegen_params` flag warranted — there is no fast form to emit |
| `stockham_fft_layout_confound.cpp` | the two streaming loops (twiddle-multiply + TensorTranspose) of stockham_fft in legacy vs experimental-readable source forms, all in one binary | **REFUTED as a lever.** stockham_fft is reported ~1.3× slower under the experimental generator at single core, but the butterfly is an identical `cblas_cgemm` and the two streaming loops tie: twiddle is bandwidth-bound, and with runtime `R,K` the transpose is scalar for BOTH forms (verified: identical FP/scalar selection in the generated `.s`). Kernel-level exp/leg = 0.999× g++, 0.995× clang — inside the layout floor. The only >1.2× is a g++-only memcpy-idiom artifact that appears only when the dims are constant-folded (which runtime symbols preclude) and favors experimental anyway. No `codegen_params` flag warranted |

`divisor_spellings.cpp` is the one to read first: it shows that a `const` local costs nothing (the
compiler propagates it), while binding the very same constant **by reference** throws away the
strength reduction — which is why `codegen_params.const_scalar_abi` is not a free choice for a value
used as a divisor.

## Why most candidates are missing

They were measured and collapsed to noise, all for the same reason: **the compiler re-derives the
fast form whenever a runtime guard can recover it.** Branch-vs-branchless was if-converted to `cmov`;
a laundered stride got a `cmp $0x1` version guard; a by-reference divisor got an IPA-CP `.constprop`
clone; `pow(x,2.0)` folded to `x*x`. That is the selection criterion for a real lever, and the
reason this directory is short: **a lever survives only when the fast form is not recoverable by a
runtime guard.** See §6 and §7 rule 6.

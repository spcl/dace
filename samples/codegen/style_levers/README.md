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
| `lever2_memcpy_idiom.cpp` | per-row `memcpy` idiom vs an inlined copy | **REFUTED here** (0.86–1.08×, inside the floor) though the idiom demonstrably fires. It measured 2.6× on another box, so it is microarch/glibc-specific — kept precisely so it can be re-run elsewhere |
| `divisor_spellings.cpp` | seven spellings of the same `b % 4` | `b % 4`, `const a = 4`, and `constexpr a = 4` all fold identically (magic/mask). `volatile`, a `const&` parameter, and an `asm`-laundered constant each **force the scalar `div`** — on g++ *and* clang. Inspect with `objdump -d`, no timing needed |

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

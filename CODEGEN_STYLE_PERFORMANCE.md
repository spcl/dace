# Codegen style vs. performance — design notes and resources

Why DaCe's CPU code generators expose *style* knobs (`compiler.cpu.codegen_params`) instead of
hardcoding "the fast form", what the literature says about this class of behavior, and reproducers
for each effect.

**One-line thesis.** Two source forms that are semantically identical can compile to materially
different code, because optimizer decisions are keyed on incidental IR *shape* — def-use topology,
instruction census, alias facts — not on program meaning. Which form wins is target-, compiler- and
kernel-dependent, and **the sign is not fixed**. So a code generator must not pick by reasoning; it
must expose the choice and measure it.

---

## 1. The calibration number you must know first

> Clang's **mean coefficient of variation across semantically-equivalent loop forms is ~16.9%**.
> — Gong et al., [*An Empirical Study of the Effect of Source-Level Loop Transformations on Compiler
> Stability*](https://iacoma.cs.uiuc.edu/iacoma-papers/oopsla18.pdf) (OOPSLA'18)

They name the ideal we are failing to meet: *"a perfect compiler would generate the same optimal
target code for all semantically-equivalent versions of a given source code."* The phenomenon has a
name — **compiler instability**.

**Consequence for us:** a ~17% delta from a source-form change is an *ordinary draw from a known
distribution*, not a discovery. Report it as a data point. Only the disassembly can carry a causal
claim, and even then see §3.

---

## 2. Our own case: `const T&` vs `const T` (`codegen_params.const_scalar_abi`)

### What changes

The readable generator binds a read-only scalar either by const reference (`by_ref`, the legacy
convention and our default) or by const value (`by_value`). Emitted difference, verbatim:

```cpp
// compiler.cpu.codegen_params.const_scalar_abi = by_ref      (default; matches legacy)
inline void inner_0_1_0(abi_demo_state_t *__state, const double&  sc, double* __restrict__ io, int N)

// compiler.cpu.codegen_params.const_scalar_abi = by_value
inline void inner_0_1_0(abi_demo_state_t *__state, const double   sc, double* __restrict__ io, int N)
```

Reproducer: [`samples/codegen/const_scalar_abi_demo.py`](samples/codegen/const_scalar_abi_demo.py)
— emits both forms from one SDFG and diffs them.

### What we measured, and what we got wrong

On Neoverse-V2 + clang, the two forms lower a masked sum-reduction (`for j: if mask[j]: sum +=
data[j]`) differently:

| binding | lowering | interleave |
|---|---|---|
| `by_ref` | predicated load `ld1w p/z` + `sel` → **unpredicated** `fadd z,z,z` | 1 |
| `by_value` | plain `ldr z` → **merging-predicated** `fadd z, p/m, z, z` | 2 |

`by_value` measured ~17% slower. The originally-claimed cause — *"merging-predicated fadd issues on
fewer FP pipelines on Neoverse-V2"* — is **RETRACTED. It is contradicted by Arm's own guide.**

> Arm Neoverse V2 Core SWOG (Issue 3.0), **Table 3-25** has ONE row covering all FADD forms:
> `FADD, FADDP, FNEG, FSUB, FSUBR | Latency 2 | Throughput 4 | Pipelines V`, where Table 3-1 defines
> `V` = FP/ASIMD 0/1/2/3 — **all four pipes**. There is no predicated-vs-unpredicated split for FP
> arithmetic anywhere in the tables.

Worse for the story, **the uop arithmetic runs the wrong way**: the `sel` + unpredicated form we
measured as *faster* consumes *more* V-pipe slots (2 uops) than the single merging-predicated `fadd`
(1 uop). And where the guide does discuss FADD + predication (§4.17, MOVPRFX fusion), predication is
*privileged*, not penalized.

**What survives:** the observation (binding mode flips the lowering; ~17% correlates). **What is
open:** the mechanism. Leading candidate — **predicate-pipe dependency**, not FP-pipe count:

> "GCC already takes advantage of this and drops predicates entirely when it can to avoid the
> dependency on the predicate pipe." — Tamar Christina (Arm),
> [gcc-patches](https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg373383.html)

The `p/m` accumulate keeps the governing predicate inside the *loop-carried dependency chain*; the
`p/z`-load + `sel` + unpredicated form lets the predicate feed the load/sel and drop out of the
accumulate chain. **Not yet excluded:** code layout (§3).

### The sign is not fixed — this is the whole argument for a flag

| Source | Direction | Evidence |
|---|---|---|
| [llvm #121262](https://github.com/llvm/llvm-project/issues/121262) | **reference wins** | Same `const T` vs `const T&` axis, post-inline, scalar. The reference form reaches the middle-end as a load-from-reference and SimplifyCFG forms a jump table; by-value stays an `icmp` chain. |
| [Lemire (2019)](https://lemire.me/blog/2019/09/05/passing-integers-by-reference-can-be-expensive/) | **value wins, 4.5×** | 5.9 vs 1.3 cyc/value. Aliasing: the compiler can't prove the referenced scalar isn't an array element. *But he notes this is "likely irrelevant if the function gets inlined."* |
| [O'Dwyer (2021)](https://quuxplusone.github.io/blog/2021/11/09/pass-string-view-by-value/) | **value wins** | `string_view` by value: kills a load indirection, kills a caller spill, and the copy cannot alias. |
| [GCC PR 115531](https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg818483.html) / [PR 115629](https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg820407.html) | **opposite preference** | Same kernel shape; GCC concludes the *reverse* of clang. |
| [WG21 N3538](https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3538.html) | *"stop deciding in source"* | Crowl names both costs and proposes handing the choice to the compiler. C++ didn't adopt it — hence the knob. |

Two corrections to how this is often described:

- **`const` is not the causal variable.** A whole-program `const`-strip of SQLite (200K+ LoC) made
  the **non-const** build ~0.5% *faster*; `const` on a parameter is not a no-alias or no-write
  guarantee (callers may cast it away).
  → [Why const Doesn't Make C Code Faster](https://theartofmachinery.com/2019/08/12/c_const_isnt_for_performance.html).
  The causal variable is **value-vs-reference binding**.
- **"It all inlines, so binding mode is erased" is false.** The inliner *deliberately re-encodes*
  parameter attributes onto the inlined body ("the conversion of noalias arguments to scoped noalias
  metadata during inlining" — nikic, [rust#82834](https://github.com/rust-lang/rust/pull/82834)), and
  #121262 is a post-inline scalar case. **Parameter form survives inlining by design.**

---

## 3. Methodology gate: exclude layout before blaming instruction selection

Two binaries are two samples from the *layout* space, not two measurements of one program.

- [Mytkowicz et al., *Producing Wrong Data Without Doing Anything Obviously Wrong!*](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) (ASPLOS'09) — link order and environment size alone move results.
- [Curtsinger & Berger, STABILIZER](https://people.cs.umass.edu/~emery/pubs/stabilizer-asplos13.pdf) (ASPLOS'13).
- [Beyls (Arm), *Towards Ameliorating Measurement Bias*](https://llvm.org/devmtg/2016-03/Presentations/Beyls2016_AmelioratingMeasurmentBias.pdf) (EuroLLVM'16) — AArch64, actionable: a layout-randomizing `MachineFunctionPass`.

Reported layout-only swings reach **7–51% on AArch64** — i.e. **our 17% fits entirely inside the
layout confound.** Randomize layout (or pin `-align-all-functions` / `-align-all-blocks`) and show
the delta survives, *before* attributing it to instruction selection.

Cost models cannot referee this either: llvm-mca is ~28% MAPE
([uiCA](https://arxiv.org/abs/2107.14210)), and LLVM's vectorizer speedup prediction correlates at
only ρ=0.58 ([Pohl et al., MASCOTS'19](https://www.cosenza.eu/papers/PohlMASCOTS19.pdf)).

---

## 4. Other knobs known to move backend decisions with zero semantic change

Every row is a candidate `codegen_params` axis, and a candidate NestForge variant axis.

| Knob | Documented effect |
|---|---|
| `noalias` / `__restrict__` | **Bimodal**: 46 tests faster / 18 slower ([rust#82834](https://github.com/rust-lang/rust/pull/82834)); `!noalias` *disabled* vectorization via GVN load-PRE ([llvm#38583](https://github.com/llvm/llvm-project/issues/38583)); shipped miscompiles repeatedly |
| `llvm.assume` / `__builtin_assume` | Inflates instruction census + use counts → defeats inlining/vectorization/unrolling; **libc++ disabled it** ([Discourse](https://discourse.llvm.org/t/llvm-assume-blocks-optimization/71609)) |
| `[[clang::trivial_abi]]` | Register vs stack-temporary passing; up to **1.6%** whole-program ([libc++ doc](https://libcxx.llvm.org/DesignDocs/UniquePtrTrivialAbi.html)) |
| Accumulator width (`int`→`long`) | TBAA disproves aliasing → store sinks out of the loop ([Godbolt's blog](https://xania.org/202512/15-aliasing-in-general)) |
| Wrapping a scalar in a 1-member struct | Kills 128-bit SLP merge despite identical layout ([llvm#54646](https://github.com/llvm/llvm-project/issues/54646)) |
| Index integer width (`int` vs `long long`) | Selects clang's per-row `memcpy` idiom — itself **2.6×** slower (7.5 vs 19.7 GiB/s); direction disagrees across compilers (see §5) |
| Templating/overloading a helper on a type | **Nothing** — GCC's identical-code folding collapses instantiations onto one body (`jmp <v2>`) |
| Making an operand compile-time-constant | GCC drops to **scalar branching** on TSVC S276 where the unknown-parameter form vectorizes ([TACO'19](https://ora.ox.ac.uk/objects/uuid:eac7b135-e92b-48dc-a1f7-4de66a441390/files/szg64tk95s)) — *more* information made it *worse* |
| `-fno-semantic-interposition` | **5–27%** on CPython, zero source change ([Fedora](https://fedoraproject.org/wiki/Changes/PythonNoSemanticInterposition)) |
| Fill/memset literal type (`0` vs `'\0'`) | **29×** (1.0 → 29.1 B/cyc): deduction makes the value type differ from the iterator type, so libstdc++'s `__fill_a` memset specialization is never selected ([Downs](https://travisdowns.github.io/blog/2020/01/20/zero.html)) — the *performance* twin of §5's silent-int32 correctness bug |
| Index extension `sext` vs `zext` | LLVM promotes GEP indices to register width and defaults to `sext` "for safety"; on x86-64 `sext` is an extra instruction, `zext` folds into the load — LLVM tells frontends to emit `zext` when the range is known ([LLVM Perf Tips](https://llvm.org/docs/Frontend/PerformanceTips.html)) |
| Non-constant-size `memcpy` from a copy loop | LoopIdiomRecognize forms a `memcpy` libcall that **no later pass can undo**; for small copies the call is slower than the loop it replaced ([llvm#87440](https://github.com/llvm/llvm-project/issues/87440)) |
| `__builtin_expect` / `[[likely]]` | Silently rot on code change and regress; LLVM shipped a **MisExpect diagnostic** because of it ([LLVM MisExpect](https://llvm.org/docs/MisExpect.html)) |
| Non-temporal / streaming stores | **Bimodal, sign flips by thread count on the *same* kernel**: STREAM Triad ~20–40% *slower* on 1 thread (NT-store buffer occupancy > L2 prefetcher), ~33–50% *faster* all-core (saves the write-allocate read) ([McCalpin](https://sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/)) |
| `schedule(dynamic,1)` vs `static` | **5×** slower on a small-iteration loop — even one that is genuinely irregular (57% CV) — from per-chunk hand-off overhead ([Eleliemy/Ciorba](https://arxiv.org/abs/1809.03188)) |
| Explicit `#pragma unroll` factor | Loses when the bottleneck is the FP/vector unit; the unrolled body evicts the µop-cache/loopback buffer and spills → *slower*, and can slow neighbouring code ([llvm#42332](https://github.com/llvm/llvm-project/issues/42332)) |
| Divide/modulo by a runtime var vs a constant | **12× measured (Zen 4, §6)**: a constant divisor becomes a Granlund–Montgomery magic-number multiply that vectorizes; a runtime divisor is a scalar `div` ([our reproducer](samples/codegen/style_levers/)) |
| `__builtin_prefetch` on a gather/indirect loop | **Bimodal, but the safe direction of bimodal** (a pure timing hint — never changes results): **2.6× (x86) / 1.55× (ARM)** on a 256 MB irregular hash-set gather; ~neutral-to-small-loss on unit-stride streams (the HW prefetcher already covers them) ([Johnny's SW Lab](https://johnnysswlab.com/the-pros-and-cons-of-explicit-software-prefetching/); Lemire's mild-irregularity case is only 10%, below the floor) |
| `ivdep` is three different soundness contracts | Intel `#pragma ivdep` ignores only *assumed* deps and **respects proven ones** (can't miscompile) ([Intel](https://www.smcm.iqfr.csic.es/docs/intel/compiler_c/main_cls/cref_cls/common/cppref_pragma_ivdep.htm)); GCC `ivdep` / clang `vectorize(assume_safety)` / `omp simd` override *all* deps and **miscompile if wrong** ([GCC](https://gcc.gnu.org/onlinedocs/gcc/Loop-Specific-Pragmas.html)) — same class as `!noalias` |
| `assume_aligned(N)` on a returned/param pointer | NOT the aligned-move (§9's ~2% myth): it lets the vectorizer **skip the alignment prologue AND epilogue and the runtime alignment branch** ([P0886](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0886r0.pdf)) — real, but sub-noise except split-prone / short-trip loops; **UB if the alignment promise is false** |
| Return a small struct by value vs via an `sret` out-pointer | `sret` leaves redundant stack `memcpy`s the middle end often can't fold; `MemCpyOpt` recovers **17–42%** of them ([llvm D140089](https://reviews.llvm.org/D140089)) — note that is a memcpy *count*, not a wall-clock number |

---

## 5. Our second case: the index type (`codegen_params.index_ctype`)

The `<array>_idx` helpers had `long long` hardcoded. The obvious-looking fix — template the helper on
the index type, or overload it for both — is the one we measured and **rejected**. Two findings, both
independent of the timings, killed it:

**Identical Code Folding makes the templated design a no-op.** GCC folds the instantiations back onto
a single body, so the "let each call site pick" design generates one function and a jump to it:

```asm
0000000000401a30 <stencil3d_v3_c32>:
  401a30:  endbr64
  401a34:  jmp    401850 <stencil3d_v2_c32>   ; the whole "variant"
```

**C++ argument deduction picks the type behind your back, silently.** Our generator emits
`for (auto i = 0; ...)` — so `i` is `int` and every call site is *already* int32. A templated helper
deduces 32-bit math there and **wraps** with no diagnostic: `4000000000` came back `-294967296`. An
explicit knob cannot do that; a deduced one does it by default.

**The headline number was a red herring, and chasing it would have been the actual bug.** The −60%
that started this was not index math at all: 64-bit indices trip clang's per-row `memcpy` idiom, and
*that idiom is itself a 2.6× pessimization* (7.5 vs 19.7 GiB/s). int32 was fast only by accidentally
defeating it. Tuning the index type here would have "fixed" a symptom of a lowering choice that
deserves its own knob. (Why glibc `memcpy` stalls at 1–3 KiB/call is open.)

So: **int64 default, int32 opt-in, never auto-selected.** Most kernels sit inside a 0.5–6% noise
floor, and the deltas clearing it disagree on sign (gcc/gather −26% *for* int32; clang/stencil +12%
*against*) — §1's ordinary draw again, not a discovery.

The safety bound is the part worth remembering, because the intuition is wrong: **int32 overflows on
the ELEMENT count, not the byte size.** An `int8` array wraps at **2 GiB** of data — reachable today,
silent, and unprovable by the generator for a symbolic shape.

| element type | overflows at |
|---|---|
| int8 | **2 GiB** |
| float32 | 8 GiB |
| float64 | 16 GiB |

Incidental trap found on the way: `INDEX_CTYPE` is `long long`, but `dace.int64.ctype` is `int64_t`
(= `long`) — *distinct* C++ types on LP64. Any overload set spanning both is ambiguous or silently
truncating. This is why `index_ctype` emits `int`/`long long` literally and matches the call sites.

Two follow-ups worth recording, one open and one new:

- **The 1–3 KiB `memcpy` stall stays OPEN — resist the obvious story.** The tempting explanation is
  glibc's `x86_rep_movsb_threshold` (default 2048, dead center of the window). It is a real, verified
  tunable, but it is the wrong tool: it is x86-only (§2's measurements are Neoverse-V2), it predicts a
  *crossover* not the *plateau* we saw on both sides, and an Arm glibc engineer's own numbers show
  `rep movsb` ~3× *slower* than AVX `memcpy` across 4–64 KiB — i.e. above where it is supposed to win.
  Shipping it would repeat the retracted Neoverse pipe mistake: a plausible vendor-sourced story that
  the vendor's own data contradicts. What *is* corroborated is that LoopIdiomRecognize's non-constant
  `memcpy` is a **one-way door** — no later pass undoes it — so a small per-row copy pays a full
  libcall ([llvm#87440](https://github.com/llvm/llvm-project/issues/87440); the mechanism, not a
  number: the source's "much slower" is unquantified).
- **The real index knob may be signedness/extension, not width.** LLVM's own frontend guide tells code
  generators that GEP indices default to `sext` "for safety", that on x86-64 `sext` costs an extra
  instruction while `zext` folds into the load, and — decisively — *"if your source language provides
  information about the range of the index, you may wish to manually extend indices … using a zext"*
  ([LLVM Performance Tips](https://llvm.org/docs/Frontend/PerformanceTips.html); GCC agrees the *sign*,
  not the width, gates promotion —
  [Walfridsson](http://kristerw.blogspot.com/2016/02/how-undefined-signed-overflow-enables.html)). We
  already hold exactly that information: canonicalization assumes symbols are nonnegative. So the next
  index axis to try is an unsigned/`zext`-friendly index form, not a wider one — filed as a candidate,
  not yet built.

---

## 6. Measured on our own hardware (Zen 4)

We built standalone reproducers for a set of candidate pessimization levers on an AMD Ryzen 7 8845HS
(Zen 4, AVX-512), g++ 15.2 and clang++ 21.1, `-O3 -march=native`. Every variant carries a
byte-identical twin of its fast form and asserts bit-identical output. Reproducers live in
[`samples/codegen/style_levers/`](samples/codegen/style_levers/).

**Noise floor, stated first.** Intra-run twin (identical code, different symbol, same process):
0.94–1.13×. Inter-run (same binary, 7 invocations): FAST-form median spread ~10–18% — matching §1's
16.9%. **Reportable threshold: ~1.2×.** Everything below is an ordinary draw.

**The organizing principle — and it is the useful result.** Most candidate levers *collapsed to
noise*, and they collapsed the same way: **the compiler re-derives the fast form whenever a runtime
guard can recover it.** We watched it happen four times, with disassembly:

| Candidate lever | How the compiler erased it | Pass |
|---|---|---|
| data-dependent branch vs branchless | if-converted to `cmov` | SimplifyCFG / early-if |
| laundered runtime stride vs unit stride | emitted a `cmp $0x1` version guard, then the fast loop | loop versioning (SCEV) |
| divide by a by-reference value | cloned a constant-divisor specialization | IPA-CP `.constprop` |
| `std::pow(x,2.0)` vs `x*x` | folded pow→mul | `expand_builtin_pow` / SimplifyLibCalls |

This is §7 rule 6 ("a knob the compiler can re-derive is not a knob") observed live. It gives the
**selection criterion for a real lever, and for NestForge's worst/best variants**: a lever survives
only when the fast form is **not recoverable by a runtime guard**. Don't spend variants on
alias/predictability/stride axes — the backend erases them.

**The three that survived** (each non-recoverable by construction):

| Lever (worst → best) | Δ g++ | Δ clang++ | Mechanism | Emittable in DaCe |
|---|---|---|---|---|
| modulo/divide by a runtime var vs a constant | **12.0–12.6×** | **9.2–11.7×** | constant divisor → Granlund–Montgomery magic-number multiply (vectorizes); runtime → scalar `div` | `sym2cpp` already picks the fast form for a constant extent — the slow form is laundering a known constant through a symbol |
| elementwise op as an out-of-line call vs an inlined tasklet | **6.3–6.5×** | **7.3–8.0×** | a TU boundary blocks the inliner → no LoopVectorize/LICM across it | real DaCe choice: inlined tasklet vs a library-node call |
| reduce into a memory location vs a local accumulator | **4.3×** | 1.13× (clang versions it) | aliasing blocks LICM scalar promotion → no reduction PHI → no vectorization | **DaCe already emits the slow form today** for `__state->`-resident accumulators (`cpu.py:2385-2389`) |

Two things to carry forward. First, the reduce-into-memory lever is not hypothetical: DaCe hard-codes
its slow side for persistent/external reduction targets, so a real vectorizable reduction is being
left on the table wherever the accumulator lives in the state struct — worth its own look (it is a
`reduction(op:var)`-clause / local-temp fix, not a style knob). Second, clang recovers that same lever
with a runtime alias check (→1.13×), so even among the survivors the **sign and magnitude are
compiler-dependent** — §1 again.

**What did NOT reproduce.** The per-row `memcpy` idiom (§5's 2.6× lead from a different box) came out
**inside the noise floor on Zen 4** — the idiom fires (we see `call memcpy@plt`), but a 64-byte glibc
`memcpy` here is fast. So the 2.6× is **microarch/glibc-specific**, and §5's "why glibc stalls at
1–3 KiB/call is open" correctly stays open — it is not even observable on this hardware.

---

## 7. Rules this imposes on a code generator

1. **Match a known-good ABI; don't reason from first principles.** Lemire and #121262 disagree about
   which binding wins — both are right, on different targets. There is no rule to derive.
2. **Never hardcode a style choice. Make it a flag, default it to the legacy form, and sweep it.**
   This is exactly why `codegen_params` exists, and why every key states `Applies to:`.
3. **A knob's default must reproduce today's output byte-for-byte**, so the flag is opt-in and the
   legacy generator is provably unaffected.
4. **Annotations are not free information.** `!noalias` disabled vectorization; `__builtin_assume`
   got banned from libc++. In a generator, an annotation is an IR perturbation with a cost-model side
   effect and occasionally a soundness one.
5. **Don't promise no-alias on data you declared may-alias.** That is a miscompile, not a win — which
   is why `may_alias` arrays are fused *without* `__restrict__`.
6. **A knob the compiler can re-derive is not a knob.** Identical Code Folding collapsed every
   templated index variant onto one body. If the choice is expressible in the type system, the
   backend owns it, and the "variant" is a jump instruction.
7. **Never let a correctness-relevant type be *deduced*.** Deduction has no diagnostic: our call sites
   say `auto i = 0`, so a templated helper picks int32 and wraps at 2**31 silently. State it.
8. **Chase the mechanism before the number.** The −60% index-type headline was clang's memcpy idiom,
   not index math — tuning the knob would have papered over a 2.6× lowering pessimization.

---

## 8. Further reading

**Framing**
- Gong et al., [Compiler stability](https://iacoma.cs.uiuc.edu/iacoma-papers/oopsla18.pdf) (OOPSLA'18) — the 16.9% number.
- Siso et al., [Evaluating Auto-Vectorizing Compilers through Objective Withdrawal of Useful Information](https://ora.ox.ac.uk/objects/uuid:eac7b135-e92b-48dc-a1f7-4de66a441390/files/szg64tk95s) (TACO'19) — the closest mechanistic twin; see the S276 case.
- Mytkowicz et al., [Producing Wrong Data…](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) (ASPLOS'09).

**The exact axis** — [llvm#121262](https://github.com/llvm/llvm-project/issues/121262) · [Lemire](https://lemire.me/blog/2019/09/05/passing-integers-by-reference-can-be-expensive/) · [N3538](https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3538.html) · [const isn't for performance](https://theartofmachinery.com/2019/08/12/c_const_isnt_for_performance.html)

**Cost models / IR shape** — [Pohl MASCOTS'19](https://www.cosenza.eu/papers/PohlMASCOTS19.pdf) · [goSLP](https://arxiv.org/pdf/1804.08733) · [uiCA](https://arxiv.org/abs/2107.14210) · [rust#82834](https://github.com/rust-lang/rust/pull/82834) · [llvm.assume](https://discourse.llvm.org/t/llvm-assume-blocks-optimization/71609)

**Idiom recognition + deduction** — [Downs, *Sometimes zero is too much zero*](https://travisdowns.github.io/blog/2020/01/20/zero.html) (the 29× `std::fill`) · [llvm#87440](https://github.com/llvm/llvm-project/issues/87440) (LoopIdiom `memcpy` is a one-way door) · [LLVM Frontend Performance Tips](https://llvm.org/docs/Frontend/PerformanceTips.html) (emit `zext`, add range info) · [Walfridsson, undefined signed overflow](http://kristerw.blogspot.com/2016/02/how-undefined-signed-overflow-enables.html) (sign, not width, gates promotion) · [LLVM MisExpect](https://llvm.org/docs/MisExpect.html)

**Vectorization surveys** — Siso et al. TACO'19 (above) · [Sakib et al. 2025](https://arxiv.org/abs/2502.11906) — 6 compilers × 2 ISAs, only 46–56% of TSVC2 vectorizes, and near-identical loops (s4112 vs s4115) get opposite decisions

**Pragmas / memory hints** — [McCalpin, non-temporal stores](https://sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/) (bimodal, sign flips by thread count) · [Eleliemy/Ciorba, OpenMP scheduling](https://arxiv.org/abs/1809.03188) (dynamic,1 → 5×) · [llvm#42332](https://github.com/llvm/llvm-project/issues/42332) (excessive unrolling) · [Burnus/Geva/Prince, vectorizer pragmas](https://gcc.gcc.gnu.narkive.com/lR8PtMw5/vectorizer-pragmas) (`omp simd` "the user rules" — overrides the cost model). REFUTED: array-`reduction`-is-slow is thread-creation overhead, not the copy ([Fortran Discourse](https://fortran-lang.discourse.group/t/openmp-efficiency-in-reduction-in-large-array/4892))

**Register pressure / branch shape** — [Shipilëv, FPU spills](https://shipilev.net/jvm/anatomy-quarks/20-fpu-spills/) (spill cost anchor, 37%) · [Lemire, mispredicted branches](https://lemire.me/blog/2019/10/15/mispredicted-branches-can-multiply-your-running-times/) (~5×) · [Algorithmica, binary search](https://en.algorithmica.org/hpc/data-structures/binary-search/) (branchless sign flips by size) · [Wennborg, switch lowering](https://llvm.org/devmtg/2015-10/slides/Wennborg-SwitchLowering.pdf) (whole-program effect sub-noise). REFUTED: alignment `movaps`-vs-`movups` is [a ~2% myth](https://lemire.me/blog/2012/05/31/data-alignment-for-speed-myth-or-reality/) on modern x86; the real mechanism is cache-line-split avoidance

**Loop addressing form** — [LLVM Passes](https://llvm.org/docs/Passes.html) (indvars canonicalizes to a single 0-step-1 IV; LSR lowers back to a walking GEP — pointer-vs-index is a non-knob) · [LLVM Vectorizers](https://llvm.org/docs/Vectorizers.html) (accepts pointer-IV form) · [ILT, signed overflow](https://www.airs.com/blog/archives/120) (signed IV lets the compiler assume no wrap) · [llvm-dev delinearization](https://groups.google.com/g/llvm-dev/c/Apd9mx3tbcU) (fails on symbolic bounds)

**Prefetch / function attributes** — [Johnny's SW Lab, software prefetch](https://johnnysswlab.com/the-pros-and-cons-of-explicit-software-prefetching/) (2.6× gather / neutral on streams) · [Lemire, is `__builtin_prefetch` useful](https://lemire.me/blog/2018/04/30/is-software-prefetching-__builtin_prefetch-useful-for-performance/) (10%, below floor; "if it helps, your code needs work") · [Intel ivdep](https://www.smcm.iqfr.csic.es/docs/intel/compiler_c/main_cls/cref_cls/common/cppref_pragma_ivdep.htm) (respects proven deps — the only non-miscompiling ivdep) · [P0886 assume_aligned](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0886r0.pdf) (skips the vectorizer peel+epilogue, not the move) · [llvm D140089](https://reviews.llvm.org/D140089) (sret memcpy elimination 17–42%) · [GCC function attributes](https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Common-Function-Attributes.html) (`const`/`pure`/`malloc`/`hot`/`cold` — soundness contracts)

**SVE / Neoverse V2** — [Arm Neoverse V2 SWOG](https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668) (Tables 3-1, 3-25; §4.17) **— read before repeating any pipe claim** · [de Smalen, Optimizing Code for Scalable Vector Architectures](https://llvm.org/devmtg/2021-11/slides/2021-OptimizingCodeForScalableVectorArchitectures.pdf) (LLVM Dev'21) · [GCC PR115531](https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg818483.html) · [predicate-pipe dependency](https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg373383.html) · [A Critical Look at SVE2](https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd) — why a V2 result does not generalize to "SVE"

**Methodology** — [STABILIZER](https://people.cs.umass.edu/~emery/pubs/stabilizer-asplos13.pdf) · [Beyls, Ameliorating Measurement Bias](https://llvm.org/devmtg/2016-03/Presentations/Beyls2016_AmelioratingMeasurmentBias.pdf)

---

## 9. Candidate knobs already located in the generator

An audit of `experimental_cpu.py` and the `cpu.py` / `cpp.py` emitters it overrides found ten more
hardcoded style points, each a place the generator picks one C++ spelling over a semantically-equal
one. This is the implementation backlog for future `codegen_params` keys — ranked by (plausible
effect × emittability × low legacy risk). Two are shipped (`const_scalar_abi`, `index_ctype`); the
rest are located but not built. **`affects: both`** means the site is in the shared emitter and the
knob's default MUST reproduce legacy byte-for-byte (§6 rule 3); **`experimental`** = zero legacy risk.

| Proposed key | Values | Site | Affects | Mechanism (hypothesis) | Cost |
|---|---|---|---|---|---|
| `heap_ptr_restrict` | restrict (def) / none | `experimental_cpu.py:476` | experimental | Alias analysis → vectorization; the doc's #1 bimodal axis (rust#82834) | trivial |
| `index_fn_qualifier` | hdfi_constexpr (def) / always_inline / static_inline | `experimental_cpu.py:41` | experimental | Guarantees the `_idx` call inlines so the vectorizer/idiom-recognizer sees `p[i*S+j]` | trivial |
| `loop_index_type` | auto (def) / int64 / int32 | `cpu.py:2609` | **both** | Index/accumulator width → TBAA disproves aliasing → store sinks out of loop; no-wrap/SCEV legality | moderate |
| `loop_bound_cmp` | lt (def) / le / ne | `cpu.py:2609` | **both** | `!=` with unknown-sign step forces a termination proof; `<` is SCEV-friendly | trivial |
| `array_access_form` | index_fn (def) / row_pointer | `experimental_cpu.py:67` | experimental | Strength reduction + the §5 per-row `memcpy` idiom; row-pointers change what alias analysis sees | structural |
| `heap_alignment` | 64 (def) / 32 / none | `cpu.py:1997`, `experimental_cpu.py:497` | **both** | NOT `movaps` vs `movups` (that is ~2% on modern x86, [a myth](https://lemire.me/blog/2012/05/31/data-alignment-for-speed-myth-or-reality/)): the real effect is cache-line-split avoidance + the vectorizer's alignment peel — expect ~0 except on split-prone access, and the peel/versioning is a cost | moderate |
| `omp_simd_clause` | off (def) / on | `cpu.py:2502` | **both** | Forces the vectorizer AND asserts no loop-carried dep — a *soundness* edge (opt-in only) | moderate |
| `loop_bound_hoist` | inline (def) / hoisted | `cpu.py:2609` | **both** | LICM; usually nil, matters only when a compound bound defeats hoisting | trivial |
| `const_scalar_binding` | fused_const (def) / mutable_decl | `experimental_cpu.py:691` | experimental | `const` binding + decl placement → regalloc/lifetime; **weak** — §2 doubts `const` is causal | moderate |
| `size_fn_qualifier` | consteval (def) / constexpr | `experimental_cpu.py:44` | experimental | UNKNOWN — both fold at compile time; expected compile-time-only | trivial |
| `loop_index_signed` | signed (def) / unsigned | `cpu.py` loop emitter | **both** | Signed IV → `nsw` → SCEV proves no-wrap → IV widening + vectorization *legality* (not just address cost). Non-re-derivable **on symbolic shapes** — DaCe's exact case. Distinct from `index_ctype`'s width/sext axis ([Walfridsson](http://kristerw.blogspot.com/2016/02/how-undefined-signed-overflow-enables.html), [ILT](https://www.airs.com/blog/archives/120)) | moderate |
| `sw_prefetch` | off (def) / gather | tasklet/gather emit | experimental | `__builtin_prefetch` on an indirect/gather access. A *pure timing hint* (not a soundness assertion), so it can be defaulted-on for detected indirection; **2.6× on irregular gather**, ~neutral on streams | moderate |

Best "free to expose, low risk" first picks: `heap_ptr_restrict`, `index_fn_qualifier`, `loop_bound_cmp`.
Highest mechanism but structural: `array_access_form` (row-pointer) — the one grounded in §5's own
memcpy-idiom finding. Note `loop_index_type` and `index_ctype` are *distinct* axes: the former types
the loop induction variable (shared emitter, legacy-affecting), the latter the helper body.

Judged NOT worth a knob (no mechanism, semantics-adjacent, or data-driven not style): brace/scope
elision and single-line tasklet collapse (lifetime/readability only); OpenMP `schedule`/`collapse(n)`/
`num_threads` (already driven by map properties, not a hardcoded spelling); reduction infix-vs-helper
(bit-exactness constraint, not neutral); `i += skip` vs `i = i + skip` (identical IR); the
`{(T)(expr)}` len-1 cast (suppresses `-Wnarrowing`, a correctness concern); and **branch vs
branchless/`select`** for a data-dependent conditional — that is a *vectorization-legality transform*
the vectorizer already owns (predication is how a masked loop becomes vectorizable), not a free
post-hoc codegen spelling, so it is not a `codegen_params` knob. (Its measured behavior is real —
~5× on genuinely unpredictable, un-if-convertible branches, sign-flipping by working-set size — but
the compiler if-converts the common case away, per §6.)

Also judged NOT a knob — **pointer-walking (`*p++` / `p += stride`) vs indexed `base[i]`, and a
non-zero pointer start / custom stride.** The backend re-derives both directions and the loop form
does not survive to the vectorizer: *"All loops are transformed to have a single canonical induction
variable which starts at zero and steps by one"* and *"Any pointer arithmetic recurrences are raised
to use array subscripts"* ([LLVM indvars](https://llvm.org/docs/Passes.html)), then loop-strength-reduce
lowers the surviving index back to a walking GEP per target, and the vectorizer accepts pointer-IV
form regardless ([LLVM Vectorizers](https://llvm.org/docs/Vectorizers.html)); on x86-64 the
scaled-index addressing mode makes the index form's address math free anyway. So a carried-increment
loop is a *readability* choice, not a performance lever — and it is only legal on a SEQUENTIAL
construct, since a walking pointer across iterations is a loop-carried dependency an OpenMP `parallel
for` forbids (each thread must derive its address from its private index). The two axes hiding next to
it that ARE real on symbolic shapes are `loop_index_signed` and the multi-dim subscript form
(linearized `A[i*N+j]` vs a delinearizable 2-D form — SCEV delinearization *"tend[s] to fail in many
common cases (involving loops with symbolic bounds)"*, [llvm-dev](https://groups.google.com/g/llvm-dev/c/Apd9mx3tbcU)).

The function-attribute family a generator could attach to its emitted helpers, and why most do not
clear the bar: `((const))`/`((pure))` on an `<array>_idx` helper would enable CSE **but is redundant
once the helper inlines** (GVN already commons the exposed arithmetic) and is a *miscompile* if
misapplied to a pointer-reading function; `((hot))`/`((cold))` are **pure code layout**, so entirely
inside §3's 7–51% confound (the profile-driven MFS ceiling is only ~1.6–2.3%); `((malloc))` on an
allocator is the return-value twin of `__restrict__` (same bimodality, a miscompile on a View-returning
helper); `((noinline))` is just the attribute spelling of §6's measured 6–8× out-of-line-call lever.
Explicitly OUT OF SCOPE as *semantic* (they change FP results, so not neutral knobs): `#pragma clang fp
reassociate`/`contract(fast)`, `__attribute__((optimize("fast-math")))`, `#pragma STDC FP_CONTRACT` —
and GCC's own manual disqualifies `((optimize(...)))` for production regardless (*"for debugging
purposes only"*).

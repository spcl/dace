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
| Making an operand compile-time-constant | GCC drops to **scalar branching** on TSVC S276 where the unknown-parameter form vectorizes ([TACO'19](https://ora.ox.ac.uk/objects/uuid:eac7b135-e92b-48dc-a1f7-4de66a441390/files/szg64tk95s)) — *more* information made it *worse* |
| `-fno-semantic-interposition` | **5–27%** on CPython, zero source change ([Fedora](https://fedoraproject.org/wiki/Changes/PythonNoSemanticInterposition)) |

---

## 5. Rules this imposes on a code generator

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

---

## 6. Further reading

**Framing**
- Gong et al., [Compiler stability](https://iacoma.cs.uiuc.edu/iacoma-papers/oopsla18.pdf) (OOPSLA'18) — the 16.9% number.
- Siso et al., [Evaluating Auto-Vectorizing Compilers through Objective Withdrawal of Useful Information](https://ora.ox.ac.uk/objects/uuid:eac7b135-e92b-48dc-a1f7-4de66a441390/files/szg64tk95s) (TACO'19) — the closest mechanistic twin; see the S276 case.
- Mytkowicz et al., [Producing Wrong Data…](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) (ASPLOS'09).

**The exact axis** — [llvm#121262](https://github.com/llvm/llvm-project/issues/121262) · [Lemire](https://lemire.me/blog/2019/09/05/passing-integers-by-reference-can-be-expensive/) · [N3538](https://open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3538.html) · [const isn't for performance](https://theartofmachinery.com/2019/08/12/c_const_isnt_for_performance.html)

**Cost models / IR shape** — [Pohl MASCOTS'19](https://www.cosenza.eu/papers/PohlMASCOTS19.pdf) · [goSLP](https://arxiv.org/pdf/1804.08733) · [uiCA](https://arxiv.org/abs/2107.14210) · [rust#82834](https://github.com/rust-lang/rust/pull/82834) · [llvm.assume](https://discourse.llvm.org/t/llvm-assume-blocks-optimization/71609)

**SVE / Neoverse V2** — [Arm Neoverse V2 SWOG](https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668) (Tables 3-1, 3-25; §4.17) **— read before repeating any pipe claim** · [de Smalen, Optimizing Code for Scalable Vector Architectures](https://llvm.org/devmtg/2021-11/slides/2021-OptimizingCodeForScalableVectorArchitectures.pdf) (LLVM Dev'21) · [GCC PR115531](https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg818483.html) · [predicate-pipe dependency](https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg373383.html) · [A Critical Look at SVE2](https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd) — why a V2 result does not generalize to "SVE"

**Methodology** — [STABILIZER](https://people.cs.umass.edu/~emery/pubs/stabilizer-asplos13.pdf) · [Beyls, Ameliorating Measurement Bias](https://llvm.org/devmtg/2016-03/Presentations/Beyls2016_AmelioratingMeasurmentBias.pdf)

# `compiler.cpu.codegen_params` — the option set, and the contract behind it

Design doc for the tuning surface of the readable CPU code generator
(`compiler.cpu.implementation = experimental_readable`).

Companion docs, no overlap:

- [`READABLE_CODEGEN_MERGE.md`](READABLE_CODEGEN_MERGE.md) — architecture: what the generator emits, which
  files, which passes.
- [`CODEGEN_STYLE_PERFORMANCE.md`](CODEGEN_STYLE_PERFORMANCE.md) — measurement: how a knob is calibrated,
  what the noise floor is, which candidates died.
- [`samples/codegen/style_levers/README.md`](samples/codegen/style_levers/README.md) — standalone reproducers.

This doc is the **options** half: what each key is, who it applies to, why it exists as a key at all,
and how to sweep them.

---

## 1. High level

`codegen_params` is a flat config group of **semantic no-ops**. Every key changes only the *spelling*
of the emitted C++, never what it computes. That is the whole point: the group is a search space for
an autotuner, not a correctness surface.

Two invariants govern every key. Both are load-bearing; break either and the group stops being safe
to sweep.

**Invariant 1 — every key states an `Applies to:` line.** The group lives under `compiler.cpu`, but
some keys reach both generators, some only the readable one, and one splits down the middle
(`decl_placement`). A reader must not have to guess. The line is the first line of every key's
description in `config_schema.yml`.

**Invariant 2 — every default reproduces today's output byte-for-byte.** Not "equivalent",
not "close" — byte-identical, for both generators. This is what makes the group additive: adding a
key cannot regress an existing build. Legacy in particular must stay byte-identical for *every*
value of an experimental-only key, which is why most keys gate on the generator before doing
anything.

A third rule comes from `CODEGEN_STYLE_PERFORMANCE.md` §7 and decides what is even allowed in:

**Rule 6 — a knob the compiler can re-derive is not a knob.** If the backend recovers the fast form
via a runtime guard, constant propagation, or an IPA clone, the axis is not a lever and does not get
a key. This is why the group is short: branchless-vs-branch, laundered strides, and
by-reference divisors all collapsed to noise and were rejected.

---

## 2. The keys

Twelve keys. Defaults in the table are the byte-identical-to-today values.

| Key | Default | Applies to |
|---|---|---|
| `const_scalar_abi` | `by_ref` | readable only |
| `index_ctype` | `int64` | readable only |
| `heap_ptr_restrict` | `restrict` | readable only |
| `index_fn_qualifier` | `inline_constexpr` | readable only |
| `loop_access_form` | `indexed` | readable only |
| `scalar_init_style` | `split` | readable only |
| `ssa_loop_scalars` | `off` | readable only |
| `loop_index_type` | `auto` | **both** |
| `loop_bound_cmp` | `lt` | **both** |
| `loop_decl_style` | `for_init` | **both** |
| `split_nsdfg_translation_units` | `false` | **both** |
| `decl_placement` | `eager` | **both** for loop counters; readable only for scalars |

They group into four sub-designs.

### 2.1 Access path — how an element is reached

Only the readable generator has these: legacy inlines its offset arithmetic at every access site and
emits no `<array>_idx` helper, so there is no access path to re-spell and it ignores all three.

- **`index_ctype`** (`int64` | `int32`) — the integer type `<array>_idx` / `<array>_size` compute the
  flat index in. **Unsafe above 2³¹**: `int32` silently wraps, and the bound is on the *element
  count*, not bytes — an int8 array overflows at 2 GiB, float64 at 16 GiB. Never selected
  automatically, because the generator cannot prove a symbolic shape stays under the bound.
- **`index_fn_qualifier`** (`inline_constexpr` | `always_inline`) — `static DACE_HDFI constexpr` vs
  additionally forcing `__attribute__((always_inline))`. Inlining is the *precondition* for the
  access to read as `p[i*S+j]` to the vectorizer; `always_inline` matters only where a helper is
  wrongly left out of line, which pins a vectorization precondition.
- **`loop_access_form`** (`indexed` | `ptr_increment`) — recompute the flat index each iteration, or
  walk a base pointer by its constant per-iteration stride. Identical accesses in identical order.
  Explicitly a **readability / variant-space form, not a speed lever** — the backend canonicalizes
  the two together. It is in the group to widen the search space, not because it is expected to win.

### 2.2 Declarations — where a name is bound, and how tightly

- **`decl_placement`** (`eager` | `late`) — the declare-near-use axis, and the only key that spans
  both generators asymmetrically. `late` moves each declaration as close to its first use as is
  **provably sound**; anything failing its gate keeps the eager declaration, so `late` is
  a best-effort narrowing, never a correctness bet. Two constructs, two independent gates:
  - *Loop counters* (both generators, shared emitter): declared in the `for`-init clause. Gate:
    exactly one owning `LoopRegion`, not inverted, plain `i = <expr>` init, and the counter is not
    used outside the loop.
  - *Scalars* (readable only): deferred to the first-use scope. Gate: single-state, absent from every
    interstate edge's free symbols and every region condition, one scope, and no read-before-write.
    **This gate is where the one real miscompile of this feature lived** — see §4.
- **`scalar_init_style`** (`split` | `fused`) — `T x;` + `x = expr;` vs `T x = expr;`. The mutable
  counterpart of the `const T x = expr;` binding that write-once scalars already get from
  `MarkConstInit`: a reassigned scalar cannot be `const`, but its first write can still *define* it.
  Later writes stay plain assignments.
- **`ssa_loop_scalars`** (`off` | `on`) — the only key that runs an **SDFG rewrite**, not an emission
  change: `PrivatizeScalars` (`ScalarFission`) versions a repeatedly-reassigned scope scalar into
  single-assignment names (`nx`, `nx_0`, …). Each version is then write-once, so `MarkConstInit`
  turns it into a `const` binding. It runs *before* `MarkConstInit` in the `codegen.py` pipeline for
  exactly that reason. Default `off` → pass not run → byte-identical.
- **`const_scalar_abi`** (`by_ref` | `by_value`) — `const T& x` vs `const T x` for a read-only scalar.
  **The sign is not fixed**, which is the entire argument for making it a key: LLVM #121262 is this
  axis with the *reference* winning, while Lemire measured by-reference 4.5× *slower* on an
  aliasing-bound loop. Measured here on Neoverse-V2 + clang: the two forms lower a masked
  sum-reduction differently (`ld1w p/z` + `sel` + unpredicated `fadd` vs `ldr z` + merging-predicated
  `fadd`), correlating with ~17%.
- **`heap_ptr_restrict`** (`restrict` | `none`) — `T* __restrict__ p` vs `T* p`. `restrict` is the
  default and the faster bet, since no-alias is what lets the vectorizer prove independence. **Never
  applied to a `may_alias` descriptor regardless of this key** — that array is deliberately reachable
  through a second pointer, so promising no-alias would be a miscompile, not a win. Genuinely
  bimodal when it does apply (46 faster / 18 slower in rust#82834).

### 2.3 Loop form — shared emitter, so both generators

- **`loop_index_type`** (`auto` | `int64` | `int32`) — `auto` deduces from the lower bound, so the
  usual `0` gives `int` (32-bit index math). Distinct from `index_ctype`, which types the *helpers*.
  Widening can let TBAA disprove aliasing so a store sinks out of the loop, and it changes the
  no-wrap facts SCEV relies on — but it is also register pressure, and pure cost on a 32-bit-safe
  loop. Sign not fixed; sweep it.
- **`loop_bound_cmp`** (`lt` | `le` | `ne`) — `i < end + 1`, `i <= end`, or `i != end + 1`, for the
  identical iteration space. `ne` carries a real correctness subtlety: naive `i != end + 1` is only
  correct when the stride divides the range, else the counter steps *over* the bound and the loop
  never terminates. For a non-unit stride the bound is normalised to the first value the counter
  actually lands on at or past the end, `begin + int_ceil(end + 1 - begin, skip) * skip`, which it is
  guaranteed to hit exactly.
- **`loop_decl_style`** (`for_init` | `hoisted`) — `for (auto i = 0; ...)` vs `auto i = 0; for (; ...)`.
  Lifetime is the only difference, and that is what makes it an axis: a longer live range is a
  different input to register allocation. It also means two sibling maps sharing a parameter name
  would collide, so `hoisted` forces the map's encapsulating brace to be emitted.

### 2.4 Translation-unit structure

- **`split_nsdfg_translation_units`** (bool, both generators) — `OutlineTopLevelNests` wraps each
  top-level map-nest / loop region in a `no_inline` nested SDFG, and each is emitted to
  `src/cpu/nsdfg/<label>.cpp` instead of the one frame `.cpp`. The frame keeps a forward declaration
  and the call.

  **The current config description is wrong and is a tracked fix (T3).** It says this is "a
  BUILD-PARALLELISM tactic, not a codegen-quality one". Build parallelism is real — one huge TU is
  compiled by exactly one core, so a big SDFG serialises the build under the Ninja generator — but
  `CODEGEN_STYLE_PERFORMANCE.md` §7 documents a size-gated **runtime** lever here too: past a
  threshold, TU size changes backend decisions (inlining budgets, register allocation), so the split
  is not runtime-neutral. The description must stop claiming it is.

---

## 3. How to use

Per-key, config or env (env wins):

```sh
DACE_compiler_cpu_implementation=experimental_readable \
DACE_compiler_cpu_codegen_params_decl_placement=late \
DACE_compiler_cpu_codegen_params_scalar_init_style=fused \
  python your_program.py
```

In Python:

```python
from dace import Config
Config.set('compiler', 'cpu', 'implementation', value='experimental_readable')
Config.set('compiler', 'cpu', 'codegen_params', 'decl_placement', value='late')
```

Read the emitted code without running it:

```python
code = sdfg.generate_code()[0].clean_code   # the frame .cpp
```

**Config leaks across tests.** The flag is process-global and is *not* part of the SDFG hash, so two
generators sharing an SDFG name will serve one's compiled binary to the other and mask a real
divergence. `tests/codegen/readable/conftest.py` provides `use_implementation` (a context manager)
and the `LEGACY` / `EXPERIMENTAL` constants; the corpus tests additionally name each SDFG per
`(kernel, implementation, target)`. Use them — do not `Config.set` a generator in a test without
restoring it.

Sweeping: `performance_regression_jobs/codegen_variants/` drives the group. Calibrate against
`CODEGEN_STYLE_PERFORMANCE.md` §1 before believing any result — clang's mean CoV across
semantically-equivalent forms is ~16.9%, the reportable threshold on the Zen 4 box is ~1.2×, and
`layout_probe.cpp` shows five *byte-identical* twins spanning 1.28×. Anything under that is a draw.

---

## 4. What this design got wrong once (keep it here)

The scalar half of `decl_placement=late` shipped with a real miscompile: the gate used
`isinstance(dfg, SDFGState)` as proof that a scalar lives in one state. It is not.
`allocate_array`'s `dfg` argument is the **first state the data appears in**, not the allocation
scope — `framecode.py:945` appends `first_state_instance` even when `curscope` is the whole SDFG. A
scalar live across three states therefore arrives with a real `SDFGState` in hand, got declared
inside one state's brace, and produced `error: 's' was not declared in this scope`.

Two lessons the option set depends on:

1. **Prove the property, don't infer it from an argument's type.** The gate now checks explicitly: no
   other state's `data_nodes()` mention it, absent from every `all_interstate_edges()` free-symbol
   set, absent from every region's `used_symbols(all_symbols=True, with_contents=False)`. Note
   `SDFGState.used_symbols(with_contents=False)` returns `set()`, so states must be asked *with*
   contents and regions *without*.
2. **Hand-built SDFG fixtures hide whole bug classes.** Every fixture was single-state, so the class
   was invisible; a `@dace.program` accumulator caught it instantly. Worse, the knob passed all its
   tests while being near-dead on real code — its gate admitted only state-top-level scalars, but
   frontends put scalars in map scopes. Pair every hand-built unit fixture with a `@dace.program` one
   and dump the C++ to confirm the feature actually fires.

---

## 5. Not knobs (and why)

Rejected by Rule 6 — the compiler re-derives the fast form: branch-vs-branchless (if-converted to
`cmov`), laundered stride (`cmp $0x1` version guard), by-reference divisor (IPA-CP `.constprop`
clone), `pow(x,2.0)` (folds to `x*x`).

Rejected by measurement — real axis, no fast form to emit: the whole heat_3d legacy-vs-readable
source-form difference. SROA/mem2reg/GVN erase the `[1]`-temp, brace, index-fn and restrict axes
before the vectorizer, so in one binary every hybrid lands 0.97–1.00×. The apparent ~1.30× is the
layout confound. Same verdict for stockham_fft.

**Explicit copy — open, and previously analysed wrongly.** `InsertExplicitCopies` lifts implicit copy
edges to a `CopyLibraryNode`, whose `ExpandAuto` selects by subset volume and storage
([`copy_node.py:85-135`](dace/libraries/standard/nodes/copy_node.py#L85-L135)):

| Case | Emission |
|---|---|
| multi-element, host CPU-resident, contiguous, same-layout | `MemcpyCPU` → `memcpy(_out, _in, N * sizeof(T))` |
| single-element | `Tasklet` → `_out = _in` |
| non-contiguous / reshape / device-scope multi-element | `MappedTasklet` (a map) |
| GPU_Shared block-cooperative | `SharedMemoryCollective` → `dace::CopyND` |

An earlier note in this repo's history framed the axis as "`dace::CopyND` vs `=`" and concluded no
key was warranted. That was wrong twice over. `dace::CopyND` is the *legacy implicit* lowering —
`SharedMemoryCollective` is, per the file, "the only remaining `dace::CopyND` user" — so it is not
what the pass emits at all; and the pass's multi-element host output is `memcpy`, with `=` reserved
for the single-element case. The conclusion that the heat_3d win is purely the semantic
"copy becomes its own `#pragma omp parallel for`" half does not follow either, since `MemcpyCPU` is
one serial `memcpy` and not a map — that reasoning only covers the `MappedTasklet` branch, and which
branch heat_3d takes is unmeasured.

The axis is therefore codegen-visible and the key remains an open question, not a closed one.
`performance_regression_jobs/ec_ab.py` is the A/B to settle it. Note `lever2_memcpy_idiom.cpp`
measures the `memcpy`-idiom component at 0.86–1.08× on Zen 4 (inside the floor) but 2.6× on another
box — microarch-specific, so the sweep must not be run on one machine and generalised.

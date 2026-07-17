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

**Invariant 2 — LEGACY output is byte-identical for every value of every key.** This is the hard
one, and it is absolute: legacy must not move, whatever the group is set to. An experimental-only key
gates on the generator before doing anything; a shared-emitter key (`loop_*`,
`split_nsdfg_translation_units`) must default to legacy's existing spelling. It is why
`loop_index_type` still defaults to `auto` even though `auto` deduces `int` from a `0` lower bound
and therefore *wraps* on a map over `0:N` with N above 2³¹ — fixing that by defaulting to `int64`
would rewrite every legacy user's loops, so it waits on a sweep.

**Invariant 2b — a new key's default reproduces today's output.** This is what makes *adding* a key
additive: it cannot regress an existing build. Note this constrains the moment of introduction, not
forever. The readable generator's defaults may be deliberately changed afterwards, and two already
are: `scalar_init_style` defaults to `fused` and `const_init` to `on`, because the readable form is
the point of that generator. Legacy is untouched by both.

A third rule comes from `CODEGEN_STYLE_PERFORMANCE.md` §7 and decides what is even allowed in:

**Rule 6 — a knob the compiler can re-derive is not a knob.** If the backend recovers the fast form
via a runtime guard, constant propagation, or an IPA clone, the axis is not a lever and does not get
a key. This is why the group is short: branchless-vs-branch, laundered strides, and
by-reference divisors all collapsed to noise and were rejected.

---

## 2. The keys

Fifteen keys.

| Key | Default | Applies to |
|---|---|---|
| `const_scalar_abi` | `by_ref` | readable only |
| `index_ctype` | `int64` | readable only |
| `heap_ptr_restrict` | `restrict` | readable only |
| `index_fn_qualifier` | `inline_constexpr` | readable only |
| `loop_access_form` | `indexed` | readable only |
| `scalar_init_style` | **`fused`** | readable only |
| `const_init` | **`on`** | readable only |
| `explicit_copy` | **`on`** | readable only |
| `ssa_loop_scalars` | `off` | readable only |
| `scalar_emission_type` | `keep` | readable only |
| `loop_index_type` | `auto` | **both** |
| `loop_bound_cmp` | `lt` | **both** |
| `loop_decl_style` | `for_init` | **both** |
| `split_nsdfg_translation_units` | `false` | **both** |
| `decl_placement` | `eager` | **both** for loop counters; readable only for scalars |

Every default leaves LEGACY byte-identical. The three in bold do *not* reproduce legacy's spelling
from the READABLE generator — see Invariant 2b. `explicit_copy` is a special case of that: unlike
`const_init`/`scalar_init_style`, whose passes already run, its pass does NOT run in codegen today, so
`on` deliberately changes the readable generator's output (see §2.5).

They group into four sub-designs.

### 2.1 Access path — how an element is reached

Only the readable generator has these: legacy inlines its offset arithmetic at every access site and
emits no `<array>_idx` helper, so there is no access path to re-spell and it ignores all three.

- **`index_ctype`** (`int64` | `int32`) — the integer type `<array>_idx` / `<array>_size` compute the
  flat index in, emitted as the exact-width `<cstdint>` types `int64_t` / `int32_t` (already in scope
  via `dace/dace.h`). `long long` would be wrong to emit here: it is only guaranteed to be *at least*
  64 bits, so it does not state the width the key names. **`int32` is unsafe above 2³¹**: it silently
  wraps, and the bound is on the *element count*, not bytes — an int8 array overflows at 2 GiB,
  float64 at 16 GiB. Never selected automatically, because the generator cannot prove a symbolic
  shape stays under the bound.
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

  **This key is NOT orthogonal to `scalar_init_style`, and cannot be.** `fused` (that key's default)
  makes a scalar's declaration *be* its first write, so it sits at the write by construction and
  there is no separate line left for `eager` to hoist. `decl_placement` therefore governs only the
  declarations that stay separate: loop counters, and the scalars `fused` could not fold (a braced
  first-use tasklet). For those, `fused` still emits the `T x;` at first use rather than the scope
  top — a scalar it meant to fold is one it means to declare near its write. So **at the default
  settings a braced-first-use scalar is not declared at its scope top**; `scalar_init_style: split`
  is how you get that. Pinned by
  `test_fused_defers_a_declaration_it_cannot_fold_even_under_eager`.
- **`scalar_init_style`** (`split` | **`fused`**, default fused) — `T x;` + `x = expr;` vs
  `T x = expr;`. The mutable counterpart of the `const T x = expr;` binding that write-once scalars
  already get from `MarkConstInit`: a reassigned scalar cannot be `const`, but its first write can
  still *define* it. Later writes stay plain assignments. Fusion is decided at **emission**, not
  predicted — a declaration folded into a tasklet that turns out to need its own `{ ... }` block
  would be scoped to that block and invisible to later readers, and whether a tasklet is brace-free
  is not known until its body is lowered; a braced candidate falls back to a plain `T x;` ahead of
  the tasklet.
- **`const_init`** (**`on`** | `off`, default on) — runs `MarkConstInit`, which classifies
  write-once-then-read-only transients for `constexpr T x[N] = {...}` (promoted to an SDFG constant,
  its dead runtime writes removed) or `const T x = expr;`. Default `on` because the pass already
  runs; the key exists to take it *out*. **Not a pure spelling axis** — a `constexpr_static` fill
  deletes the initializing writes, so `off` emits strictly more work.
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
  usual `0` gives `int` (32-bit index math); `int64`/`int32` emit `int64_t`/`int32_t`. Distinct from
  `index_ctype`, which types the *helpers*. Widening can let TBAA disprove aliasing so a store sinks
  out of the loop, and it changes the no-wrap facts SCEV relies on — but it is also register
  pressure, and pure cost on a 32-bit-safe loop. Sign not fixed; sweep it.
  **The `auto` default is a known wart**: 32-bit index math wraps on a map over `0:N` with N above
  2³¹. It stays `auto` anyway because this key reaches the shared emitter, so an `int64` default
  would rewrite every legacy user's loops (Invariant 2) for an unmeasured perf tradeoff.
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

  **Measured (16-core box, under ~load 11, cold cache).** The split's compile-time direction depends
  entirely on per-nest body size versus the fixed per-TU cost of re-parsing `<dace/dace.h>`:

  | Workload | Files | Compile off → on |
  |---|---|---|
  | 16 tiny nests (short arithmetic each) | 1 → 17 | 6.4s → 17.3s (**0.37×, a loss**) |
  | 6 heavy nests (long chains each) | 1 → 7 | 7.8s → 6.9s (**1.14×, a win**) |

  Both produce identical numbers to the single-TU build. The takeaway: splitting pays only once each
  nest's own compile dominates the ~1s header re-parse it now pays per file; many small nests make it
  strictly worse. A quiet machine widens the heavy-nest win (here 5 of 16 cores were busy).

### 2.5 Copy lowering

- **`explicit_copy`** (`on` | `off`, **default on**, readable only) — runs `InsertExplicitCopies`,
  lifting each implicit copy (AccessNode→AccessNode, View endpoints, map staging) to a
  `CopyLibraryNode`, then expands only those nodes. `ExpandAuto` then lowers each copy on its merits:
  a single-element copy to a plain `=` tasklet (whose connectors the readable generator then inlines
  like any other), a contiguous same-layout copy to `std::memcpy`. `off` leaves every copy on the
  implicit `dace::CopyND` path.

  **This default breaks the "new key reproduces today's output" rule, on purpose.** `const_init` and
  `scalar_init_style` default to their non-legacy spelling but run passes that *already* run today;
  `InsertExplicitCopies` does **not** run in `codegen.py` today, so `on` genuinely changes the
  readable generator's output. The absolute invariant still holds: the gate lives inside the
  `implementation == 'experimental_readable'` block, so **legacy stays byte-identical** for either
  value (verified: identical SHA across `on`/`off`). Measured on a two-copy program, readable output
  goes from 4×`dace::CopyND` / 0×`memcpy` (`off`) to 0×`dace::CopyND` / 2×`memcpy` (`on`).

  The pass is **shared**: its other callers (`transformation/layout/prepare.py`, the
  GPU-specialization pipeline) need the `CopyLibraryNode` left **unexpanded** so `RewriteCopyForLayout`
  can turn it into a `TensorTranspose`. That is why the expansion lives in this key's block in
  `codegen.py`, restricted by predicate to the copy nodes just inserted, and not in the pass itself.

### 2.6 Descriptor form — single-value transients

- **`scalar_emission_type`** (`keep` | `scalar` | `len1_array`, default `keep`) — which C++ form a
  single-value TRANSIENT is emitted in: a by-value `Scalar` (`T x;`) or a length-1 `Array` (`T x[1];`,
  indexed `x[0]`). A frontend leaves a mix of the two; this normalizes it. `keep` runs neither
  conversion (byte-identical). `scalar` runs `ConvertLengthOneArraysToScalars`; `len1_array` runs the
  inverse `ConvertScalarsToLengthOneArrays` — both existing, tested passes.

  Two properties are load-bearing. **Signature is never touched**: both conversions are
  `transient_only`, because a non-transient descriptor is the call contract (the caller binds a Python
  scalar to a `Scalar`, a 1-element buffer to an `Array`). **GPU kernel outputs stay length-1 arrays**:
  `scalar` chains `PromoteGPUScalarsToArrays` after the scalarization, which widens a GPU_Global
  length-1 output straight back — a by-value `Scalar` cannot live in device memory. So only host-side
  transients become `Scalar`s.

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

## 4. What this design got wrong (keep it here)

### 4.1 A pass that decided on one graph and mutated another

`const_init`'s `MarkConstInit` speculatively unrolls a small constant-fill map so its classifier sees
element-wise writes instead of a map producer. The classifier can then still decline the target — a
second write, overlapping subsets, a read before the write — by which point the map is gone, and with
it its schedule. On a GPU-transformed SDFG that schedule is the only thing putting the write on the
device, so a declined target left host tasklets writing `GPU_Global` memory. It broke plain CPU too:
an eight-line `@dace.program` using `np.full` corrupted the graph outright (`Isolated node`). The
corpus never caught either, because a candidate needs a compile-time extent ≤16 and nussinov's
`np.full` is `(N, N)`.

The fix is to decide on a throwaway copy and unroll only what pays. That is not sufficient on its own,
and the first attempt was wrong in a way worth recording: the probe unrolls **every** candidate, but
the real run unrolls only the paying **subset** — two different graphs, which can disagree. A map
writing one paying name and one declined name is held back, and its surviving `MapExit` then makes its
other target a runtime multi-write, declining a name the probe had accepted. So the decision is
iterated to a **fixed point**. It terminates because unrolling fewer maps can only decline more names,
so the set shrinks monotonically.

**The lesson:** "decide on a copy" only works if the copy is the graph you will actually produce.

### 4.2 An argument's type mistaken for proof of scope

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

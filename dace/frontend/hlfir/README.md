# HLFIR → DaCe frontend

## Design rationale

Flang already does all Fortran parsing, name binding, type inference,
and elemental-intrinsic lowering. Re-parsing Fortran in Python
duplicates that work and drifts against the standard. This frontend
consumes Flang's already-elaborated HLFIR (MLIR) instead, walks it
into a DaCe SDFG, and regenerates a Fortran wrapper that preserves
the caller's original interface.

The pipeline is five sequential stages. Each stage strengthens an
invariant; the Python walker only has to understand one narrow IR
shape because the earlier stages normalised away the irregularities.

```
.f90 ── flang -fc1 -emit-hlfir ──▶ .hlfir file(s)
                                       │
                            (1) parse + snapshot interface
                                       │
                        (2) inline + verify + DCE
                                       │
                       (3) normalise (HLFIR rewrites)
                                       │
                            (4) build SDFG
                                       │
                          (5) regen binding
                                       ▼
              <entry>_bindings.f90  +  compiled SDFG .so
```

## Stages

**(1) Parse.** One or more `.hlfir` files are loaded into a shared
`MLIRContext` and merged into a single `ModuleOp` via
`SymbolTable::insert`. A pre-pass snapshot of the caller-visible
`hlfir.declare`s — dummy names, intents, derived-type layouts — is
captured as `FortranInterface`. Entry points: `HLFIRModule.parse_files`,
`HLFIRModule.get_fortran_interface`.

**(2) Inline + link.** `hlfir-inline-all` flattens the call tree
into the pinned entry. `hlfir-verify-no-unresolved-calls` fails if
any `fir.call` survives outside the Flang-runtime / libm / C-stdlib
allowlist. `symbol-dce` strips the dead siblings. Requires the
`DialectInlinerInterface` extensions for `fir` / `func` / `LLVM` to
be attached to the context; the bridge constructor does this once.

**(3) Normalise.** HLFIR rewrites in strict order — each depends on
its predecessors having reshaped the IR:

| Pass | Purpose |
| --- | --- |
| `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elemental / scalar-arg procedures |
| `symbol-dce` | Drop now-private callee bodies once `hlfir-inline-all` has folded them in |
| `hlfir-flatten-structs` | AoS → SoA; emits `hlfir.flatten_plan` module attribute |
| `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols |
| `hlfir-default-intent` | Intent-less dummies default to `intent_inout` |
| `lift-cf-to-scf` | Raw-CFG loops (`DO WHILE`, `DO…EXIT`) → `scf.while` + `scf.if` |
| `sccp` → `canonicalize` → `cse` | Fold + simplify + dedupe after HLFIR exposed its constants |

**(4) Build SDFG.** `bridge/extract_vars.cpp` classifies every
`hlfir.declare` as `array`, `symbol`, or `scalar` (rules below).
`bridge/extract_ast.cpp` walks the function body into a recursive
`ASTNode` tree covering `loop` / `while` / `conditional` / `assign` /
`copy` / `memset` / `libcall` / `reduce` / `break` / `return`.
`builder/SDFGBuilder` emits the SDFG from that tree, then snapshots
`sdfg.arglist()` + free symbols into a `FrozenSignature` and pins
it on the SDFG.

**(4b) Post-generation cleanup.** A scalar-promotion pass runs over
the freshly-built SDFG to fold every length-1 array into a true
scalar.  ``REAL(8), VALUE :: x`` lands as a 1-element array on the
SDFG signature (the ABI Fortran uses for pass-by-value scalar dummies);
without this cleanup the caller has to wrap each constant in a numpy
``array([v])`` to satisfy DaCe's array-arg runtime check, and the
generated C++ touches ``x[0]`` instead of just ``x``.  The cleanup
walks every non-transient ``Array`` whose shape is ``(1,)``, rewrites
its descriptor to a ``Scalar``, and rewrites every memlet subset that
references it.  Lifted from the ``yakup/dev`` ``Specialize scalar
utility function`` work.  Runs after ``SDFGBuilder.build()`` returns
and before the ``FrozenSignature`` snapshot is taken so the bindings
emitter sees the post-cleanup signature.

**(5) Regen binding.** `bindings/emit_bindings` reads three artefacts
— `FortranInterface` (outer caller surface), `FrozenSignature`
(inner SDFG surface), and the `FlattenPlan` from stage 3 — and writes
`<entry>_bindings.f90`: a ref-counted Fortran module that preserves
the caller's original signature, populates SDFG symbols via
`size` / `lbound`, and for each struct member picks between a zero-copy
`c_f_pointer` alias and a Fortran `do`-loop deep copy based on the
recipe.

## Data artefacts

These are the structured records that flow between stages. They are
the frontend's stable contract surface — new features extend them,
they do not invent parallel channels.

| Artefact | Produced at | Consumed at | Role |
| --- | --- | --- | --- |
| `FortranInterface` | (1) snapshot | (5) emit | Caller-facing dummy list + derived-type layouts |
| `FlattenPlan` (MLIR attr) | (3) `hlfir-flatten-structs` | (5) emit | Per-dummy AoS→SoA unpack recipe |
| `VarInfo[]` | (4) `extract_vars` | (4) `SDFGBuilder` | Classification + shape + intent per variable |
| `ASTNode` tree | (4) `extract_ast` | (4) `SDFGBuilder` | Normalised CFG + assigns + library-op references |
| `FrozenSignature` | (4) end of `build()` | codegen, (5) emit | SDFG arglist snapshot — drift check at codegen time |

## Mechanisms

**Symbol vs scalar classification.** A Fortran integer is a *symbol*
iff it's a DO induction variable, an array shape extent, a DO bound
(upper or lower), an `hlfir.designate` index, or feeds a control-flow
condition. Everything else integer is a *scalar*. Writes to symbols
become interstate-edge assignments; writes to scalars become
tasklets. Only symbols can appear as array indices — so classification
drives whether a write changes the state machine or mutates data in
place.

**Fortran lbound handling.** Every array descriptor carries
`shape_symbols` and `lower_bounds`. Memlet subsets use Fortran-native
indices; `access.py::build_memlet_index` folds the lbound offset
once at subset-build time. DaCe's own descriptor `offset` field stays
at zero — keeping a single-site offset arithmetic so downstream
transformations don't have to reason about two conventions.

**Assumed-shape alias re-basing.** After `hlfir-inline-all` splices
an assumed-shape callee (`arr(:)`) into a caller whose actual has
custom bounds (`x(-2:2)`), Flang emits a second `hlfir.declare` —
no shape operand, memref via `fir.convert` of the outer declare.
The bridge detects this (`trace_utils::asAssumedShapeAlias`), skips
it in `extract_vars` (one SDFG descriptor per storage), walks
through it in `traceToDecl` (accesses resolve to the outer name),
and rewrites each access's index by `outer_lbound − inner_lbound` at
`buildDesignateIndexExpr`. The caller-side lbound fold then fires
uniformly.

**Exponentiation.** `math.fpowi` / `math.powf` / `math.powi` /
`math.ipowi` all surface as Python `(a ** b)` in the tasklet. A
downstream SDFG simplify pass picks the concrete lowering from the
tasklet's connector types. `hlfir.no_reassoc` (Flang's FP-reassoc
barrier) is transparent in `buildExpr`.

**Section reductions.** Whole-array `SUM` / `PRODUCT` / `ANY` / `ALL`
lower to DaCe's `standard.Reduce`. Section reductions
(`ANY(mask(lo:hi, jk))`) synthesise a loop-accumulator AST — an
init-to-identity assign plus a `kind="loop"` whose body accumulates
via the appropriate Python operator. DaCe's Reduce can't express a
dynamic-section input directly.

**ELEMENTAL procedures → loop + scalar tasklet.** A Fortran
`elemental` procedure is a scalar body that, when called on array
actuals, Flang lowers as `fir.do_loop { hlfir.designate per-arg;
fir.call scalar_body }` for the subroutine form, or as
`hlfir.elemental { hlfir.designate; fir.call; yield_element }` for
the function form. `hlfir-inline-all` splices the callee's body in;
the per-element `hlfir.declare`s that remain (callee dummies named
after outer array elements) are semantic no-ops that carry the
callee's Fortran names into the body. `hlfir-fold-element-aliases`
erases those — whatever reads them rewires to the outer array's
`hlfir.designate`, so the SDFG builder sees the same shape as a
hand-written per-element loop. `fir.do_loop` block args with no
store-to-alloca sibling (produced by this fold) get a synthetic
iter name (`_doit_N`) pushed onto `indexStack()` so `resolveIndex`
still resolves raw block-arg uses inside the body.

**Sibling-assign RAW hazards in loop bodies.** When a `fir.do_loop`
body contains multiple assigns (`f = c*c; a = a - b*f; …`) into
`hlfir.declare`-backed storage, the naïve "one tasklet per assign,
all in one body state" wiring races: non-transient access nodes
share the underlying SDFG array, so a second tasklet's write can
clobber a first tasklet's read even with distinct access nodes. The
loop emitter detects any read-write name overlap across siblings
and serialises them into a chain of states (one tasklet per state,
interstate edges between). Siblings with no hazard still share a
single state — the check is per-loop-body, not a blanket pessimisation.

**OPTIONAL dummies → companion present-flag.** Fortran `OPTIONAL`
args compile to `hlfir.declare` with `fortran_attrs = optional`, and
`present(x)` lowers to `fir.is_present %x : (!fir.ref<T>) -> i1`.
The bridge's `buildBoolExpr` renders that op as the identifier
`<name>_present`, and `extract_vars` registers a corresponding
`int32` symbol `VarInfo` right after each optional declare. The
flag lands on the SDFG signature alongside its host (`sdfg(a=…,
a_present=…)`); non-zero = present, zero = absent. The existing
if/else lowering reads the flag exactly like any other scalar
condition — no new AST kind. Intent-less optionals default to
`intent_in` so `descriptors.py` doesn't misclassify them as
transients. Correctness relies on Fortran's guarantee that every
non-`present()` use of an absent optional is dominated by a
`present` check; the SDFG simply threads that check through.

**AoS → SoA flattening.** `hlfir-flatten-structs` is the bridge's
answer to Fortran derived types: DaCe has no record/struct data
descriptor, so every member is hoisted out as its own top-level
dummy. Each struct-typed `hlfir.declare` is replaced by one
`hlfir.declare` per member (plus shape-ferrying ops), every
`hlfir.designate` onto a struct field rewires to the lifted member,
and the pass stamps a `hlfir.flatten_plan` module attribute
recording the original AoS shape: which dummy each member came from,
its offset in the record, whether it's aliasable (contiguous,
pointer-valid), and the scratch dtype for non-aliasable spill. The
downstream SDFG sees only flat arrays, no knowledge of the parent
type. Stage (5) re-assembles the AoS view on the caller side:
`bindings/loop_copy.py` reads the `FlattenPlan` and emits, per
member, either a zero-copy `c_f_pointer` alias (contiguous +
lifetime-compatible) or an explicit Fortran `do`-loop copy into a
scratch member-typed array. The original caller's signature stays
intact — the binding module is the AoS↔SoA boundary, the SDFG only
ever sees SoA.

**Signature freezing.** `codegen.generate_code` verifies
`sdfg._frozen_signature` before emitting the C++ header. Drift from
the snapshot raises `SignatureDriftError`. Transformations mutate
SDFGs freely, but a header that disagrees with the emitted Fortran
binding cannot ship.

**Defensive walk budgets.** SSA-tracing and expression-reconstruction
helpers guard against pathological IR via `trace_utils::limits::*`
depth constants (`kBuildExprDepth=128`, `kConvertChainDepth=32`,
`kTraceToDeclMax=1024`, etc.). Bumping them never changes semantics
on well-formed HLFIR; it only reduces false-`?` fallbacks on deep
post-optimiser chains.

## Components

```
dace/frontend/hlfir/
├── bridge/            C++ — HLFIR parser + classifier + walker (nanobind)
│   ├── bridge.cpp              MLIRContext, pass pipeline, Python exports
│   ├── extract_vars.cpp        hlfir.declare → VarInfo[]
│   ├── extract_ast.cpp         function body → ASTNode tree
│   └── trace_utils.cpp         SSA tracing + alias helpers + depth limits
├── passes/            C++ — HLFIR → HLFIR rewrites
│   ├── InlineAll.cpp
│   ├── FoldElementAliases.cpp  erase elemental-body alias declares
│   ├── FlattenStructs.cpp      stamps hlfir.flatten_plan
│   ├── PropagateShapes.cpp
│   ├── DefaultIntent.cpp
│   ├── VerifyNoUnresolvedCalls.cpp
│   └── Passes.cpp              registerAllBridgePasses()
├── builder/           Python — SDFG emission (stage 4)
│   ├── __init__.py             SDFGBuilder, _emit dispatch, pipelines
│   ├── context.py              _Ctx (state, pending assigns, iter_map)
│   ├── descriptors.py          add_descriptors, DTYPE mapping
│   ├── access.py               build_memlet_index, indirect-symbol lifting
│   ├── emit_tasklet.py         per-occurrence tasklet + emit_scalar_assign
│   ├── emit_cfg.py             assign / loop / while / conditional
│   └── emit_library.py         copy / memset / libcall / reduce / break / return
├── intrinsics/        Python — Fortran intrinsic registry (consumed by bindings)
├── bindings/          Python — Fortran wrapper emitter (stage 5)
│   ├── frozen_signature.py     FrozenArg + FrozenSignature + drift check
│   ├── fortran_interface.py    OriginalInterface (outer surface)
│   ├── flatten_plan.py         FlattenPlan + to_dict / from_dict
│   ├── block_builders.py       per-Fortran-section emitters
│   ├── loop_copy.py            alias vs deep-copy renderers
│   └── emit_bindings.py        → <entry>_bindings.f90
├── hlfir_to_sdfg.py   compat shim — re-exports from builder/
└── fortran_parser.py  top-level entry: generate_sdfg(...)
```

## Entry point

```python
from dace.frontend.hlfir.fortran_parser import generate_sdfg

# Multi-file (production)
sdfg, bindings_f90, frozen_sig_json = generate_sdfg(
    entry="compute_tendencies",
    hlfir_files=["kernel.hlfir", "math_utils.hlfir"],
    out_dir="build/",
)

# Single-file (experiments; skips binding emission)
sdfg = generate_sdfg("code.hlfir")
sdfg = generate_sdfg("code.hlfir", pipeline="hlfir-propagate-shapes")
```

## Extending the frontend

| If you're adding… | Change here | Then cover in |
| --- | --- | --- |
| a new `math.*` intrinsic | `extract_ast.cpp` `unary_math` / `binary_math` tables | `tests/hlfir/elemwise_intrinsics_test.py` |
| a new reducer | `extract_ast.cpp::kRedTable` (+ `buildSectionReduceAssign` for section form) | `tests/hlfir/reduce_intrinsics_test.py` |
| a new CFG op | `extract_ast.cpp::buildAST` dispatch + `builder/__init__.py::_EMIT_DISPATCH` + emitter in `builder/emit_cfg.py` | ports from `ported_from_f2dace_windmill_test.py` |
| a new binding layout rule | `bindings/loop_copy.py` + new `FlattenRecipe` field | `tests/hlfir/bindings/emit_bindings_test.py` |
| a new HLFIR pass | file in `passes/`, register in `Passes.cpp`, slot into `DEFAULT_PIPELINE` | `tests/hlfir/<pass>_test.py` |

## Non-goals

- Re-parsing Fortran in Python. Flang is authoritative.
- GPU target bindings (would need OpenACC shim emission).
- Fortran SIMD / COARRAY semantics.
- Cross-kernel fusion across translation-unit boundaries — inline-all
  handles intra-TU fusion; cross-TU is the binding emitter's problem.

## Testing

Every supported construct has a seeded numerical test against
gfortran / f2py under `tests/hlfir/`. Binding-specific tests live
in `tests/hlfir/bindings/`. The six E6 velocity-advection
representative loopnests have SDFG-vs-f2py comparisons in
`tests/hlfir/velocity_loopnests/`. All executable-Fortran tests
compile with `gfortran` (Ubuntu's `flang-new-21` ships without
`libflang_rt` so it's emit-HLFIR-only).

```bash
# dump built SDFGs for inspection
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/hlfir/
```

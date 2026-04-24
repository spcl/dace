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

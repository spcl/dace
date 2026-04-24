# HLFIR → DaCe frontend

Fortran source → Flang HLFIR (MLIR) → DaCe SDFG → Fortran binding,
without re-parsing Fortran in Python. Flang does all parsing and
semantic analysis; we walk the already-elaborated HLFIR, emit DaCe
nodes, and regenerate a Fortran wrapper that preserves the user's
original interface.

## Pipeline

```
  parse  →  inline + link  →  preprocess  →  build  →  regen interface
   (1)          (2)              (3)         (4)            (5)
```

Each stage assumes its predecessors' invariants hold. The Python
walker only has to understand one narrow IR shape because every stage
ahead of it has normalised away the irregularities.

### 1. Parse

One or more `.hlfir` files produced by `flang-new -fc1 -emit-hlfir`
are loaded into a shared `MLIRContext` and merged into a single
`ModuleOp` via `SymbolTable::insert`. A pre-pass snapshot of the
caller-facing `hlfir.declare`s is captured as `FortranInterface`
— dummy names, intents, derived-type layouts — before any rewrite
runs.

Bridge entry points: `HLFIRModule.parse_files(paths)` and
`HLFIRModule.get_fortran_interface(entry)`.

### 2. Inline + link

| Pass | Purpose |
| --- | --- |
| `hlfir-inline-all` | Whole-program bottom-up inlining; remaining `fir.call`s are externals / intrinsics only. |
| `hlfir-verify-no-unresolved-calls` | Fails if any `fir.call` is unresolved outside the Flang-runtime / libm / C-stdlib allowlist. |
| `symbol-dce` | Strips every `func.func` except the pinned entry. |

Requires the upstream dialect `DialectInlinerInterface` extensions for
`fir` / `func` / `LLVM` to be attached to the `MLIRContext` — the
bridge's constructor does this once.

### 3. Preprocess

| Pass | Purpose |
| --- | --- |
| `hlfir-flatten-structs` | AoS → SoA on derived types + ELLPACK jagged variants. Struct dummies become per-member flat args; records the unpack as a `FlattenPlan` attribute on the module for the binding emitter to replay. |
| `hlfir-propagate-shapes` | Assumed-shape `(:,:)` dummies pick up real Fortran symbol names through call-graph shape propagation. |
| `hlfir-default-intent` | Intent-less dummies get `intent_inout` so downstream classification sees uniform attributes. |
| `lift-cf-to-scf` | Flang's `DO WHILE` / `DO…EXIT` raw-CFG shape → `scf.while` + nested `scf.if`. |
| `sccp` → `canonicalize` → `cse` | Constant fold + simplify + dedupe, run after every HLFIR rewrite has exposed its constants. |

### 4. Build

Python `SDFGBuilder` walks the normalised HLFIR:

- `bridge/extract_vars.cpp` classifies variables as
  `array` / `symbol` / `scalar`. A variable is a symbol if it's a DO
  induction variable, an array shape extent, a DO bound (upper or
  lower), an `hlfir.designate` index, or feeds a control-flow
  condition. Writes to symbols become interstate-edge assignments;
  writes to scalars become tasklets.
- `bridge/extract_ast.cpp` produces a recursive `ASTNode` tree covering
  `loop` / `while` / `conditional` / `assign` / `copy` / `memset` /
  `libcall` / `reduce` / `break` / `return`.
- `builder/` emits the SDFG from the tree.

**Array descriptors** carry Fortran column-major strides and
per-dimension lower bounds. Accesses use Fortran-native indices in
memlet subsets and the caller-side `access.py::build_memlet_index`
applies the lbound offset when composing the subset. DaCe's own
`offset` field is left at zero — we keep a single descriptor-layer
convention so the rest of the pipeline doesn't need to reason about
two kinds of offset arithmetic.

**Assumed-shape alias re-basing.** After `hlfir-inline-all` splices
an assumed-shape callee into a caller with custom-bounded actuals,
Flang leaves a second `hlfir.declare` (no shape operand, memref via
`fir.convert` of the outer declare). The bridge detects the pattern
(`trace_utils::asAssumedShapeAlias`), skips it in
`extract_vars` (one SDFG descriptor per storage), walks through it
in `traceToDecl` (accesses resolve to the outer name), and rewrites
each index expression by `outer_lbound - inner_lbound` at
`extract_ast::buildDesignateIndexExpr`. The caller's lbound offset
then fires uniformly for every access, aliased or not.

**Exponentiation.** `math.fpowi` / `math.powf` / `math.powi` /
`math.ipowi` all lower to Python `**` in the tasklet. An SDFG-level
simplify pass chooses between a multiplication chain and a libm
`pow` call based on the tasklet's connector types. `hlfir.no_reassoc`
(the wrapper Flang emits to block FP reassociation) is a transparent
pass-through in `buildExpr`.

**Section reductions.** `ANY` / `ALL` / `SUM` / `PRODUCT` over a
section (`ANY(mask(lo:hi, jk))`) synthesise a loop-accumulator AST —
an init-to-identity assign followed by a `kind="loop"` whose body
accumulates via the right Python operator (`or` / `and` / `+` / `*`).
Whole-array reductions still go through DaCe's `standard.Reduce`.

At the end of `build()` the SDFG's `arglist()` + free symbols are
snapshotted into a `FrozenSignature` and pinned on the SDFG.

### 5. Regen interface

The binding emitter reads `FortranInterface` (outer surface — the
struct-visible dummies the caller wrote) AND `FrozenSignature` (inner
surface — what the SDFG actually expects), plus the `FlattenPlan`
attribute left by `hlfir-flatten-structs`, and emits
`<entry>_bindings.f90`:

- **Alias path** — same rank, same element type → `c_f_pointer(c_loc(st%u), st_u, shape)`. Zero-copy.
- **Deep-copy path** — element-type or rank mismatch (e.g.
  `complex(c_double)` → two `real(c_double)` arrays) → Fortran `do`
  loop that splits / repacks. Reverse loop only when intent is
  `out` / `inout`.
- **Symbol population** — SDFG free symbols (`nproma`, `nlev`, …)
  populated via Fortran intrinsics (`size(arg, dim=d)`, `lbound`).
- **Ref-counted init** — `init_count` counter + `c_null_ptr` handle
  so multiple callers share one DaCe state; finalise when the last
  caller signs off.

**Signature freezing.** `codegen.generate_code` inspects
`sdfg._frozen_signature` before emitting the C++ header. Any drift
from the snapshot raises `SignatureDriftError` — the contract is
compile-time, not SDFG-time, so transformations are free to mutate
the SDFG but cannot ship a header that disagrees with the emitted
binding.

## Components

```
dace/frontend/hlfir/
├── bridge/            # C++ — HLFIR parser + classifier + AST walker (nanobind)
│   ├── bridge.cpp             # MLIRContext owner, pass pipeline, Python exports
│   ├── extract_vars.cpp       # hlfir.declare → VarInfo (name, role, intent, …)
│   ├── extract_ast.cpp        # function body → recursive ASTNode tree
│   └── trace_utils.cpp        # traceToDecl, buildExpr, alias-declare helpers
├── passes/            # MLIR passes (C++)
│   ├── InlineAll.cpp
│   ├── FlattenStructs.cpp     # stamps hlfir.flatten_plan on the module
│   ├── PropagateShapes.cpp
│   ├── DefaultIntent.cpp
│   ├── VerifyNoUnresolvedCalls.cpp
│   └── Passes.cpp             # registerAllBridgePasses()
├── builder/           # SDFG emission (Python, stage 4)
│   ├── __init__.py            # SDFGBuilder + _emit dispatch
│   ├── context.py             # _Ctx (state, pending assigns, iter_map)
│   ├── descriptors.py         # add_descriptors, DTYPE, auto_declare_synth
│   ├── access.py              # build_memlet_index + indirect-symbol lifting
│   ├── emit_tasklet.py        # per-occurrence tasklet + emit_scalar_assign
│   ├── emit_cfg.py            # assign / loop / while / conditional
│   └── emit_library.py        # copy / memset / libcall / reduce / break / return
├── intrinsics/        # Fortran intrinsic registry
│   ├── elementwise.py         # sin, cos, exp, sqrt, abs, min, max, …
│   ├── reduction.py           # sum, product, minval, maxval, any, all, count
│   ├── linalg.py              # matmul, transpose, dot_product
│   └── direct.py              # SIZE, LBOUND, …
├── bindings/          # Fortran wrapper emitter (Python, stage 5)
│   ├── frozen_signature.py    # FrozenArg + FrozenSignature + drift check
│   ├── fortran_interface.py   # OriginalInterface (outer surface)
│   ├── flatten_plan.py        # FlattenPlan + to_dict / from_dict
│   ├── block_builders.py      # per-Fortran-section emitters
│   ├── loop_copy.py           # alias vs deep-copy renderers
│   ├── emit_bindings.py       # top-level → <entry>_bindings.f90
│   └── templates/*.f90.in
├── hlfir_to_sdfg.py   # compat shim — re-exports from builder/
└── fortran_parser.py  # top-level: generate_sdfg(entry=..., hlfir_files=[...])
```

## Entry point

```python
from dace.frontend.hlfir.fortran_parser import generate_sdfg

sdfg, bindings_f90, frozen_sig_json = generate_sdfg(
    entry="compute_tendencies",
    hlfir_files=["kernel.hlfir", "math_utils.hlfir"],
    out_dir="build/",
)
```

Returns the validated `SDFG`, the path to the generated Fortran
binding, and the path to the frozen-signature snapshot.

For quick experiments the single-file form still works and skips
binding emission:

```python
sdfg = generate_sdfg("code.hlfir")                         # default pipeline
sdfg = generate_sdfg("code.hlfir", pipeline="hlfir-propagate-shapes")
```

## Supported Fortran constructs

| Fortran | HLFIR | DaCe |
| --- | --- | --- |
| `DO i = lo, hi` | `fir.do_loop` | `LoopRegion(loop_var, init, update, cond)` |
| `DO WHILE (…)` | `scf.while` (post `lift-cf-to-scf`) | `LoopRegion(condition_expr)` |
| `IF / ELSE` | `fir.if` / `scf.if` | `ConditionalBlock` + `ControlFlowRegion` per branch |
| `SELECT CASE` | `fir.select_case` | nested `ConditionalBlock` chain |
| `EXIT` / `RETURN` | `scf.while` break-yield / `cf.br` to exit | `BreakBlock` / `ReturnBlock` |
| `b = a` (whole array) | `hlfir.assign` array→array | `CopyLibraryNode` |
| `c = 0.0` (zero fill) | `hlfir.assign` zero→array | `MemsetLibraryNode` |
| `res(a:b) = 42` (section assign, scalar RHS) | `hlfir.assign` → section `hlfir.designate` | nested `LoopRegion` + indexed tasklet |
| Elementwise intrinsics (`sin`, `cos`, …) | `hlfir.elemental` + `hlfir.apply` | nested `LoopRegion` + Python tasklet |
| `a ** b` (integer/float exponent) | `math.fpowi` / `math.powf` / `math.ipowi` | tasklet `(a ** b)`; downstream simplify pass picks the lowering |
| `sum` / `product` / `minval` / `maxval` (whole-array) | `hlfir.sum` / … | `state.add_reduce(wcr, axes, identity)` |
| `ANY` / `ALL` / `SUM` / `PRODUCT` over section | `hlfir.any` / `hlfir.all` / … on `hlfir.designate` with triplets | loop-accumulator AST |
| `matmul` / `transpose` / `dot_product` | `hlfir.matmul` / … | `blas.MatMul` / `standard.Transpose` / `blas.Dot` |
| Indirect `a(idx(j))` | chained `hlfir.designate` | interstate-edge symbol + per-occurrence connector |
| AoS structs | `fir.type` + `fir.coordinate_of` | flattened by `hlfir-flatten-structs` before walker sees them |
| Assumed-shape callee inlined into a caller with custom lbounds | second `hlfir.declare` aliasing the outer box | bridge skips the alias, rewrites indices by `outer_lbound - inner_lbound` |

## Not yet supported

- `OPTIONAL` dummy arguments (`fir.is_present` → SDFG conditional + nullable descriptor).
- User-defined `ELEMENTAL` procedures (per-element synthesis around an `hlfir.elemental` body that's a scalar call).
- Literal integer indices in the middle dim of a 3D access (`vn(je, 1, jb)`) — `buildIndexExpr` / `resolveIndex` leak `?` into memlet subsets.
- 3D integer indirection tables (`icidx(je, jb, 1..2)`) on the read side — memlet subset resolution gap.
- 4D output arrays (`ddt_vn_apc_pc(…, ntnd)`) — extra rank in the write access path.
- GPU target bindings (OpenACC shim emission).

## Tests

Every supported construct has a seeded numerical test against
gfortran + f2py in `tests/hlfir/`. Binding-specific tests live in
`tests/hlfir/bindings/`. The six E6 velocity-advection representative
loopnests have their own SDFG-vs-f2py comparisons in
`tests/hlfir/velocity_loopnests/`.

### Dumping built SDFGs

```bash
# dump to /tmp/hlfir_test_sdfgs/<subroutine>.sdfg
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/

# custom dir
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/hlfir/
```

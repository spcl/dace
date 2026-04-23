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

Each stage consumes what the previous stage produces and assumes all
its predecessors' invariants hold. The Python walker only has to
understand one narrow IR shape because every stage ahead of it has
normalised away the irregularities.

### 1. Parse

One or more `.hlfir` files produced by the user's build
(`flang-new-21 -fc1 -emit-hlfir`) are loaded into a shared
`MLIRContext` and merged into a single `ModuleOp` via
`SymbolTable::insert`. A **pre-pass snapshot** of `hlfir.declare`
captures the original Fortran-visible interface — dummy names,
intents, derived-type layouts — before any rewrite runs. This
snapshot (`FortranInterface`) is what the binding emitter later uses
to reconstruct the caller-facing signature.

Bridge entry points:
- `HLFIRModule.parse_files(paths)` — multi-file load.
- `HLFIRModule.get_fortran_interface(entry)` — snapshot the pre-pass
  outer surface.

### 2. Inline + link

Two passes around our existing inliner, plus upstream DCE:

| Pass | One-liner |
| --- | --- |
| `hlfir-inline-all` | Bottom-up whole-program inlining: every `fir.call` with a body becomes inlined, until only externals / intrinsics remain. |
| `verify-no-unresolved-calls` | Walks remaining `fir.call` ops; raises `UnresolvedCallError` on any non-whitelisted target. Allowed: `_Fortran*` (Flang runtime), `sin`/`cos`/`sqrt`/... (libm), `malloc`/`free`/`memcpy`/`memset`. |
| `--symbol-dce` | Strips every `func.func` except the pinned entry. |

### 3. Preprocess

MLIR passes in strict order — each depends on its predecessors
having normalised the IR:

| Pass | Purpose |
| --- | --- |
| `hlfir-flatten-structs` | AoS → SoA on derived types + ELLPACK jagged variants. Struct dummies become per-member flat args; `fir.coordinate_of` chains get rewritten onto the new dummies. |
| `hlfir-propagate-shapes` | Assumed-shape `(:,:)` dummies pick up real Fortran symbol names through call-graph shape propagation. |
| `hlfir-default-intent` | Intent-less dummies get `intent_inout` so the classifier sees uniform attributes. |
| `lift-cf-to-scf` | Flang's `DO WHILE` / `DO+EXIT` raw-CFG shape → `scf.while` with nested `scf.if`. |
| `sccp` → `canonicalize` → `cse` | Constant fold + simplify + dedupe, run AFTER every HLFIR rewrite has exposed its constants (inlining reveals parameter constants, flatten-structs exposes scalar member loads, lift-cf-to-scf makes IV bounds visible). |

### 4. Build

Python `SDFGBuilder` walks the normalised HLFIR:

- `bridge/extract_vars.cpp` — classify variables as
  `array` / `symbol` / `scalar`. A variable is a **symbol** if it's
  a Fortran DO induction variable, an array shape extent, a DO upper
  bound, an `hlfir.designate` index, or reads into a control-flow
  condition (scf.if / fir.if / scf.while). Everything else scalar is
  `scalar` (pure value, no state-change on write). Only symbols can
  appear as array indices — scalars can't — so classification drives
  whether a write becomes an interstate-edge assignment or a tasklet.
- `bridge/extract_ast.cpp` — recursive `ASTNode` tree covering
  `loop` / `while` / `conditional` / `assign` / `copy` / `memset` /
  `libcall` / `reduce` / `break` / `return`.
- `builder/` package — per-concern SDFG emission (see **Components**).
- End of `build()` — snapshot `SDFG.arglist()` + free symbols into a
  `FrozenSignature` and pin it on the SDFG.

### 5. Regen interface

The binding emitter reads `FortranInterface` (outer surface — the
struct-visible dummies the caller wrote) AND `FrozenSignature` (inner
surface — what the SDFG actually expects), decides per-member whether
to alias or copy, and writes `<entry>_bindings.f90`:

- **Alias path** — same rank + same element type → `call
  c_f_pointer(c_loc(st%u), st_u_flat, shape)`. Zero-copy.
- **Deep-copy path** — element-type or rank mismatch (e.g.
  `complex(c_double)` → two `real(c_double)` arrays) → Fortran `do`
  loop that splits / repacks. Reverse loop only when intent is `out`
  / `inout`.
- **Symbol population** — SDFG free symbols (`nproma`, `nlev`, …) set
  via Fortran intrinsics (`size(arg, dim=d)`, `lbound`).
- **Ref-counted init** — `init_count` counter + `c_null_ptr` handle,
  so multiple callers share one DaCe state object; finalize when the
  last caller signs off.

**Signature freezing.** `codegen.generate_code` inspects
`sdfg._frozen_signature` before emitting the C++ header. Any drift
from the snapshot raises `SignatureDriftError` — the contract is
compile-time, not SDFG-time, so transformations are free to mutate
the SDFG but can't ship a header that disagrees with the emitted
binding.

## Components

```
dace/frontend/hlfir/
├── bridge/           # C++ — HLFIR parser + classifier + AST walker (nanobind)
│   ├── bridge.cpp            # MLIRContext owner, pass pipeline, Python exports
│   ├── extract_vars.cpp      # hlfir.declare → VarInfo (name, role, intent, …)
│   ├── extract_ast.cpp       # function body → recursive ASTNode tree
│   └── trace_utils.cpp       # shared: traceToDecl, buildExpr, buildBoolExpr
├── passes/           # MLIR passes (C++)
│   ├── InlineAll.cpp
│   ├── FlattenStructs.cpp
│   ├── PropagateShapes.cpp
│   ├── DefaultIntent.cpp
│   ├── VerifyNoUnresolvedCalls.cpp
│   └── Passes.cpp            # single registerAllBridgePasses() entry
├── builder/          # SDFG emission (Python, stage 4)
│   ├── __init__.py           # SDFGBuilder class body + _emit dispatch
│   ├── context.py            # _Ctx (state + pending + iter_map)
│   ├── descriptors.py        # add_descriptors, DTYPE, auto_declare_synth
│   ├── access.py             # acc, build_memlet_index, indirect-symbol lifting
│   ├── emit_tasklet.py       # per-occurrence tasklet + emit_scalar_assign
│   ├── emit_cfg.py           # assign / loop / while / conditional
│   └── emit_library.py       # copy / memset / libcall / reduce / break / return
├── intrinsics/       # Fortran intrinsic registry
│   ├── elementwise.py        # sin, cos, exp, sqrt, abs, min, max, …
│   ├── reduction.py          # sum, product, minval, maxval
│   ├── linalg.py             # matmul, transpose, dot_product
│   └── direct.py             # SIZE, LBOUND, … (Phase 4 stub)
├── bindings/         # Fortran wrapper emitter (Python, stage 5)
│   ├── __init__.py
│   ├── frozen_signature.py   # FrozenArg + FrozenSignature + JSON I/O + drift check
│   ├── fortran_interface.py  # OriginalInterface (outer surface) dataclasses
│   ├── layout_match.py       # per-arg alias vs deep-copy strategy
│   ├── emit_bindings.py      # string-template emitter → <entry>_bindings.f90
│   └── templates/*.f90.in
├── hlfir_to_sdfg.py  # compat shim — re-exports from builder/
└── fortran_parser.py # top-level: generate_sdfg(entry=..., hlfir_files=[...])
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
binding (`<entry>_bindings.f90`), and the path to the frozen-signature
snapshot (`<entry>.sig.json`).

For quick experiments the single-file signature still works and skips
binding emission:

```python
sdfg = generate_sdfg("code.hlfir")                         # default pipeline
sdfg = generate_sdfg("code.hlfir", pipeline="")            # no passes
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
| `sin` / `cos` / … (elementwise) | `hlfir.elemental` + `hlfir.apply` | nested `LoopRegion` + Python tasklet (bare names resolved via `_ALLOWED_MODULES` to `dace::math::*`) |
| `sum` / `product` / `minval` / `maxval` | `hlfir.sum` / … | `state.add_reduce(wcr, axes, identity)` → `standard.Reduce` |
| `matmul` / `transpose` / `dot_product` | `hlfir.matmul` / … | `blas.MatMul` / `standard.Transpose` / `blas.Dot` |
| Indirect `a(idx(j))` | `hlfir.designate` chained | interstate-edge symbol + per-occurrence connector |
| AoS structs | `fir.type` + `fir.coordinate_of` | flattened by `hlfir-flatten-structs` before walker sees them |

## Not yet supported

- Array-section assigns `a(1:n) = …` as explicit DaCe Maps.
- `OPTIONAL` dummy arguments / user-defined `ELEMENTAL` procedures.
- Mask / dim reductions (`count`, `all`, `any`, `sum(a, dim=2)`).
- GPU target bindings (OpenACC shim emission).

## Tests

Every supported construct has a seeded numerical test against
gfortran + f2py in `tests/hlfir/`. Binding-specific tests live in
`tests/hlfir/bindings/`.

### Dumping built SDFGs

```bash
# dump to /tmp/hlfir_test_sdfgs/<subroutine>.sdfg
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/

# custom dir
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/hlfir/

# standalone walker over every .f90 fixture
python3 tools/dump_hlfir_sdfgs.py [output_dir]
```

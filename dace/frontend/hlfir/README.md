# HLFIR → DaCe frontend

Fortran source → Flang HLFIR (MLIR) → DaCe SDFG, without going through
Fortran AST parsing in Python. The frontend leans on Flang to do all the
parsing / semantic work and only walks the already-elaborated HLFIR to
produce SDFG nodes.

## Aim

Replace the legacy pure-Python Fortran frontend (`dace/frontend/fortran/`)
with a thinner pipeline that lets Flang's own optimisation and
structurisation passes do the heavy lifting. Specifically:

- Parse once, outside Python, via `flang-new-21 -emit-hlfir`.
- Normalise the IR with a small pipeline of MLIR/HLFIR passes (see below)
  so the Python walker only has to handle a narrow, predictable shape.
- Emit DaCe library nodes directly (`MatMul`, `Dot`, `Transpose`,
  `Reduce`, `CopyLibraryNode`, `MemsetLibraryNode`) for every Fortran
  intrinsic that has a library-node equivalent, rather than lowering
  through tasklet-and-loop and pattern-matching later.
- Keep the SDFG structure shape-preserving: every Fortran `DO` becomes a
  `LoopRegion` with the exact Fortran bounds, every `IF` becomes a
  branching pair of interstate edges, every intrinsic a single node.

## Components

| File / dir | Role |
| --- | --- |
| `bridge/bridge.cpp` | nanobind shim — owns an `MLIRContext`, parses HLFIR, runs the pass pipeline, exposes `HLFIRModule.get_variables()` / `get_ast()` / `dump()` to Python. |
| `bridge/extract_vars.cpp` | Walks `hlfir.declare` ops → `VarInfo` (fortran_name, rank, dtype, intent, shape_symbols, role). |
| `bridge/extract_ast.cpp` | Walks the subroutine body → recursive `ASTNode` tree (loop / assign / while / conditional / call / reduce / copy / memset / libcall). |
| `bridge/trace_utils.cpp` | Shared helpers: `traceToDecl`, `buildExpr`, `buildIndexExpr`, `buildBoolExpr`. |
| `passes/` | Custom MLIR passes registered via `registerAllBridgePasses()`: `hlfir-inline-all`, `hlfir-flatten-structs`, `hlfir-propagate-shapes`, `hlfir-default-intent`, `lift-cf-to-scf`. |
| `hlfir_to_sdfg.py` | `SDFGBuilder` — the Python side. Parses + runs the bridge pipeline, classifies variables, walks the AST, emits DaCe constructs (`LoopRegion`, `SDFGState`, tasklets, library nodes). |
| `intrinsics/` | Per-family Fortran-intrinsic registry. See below. |
| `build_bridge.py` | Loads the compiled `hlfir_bridge.so`. |
| `fortran_parser.py` | Top-level entry `generate_sdfg(hlfir_path)`. |

## Default pass pipeline

Run in order before `get_ast()` / `get_variables()`:

| Pass | What it does |
| --- | --- |
| `hlfir-inline-all` | Inlines every `fir.call` whose callee is in the same module. Only local callees — cross-TU inlining is not attempted. |
| `hlfir-flatten-structs` | AoS→SoA on derived types and jagged ELLPACK variants, exposes scalar member loads for later constant folding. |
| `hlfir-propagate-shapes` | Fills in assumed-shape `(:,:)` dummies with real Fortran symbol names so the SDFG can carry symbolic dimensions. |
| `hlfir-default-intent` | Assigns `intent(inout)` to dummies that declared no intent. |
| `lift-cf-to-scf` | Rewrites `cf.br` / `cf.cond_br` irreducible loops (Flang's `DO WHILE` shape) into `scf.while` so `extract_ast` can walk them. |
| `sccp`, `canonicalize`, `cse` | Constant propagation + fold + CSE *after* every HLFIR rewrite has exposed the constants it will expose. |

Override per call:

```python
generate_sdfg("code.hlfir",
              pipeline="hlfir-propagate-shapes")          # one pass only
generate_sdfg("code.hlfir", pipeline="")                   # skip passes
```

## Supported Fortran constructs

| Fortran | HLFIR | DaCe emission |
| --- | --- | --- |
| `DO i = lo, hi` | `fir.do_loop` | `LoopRegion(condition_expr, loop_var, init, update)` |
| `DO WHILE (…)` | `scf.while` (after `lift-cf-to-scf`) | `LoopRegion(condition_expr)` |
| `IF (…) / ELSE` | `fir.if` / `scf.if` | two branch states + interstate edges on `cond` / `not cond` |
| `b = a` (whole array) | `hlfir.assign` (array→array) | `standard.CopyLibraryNode` |
| `c = 0.0d0` (zero fill) | `hlfir.assign` (zero→array) | `standard.MemsetLibraryNode` |
| Elementwise intrinsics (`sin`/`cos`/`tan`/`sqrt`/`exp`/`log`/`abs`/`floor`/`min`/`max`/…) | `hlfir.elemental` + `hlfir.apply` | nested `LoopRegion`s + Python tasklet with bare names, resolved via DaCe `_ALLOWED_MODULES` to `dace::math::*` |
| `sum` / `product` / `minval` / `maxval` | `hlfir.sum` / `.product` / `.minval` / `.maxval` | `state.add_reduce(wcr, axes, identity)` → `standard.Reduce` |
| `matmul(a, b)` | `hlfir.matmul` | `blas.MatMul` (specializes to GEMM / GEMV internally) |
| `transpose(a)` | `hlfir.transpose` | `standard.Transpose` |
| `dot_product(u, v)` | `hlfir.dot_product` | `blas.Dot` |
| Indirect `a(idx(j,1))` | `hlfir.designate` chained through a load | Interstate-edge symbol assignment + tasklet with per-occurrence connectors |
| AoS struct flattening | `fir.coordinate_of` | `hlfir-flatten-structs` pass transforms struct-of-arrays before the walker sees it |

## Intrinsics registry (`intrinsics/`)

Flat package exposing `is_elementwise`, `is_reduction`, `is_libnode`,
`is_intrinsic`, `render_call`, `reduction_spec`, `libnode_spec`. Each
family lives in its own subpath so adding a new intrinsic means editing
one file:

```
intrinsics/
├── elementwise.py              # sin, cos, tan, exp, log, sqrt, abs, min, max, …
├── reductions/
│   └── scalar_reductions.py    # sum, product, minval, maxval
├── linalg/
│   ├── matmul.py
│   ├── transpose.py
│   └── dot_product.py
└── direct/                     # Phase 4 — SIZE, LBOUND, UBOUND, PRESENT, ALLOCATED (stubs)
```

## Not yet supported

- SELECT CASE (will lower to chained `if/elif/else`).
- Array-section assignment `a(1:n) = …` as an explicit construct (will lower to a DaCe Map).
- CYCLE / EXIT in non-trivial placements where Flang can't structurise.
- OPTIONAL dummy arguments, user-defined ELEMENTAL procedures.
- Scalar-target assigns that read array elements (`s = d(2,1) + 1.0`) —
  `emit_scalar_assign` doesn't wire array-element reads yet.
- Mask / dim reductions: `count`, `all`, `any`, `sum(a, dim=2)`.

## Entry points

```python
from dace.frontend.hlfir.fortran_parser import generate_sdfg

sdfg = generate_sdfg("code.hlfir")           # validates, returns dace.SDFG
```

or directly via `SDFGBuilder`:

```python
from dace.frontend.hlfir.hlfir_to_sdfg import SDFGBuilder, DEFAULT_PIPELINE

sdfg = SDFGBuilder("code.hlfir").build()
```

## Tests

Every supported construct has a seeded numerical test against
gfortran + f2py in `tests/hlfir/`. Ad-hoc `/tmp/foo.f90` probes are
**not** the right pattern — save every shape worth exploring as a
permanent `tests/hlfir/<feature>.f90` + `_test.py` pair.

### Dumping built SDFGs for inspection

Set `__DACE_HLFIR_GEN_TEST_SDFGS` before running the suite and every
test that calls `_util.build_sdfg(...).build()` will also save its
constructed SDFG for offline inspection:

```bash
# dump to /tmp/hlfir_test_sdfgs/<subroutine>.sdfg
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/

# dump to a custom directory
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/my_sdfgs python3 -m pytest tests/hlfir/
```

A standalone dumper that walks every `tests/hlfir/*.f90` fixture
without running the tests is also available:

```bash
python3 tools/dump_hlfir_sdfgs.py [output_dir]
```

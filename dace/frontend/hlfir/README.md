# HLFIR -> DaCe frontend

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
.f90 -- flang -fc1 -emit-hlfir --> .hlfir file(s)
                                       |
                            (1) parse + snapshot interface
                                       |
                        (2) inline + verify + DCE
                                       |
                       (3) normalise (HLFIR rewrites)
                                       |
                            (4) build SDFG
                                       |
                          (5) regen binding
                                       v
              <entry>_bindings.f90  +  compiled SDFG .so
```

## Stages

**(0) Pre-process source.** `dace.frontend.hlfir.preprocess` holds
three text rewrites that must run before flang (they change what
flang accepts, or what arithmetic each backend is free to pick).
They are SED-style regex transforms, not a Fortran parser, so each
is deliberately narrow; comment- and string-awareness is shared via
`_scan_line` so a `!` or `**` inside a character literal is never
touched.

- `rewrite_integer_powers` -- expands an integer-valued REAL-literal
  power (`x**2.0` -> `(x*x)`, `(p-q)**3.0` -> `((p-q)*(p-q)*(p-q))`).
  Runs **unconditionally** in `compile_to_hlfir`: algebraically exact
  and removes a backend-dependent `pow(x, 2.0)` vs `x*x` rounding
  difference against the gfortran reference. A bare-integer exponent
  (`x**2`) is left for flang's own (bit-identical) integer-power
  lowering; genuine fractional powers (`**0.5`) stay as `pow()`; a
  base containing a call/array reference is left alone (duplicating it
  would invoke it twice).
- `promote_real_literals_to_double` -- single/default REAL literals to
  an explicit double form (`2.0` -> `2.0D0`). A standalone utility
  applied directly to kernel source on disk when a codebase must be
  globally double; **not** wired into the build path.
- `preprocess_fortran` -- `IF (intvar)` -> `IF (intvar /= 0)` for
  INTEGER scalars. flang-new-21 rejects bare INTEGER as an IF
  condition (only LOGICAL is legal); ECRAD / CloudSC (`IF
  (laericeauto)`) / ICON scaffolding ship this. **Opt-in** per call
  site (`compile_to_hlfir(..., preprocess=True)`); off by default so
  we don't paper over real issues in clean source.

**(1) Parse.** One or more `.hlfir` files are loaded into a shared
`MLIRContext` and merged into a single `ModuleOp` via
`SymbolTable::insert`. A pre-pass snapshot of the caller-visible
`hlfir.declare`s  --  dummy names, intents, derived-type layouts  --  is
captured as `FortranInterface`. Entry points: `HLFIRModule.parse_files`,
`HLFIRModule.get_fortran_interface`.

**(2) Inline + link.** `hlfir-inline-all` flattens the call tree
into the pinned entry. `symbol-dce` strips the dead siblings. The
multi-file pipeline additionally runs `hlfir-verify-no-unresolved-calls`
to fail if any `fir.call` survives outside the Flang-runtime / libm /
C-stdlib allowlist. Inlining requires the `DialectInlinerInterface`
extensions for `fir` / `func` / `LLVM` to be attached to the context;
the bridge constructor does this once.

**(3) Normalise.** HLFIR rewrites in strict order  --  each depends on
its predecessors having reshaped the IR.  This is the actual
`DEFAULT_PIPELINE` order (see `builder/__init__.py`); stage (2)'s
`hlfir-inline-all` is the second entry in the same pipeline, listed
separately above only to keep the conceptual flow readable:

| Pass | Purpose |
| --- | --- |
| `lower-fir-select-case` | Lower `fir.select_case` to `cf.cond_br` BEFORE inlining (the inliner's block-operand remap segfaults on a callee containing a select-case) |
| `hlfir-inline-all` | Splice every callee body into the pinned entry |
| `hlfir-fold-element-aliases` | Erase element-scoped alias declares left by inlined elemental / scalar-arg procedures |
| `hlfir-expand-vector-subscript-gather` | Replace `hlfir.associate` of an `hlfir.elemental` (Flang's gather temp for noncontiguous slice arguments) with an explicit `fir.alloca` + gather DO loop |
| `hlfir-expand-vector-subscript-scatter` | Replace `hlfir.region_assign` with an `hlfir.elemental_addr` destination (vector-subscripted scatter `d(cols) = source`) by an explicit DO loop |
| `symbol-dce` | Drop private callee bodies once `hlfir-inline-all` has folded them in |
| `fir-polymorphic-op` | Statically devirtualise resolvable `fir.dispatch` / `fir.select_type`; lowers the rest to an indirect-call shape that the next pass catches |
| `hlfir-reject-polymorphism` | Loud-fail on any surviving `fir.dispatch`, `fir.select_type`, or `fir.box_tdesc` (residual indirect dispatch from `fir-polymorphic-op`)  --  the bridge supports CLASS-as-monomorphic-box only |
| `hlfir-flatten-structs` | AoS -> SoA; emits `hlfir.flatten_plan` module attribute. Peels `fir.class<T>` via `BaseBoxType` so monomorphic CLASS receivers flatten through the same path as TYPE |
| `hlfir-propagate-shapes` | Assumed-shape dummies acquire real extent symbols |
| `hlfir-default-intent` | Intent-less dummies default to `intent_inout` |
| `lift-cf-to-scf` | Raw-CFG loops (`DO WHILE`, `DO...EXIT`) -> `scf.while` + `scf.if` |
| `sccp` -> `canonicalize` -> `cse` | Fold + simplify + dedupe after HLFIR exposed its constants |

**(4) Build SDFG.** `bridge/extract_vars.cpp` classifies every
`hlfir.declare` as `array`, `symbol`, or `scalar` (rules below).
`bridge/extract_ast.cpp` is the dispatcher; the AST extraction itself
is split into focused translation units under `bridge/ast/`  --
`expressions.cpp` (RHS rendering), `assigns.cpp` (assignment-shape
builders), `elementals.cpp` (reductions + libcall + select-case),
`control_flow.cpp` (cmp / boolean / scf-while / merge), `dispatch.cpp`
(top-level walker).  Cross-TU API + thread-local state lives in
`ast/ast_helpers.h`; internal cross-TU helpers in `ast/ast_internal.h`.
The walker produces a recursive `ASTNode` tree covering `loop` /
`while` / `conditional` / `assign` / `copy` / `memset` / `libcall` /
`reduce` / `break` / `return`.

**Loop bounds + IF conditions are hoisted to symbols.** Every
non-trivial loop bound (anything beyond a bare identifier or integer
literal) and every non-trivial IF condition is materialised as an SDFG
symbol on a state-change before the block. Names follow a global
counter: `loopbegin_<N>` / `loopend_<N>` for loop bounds and
`if_cond_<N>` for branch guards. The `LoopRegion` / `ConditionalBlock`
itself then references **only** the symbol  --  no expression rewriting
in the bound or condition. This:
  * keeps the bridge's emitters small (no iter-rename plumbing in
    bound expressions, no ad-hoc `[0]` subscripting in IF conditions);
  * funnels indirect-array reads through the existing symbol-staging
    machinery (a bound containing `row_ptr[i+1] - 1` becomes one
    interstate-edge assignment that the C++ codegen renders correctly
    via the array-aware sympy printer);
  * gives the SSA loop-iter pass a uniform input shape.

`builder/SDFGBuilder` emits the SDFG from that tree, then runs the
post-generation cleanup pipeline below, then snapshots
`sdfg.arglist()` + free symbols into a `FrozenSignature` and pins
it on the SDFG.

**(4b) Post-generation cleanup.** Two passes run over the freshly-
built SDFG, in order:

1. **`SSALoopIterators`** (`dace.transformation.passes.ssa_loop_iterators`).
   Renames every `LoopRegion.loop_variable` to a globally-unique
   `_it_<N>` symbol and propagates the rename through the body
   (memlet subsets, tasklet bodies, interstate-edge assignments,
   nested SDFG symbol mappings).  Adds a reconstruction state after
   each loop that re-asserts `<original_var> = <loop_end>` so
   downstream code reading the un-renamed name sees the correct
   post-loop value.  Skips while-shape loops (no induction variable).
   Renders the reconstruction RHS via `dace.symbolic.symstr(arrayexprs=...)`
   so an array-subscripted bound like `row_ptr[i+1] - 1` renders with
   `[]` (not `()`, which sympy would print and the C++ codegen would
   reject).  The bridge consequently emits each `LoopRegion` using the
   source-Fortran iter name (`jk`, `je`, ...) verbatim and lets this
   pass handle the uniquification  --  no `iter_map` plumbing in the
   emitters.

2. **`replace_length_one_arrays_with_scalars`**
   (`dace.sdfg.construction_utils`).  Walks every length-1 ``Array``
   on the SDFG and rewrites the descriptor to a true ``Scalar``,
   stripping leftover ``[0]`` subscripts from interstate-edge
   assignments, conditional-block guards, and loop-region condition
   expressions.  Runs with **`transient_only=True`** at the top
   level: only LOCAL 1-element transients (loop accumulators left as
   length-1 arrays) get folded.  Signature scalars follow the bridge's
   I/O convention  --  `intent(in)` / `VALUE` are emitted directly as
   `Scalar` by `descriptors.py`, while `intent(out)` / `intent(inout)`
   stay as length-1 ``Array`` so callers can pass a numpy 1-element
   buffer to receive the value.  Recurses into nested SDFGs (their
   transient-only sub-cleanup follows the same rule).

The pipeline runs **before** the `FrozenSignature` snapshot is taken
so the bindings emitter sees the post-cleanup signature.

**Loop iterator validation.** SDFG validation rejects writing to a
`LoopRegion.loop_variable` from an interstate-edge assignment inside
its own region.  The `LoopRegion` already owns the iterator update
via `init_expr` / `update_expr`; mutating it elsewhere races with
that machinery and breaks the SSA invariant the iter pass relies on.

**(5) Regen binding.** `bindings/emit_bindings` reads three artefacts
 --  `FortranInterface` (outer caller surface), `FrozenSignature`
(inner SDFG surface), and the `FlattenPlan` from stage 3  --  and writes
`<entry>_bindings.f90`: a ref-counted Fortran module that preserves
the caller's original signature, populates SDFG symbols via
`size` / `lbound`, and for each struct member picks between a zero-copy
`c_f_pointer` alias and a Fortran `do`-loop deep copy based on the
recipe.

## Data artefacts

These are the structured records that flow between stages. They are
the frontend's stable contract surface  --  new features extend them,
they do not invent parallel channels.

| Artefact | Produced at | Consumed at | Role |
| --- | --- | --- | --- |
| `FortranInterface` | (1) snapshot | (5) emit | Caller-facing dummy list + derived-type layouts |
| `FlattenPlan` (MLIR attr) | (3) `hlfir-flatten-structs` | (5) emit | Per-dummy AoS->SoA unpack recipe (`flat_names`, `read_exprs`, `shape_exprs`, `aliasable`, `aos_alloc`+`cap_symbol` for padding-to-max) |
| `VarInfo[]` | (4) `extract_vars` | (4) `SDFGBuilder` | Classification + shape + intent per variable |
| `ASTNode` tree | (4) `extract_ast` | (4) `SDFGBuilder` | Normalised CFG + assigns + library-op references |
| `FrozenSignature` | (4) end of `build()` | codegen, (5) emit | SDFG arglist snapshot  --  drift check at codegen time |

## Mechanisms

**Symbol vs scalar classification.** A Fortran integer is a *symbol*
iff it's a DO induction variable, an array shape extent, a DO bound
(upper or lower), an `hlfir.designate` index, or feeds a control-flow
condition. Everything else integer is a *scalar*. Writes to symbols
become interstate-edge assignments; writes to scalars become
tasklets. Only symbols can appear as array indices  --  so classification
drives whether a write changes the state machine or mutates data in
place.

**Fortran lbound handling.** Every array descriptor carries
`shape_symbols` and `lower_bounds`. Memlet subsets use Fortran-native
indices; `access.py::build_memlet_index` folds the lbound offset
once at subset-build time. DaCe's own descriptor `offset` field stays
at zero  --  keeping a single-site offset arithmetic so downstream
transformations don't have to reason about two conventions.

**Assumed-shape alias re-basing.** After `hlfir-inline-all` splices
an assumed-shape callee (`arr(:)`) into a caller whose actual has
custom bounds (`x(-2:2)`), Flang emits a second `hlfir.declare`  --
no shape operand, memref via `fir.convert` of the outer declare.
The bridge detects this (`trace_utils::asAssumedShapeAlias`), skips
it in `extract_vars` (one SDFG descriptor per storage), walks
through it in `traceToDecl` (accesses resolve to the outer name),
and rewrites each access's index by `outer_lbound - inner_lbound` at
`buildDesignateIndexExpr`. The caller-side lbound fold then fires
uniformly.

**Exponentiation.** `math.fpowi` / `math.powf` / `math.powi` /
`math.ipowi` all surface as Python `(a ** b)` in the tasklet. A
downstream SDFG simplify pass picks the concrete lowering from the
tasklet's connector types. `hlfir.no_reassoc` (Flang's FP-reassoc
barrier) is transparent in `buildExpr`.

**Section reductions.** Whole-array `SUM` / `PRODUCT` / `ANY` / `ALL`
lower to DaCe's `standard.Reduce`. Section reductions
(`ANY(mask(lo:hi, jk))`) synthesise a loop-accumulator AST  --  an
init-to-identity assign plus a `kind="loop"` whose body accumulates
via the appropriate Python operator. DaCe's Reduce can't express a
dynamic-section input directly.

**ELEMENTAL procedures -> loop + scalar tasklet.** A Fortran
`elemental` procedure is a scalar body that, when called on array
actuals, Flang lowers as `fir.do_loop { hlfir.designate per-arg;
fir.call scalar_body }` for the subroutine form, or as
`hlfir.elemental { hlfir.designate; fir.call; yield_element }` for
the function form. `hlfir-inline-all` splices the callee's body in;
the per-element `hlfir.declare`s that remain (callee dummies named
after outer array elements) are semantic no-ops that carry the
callee's Fortran names into the body. `hlfir-fold-element-aliases`
erases those  --  whatever reads them rewires to the outer array's
`hlfir.designate`, so the SDFG builder sees the same shape as a
hand-written per-element loop. `fir.do_loop` block args with no
store-to-alloca sibling (produced by this fold) get a synthetic
iter name (`_doit_N`) pushed onto `indexStack()` so `resolveIndex`
still resolves raw block-arg uses inside the body.

**Sibling-assign RAW hazards in loop bodies.** When a `fir.do_loop`
body contains multiple assigns (`f = c*c; a = a - b*f; ...`) into
`hlfir.declare`-backed storage, the naive "one tasklet per assign,
all in one body state" wiring races: non-transient access nodes
share the underlying SDFG array, so a second tasklet's write can
clobber a first tasklet's read even with distinct access nodes. The
loop emitter detects any read-write name overlap across siblings
and serialises them into a chain of states (one tasklet per state,
interstate edges between). Siblings with no hazard still share a
single state  --  the check is per-loop-body, not a blanket pessimisation.

**OPTIONAL dummies -> companion present-flag.** Fortran `OPTIONAL`
args compile to `hlfir.declare` with `fortran_attrs = optional`, and
`present(x)` lowers to `fir.is_present %x : (!fir.ref<T>) -> i1`.
The bridge's `buildBoolExpr` renders that op as the identifier
`<name>_present`, and `extract_vars` registers a corresponding
`int32` symbol `VarInfo` right after each optional declare. The
flag lands on the SDFG signature alongside its host (`sdfg(a=...,
a_present=...)`); non-zero = present, zero = absent. The existing
if/else lowering reads the flag exactly like any other scalar
condition  --  no new AST kind. Intent-less optionals default to
`intent_in` so `descriptors.py` doesn't misclassify them as
transients. Correctness relies on Fortran's guarantee that every
non-`present()` use of an absent optional is dominated by a
`present` check; the SDFG simply threads that check through.

**AoS -> SoA flattening.** `hlfir-flatten-structs` is the bridge's
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
intact  --  the binding module is the AoS<->SoA boundary, the SDFG only
ever sees SoA.

**AoS + allocatable members  --  padding-to-max contract (Phase 5c).**
A struct with allocatable / pointer array members on an AoS dummy
(`type t :: real, allocatable :: w(:); type(t) :: A(N)`) can't
flatten to a single static-shape companion: each instance owns its
own descriptor with a runtime size, and the sizes generally differ.
Two sub-cases, one shared contract:

  * **5c-A  --  local instance, kernel-internal allocate.**  When `A`
    is a local `fir.alloca` and every `allocate(A(i)%w(M))` site
    uses the same compile-time constant `M`, the pass synthesises a
    fully static 2D companion `A_w(N, M)` and erases the per-instance
    allocate / freemem chain (the buffer is pre-allocated at the
    static shape, so each `allocate` becomes a semantic no-op).  The
    helpers `aosAllocUniformConstSize`, `rewriteAosWholeMemberAssign`,
    `collapseAosAllocReads`, `eraseAosAllocDeallocChain`, and
    `stripReallocOnAosMember` together turn the original IR into a
    flat 2-index designate over `A_w`.

  * **5c-B  --  true SDFG-boundary dummy.**  When `A` is an
    `intent(inout)` (or `intent(in)`) AoS dummy on the SDFG entry
    itself, per-instance sizes are caller-determined and may differ
    at runtime.  The pass emits one FlattenEntry per allocatable
    member with `aos_alloc=True` and `cap_symbol="cap_<base>_<m>"`,
    and inserts two block args per member  --  a runtime cap (`index`
    ref) and a 2D data buffer `ref<array<N x ?xT>>`.  The cap declare's
    `uniq_name = cap_<base>_<m>` makes `traceToDecl` resolve the
    inner extent to that symbol on the SDFG signature.  Stage (5)'s
    bindings emitter (`render_aos_alloc_pack_in` /
    `render_aos_alloc_pack_out` in `bindings/loop_copy.py`) computes
    `cap = max_i(merge(size(A(i)%w), 0, allocated(A(i)%w)))`,
    allocates `A_w(N, cap)`, packs each allocated row's live region
    in, and (for `out`/`inout`) unpacks back on return.  Mixed
    allocation states are allowed  --  unallocated rows stay as the
    zero-padded default and the user's program logic is responsible
    for not reading them.  An empty-batch sentinel (`cap == 0 -> 1`)
    keeps the buffer non-degenerate.  Mixed structs (one allocatable
    + one plain member) split into one `aos_alloc=True` entry per
    allocatable plus one regular aliasable entry covering the rest;
    `recordStructArgEntry` takes an exclude-set to skip members
    already covered by the per-member path.

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
|--- bridge/            C++  --  HLFIR parser + classifier + walker (nanobind)
|   |--- bridge.cpp              MLIRContext, pass pipeline, Python exports
|   |--- extract_vars.cpp        hlfir.declare -> VarInfo[]
|   |--- extract_ast.cpp         entry point; calls into ast/dispatch.cpp
|   |--- trace_utils.cpp         SSA tracing + alias helpers + depth limits
|   \--- ast/                    AST extraction split per-responsibility (real TUs)
|       |--- ast_helpers.h         cross-TU public API + inline thread_local state
|       |--- ast_internal.h        cross-TU internal helper declarations
|       |--- expressions.cpp       buildExpr, buildIndexExpr, lowerIsPresent
|       |--- assigns.cpp           buildAssignNode, copy/memset/libcall, sections
|       |--- elementals.cpp        reductions, elemental walks, select-case chains
|       |--- control_flow.cpp      cmp predicates, buildBoolExpr, scf.while/if walkers
|       \--- dispatch.cpp          top-level walker; calls into the others
|--- passes/            C++  --  HLFIR -> HLFIR rewrites
|   |--- LowerFirSelectCase.cpp  fir.select_case -> cf.cond_br (pre-inline)
|   |--- InlineAll.cpp
|   |--- FoldElementAliases.cpp  erase elemental-body alias declares
|   |--- ExpandVectorSubscriptGather.cpp  hlfir.associate(elemental) -> alloca + gather loop
|   |--- ExpandVectorSubscriptScatter.cpp  hlfir.region_assign(elemental_addr) -> scatter loop
|   |--- RejectPolymorphism.cpp  loud-fail on residual virtual dispatch / SELECT TYPE
|   |--- FlattenStructs.cpp      stamps hlfir.flatten_plan
|   |--- PropagateShapes.cpp
|   |--- DefaultIntent.cpp
|   |--- VerifyNoUnresolvedCalls.cpp
|   \--- Passes.cpp              registerAllBridgePasses()
|--- builder/           Python  --  SDFG emission (stage 4)
|   |--- __init__.py             SDFGBuilder, _emit dispatch, pipelines
|   |--- context.py              _Ctx (state, pending assigns, iter_map)
|   |--- descriptors.py          add_descriptors, DTYPE mapping
|   |--- access.py               build_memlet_index, indirect-symbol lifting
|   |--- emit_tasklet.py         per-occurrence tasklet + emit_scalar_assign
|   |--- emit_cfg.py             assign / loop / while / conditional
|   \--- emit_library.py         copy / memset / libcall / reduce / break / return
|--- intrinsics/        Python  --  Fortran intrinsic registry (consumed by bindings)
|--- bindings/          Python  --  Fortran wrapper emitter (stage 5)
|   |--- frozen_signature.py     FrozenArg + FrozenSignature + drift check
|   |--- fortran_interface.py    OriginalInterface (outer surface)
|   |--- flatten_plan.py         FlattenPlan + to_dict / from_dict
|   |--- block_builders.py       per-Fortran-section emitters
|   |--- loop_copy.py            alias vs deep-copy renderers
|   \--- emit_bindings.py        -> <entry>_bindings.f90
|--- hlfir_to_sdfg.py   compat shim  --  re-exports from builder/
\--- fortran_parser.py  top-level entry: generate_sdfg(...)
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

| If you're adding... | Change here | Then cover in |
| --- | --- | --- |
| a new `math.*` intrinsic | `extract_ast.cpp` `unary_math` / `binary_math` tables | `tests/hlfir/elemwise_intrinsics_test.py` |
| a new reducer | `extract_ast.cpp::kRedTable` (+ `buildSectionReduceAssign` for section form) | `tests/hlfir/reduce_intrinsics_test.py` |
| a new CFG op | `extract_ast.cpp::buildAST` dispatch + `builder/__init__.py::_EMIT_DISPATCH` + emitter in `builder/emit_cfg.py` | ports from `baseline_*_test.py` |
| a new binding layout rule | `bindings/loop_copy.py` + new `FlattenRecipe` field | `tests/hlfir/bindings/emit_bindings_test.py` |
| a new HLFIR pass | file in `passes/`, register in `Passes.cpp`, slot into `DEFAULT_PIPELINE` | `tests/hlfir/<pass>_test.py` |

## Future support map

Current status by feature.  [OK] supported, [!] planned (tracked in
xfails), [X] never (out of scope).

### Types

| Feature | Status | Notes |
|---|---|---|
| `INTEGER(1/2/4/8)` | [OK] | mapped to `int8/16/32/64` |
| `REAL(4/8)` | [OK] | mapped to `float32/64` |
| `LOGICAL(1/4/8)` | [OK] | surfaced as `uint8/int32/int64` for f2py ABI |
| `COMPLEX(4/8)` | [OK] | arrays only (Python 3.12 ctypes lacks `c_double_complex`; scalar by-value is a DaCe-core gap) |
| `CHARACTER(*)` | [X] | string handling out of scope |
| Module-level derived type, flat members | [OK] | Phase 1, lowered by `hlfir-flatten-structs` |
| Module-level derived type, nested | [OK] Phase 2 | recursive `collectFlatLeaves` synthesises one declare per leaf with path-flattened name `base_m1_m2_..._leaf` |
| Array-of-struct with array members (`A(N)%w(M,M)`) | [OK] Phase 1.5 | shape concatenation; designate chains `A(i)%w(j,k)` rewrite to `A_w(i,j,k)` |
| Whole-member access on AoS (`A(i)%w = ...`) | [OK] Phase 1.5 | rewriter emits triplet section `A_w(i, 1:M:1, ...)` |
| Cross-subroutine struct args (incl. AoS) | [OK] Phase 2.2 | function signature rewrite + per-member block args + `hlfir.flatten_plan` attribute carries the recipe |
| Module-level derived type, allocatable members | [OK] Phase 5a/5b | scalar struct + 1-D allocatable / pointer member  --  flat top-level allocatable companion + per-allocate-site rename |
| AoS + allocatable, uniform compile-time size (`type(t) :: A(N); allocate(A(i)%w(M))` with constant `M`) | [OK] Phase 5c-A | static 2D companion `A_w(N, M)`; allocate / freemem chain erased post-flatten |
| AoS + allocatable as inlined-kernel dummy | [OK] Phase 5c-B (inlined) | `collapseAosAllocReads` follows alias chains through `hlfir-inline-all`'s declare aliases |
| AoS + allocatable as true SDFG-boundary dummy | [OK] Phase 5c-B | padding-to-max contract: bindings layer computes `cap = max_i(size(A(i)%w))`, packs/unpacks live regions; runtime cap symbol on SDFG signature |
| AoS + allocatable, kernel-internal reallocation (`intent(out)` first allocation inside kernel) | [!] Phase 5c-C | needs an HLFIR shape-discovery pre-pass + caller-side stub interface |
| Batched CSR (jagged AoS with per-instance allocatable members of differing sizes) | [!] Phase 5c+ | xfailed contract test pins the design; padding-to-max works mechanically but the two-different-members `(rowptr, colidx, val)` shape needs separate cap symbols (in scope but not yet exercised) |
| Derived type with parametric array dim from struct field | [!] Phase 4 | 1 xfail |
| Circular type definitions | [X] | recursion through pointer chain  --  out of scope |

### Control flow

| Feature | Status | Notes |
|---|---|---|
| `DO`, `DO WHILE`, `DO CONCURRENT` | [OK] | LoopRegion + scf.while |
| `IF` / `ELSE IF` / `ELSE` | [OK] | scf.if |
| `SELECT CASE` | [OK] | `lower-fir-select-case` lifts to cf.cond_br |
| `SELECT TYPE` | [X] | requires polymorphic dispatch  --  never |
| `EXIT`, `CYCLE` | [OK] | |
| `GOTO` | [X] | rely on flang's `lift-cf-to-scf`; unstructured GOTO won't lift |
| Statement functions (`f(x) = ...`) | [!] | 1 xfail, [function_statement](../../../tests/hlfir/ported/fortran_language_test.py) |

### Subprograms / linkage

| Feature | Status | Notes |
|---|---|---|
| Module-contained `SUBROUTINE` / `FUNCTION` | [OK] | inlined by `hlfir-inline-all` |
| Internal subprograms (`CONTAINS` inside subroutine) | [OK] | needs `fir.embox` peeling in alias walk (added 2026-04-28) |
| `INTERFACE` blocks | [OK] | resolved at flang time |
| `EXTERNAL` statements | [X] | use modules |
| `USE`, `USE ... ONLY:` | [OK] | flang resolves at lowering |
| `OPTIONAL` dummy + `PRESENT` | [OK] | folded statically post-inline |
| `ALLOCATABLE`, `ALLOCATE`, `DEALLOCATE` | [OK] | local + dummy |
| `POINTER` | [X] probably | requires SSA-breaking aliasing |
| BLAS/LAPACK via `EXTERNAL` | [X] | use module-contained or DaCe libnodes |

### Polymorphism

| Feature | Status | Notes |
|---|---|---|
| `CLASS(t)` as a monomorphic box (no virtual dispatch) | [OK] | `FlattenStructs` peels `fir.class<T>` via `BaseBoxType` |
| Type-bound procedure with statically-known receiver (`c%area()` where `c : type(circle_t)`) | [OK] | `fir-polymorphic-op` devirtualises before flatten; tested in `derived_type_test::test_static_polymorphism_devirtualised` |
| Truly virtual dispatch (`class(shape_t) :: p; p%area()` where the receiver type is set by a caller) | [X] | `fir-polymorphic-op` lowers the dispatch to an indirect `fir.call` through the type-info table; `hlfir-reject-polymorphism` fires loudly on the residual `fir.box_tdesc`.  Tested in `noncontig_unsupported_test::test_virtual_dispatch_bails_loudly`. |
| `SELECT TYPE` / runtime type discrimination | [X] | rejected by `hlfir-reject-polymorphism` |

### Slicing / array ops

| Feature | Status | Notes |
|---|---|---|
| Contiguous slice `a(i:j, k:l)` | [OK] | |
| Whole-array assign `a = b` | [OK] | hlfir.elemental + emit_library |
| Elementwise intrinsics (sin/cos/exp/sqrt/...) on real/complex | [OK] | added complex variants 2026-04-28 |
| Reductions (sum/product/min/max/all/any/count/minval/maxval) | [OK] | |
| BLAS/LAPACK (matmul, transpose) | [OK] | dense -> libnode, strided -> explicit DO loop |
| Noncontiguous slice via index array `a(idx, :)`  --  rank-1, **constant** gather extent | [OK] | Lowered by `hlfir-expand-vector-subscript-gather` (replaces flang's `hlfir.associate` of an `hlfir.elemental` with an explicit `fir.alloca` + gather DO loop, then reuses DaCe indirection memlets `<arr>_at<gid>`).  See pass header for shape constraints. |
| Noncontiguous slice  --  rank-2+ gather (`d(cols2, cols)`) | [!] Phase 2 | 5 xfails  --  pass currently bails with a clear error |
| Noncontiguous slice  --  **symbolic** extent | [X] | DaCe can't express runtime-sized symbol arrays.  Pass aborts loudly via `op.emitError`; covered by [noncontig_unsupported_test.py](../../../tests/hlfir/noncontig_unsupported_test.py) |
| Noncontiguous slice  --  INTENT(out) scatter-back | [!] | rare in real code; not yet modelled (the i1 must-finalise flag is hard-coded to false) |
| `ASSOCIATE` block | [!] | not currently exercised; relative indexing only |

### Codegen targets

| Feature | Status | Notes |
|---|---|---|
| CPU C++ tasklets | [OK] | |
| GPU CUDA | [X] | would need OpenACC-style shim emission |
| OpenMP / `!$OMP` directives | [X] | DaCe handles parallelism |
| COARRAY | [X] | |

## Non-goals

- Re-parsing Fortran in Python. Flang is authoritative.
- GPU target bindings (would need OpenACC shim emission).
- Fortran SIMD / COARRAY semantics.
- Cross-kernel fusion across translation-unit boundaries  --  inline-all
  handles intra-TU fusion; cross-TU is the binding emitter's problem.

## Testing

Every supported construct has a seeded numerical test against
gfortran / f2py under `tests/hlfir/`. Binding-specific tests live
in `tests/hlfir/bindings/`. The six E6 velocity-advection
representative loopnests have SDFG-vs-f2py comparisons in
`tests/hlfir/icon_loopnests/`. All executable-Fortran tests
compile with `gfortran` (Ubuntu's `flang-new-21` ships without
`libflang_rt` so it's emit-HLFIR-only).

```bash
# dump built SDFGs for inspection
__DACE_HLFIR_GEN_TEST_SDFGS=1 python3 -m pytest tests/hlfir/
__DACE_HLFIR_GEN_TEST_SDFGS=/tmp/mine python3 -m pytest tests/hlfir/
```

# HLFIR bridge — remaining work, prioritized

**Last updated:** 2026-05-02 (post-Phase H + I).
**Sweep:** 553 passed / 13 xfailed / 0 failed.
**SOTA velocity_tendencies SDFG:** `/tmp/probe_velocity/velocity_tendencies_partial.sdfg` (608 KB).
**Sources:** ICON-DACE at `/home/primrose/Work/icon-dace/`; velocity probe at `/tmp/probe_velocity/velocity_tendencies.f90`.

The single highest-leverage outcome is **getting `velocity_tendencies` to fully
build a validated SDFG**. Everything else is below that. Within each tier,
items are ordered by likely effort (smallest first).

Legend:
- 🚫 currently failing the build / blocker
- 🟡 currently xfailed (test exists, marked xfail)
- 🟢 follow-up / hardening (sweep stays green)
- ⛔ truly out of scope

---

## Tier 0 — Velocity_tendencies SDFG fully builds

### 0.1 — Phase F: alias-prefix chain rewrite into inlined-callee bodies 🚫
**The actual velocity probe blocker.**

**Symptom.** `SDFGBuilder.build()` dies at `_attach_frozen_signature` →
`sdfg.arglist()` with `KeyError: 'p_int'`. The `set_zeta` tasklet
expression in the inlined `rot_vertex_ri` body renders as
`(_in_vec_e_0 * p_int) + ...` — bare `p_int` instead of the synthesised
flat companion `p_int_geofac_rot`.

**Inferred root cause.** `t_int_state` members `geofac_rot` and
`cells_aw_verts` are referenced ONLY through the inlined callee body
(`ptr_int%geofac_rot(...)`, never directly in the caller's body).
`traceAliasPrefixToDecl` /
[FlattenStructs.cpp:933-985](dace/frontend/hlfir/passes/FlattenStructs.cpp#L933-L985)
walks the alias-prefix chain through `convert / embox / designate /
declare` ops, but apparently misses some op kind in the inlined body's
chain — possibly `fir.rebox`, possibly an additional `hlfir.declare`
intermediate. The 5 caller-touched members (`c_lin_e`, `e_bln_c_s`,
etc.) flatten correctly because they have direct caller references too.

**Investigation steps (do BEFORE coding):**
1. Dump post-inline pre-flatten IR for the velocity probe. Find the
   inlined `rot_vertex_ri` body's `ptr_int` declare; trace its memref
   chain.
2. Find the `hlfir.designate %ptr_int_inlined{"geofac_rot"}(jv, k, jb)`
   in the IR. Its memref should resolve to `p_int_decl.getResult(0/1)`
   through `traceAliasPrefixToDecl`. If it doesn't, identify the
   missing op kind.
3. Confirm `leafBase["geofac_rot"]` exists at flatten time
   (`collectFlatLeaves` should enumerate all 7 members).

**Fix.** Extend `traceAliasPrefixToDecl` (or `rewriteChainsRootedAt`'s
alias discovery) with the missing op-walk case.

**Test.** New `tests/hlfir/inlined_callee_member_alias_test.py`:
2-routine fixture, caller passes struct `s`, inner subroutine reads
`s%onlyhere` (a member the caller never touches directly), assert
SDFG references the flat `s_onlyhere` companion + numerical
correctness against gfortran.

**Effort.** ~50–100 LoC + 1 unit test. Investigation likely consumes
half the time.

**Acceptance.** Velocity probe gets past `arglist()` with no `KeyError`.

---

### 0.2 — Whatever Phase F surfaces next 🚫
Re-probe; bucket the next concrete failure. Likely candidates:
- `sdfg.validate()` errors on the connected SDFG.
- Phase G (runtime-shape symbols for ALLOCATABLE struct members) becomes
  a real bug instead of cosmetic.
- A second alias-resolution gap in a different inlined-callee.

Iterate until the build returns a validated SDFG.

---

### 0.3 — End-to-end velocity_tendencies numerical test 🟢
**Once the SDFG builds.** Write
`tests/hlfir/velocity_tendencies_e2e_test.py`:
- Compile `/tmp/probe_velocity/velocity_tendencies.f90` via gfortran
  (kernel only, driver supplies pre-allocated struct members).
- Run both the SDFG and the gfortran reference on identical random
  inputs at realistic ICON sizes (`nproma=32, nlev=16, nblks=4`).
- Compare output buffers with appropriate `rtol`.

Currently no such test exists — only structural xfail probes.
Without it, P0.1+P0.2 could land a buildable-but-wrong SDFG.

---

## Tier 1 — Currently-xfailed tests, concrete known fixes (small wins)

### 1.A — Phase A: module-level / local type with pointer member 🟡
**Test:** [global_test.py:19](tests/hlfir/ported/global_test.py#L19) — `test_fortran_frontend_global`.

**Shape.** `type(t) :: ptr_patch` at function scope where `t` has
`double precision, pointer :: w(:,:,:)`. Caller writes
`ptr_patch%w(:,:,:)`, passes `ptr_patch%w` by reference, reads scalar
elements.

**Fix.** Lift the existing Phase 5b dummy-arg synthesis (already
handles `pointer :: w(:,:,:)` members on dummies, see
[FlattenStructs.cpp:3087-3110](dace/frontend/hlfir/passes/FlattenStructs.cpp#L3087-L3110))
into the local-allocation walk in `splitLocal`. Reuse
`isAllocatableArrayMember` predicate verbatim.

**Effort.** ~30 LoC + new `tests/hlfir/local_pointer_member_test.py`.

**Acceptance.** Xfail decorator drops; new test passes against f2py
reference.

---

### 1.B — Phase B: plain pointer dummy as runtime-shape array 🟡
**Test:** [nested_array_test.py:70](tests/hlfir/ported/nested_array_test.py#L70) — `test_fortran_frontend_nested_array_access_pointer_args_2`.

**Shape.** `integer, pointer, intent(inout) :: test(:,:,:)` — top-level
pointer dummy, never a struct member.

**Fix.** Drop the `fir.box<ptr<array<?>>>` rejection in
[extract_vars.cpp:621-651](dace/frontend/hlfir/bridge/extract_vars.cpp#L621-L651);
peel like allocatable. Confirmed by the old
`dace/frontend/fortran/`'s no-op `pointer_stmt` — pointer dummies are
just decoration on a runtime-shape array.

**Effort.** ~20 LoC; relax classifier guard, add 1 unit test.

**Acceptance.** Xfail decorator drops; numerical correctness against
gfortran.

---

### 1.J — Phase J: class-to-type demotion for monomorphic CLASS 🟡
**Tests:**
- [fortran_class_test.py:20](tests/hlfir/ported/fortran_class_test.py#L20) — `test_fortran_frontend_class`.
- [elemental_test.py:135](tests/hlfir/ported/elemental_test.py#L135) — `test_fortran_frontend_elemental_ecrad_range`.

**Insight.** Saved memory's "OOP unsupported" was overly broad. In
Fortran, if no `TYPE EXTENDS(T)` declares a child of `T` anywhere in
the program, every `CLASS(T)` reference is in practice monomorphic.
Demote the dummy to `TYPE(T)` and the existing flatten machinery
applies verbatim.

**Both failing tests are monomorphic** (verified by inspection):
- `t_comm_pattern_orig` is not extended; calls are direct subroutine
  calls (no `obj%method()` dispatch).
- `pdf_sampler_type` is not extended; caller passes a plain `TYPE`
  instance into a `CLASS(...)` dummy.

ICON broadly uses `CLASS` mostly monomorphically; `mo_ecrad.f90`'s
`del_opt_ptrs(CLASS(t_opt_ptrs))` is monomorphic. The velocity probe
uses **0** `CLASS`, so this phase is xfail-flipping not
velocity-blocking.

**Approach.** New MLIR pass `hlfir-demote-monomorphic-class`, runs
AFTER Flang's upstream `polymorphic-op-conversion` (already in
[Passes.cpp:75-77](dace/frontend/hlfir/passes/Passes.cpp#L75-L77))
and BEFORE `hlfir-reject-polymorphism`:

1. Walk the module; collect every `fir.RecordType` that's the parent
   of some other `RecordType` via `EXTENDS` (the "extended" set).
2. For every `hlfir.declare` whose type carries `fir.class<T>` where
   `T` is NOT in the extended set AND no `fir.dispatch` /
   `fir.select_type` in the function targets `T`:
   - Rewrite the type from `fir.class<T>` to `fir.box<T>` (or plain
     `T` for non-allocatable / non-pointer dummies).
   - Update transitive uses (`fir.embox` / `fir.rebox` / load / store).
3. Each demoted declare carries a metadata note ("Demoted CLASS(T) to
   TYPE(T) — T has no extensions in this program") so adding an
   extension later surfaces the silent unsafety at build time.

Surviving `fir.dispatch` after both upstream conversion AND this
demotion is genuinely runtime-polymorphic and stays rejected by
`hlfir-reject-polymorphism`.

**Tests.** 3 unit tests:
- monomorphic CLASS dummy → flattens correctly (positive).
- CLASS dummy where T has extensions but no dispatch at this site →
  still demoted at this call.
- CLASS dummy with `obj%method()` dispatch → stays rejected.

**Effort.** ~80 LoC pass + ~30 LoC for the extended-set computation +
3 unit tests + 2 xfail flips.

---

### 1.K — Phase K: cycle-detected flatten + opaque self-pointer leaves 🟡
**Test:** [type_test.py:237](tests/hlfir/ported/type_test.py#L237) — `test_fortran_frontend_circular_type`.

**Insight.** Fortran circular types MUST use `POINTER` (or
`ALLOCATABLE`) at the cycle-breaking edge — the language requires it
because non-pointer recursion would have infinite size. That same
pointer breaks the recursion at flatten time too. The xfailed test
has the circular pointer assignments commented out and only uses
non-circular paths (`s%w`, `b(1)%x`); steps 1+2 below flip it.

**Steps 1+2 (in scope here):**

1. **Cycle detection in `collectFlatLeaves`.** Add a `seenTypes`
   `DenseSet<Type>` parameter; on entry to a record type insert it
   and return early (treat as opaque leaf) if already present; on
   exit erase. Today's recursion at
   [FlattenStructs.cpp:499-523](dace/frontend/hlfir/passes/FlattenStructs.cpp#L499-L523)
   would otherwise infinite-loop or hit `kFlattenMaxDepth`.

2. **Pointer-to-RecordType as opaque leaf.** Extend `isFlatMemberType`
   to accept `fir.box<fir.ptr<fir.RecordType>>` as an opaque leaf
   when the record is on the recursion path. Synthesise a flat
   companion that's just the pointer descriptor — no per-field
   flattening of the pointee.

**Step 3 (out of scope here, see ⛔ tier).** Use-driven flatten through
cycles — only needed if a kernel actually dereferences the
self-referential pointer (`s%b%a%w`).

**Tests.** Add to `tests/hlfir/circular_type_test.py`:
- Non-cyclic-path access on a circular type → builds correctly.
- Same with `b(N)%x` indexed access → builds correctly.

**Effort.** ~50 LoC + 1 xfail flip + 2 new unit tests.

---

### 1.G — Phase G (spike): runtime-shape symbols for flattened ALLOCATABLE struct members 🟢
**Source.** Velocity probe — flattened ALLOCATABLE members
(`p_int_c_lin_e` etc.) surface with shape `(1,)` placeholder rather
than `(p_int_c_lin_e_d0, _d1, _d2)` runtime-shape symbols. Static-shape
members of `t_patch` (`p_patch_edges_cell_idx` etc.) get proper symbols.

**Open question.** Cosmetic or real bug?
- If cosmetic: SDFG body works because each access resolves to a
  scalar memlet; bindings emitter synthesises shape symbols at the
  call site.
- If real: the bindings emitter rejects the shape `(1,)` companion
  for an array dummy, or arglist signature has wrong types.

**Approach.** Spike — read the bindings emitter; trace what happens
when it sees an ALLOCATABLE companion with shape `(1,)`. If broken,
fix the descriptor → shape-symbol extraction in
[FlattenStructs.cpp:3087-3110](dace/frontend/hlfir/passes/FlattenStructs.cpp#L3087-L3110)
to mint per-dim symbols on the synthesised declare. If cosmetic,
document and defer.

**Effort.** Spike: ≤2h. Fix if needed: ~30 LoC.

---

## Tier 2 — Currently-xfailed tests, more design needed

### 2.array_slice_inliner 🟡
**Test:** [array_test.py:134](tests/hlfir/ported/array_test.py#L134) — `nested_call_array_slice`.

**Status note.** "Inliner leaves callee dummy as uninitialised
transient." After inlining a callee that takes a `(:)` slice, the
bridge sees the dummy as a fresh transient with no upstream provenance.

**Likely fix.** In `hlfir-inline-all` post-processing or in
`extract_vars`, trace the inlined dummy's box back to the caller-side
argument and rewrite the alias (similar in shape to Phase F but
specifically for slice-typed dummies).

**Effort.** Investigation needed — probably ~80 LoC.

---

### 2.module_dt_slice 🟡
**Test:** [type_array_test.py:20](tests/hlfir/ported/type_array_test.py#L20) — `module_contained_dt_slice`.

**Shape.** `conf%fraction(1,:)` where `conf` is a derived-type
instance defined in a module's CONTAINS section (not local, not a
dummy).

**Likely fix.** Module-contained DT instances need their own flatten
path beyond Phase 1's local-instance walk. Shares structure with
Phase A (module-level type) but the storage class differs.

**Effort.** ~50 LoC after Phase A lands; shares its synthesis.

---

### 2.noncontig_pardecls 🟡
**Tests:** [noncontig_pardecls_test.py:218,328](tests/hlfir/ported/noncontig_pardecls_test.py#L218) — 2 xfails.

- "noncontiguous + derived type with cols field"
- "nested noncontiguous + transpose ECRAD pattern"

**Status.** Combines noncontiguous-section handling, derived-type
flatten, and transpose intrinsic. Each is exercised separately
elsewhere; the combination needs end-to-end thinking.

**Effort.** Substantial — each test likely surfaces 2-3 distinct
sub-issues.

---

### 2.view_reshape 🟡
**Test:** [view_reshape_test.py:20](tests/hlfir/ported/view_reshape_test.py#L20).

**Shape.** Fortran storage-association reshape: passing a 2D section
`d(:,:,1)` to a 1D dummy. The bridge sees a 2D source and a 1D
consumer; no current path bridges them.

**Status.** Distinct gap; needs a reshape-aware adapter at the call
site or an SDFG-level view of the source array.

**Effort.** ~100 LoC; new code path.

---

### 2.E — Phase E: ALLOCATABLE + INTERFACE 🟡
**Test:** [intrinsic_bound_test.py:157](tests/hlfir/ported/intrinsic_bound_test.py#L157).

**Status.** Likely call-site / signature-resolution issue with
ALLOCATABLE arrays through INTERFACE blocks; not alloc-storage
proper. Investigation needed before committing to an approach.

**Effort.** Investigation: ~2h. Likely fix: ~50 LoC.

---

## Tier 3 — Deferred (rare in real ICON kernels)

### 3.C — Phase C: nested AoS + allocatable struct member 🟡
**Test:** [type_test.py:425](tests/hlfir/ported/type_test.py#L425) — `type_arg`.

`type%pprog(1)%w(1,1)` — combines Phase 5c-A (AoS-allocatable, already
landed for the uniform-size case) with Phase 5b (allocatable scalar
member, also landed). Real ICON kernels rarely use
array-of-struct-with-pointer; revisit when forced.

### 3.D — Phase D: nested derived types + allocatable scalar 🟡
**Test:** [type_array_test.py:170](tests/hlfir/ported/type_array_test.py#L170) — `type3_array`.

Nested derived types with allocatable scalar members + pointer-rebind
in deepest inlined callee. Same deferral reasoning.

---

## Tier ⛔ — Truly out of scope

### Genuine runtime polymorphic dispatch
`fir.dispatch` / `fir.select_type` whose target depends on runtime
type — needs per-target codegen + runtime type info at the SDFG level.
Architectural change, not a pass. **Stays rejected** by
`hlfir-reject-polymorphism` after both upstream conversion AND Phase J
demotion. Tests that hit this:

- Whatever subset of `fortran_class_test` / `elemental_ecrad_*` survives
  Phase J with genuine dispatch.
- Any future ICON kernel that uses `obj%method()` over a class with
  multiple implementations.

### Use-driven flatten through cycles (Step 3 of Phase K)
Kernels that actually dereference self-referential pointers
(`s%b%a%w` traversing the cycle). Needs use-driven flatten instead of
type-driven; substantial refactor. Stays out unless forced by a real
kernel.

---

## Cross-cutting / hardening

### Isolated / duplicated constant-init states 🚫
**Source.** Velocity probe SDFG inspection.

**Symptom — broader than initially observed.** **11 top-level states
in the velocity SDFG carry content but have zero in/out edges at
their containing-region level**, and most of them are content
duplicates of states that DO live in the right region:

| Top-level isolated | Content | Duplicate of |
|---|---|---|
| `s_20` (id=15) | `set___assoc_scalar_4`, `set_cells2verts_scalar_ri_slev` | id=1 inside `if_19` |
| `s_114` (id=25) | `set_velocity_tendencies_rl_start/end` | id=1 inside `if_113` |
| `s_150` (id=26) | same | duplicate of s_114 |
| `s_187` (id=34) | same | duplicate again |
| `s_356` (id=44) | same | duplicate again |
| `s_69` (id=16) | `set___assoc_scalar_8/9` | another assoc-scalar copy |
| `post_rot_vertex_ri_elev_71` (id=18) | `set_rot_vertex_ri_rl_start/end` | const-init copy |
| `post_velocity_tendencies_i_endblk_189/440` | const-init / `set__QQred_lift_0` | each isolated |
| `post_jg_9` | `t_10` | — |
| `s_442` (id=52) | `set_max_vcfl_dyn`, `set_p_diag_max_vcfl_dyn` | the Phase I state, but emitted at top level rather than inside the loop body where it semantically belongs |

`set_velocity_tendencies_rl_start` alone appears in **5 states**: one
legitimate copy inside `if_113`, four orphan duplicates at top level.

**Implication.** This isn't two stray states — it's a systemic
CFG-wiring bug. Every constant-init tasklet emission appears to land
in BOTH the inside-region copy AND a fresh top-level state that's
never connected. The Phase I fix (`f8a6637bf`) corrected the
data-flow inside `s_442` but `s_442` itself is still mis-placed at
the top level.

This crosses from "🟢 hardening" into "🚫 likely real bug" — the
duplication strongly suggests the `ctx.flush` / `ctx.ensure` /
`region.add_state` machinery is creating orphan states alongside the
real ones, or wiring the wrong region for some emit paths.

**Hypothesis.** Two related CFG-wiring gaps:
1. `ctx.flush(builder)` (no `region` arg) emits pending scalar
   assigns into the SDFG-level current state, even when the calling
   code intended them to land inside a region. See
   [context.py:48-55](dace/frontend/hlfir/builder/context.py#L48-L55) — the
   `region=None` default routes to `self.sdfg`, not the caller's
   region.
2. The post-emit `ctx.ensure(region)` then creates a new region-level
   state but the prior flush already created a top-level state with
   the same content.

**Investigation steps:**
1. Add a structural pytest that walks every state in
   `tests/hlfir/inline_reduction_test.py::test_inline_maxval_in_max_expression`'s
   produced SDFG and asserts no isolated-with-content state at top
   level. The Phase I read-then-writeback pattern likely reproduces
   the issue at minimal scale.
2. Trace which call site emits each duplicate. Likely candidates:
   `emit_cfg.py:emit_assign` symbol-target branch
   ([emit_cfg.py:42-81](dace/frontend/hlfir/builder/emit_cfg.py#L42-L81))
   and `ctx.flush` calls without an explicit `region`.
3. Audit every `ctx.flush(builder)` (no region arg); pass the active
   region in.

**Effort.** ~50 LoC + structural test + investigation.

**Acceptance.** No top-level isolated states with content in the
velocity probe SDFG; all const-init tasklets live in the region whose
body actually executes them. New structural test pins this.

---

### Convert /tmp/probe_velocity probe into a tracked test 🟢
**Currently.** The probe lives at `/tmp/probe_velocity/`, not in the
repo. After Phase F + 0.2 land, port it as
`tests/hlfir/velocity_tendencies_probe_test.py` (initial structural
check that the SDFG builds; full numerical e2e covered by 0.3).

This pins the velocity_tendencies build state in CI so a future
regression can't silently break it.

---

## Recommended order

1. **Tier 0** end-to-end (Phase F → re-probe → e2e numerical test) —
   the highest-leverage outcome.
2. **Tier 1** in any order; J + K are independent; A + B share Phase
   5b synthesis machinery; G is a quick spike.
3. **Tier 2** items only as forced by deeper ICON probes; each is
   substantial and unlocks a smaller surface area.
4. **Tier 3** deferred until everything above closes.
5. **⛔** stays out.

---

## Reference: SOTA SDFG inspection results

| Metric | Pre-Phase-H+I | Post (current SOTA) |
|---|---|---|
| `_allocated` tracker arrays | 30 (all dead) | 0 |
| `post_*_allocated_<n>` orphan states | 34 | 0 |
| `s_442` `p_diag_max_vcfl_dyn` access nodes | 1 (cycle) | 2 (input + output) |
| Build outcome | partial-save, fails at `arglist()` | same; same `KeyError: 'p_int'` |
| Total arrays in SDFG | 174 | 144 |
| Sweep | 553P / 13xF / 0F | 553P / 13xF / 0F |

The SOTA SDFG is **structurally clean** — body-level dead weight is
gone, the read-then-writeback cycle is resolved. **Phase F is what's
left** to make `arglist()` succeed; after that the build should
finalise.

---

## Commits this session (for reference)

| Commit | Phase / topic |
|---|---|
| `e3cfbcc68` | Lift reduction operands MLIR pass + 6 unit tests |
| `19b45d530` | Initial pointer/alloc plan (A/B/C/D/E/F/G) |
| `6e55e705e` | Plan amended with H + I from velocity inspection |
| `36f08e281` | Phase H — gate `_allocated` tracker on actual usage |
| `f8a6637bf` | Phase I — split read+write access nodes |
| `1663f32b3` | Plan status table updated |
| `df335f359` | Plan + Phase J + Phase K + isolated-states finding |

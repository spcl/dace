# Velocity_tendencies SOTA + open work — full prioritized plan

**Last updated:** 2026-05-02 (post Phase H + I).
**Sweep:** 553 passed / 13 xfailed / 0 failed.
**Current SOTA SDFG:** `/tmp/probe_velocity/velocity_tendencies_partial.sdfg` (608 KB,
also snapshotted at `velocity_tendencies_post_HI_20260502_0213.sdfg`).

The single highest-priority item is **getting the velocity_tendencies SDFG to fully
build** (i.e. `SDFGBuilder.build()` returns a validated SDFG, not a partial-saved
one). Everything else is below that.

---

## P0 — Velocity_tendencies SDFG fully builds

The probe currently dies at `_attach_frozen_signature → sdfg.arglist()` with
`KeyError: 'p_int'`. The SDFG body itself builds; the failure is at
arglist-assembly because some tasklet expression carries a bare `p_int`
identifier as a free symbol.

### P0.1 — Phase F: alias-prefix chain rewrite into inlined-callee bodies

**Symptom.**

In the inlined `rot_vertex_ri` body, the `zeta(...)` tasklet expression renders as

```
_out_zeta = ((((((_in_vec_e_0 * p_int) + (_in_vec_e_1 * p_int)) + …) + (_in_vec_e_5 * p_int)))
```

— bare `p_int` instead of the synthesised flat companion `p_int_geofac_rot`.

**Inferred root cause.** Members of `t_int_state` referenced ONLY through the
inlined callee body (`geofac_rot`, `cells_aw_verts`) bypass the alias-prefix
chain rewriter. The 5 members the caller body also touches directly
(`c_lin_e`, `e_bln_c_s`, `geofac_grdiv`, `geofac_n2s`, `rbf_vec_coeff_e`) all
flatten correctly to `_in_p_int_<member>`. Pipeline order is fine
(`hlfir-inline-all` runs before `hlfir-flatten-structs`), so the alias declare
IS in the same function — yet `traceAliasPrefixToDecl` /
`rewriteChainsRootedAt` aren't reaching this designate.

**Investigation steps (do these first, before coding):**
1. Dump the post-inline pre-flatten IR for the velocity probe. Look at the
   `hlfir.declare` for `ptr_int` in the inlined `rot_vertex_ri` body and trace
   its memref chain. Does it bottom out at the `p_int` declare via a path
   `traceAliasPrefixToDecl` recognises (convert / embox / designate /
   declare)? If not, identify the missing op kind.
2. Look at the `hlfir.designate` for `ptr_int%geofac_rot(jv, k, jb)` inside the
   inlined body. Its memref should be `ptr_int_inlined_decl.getResult(0/1)`.
   Verify with `clusterFix.cpp` style IR dump.
3. If the alias declare IS in `equivalentRoots`, why does the leaf rewrite
   fail? `leafBase["geofac_rot"]` should exist (collectFlatLeaves walks all 7
   members).

**Concrete change (after investigation confirms hypothesis):** Extend
`traceAliasPrefixToDecl` at
[FlattenStructs.cpp:933-985](dace/frontend/hlfir/passes/FlattenStructs.cpp#L933-L985)
to handle the missing op kind (likely `fir.rebox` or a wrapping
`hlfir.declare` chain). If the gap is in `rewriteChainsRootedAt`'s alias
discovery, add the missing walker pattern there.

**Test.** New `tests/hlfir/inlined_callee_member_alias_test.py` with a
2-routine fixture: caller passes struct `s`, inner subroutine reads `s%onlyhere`
(a member the caller never directly touches). Pre-fix the test should fail
with bare-struct-name leak; post-fix the SDFG should reference the flat
companion.

**Acceptance.** Velocity probe gets past `arglist()` with no `KeyError`.

### P0.2 — Whatever Phase F surfaces next

Once Phase F unblocks `arglist()`, run the velocity probe again and bucket the
next concrete failure. Likely candidates (rough priority):

- **Phase G — runtime-shape symbols for flattened ALLOCATABLE struct members.**
  Currently `p_int_c_lin_e` etc. surface as shape `(1,)` placeholder; if
  bindings emission expects `_d0/d1/d2` symbols, this becomes a real bug
  rather than a cosmetic one. Spike first.
- Validation errors on the connected SDFG (now that the cycle and dead
  trackers are gone, `sdfg.validate()` may still surface secondary issues).
- Numerical correctness once the build succeeds — compare against a
  reference run.

### P0.2.bis — Isolated states inside conditional bodies

**Source of finding:** velocity probe SDFG inspection. States `s_20`
(inside `if_19`'s if-branch) and `s_114` (inside `if_113`'s if-branch)
contain content (constant-init tasklets like `set_velocity_tendencies_rl_start`,
`set___assoc_scalar_4`) but have **zero incoming and zero outgoing
interstate edges** within their containing conditional region. They're
orphaned within the branch body — the runtime never reaches them, so
the constants they would set are never written.

**Hypothesis.** The bridge synthesises a state for each constant-init
ASTNode (e.g. `set_<rl_start>`) before walking the IF body, but
doesn't wire it into the branch's CFG. Likely a missing `region.add_edge`
in the `if`-branch context-flush path, similar in shape to the
`s_442` cycle (Phase I, `f8a6637bf`) but in the CFG-wiring code rather
than the access-node-cache code.

**Acceptance.** Both isolated-state instances disappear; their tasklets
get wired into the branch's CFG so the constants are actually
assigned at branch entry.

### P0.3 — End-to-end velocity_tendencies numerical test

Once the SDFG builds, write `tests/hlfir/velocity_tendencies_e2e_test.py` that:
- Compiles the same `/tmp/probe_velocity/velocity_tendencies.f90` via gfortran
  (the kernel only — driver supplies pre-allocated arrays).
- Runs both the SDFG and the gfortran reference on identical random inputs
  with realistic ICON sizes (nproma=32, nlev=16, nblks=4 or so).
- Compares output buffers bit-exact (or with appropriate `rtol`).

Currently no such test exists; only structural xfail probes.

---

## P1 — Currently-xfailed tests with concrete known fixes

These have a clear path to passing once the prerequisite phase lands.

### P1.A — Phase A: module-level type with pointer member
**Test:** `tests/hlfir/ported/global_test.py::test_fortran_frontend_global`.
**Status:** xfail — Phase 1 doesn't lower POINTER members on local/module-level
instances; only on dummies.
**Fix:** lift the existing Phase 5b dummy-arg synthesis (already handles
`pointer :: w(:,:,:)` members on dummies) into the local-allocation walk in
`splitLocal`. Reuse `isAllocatableArrayMember` predicate verbatim.
**Effort:** ~30 LoC + new `tests/hlfir/local_pointer_member_test.py` with f2py
reference.

### P1.B — Phase B: plain pointer dummy as runtime-shape array
**Test:** `tests/hlfir/ported/nested_array_test.py::test_fortran_frontend_nested_array_access_pointer_args_2`.
**Status:** xfail — `fir.box<ptr<array<?>>>` dummies rejected.
**Fix:** drop the rejection in `extract_vars`; peel `fir.box<ptr<...>>` like
`fir.box<heap<...>>`. Confirmed by the old `dace/frontend/fortran/`'s no-op
`pointer_stmt` — pointer dummies are just decoration.
**Effort:** ~20 LoC.

---

## P2 — Currently-xfailed tests requiring more design

### P2.array_slice_inliner — `array_test::nested_call_array_slice`
**Status:** xfail — "inliner leaves callee dummy as uninitialised transient".
After inlining a callee that takes a `(:)` slice, the bridge sees the dummy
as a fresh transient with no upstream provenance.
**Likely fix:** in `hlfir-inline-all` post-processing or in `extract_vars`,
trace the inlined dummy's box back to the caller-side argument and rewrite
the alias.

### P2.module_dt_slice — `type_array_test::module_contained_dt_slice`
**Status:** xfail — module-contained derived-type array slice
(`conf%fraction(1,:)`) not lowered. Module-contained DT instances probably
need their own flatten path beyond Phase 1's local-instance walk.

### P2.noncontig_pardecls — `noncontig_pardecls_test` (2 tests)
**Tests:**
- noncontiguous + derived type with cols field
- nested noncontiguous + transpose ECRAD pattern

**Status:** xfail. Combines noncontiguous-section handling, derived-type
flatten, and transpose intrinsic. Each is exercised separately elsewhere;
the combination needs end-to-end thinking.

### P2.view_reshape — `view_reshape_test`
**Status:** xfail — Fortran storage-association reshape (passing 2D section
`d(:,:,1)` to a 1D dummy). Distinct from anything we've handled; the bridge
sees a 2D source and a 1D consumer, no current path bridges them.

### P2.intrinsic_bound_alloc_iface — `intrinsic_bound_test::ALLOCATABLE+INTERFACE` (Phase E)
**Status:** xfail — likely call-site signature-resolution issue with
ALLOCATABLE arrays through INTERFACE blocks. Investigation needed before
classifying as alloc-storage vs. call-resolution.

---

## P3 — Currently-xfailed tests deferred (rare in real ICON kernels)

### P3.C — `type_test::type_arg` — Phase C
Nested AoS + allocatable struct member: `type%pprog(1)%w(1,1)`. Combines
Phase 5c-A (AoS-allocatable) + Phase 5b (allocatable scalar member).

### P3.D — `type_array_test::type3_array` — Phase D
Nested derived types + allocatable scalar + pointer-rebind in deepest
inlined callee.

---

## P1.J — Class-to-type demotion for monomorphic CLASS dummies [NEW]

**Insight.** "OOP unsupported" was overly broad. **Monomorphic CLASS use
is tractable**: in Fortran, if no `TYPE EXTENDS(T)` declares a child of
`T` anywhere in the program, every `CLASS(T)` reference is in practice
monomorphic — its dynamic type is always `T`. Demote the dummy to
`TYPE(T)` and the existing flatten machinery applies verbatim.

**Both failing OOP tests are monomorphic:**
- `fortran_class_test::test_fortran_frontend_class` —
  `class(t_comm_pattern_orig)` dummies; `t_comm_pattern_orig` has no
  extensions; no `obj%method()` dispatch (only direct subroutine
  calls).
- `elemental_test::elemental_ecrad_range` —
  `class(pdf_sampler_type)` in one elemental subroutine;
  `pdf_sampler_type` not extended; caller passes a plain `TYPE`
  instance.

**ICON / ECRAD coverage gain.** ICON uses `CLASS` heavily but mostly
monomorphically. `mo_ecrad.f90`'s `del_opt_ptrs(CLASS(t_opt_ptrs))` is
monomorphic. The velocity probe itself uses **zero** `CLASS` — this
phase doesn't help velocity, but it flips both OOP xfails and unlocks a
broad ICON surface that's currently rejected.

**Approach.** New MLIR pass `hlfir-demote-monomorphic-class`, runs
BEFORE `hlfir-reject-polymorphism` and AFTER Flang's upstream
`polymorphic-op-conversion`:
1. Walk the module, collect every `fir.RecordType` that's the parent
   of some other `RecordType` via `EXTENDS` (an "extended" type set).
2. For every `hlfir.declare` whose type carries `fir.class<T>` where
   `T` is NOT in the extended set:
   - Rewrite the type from `fir.class<T>` to `fir.box<T>` (or plain
     `T` for non-allocatable / non-pointer dummies).
   - Update all uses transitively (`fir.embox` / `fir.rebox` / load /
     store).
3. After this pass, `hlfir-flatten-structs` sees a plain TYPE dummy
   and the existing flatten machinery applies.
4. Surviving `fir.dispatch` / `fir.select_type` after both
   `polymorphic-op-conversion` AND this demotion are genuinely
   runtime-polymorphic and continue to be rejected by
   `hlfir-reject-polymorphism`.

**Gate.** The pass MUST refuse to demote if the function body contains
ANY `fir.dispatch` or `fir.select_type` whose target type is the
candidate `T` — those are real runtime polymorphism uses regardless of
the extension set.

**Diagnostic.** Each demoted declare carries a metadata note ("Demoted
CLASS(T) to TYPE(T) — T has no extensions in this program"). When
someone later adds an extension and the demotion silently stops being
safe, the diagnostic shows up at build time.

**Effort.** ~80 LoC for the pass + ~30 LoC for the extended-set
computation + 3 tests:
- monomorphic CLASS dummy → flattens correctly (positive).
- CLASS dummy where T has extensions but no dispatch → still
  monomorphic at this call site, demote safely.
- CLASS dummy with `obj%method()` dispatch → stays rejected.

**Acceptance.** Both currently-xfailed OOP tests flip to passing.

---

## P1.K — Cycle-detected demand-driven flatten for circular types [NEW]

**Insight.** Self-referential members in Fortran MUST be `POINTER` (or
`ALLOCATABLE`) — the language requires it because non-pointer
recursion would have infinite size. That same pointer breaks the
recursion at flatten time too.

**The xfailed test (`type_test::test_fortran_frontend_circular_type`)
has the circular pointer assignments commented out** and only uses
`s%w(1,1,1)` + `b(1)%x` — both NON-circular paths. Steps 1+2 below
flip this xfail.

**Approach.**
1. **Cycle detection in `collectFlatLeaves`.** Add a `seenTypes`
   `DenseSet<Type>` parameter; on entry to a record type, insert it
   and bail (treat as opaque leaf) if already present. On exit, erase.
   Today's recursion at
   [FlattenStructs.cpp:499-523](dace/frontend/hlfir/passes/FlattenStructs.cpp#L499-L523)
   would otherwise infinite-loop (or hit `kFlattenMaxDepth` and bail
   the entire flatten).
2. **Pointer-to-RecordType as opaque leaf.** Extend
   `isFlatMemberType` to accept `fir.box<fir.ptr<fir.RecordType>>` as
   an opaque leaf when the record is on the recursion path. Synthesise
   a flat companion that's just a pointer descriptor — no per-field
   flattening of the pointee.

**Out of scope.** Step 3 (use-driven flatten through cycles) — only
needed if a real kernel actually dereferences self-referential pointers
to read fields through them. Defer until forced.

**Effort.** ~50 LoC + new test mirroring the existing circular fixture
but actually buildable.

---

## P4 — Truly out of scope

### P4.runtime-polymorphic-dispatch
Genuine `fir.dispatch` or `fir.select_type` whose target depends on
runtime type — needs per-target codegen + runtime type info at the
SDFG level. Architectural change, not a pass. Stays rejected.

### P4.use-driven-cyclic-flatten
Kernels that actually dereference self-referential pointers
(`s%b%a%w` traversing the cycle). Step 3 of P1.K above. Needs
use-driven flatten instead of type-driven; substantial refactor.
Stays out unless forced.

---

## Recommended order

1. **P0.1 Phase F** — investigate IR + fix (the velocity blocker).
2. **P0.2** — re-probe velocity, fix whatever surfaces next.
3. **P0.3** — write the end-to-end velocity_tendencies numerical test.
4. **P1.A + P1.B** — both small, both flip currently-xfailed tests.
5. **P2** items only as forced by deeper ICON probes.
6. **P3** deferred until P0/P1/P2 close.
7. **P4** stays xfailed.

---

## Reference: SOTA SDFG inspection results

| Metric | Pre-Phase-H+I | Post (current SOTA) |
|---|---|---|
| `_allocated` tracker arrays | 30 (all dead) | 0 |
| `post_*_allocated_<n>` orphan states | 34 | 0 |
| `s_442` `p_diag_max_vcfl_dyn` access nodes | 1 (cycle) | 2 (input + output) |
| Build outcome | partial-save, fails at arglist | same; same `KeyError: 'p_int'` |
| Total arrays in SDFG | 174 | 144 |
| Sweep | 553 / 13 xF / 0 F | 553 / 13 xF / 0 F |

The SOTA SDFG is **structurally clean** — body-level dead weight is gone, the
read-then-writeback cycle is resolved. Phase F is what's left to make
`arglist()` succeed, and after that the build should finalise.

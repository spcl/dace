# ICON pointer / allocatable support — revised plan (2026-05-02)

## Where we are

Sweep at **553 passed / 13 xfailed / 0 failed** after `hlfir-lift-reduction-operands`
landed. The 13 xfailed cases bucket as:

| Bucket | Tests | Status |
|---|---|---|
| Module-level type with pointer member | `global_test::global` | **Phase A** |
| Plain pointer dummy arg | `nested_array_test::pointer_args_2` | **Phase B** |
| Nested AoS + allocatable member | `type_test::type_arg` | Phase C (deferred) |
| Nested allocatable scalar | `type_array_test::type3_array` | Phase D (deferred) |
| ALLOCATABLE + INTERFACE | `intrinsic_bound_test` | Phase E (separate machinery) |
| Mutually-recursive types | `type_test::circular_type` | out of scope (per memory) |
| CLASS polymorphic | `elemental_test::elemental_ecrad_*` | out of scope (per memory) |

## Phase 0 — velocity_tendencies probe result

Drove the full distilled `velocity_tendencies` (12 subroutines, includes
`rot_vertex_ri`, `cells2verts_scalar_ri`, `velocity_tendencies` itself,
all surrounding modules, ~655 LoC) through `SDFGBuilder.build()`. The
SDFG body builds successfully — past the previous MAXVAL-as-operand
blocker. New blocker is at `_attach_frozen_signature` →
`sdfg.arglist()`:

```
KeyError: 'p_int'
```

bare `p_int` appears as a **free symbol** of the SDFG. Searched the
saved partial SDFG for the source: one tasklet expression in the
`zeta` computation reads

```
_out_zeta = ((((((_in_vec_e_0 * p_int) + (_in_vec_e_1 * p_int))
                + (_in_vec_e_2 * p_int)) + (_in_vec_e_3 * p_int))
              + (_in_vec_e_4 * p_int)) + (_in_vec_e_5 * p_int))
```

This is the inlined body of `rot_vertex_ri`'s

```
rot_vec(jv, jk, jb) = vec_e(...) * ptr_int % geofac_rot(jv, 1, jb)
                    + ... + vec_e(...) * ptr_int % geofac_rot(jv, 6, jb)
```

After inline + flatten, the `vec_e(...)` references are correctly
flattened (`_in_vec_e_0` through `_in_vec_e_5` matching the six
indirect indices), but `ptr_int % geofac_rot(jv, k, jb)` collapses to
bare `p_int` instead of the synthesised `p_int_geofac_rot` companion.

**Inferred root cause.** `geofac_rot` (and `cells_aw_verts`) of
`t_int_state` is referenced **only** inside the inlined `rot_vertex_ri`
body — never directly in the caller `velocity_tendencies`. The
alias-prefix chain rewrite (extracted in Stage 2 to a helper) walks
caller-side designate chains but isn't reached for members consumed
exclusively through the inlined-callee body. Other members
(`c_lin_e`, `e_bln_c_s`, `geofac_grdiv`, `geofac_n2s`,
`rbf_vec_coeff_e`) all have direct caller references AND inlined-body
references; those flatten correctly to `_in_p_int_<member>` in the
SDFG. The two members touched only by inlined-body references stay
unrewritten.

**Secondary observation.** All seven flattened ALLOCATABLE members of
`t_int_state` (`p_int_c_lin_e`, …) end up with shape `(1,)` in the SDFG
arrays, while the static-sized members of `t_patch` get proper
`*_d0/d1/d2` runtime symbols. The `(1,)` placeholder is fine for the
SDFG body but means the runtime descriptor → symbol extraction isn't
firing for these members. Whether this is a correctness bug or just a
cosmetic-shape issue depends on how `arglist()` and the bindings layer
consume them.

## Revised phases

### Phase A — Module-level type with pointer / allocatable member
**Test:** `global_test::test_fortran_frontend_global` (xfailed).
**Shape:** `type(t) :: ptr_patch` at function scope where `t` has
`double precision, pointer :: w(:,:,:)`. Caller does
`ptr_patch%w(:,:,:) = 5.5`, passes `ptr_patch%w` by reference, reads
`ptr_patch%w(3,3,3)`.
**Approach.** Phase 5b dummy-arg synthesis already handles the
allocatable/pointer member case for dummies. The local-allocation walk
added in Phase 1 only handles flat scalar/array members; lift the
guard so allocatable/pointer members on local instances flow through
the same Phase 5b synth (`fir.alloca<box<heap|ptr<array<?xT>>>>` +
declare with the matching `fortran_attrs`).
**Files.** `dace/frontend/hlfir/passes/FlattenStructs.cpp` —
local-allocation per-member loop, single new branch reusing existing
synthesis path.
**New tests.** Drop the xfail decorator on `global_test::global` if it
flips green; add a `tests/hlfir/local_pointer_member_test.py` mirror
that exercises the pattern with f2py reference.

### Phase B — Plain pointer dummy arg as runtime-shape array
**Test:** `nested_array_test::test_fortran_frontend_nested_array_access_pointer_args_2`.
**Insight from old `dace/frontend/fortran/`.** The old frontend's
`pointer_stmt` is a no-op pass-through; `POINTER` on a regular
variable declaration is treated as decoration. Inside an SDFG kernel,
the pointer attribute on a dummy carries no semantics beyond "this is
an array I can read and write." The HLFIR bridge currently rejects
`fir.box<ptr<array<?xT>>>` dummies; the simplest correct lowering is
to peel like an allocatable.
**Approach.** No new MLIR pass. In `extract_vars.cpp`, the pointer
peel guard at line 624–629 currently restricts peeling to declares
whose results are explicitly classified. Extend the same peel to
plain `fir.box<ptr<array<?xT>>>` dummy args. Treat `fortran_attrs =
pointer` and absent-attrs descriptor-only dummies the same as
`fortran_attrs = allocatable` for shape-symbol generation and
classification. Drop any reject-paths that fire on plain pointer
dummies.
**Files.** `dace/frontend/hlfir/bridge/extract_vars.cpp` — relax the
classifier; possibly one peel-step in `peelToElement`.
**New tests.** Drop xfail on `pointer_args_2`; add unit test for
`integer, pointer, intent(inout) :: a(:)` dummy with both read and
write through the dummy.

### Phase F — Alias resolution into inlined-callee bodies for partially-used members [NEW]
**Source of finding:** Phase 0 velocity_tendencies probe.
**Symptom.** A struct member referenced **only** through an inlined
subroutine's parameter alias (e.g. `ptr_int%geofac_rot` in
`rot_vertex_ri` when `ptr_int` aliases `p_int`) collapses to bare
`p_int` in the resulting tasklet expression instead of resolving to
the synthesised `p_int_geofac_rot` flat companion.
**Hypothesis.** `traceAliasPrefixToDecl` / `rewriteChainsRootedAt`
walk caller-side designate chains; they don't see designate chains
that originate inside an inlined-callee body whose parameter aliases
the caller's struct argument. The chain rooting therefore misses
members consumed exclusively through the inlined body.
**Approach.** Extend the alias-prefix chain walk to follow the
inliner's `BlockArgument → SSA value` substitution back to the caller
declare, so designate chains rooted at the inlined-callee's parameter
alias get rewritten the same as caller-side chains.
**Files.** `dace/frontend/hlfir/passes/FlattenStructs.cpp` — the
alias-chain rewrite helper extracted in Stage 2.
**Verification.** velocity_tendencies probe passes through `arglist()`
without `KeyError: 'p_int'`. Add a focused unit test in
`tests/hlfir/inlined_callee_member_alias_test.py` with a 2-routine
fixture: caller passes `s` to `inner(t)`, `inner` reads `t%onlyhere`
that the caller never touches directly. Reference via f2py.

### Phase G — Runtime-shape symbols for flattened ALLOCATABLE struct members [NEW, lower priority]
**Source of finding:** Phase 0 probe — `p_int_c_lin_e` and friends end
up with shape `(1,)` instead of `(p_int_c_lin_e_d0, ..._d1, ..._d2)`.
**Open question.** Is this cosmetic (the SDFG body still works
because each access reads a scalar memlet, and shape symbols are
synthesised at call-site bindings) or a real bug for arglist /
binding emission?
**Approach.** Spike: inspect the bindings emitter's behaviour on a
synthesised flattened ALLOCATABLE companion. If shape `(1,)` causes
the call wrapper to misclassify the buffer, fix the descriptor →
shape-symbol extraction to fire on members synthesised under Phase 5b
the same way it fires for direct ALLOCATABLE dummies. If only
cosmetic, document and defer.

### Phase C / D / E — deferred unchanged
- C (`type_arg`): nested AoS-allocatable + nested pointer member.
  Compounds Phase 5c-A and 5b. Real ICON kernels rarely use
  array-of-struct-with-pointer; deprioritise until a probe forces it.
- D (`type3_array`): nested allocatable scalar. Same reasoning.
- E (`intrinsic_bound + ALLOCATABLE`): orthogonal — likely a
  call-site / signature-resolution issue, not alloc-storage. Park
  until Phase F unblocks the velocity probe.

## Recommended order
1. **Phase F** first — it's the actual blocker for the velocity
   probe and yields the next concrete failure on a real ICON
   workload.
2. **Phase A + Phase B** in parallel — both small, both unlock
   currently-xfailed tests, both reuse existing synthesis paths.
3. **Phase G** as a spike after F unblocks the probe, only if shape
   `(1,)` causes a downstream failure.
4. **Phase H + Phase I** (below) after F lands — both are SDFG-shape
   correctness issues independent of the pointer/alloc work, but
   surfaced by the velocity probe inspection.

### Phase H — Drop unconditional `_allocated` tracking on non-allocatable struct members [NEW]
**Source of finding:** velocity probe SDFG inspection.
**Symptom.** Every flattened static-shape member of `t_patch` (30 of
them — `cell_idx`, `edge_blk`, `f_e`, etc.) gets:
* a `<member>_allocated : int32` scalar in `sdfg.arrays`, default 0,
* a `post_<member>_allocated_<n>` empty state with no in/out edges.

The members aren't ALLOCATABLE — `t_patch` declares them as plain
fixed-shape arrays (`integer :: cell_idx(nproma, nblks, 2)`). The
allocate-tracking scalar + post-allocate state is dead weight for any
member without a `fortran_attrs = allocatable` (or pointer) on the
synthesised companion declare.
**Approach.** Gate the allocate-tracking-scalar emission on the
companion declare carrying `fortran_attrs = allocatable | pointer`.
Static-shape members skip the tracker entirely; their data is
allocated at the SDFG arglist and never has a "not yet allocated"
state.
**Files.** `dace/frontend/hlfir/bridge/extract_vars.cpp` (or wherever
allocate-tracking scalars are synthesised) — single conditional gate.
**Verification.** Re-run the velocity probe; expect 30 fewer
`<member>_allocated` scalars + 30 fewer `post_<member>_allocated_<n>`
orphan states. Sweep stays at 553 P / 13 xF.

### Phase I — Read-then-writeback to same struct member needs distinct access nodes [NEW]
**Source of finding:** velocity probe SDFG inspection (state
`s_476`).
**Symptom.** Two-statement Fortran source
```
max_vcfl_dyn = MAX(p_diag % max_vcfl_dyn,
                   MAXVAL(vcflmax(i_startblk : i_endblk)))
p_diag % max_vcfl_dyn = max_vcfl_dyn
```
collapses into one SDFG state where the SINGLE
`p_diag_max_vcfl_dyn` access node is BOTH:
* read by `set_max_vcfl_dyn` (the MAX tasklet's `_in_` edge), AND
* written by `set_p_diag_max_vcfl_dyn` (the writeback tasklet's
  `_out_` edge).

This forms a cycle on the access node — the same data container
appears as both source and sink within one state with no snapshot
between, which is invalid SDFG topology.

The cycle is an interaction with the new
`hlfir-lift-reduction-operands` pass: the lifted
`_QQred_lift_0 = MAXVAL(...)` runs first, then the consuming
two-statement read/writeback emits into a single state where the
bridge re-uses the same access node for both endpoints.

**Approach.** When buildAssignNode encounters a writeback whose RHS
reads from the SAME data container as the LHS (directly or through
struct-member flattening), emit two distinct access nodes for that
container — an input read-snapshot used by all reads in the state,
and an output write-target consumed only by the writeback tasklet.
Topologically these become a chain: `[in] → tasklet → [tmp] →
writeback → [out]`. Or split into two states.
**Files.** `dace/frontend/hlfir/bridge/extract_ast.cpp` (assign-node
emission and access-node bookkeeping).
**Verification.** Re-run the velocity probe; the s_476 cycle
disappears. Add a focused unit test in
`tests/hlfir/read_then_writeback_test.py`:
```fortran
subroutine kernel(x, arr, n)
  ...
  x = max(x, maxval(arr(1:n)))   ! read x + write x
end subroutine
```
SDFG should have two `x` access nodes (input + output) connected
through the MAX tasklet.

## What is **not** carried over from the old frontend

The old `dace/frontend/fortran/` defers ALLOCATABLE arrays at
declare-time and creates the SDFG array at `ALLOCATE(...)` with
literal sizes pulled from the allocate-call shape spec. This model is
**incompatible with ICON**: ICON allocates struct members in the
driver (`ALLOCATE(p_diag%w(nproma, nlev, nblks))`) and the kernel
never sees the ALLOCATE — only the live `fir.box` descriptor on the
incoming struct. The bridge's descriptor-driven runtime-shape model
is the right architecture for ICON; the old frontend's approach was
borrowed intact only for **Phase B** (treat the pointer attribute as
decoration on a regular array).

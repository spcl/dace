# cloudsc `zqe_5` miscompile — root cause and fixes

**Symptom.** The cloudsc_variants measurement job died in Phase B: the sequential
reference build failed with `error: use of undeclared identifier 'zqe_5'`
(generated `cloudsc_ref_seq.cpp`: decl at a state-scope brace that closes before
the read in the next state's brace). Both `legacy` AND `experimental_readable`
CPU codegen emit the same broken scoping; `make_sequential` is irrelevant (the
parallel form fails identically — verified byte-identical around the fault).

**Malformed SDFG shape.** In nested SDFG `loop_body` (created by LoopToMap),
tasklet `__min2` in state `slice_zqsice_1124` carries the code
`__out = min(zqe_5, __in_b)` where `zqe_5` is **raw code text** — not a
connector, no memlet, no AccessNode. The only real AccessNode of `zqe_5` is the
write in sibling state `BinOp_1119`. The final cache holds 3 such dangling
tasklets (`zqe_5`, `zqe_4`, `max_0_0_expr_1`); the other two compiled by luck
of hoisting.

**Which pass causes it** (bisected via per-stage snapshots, job 4227883, +
structural detector in `cloudsc_bisect_check.py`):

1. Stages 03–13: `zqe` is a **symbol** (interstate-edge-assigned). The
   `__min2` code read `min(zqe, __in_b)` is *legal* — symbols need no
   connectors. All 14 pre-final snapshots are structurally clean.
2. Stage 13 (`loop_to_map`) nests the loop body; the edge
   `BinOp_1119 -> slice_zqsice_1124` assigns `zqe = zqe_5`, where `zqe_5` is a
   **mutated transient Scalar of the nested SDFG** (written in `BinOp_1119`).
3. Stage 14: `LoopToScan` internally invokes **`SymbolPropagation`**
   (loop_to_scan.py:403–405), which folds `zqe -> zqe_5` into the tasklet code
   and deletes the symbol — producing the connector-less container read.
   Reproduced standalone: `SymbolPropagation().apply_pass()` on the stage-13
   snapshot performs the fold.

**Why SymbolPropagation's guard failed.** The mutated-scalar filter
(symbol_propagation.py) exists to refuse exactly this fold, but `_get_in_syms`
received the **top-level** SDFG for every block while `all_cfg_blks` spans all
nested SDFGs: `scalars('zqe_5', root.arrays)` is empty (`zqe_5` lives in
`loop_body.arrays`), so the guard no-opped for every nested-SDFG assignment
(the View filter had the same wrong-level lookup).

**Fixes (both applied):**

1. `dace/codegen/targets/framecode.py` — `determine_allocation_lifetime` now
   counts **code-only container uses** (CodeNode.free_symbols ∩ arrays) as
   access instances and in the Scope-lifetime state scan. Uses in two states →
   declaration hoists to function/SDFG scope. Verified: synthetic repro
   compiles under Sequential AND CPU_Multicore; well-formed SDFGs produce
   byte-identical output (no-regression diff clean); the real cached cloudsc
   SDFG now scopes `zqe_5` correctly under BOTH codegens. Regression test:
   `tests/codegen/allocation_lifetime_test.py::test_code_only_container_read_scope`.
2. `dace/transformation/passes/symbol_propagation.py` — the mutated-scalar and
   View filters now consult the **owning** SDFG (`cfg_blk.sdfg`) instead of the
   top-level one. Verified: standalone SymbolPropagation on the stage-13
   snapshot refuses the fold (code stays `min(zqe, __in_b)`, `zqe = zqe_5`
   assignment kept); a rebuilt cache would no longer contain the dangling shape.

The existing `cache/cloudsc_parallel.sdfgz` (built pre-fix, contains the folded
shape) stays valid: fix 1 makes codegen emit correct scoping for it. Fix 2
prevents the shape from being produced on the next cache rebuild.

**Repro/attribution tooling** (this directory): `cloudsc_bisect_chain.py`
(compute-node job, per-stage snapshots — SDFG regen never on login),
`slurm_cloudsc_bisect.sh`, `cloudsc_bisect_check.py` (login-side: structural
dangling-read detector + compile check, binary search or `--linear`).

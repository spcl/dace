# External SDFG + external compile

Design plan. NOT implemented. Model 2 (standalone SDFGs), confirmed codegen-agnostic.

## Goal

Split a program into separate translation units, one per top-level control node, each compiled to its
own object file and linked in-binary. Works on old+new CPU and old+new GPU codegen with **zero codegen
changes** — each piece is a standalone SDFG run through whatever codegen it already uses.

## Heuristic

- Every TOP-LEVEL `if` (`ConditionalBlock`) and `for` (`LoopRegion`) in `root.nodes()` → its own
  external SDFG.
- Each GPU kernel → its own external SDFG (→ its own `.cu` for free, since a standalone SDFG is one
  codegen invocation = one `.cu`).
- DEPTH 1: an external SDFG is NOT itself split further. No external SDFG inside an external SDFG.

## Model 2 = standalone SDFG (NOT nested-SDFG outline)

Why not model 1 (extend `split_nsdfg_translation_units` / `no_inline` nested SDFGs): the GPU codegen
(`cuda.py:161`) emits ONE `.cu` per SDFG regardless of `no_inline`, so per-kernel GPU TUs there need a
GPU-codegen change — forbidden (`new-gpu-codegen-dev` = never touch).

Model 2: each top-level `if`/`for`/kernel is lifted to a SEPARATE top-level SDFG. Parent keeps a call
site. Each standalone SDFG is codegen'd on its own → own CodeObject(s) → own object file. Parent calls
it via a forward-declared, `DACE_HIDDEN`, external-linkage in-binary function (same call contract
`split_nsdfg` already emits; NOT `dlopen`, NOT `ext_sdfg_path`).

## Architecture

1. **Extract pass** (new, codegen-agnostic). Walk `root.nodes()`; for each top-level `ConditionalBlock`
   / `LoopRegion` / GPU-kernel scope, move its body into a fresh `SDFG`, replace the site in the parent
   with a call node carrying the argument list (reuse `no_inline` nested-SDFG arg marshaling from
   `OutlineTopLevelNests` for the boundary, but the target is a standalone SDFG, not a nested one).
   Record parent→child arg mapping. Cap at depth 1.
2. **Orchestration** in `codegen.py generate_code`: after the extract pass, run codegen on each child
   SDFG independently, collect all CodeObjects (parent + children). Children keep their own target
   (CPU child → `.cpp`, GPU child → `.cu`). No codegen file changes — just call `generate_code` per SDFG.
3. **Call wiring**: parent emits a forward decl + call for each child's entry function. Reuse the
   in-binary contract from `split_nsdfg` (`DACE_HIDDEN`, external linkage, static-linked).
4. **Build (ONE project, flat source list)** — `compiler.py` + `codegen/CMakeLists.txt`. There is
   exactly ONE CMake project. Its source list is a FLAT collection of every source from the parent and
   every external SDFG (recursively), CPU `.cpp` and GPU `.cu` together. No per-external-SDFG
   sub-project, no intermediate `.a`/`.so` — each source compiles to an object linked into the single
   library. `DACE_FILES` is already exactly this shape; the only new work is the RECURSIVE COLLECTION
   that feeds it: walk parent → all external SDFGs → their CodeObjects, flatten to one list.
   - **Native build too** (`build_mode: native`, the no-CMake path — see
     `dace/codegen/compiler.py`): its source-gathering must run the SAME recursive walk over external
     SDFGs, so both the CMake and native builders see the identical flat CPU+GPU file set. Put the
     collection in one shared helper both call — do not duplicate the walk.

## Reuse (do not rebuild)

- `OutlineTopLevelNests` (`transformation/passes/outline_top_level_nests.py`) — boundary/arg marshaling
  and the top-level walk. Adapt its nest→arg logic; retarget to standalone SDFG.
- `split_nsdfg_translation_units` emission (`cpu.py`, `target_type='nsdfg'`) — the in-binary call
  contract + per-file routing. Model 2's parent call is the same shape.
- `CMakeLists.txt` `DACE_FILES` foreach — already per-object, already one `.so`, no per-nest lib.

## Toggle

New `codegen_params` key (e.g. `external_translation_units: off|on`, default off → byte-identical).
Gate the extract pass + orchestration on it. Applies to both generators.

## Test plan (all 4 codegens)

- Extract: 2 top-level fors + 1 top-level if → 3 child SDFGs + parent; depth-1 (a for inside a child is
  NOT re-extracted). Assert child SDFG count + parent call sites.
- CPU old+new: compile + run, bit-exact vs single-TU. Assert N object files in `.dacecache/.../src`.
- GPU old+new: one `.cu` per kernel-child; compile + run on RTX 4050, allclose vs single-TU.
- CMake: assert each child object in `DACE_FILES`, no `.a`/per-nest lib target.

## Open items

- Arg marshaling for a GPU-kernel child: the kernel's inputs/outputs (GPU_Global pointers, symbols)
  become the child SDFG signature. Confirm `apply_gpu_transformations` ordering vs the extract pass.
- Symbol/free-symbol threading across the call boundary (child needs parent's defined symbols as args).
- Streams / persistent state: a child touching a persistent buffer needs the state struct passed
  through, like `split_nsdfg`'s `<name>_state_t`.

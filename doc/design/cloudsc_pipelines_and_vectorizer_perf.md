# CloudSC pipelines + vectorizer performance (handoff, 2026-09-28)

## Goal

canon-cpu, canon-gpu, vectorize-cpu, vectorize-gpu work on the CloudSC SDFG; one checkpointed recipe
reproduces all four and runs them against the un-transformed reference.

## Where

- Recipe: `tests/corpus/cloudsc/reproduce.py`. Every step saves `<out>/<step>.sdfgz` and reloads it.
- How to run: `tests/corpus/cloudsc/QUICKSTART.md`.
- Test: `tests/corpus/cloudsc/cloudsc_target_pipelines_test.py` (five legs, same recipe).
- CI: host legs in `integration-tests-ci.yml` (`cloudsc-host-pipelines`); GPU legs added to
  `ci/cscs_gpu.yml` (`test_cscs_gh200_cloudsc_pipelines`). CSCS job not seen running on PR #2475 yet.

## Status

| Step      | Numeric check vs reference                  | Build time (local, load ~35) |
|-----------|---------------------------------------------|------------------------------|
| canon_cpu | OK, bit-exact                               | 403 s                        |
| canon_gpu | OK on device, worst abs 9.1e-13             | 430 s                        |
| vec_cpu   | OK, bit-exact (local + CI)                  | >2 h -> 1078 s               |
| vec_gpu   | was MISMATCH; fixed, re-check: see Open     | died -> 2243 s               |

## Vectorizer contract (changed)

- Input = canonicalized or `ParallelizePipeline` output. Vectorizer re-runs no recipe pass:
  no `ParallelizeLoops`/LoopToMap, no IV substitution, no `StructuralCleanup`.
- Loop left in a map body -> body stays un-tiled.
- `ParallelizePipeline` now runs `IvSubstitutionFissionFixpoint` + `StructuralCleanup` itself.
- `loop_to_map_permissive` knob removed; a caller that can vouch for a scatter runs
  `ParallelizeLoops(permissive=True)` first.
- `WCRToAugAssign` kept: canonical reductions are in-body WCR, tile emitters need explicit RMW.

## Speed-ups (all verified by the vectorization suite)

- State fusion fixpoint: scoped to map-body SDFGs; per-state side-effect verdict cached
  (`StateFusionExtended.side_effect_cache`). Was ~40% of vectorize.
- Symbol-definition scan (`build_symbol_definition_map`) once per pass, not per map/access:
  select-then-apply in StrideMap, IterationMask, `tile_body_nsdfgs`; per-phase maps in
  InsertTileLoadStore, per-body map in WidenAccesses. Was ~60%.
- `DemoteDataReadingInterstateSymbols`: `free_symbols` computed once per SDFG
  (`demote_symbol_to_scalar(..., free_symbols=)`). Was ~13%.
- No `reset_cfg_list` in vectorizer passes (add/remove keep it current; harness asserts it).
  Fixpoints read `cfg_list` instead of re-walking all regions.

## Bugs fixed on the way

- `6d0d130dd` (ExpandNestedSDFGInputs stride scaling) double-scaled canonical strided kernels
  (`src[2*i]` -> `src[4*i]`). Scale only a compact view (inner stride = outer stride * step);
  steps taken before rename (in-place arrays).
- `FuseBranchedTailRemainder`: connector name != outer array name (`paph` fed by `gpu_paph`).
- `detect_cudatest`: build like a generated TU (`dace.h` first, C++20).
- vec_gpu MISMATCH root cause: GPU canonical form lowers a per-row sum (`acc += a[jl, jm]`, CloudSC
  fluxes) to a `Reduce` over a row view `x[0:M]` inside a nested SDFG. `ExpandNestedSDFGInputs` widened
  `x` to `a[jl, 0:M]` but (1) left `Reduce.axes=[0]` -> reduced the length-1 dim = copy, (2) left the
  result copy `tmp -> y` (no `other_subset`) writing `s[0]`. Fix: remap `Reduce.axes` onto surviving
  dims; rewrite the far side of copies into/out of the widened array. Repro:
  `tests/passes/vectorization/gpu_row_reduction_cudatest.py` (fails before, passes after).

## Open

- vec_gpu full-CloudSC numeric re-check after the fix: log `~/.cache/cloudsc_repro/vec_gpu.log`.
- Remaining hotspots (profile `vec_cpu_r5`): map gate scan ~12% (one per pass; sharing across passes
  needs per-pass invalidation), `ExpandNestedSDFGInputs` fixpoint ~10%, `nest_state_subgraph` ~7%.
- Local box rebooted 3x under load (other sessions' CUDA builds); run long jobs as
  `systemd-run --user` services, `pytest -n2`.

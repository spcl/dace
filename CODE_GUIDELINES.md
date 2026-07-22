# Code Guidelines — Canonicalization Pipeline work

Binding for all implementation under the SDFG Canonicalization Pipeline
(see `CanonicalizationPipeline.md`).

## Conventions

- **`CONTRIBUTING.md` is authoritative.** Read `/home/primrose/Work/dace/
  CONTRIBUTING.md` at the start of each work session; its conventions are
  binding and override this file on any conflict.
- **Docstrings required** for every file, class, and function. Use Sphinx field
  format: `:param x:`, `:returns:`, `:raises:`.
- **Type hints**: annotate parameters and non-`None` return types only; omit
  `-> None`.
- **Module-level constants** are `ALL_CAPS` (prefix `_` if private).
- **Comments describe the current code only** — no banner comments, no
  "what changed from the previous version" narration.
- **Semantically loaded new names** (`Untile`, `CanonicalizePass`) get a
  one-line explanation that disambiguates them from existing framework concepts
  (`MapTiling`, `SimplifyPass`).

## Reuse over new code

- Only `Untile` and `CanonicalizePass` are new.
- Every other stage is an existing `yakup/dev` pass — used as-is or **hardened
  in place**. Never fork a parallel copy of an existing pass.
- `SimplifyInductionVariables` builds on existing `loop_analysis` primitives;
  `LoopInvariantCodeMotion` builds on existing `sdfg.utils` scope/used-data
  helpers. Do not reinvent loop or dataflow analysis.

## Process

- No overengineering: prefer inline logic co-located with existing framework
  concepts over speculative public helpers/contracts. Hypothetical performance
  wins need measurement.
- Probes and `/tmp` snippets used while developing land as proper pytest tests,
  never throwaway scripts.
- Never edit a test's assertions or tolerances to make a failing test pass; fix
  the pass instead. Never remove an `xfail` / flip `strict=True→False` to go
  green.
- Commits: no `Co-Authored-By` Claude trailer.
- Refactor scope stays within files the work touches; do not act on agent
  "reuse this existing util" suggestions blindly.

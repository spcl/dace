# DaCe formal semantics (Lean 4)

This directory holds a Lean 4 formalization of DaCe's intermediate
representations:

- `Dace/Core`: shared data model (symbolic expressions, subsets, data
  descriptors, memlets, memory, access logs and their parallel merge, tasklets).
- `Dace/SDFG`: syntax, well-formedness, semantics, interpreter and JSON import
  of stateful dataflow multigraphs.
- `Dace/STree`: the same for schedule trees.
- `Dace/Equiv`: observational equivalence and the verified pair checker
  (`lake exe dace-equiv`).
- `Dace/Decisions.md`: the semantic decision register (every ambiguity between
  documentation, code generator and model, with the chosen resolution).

## Building

Install [elan](https://github.com/leanprover/elan); the toolchain is pinned in
`lean-toolchain`. Then:

```bash
lake exe cache get   # prebuilt Mathlib, once per toolchain bump
lake build
scripts/check_axioms.sh   # build + sorry/axiom/import-isolation checks
```

## Conventions

- Only `Dace/Prelude.lean` imports Mathlib.
- No `sorry` in committed code; `scripts/check_axioms.sh` is run in CI.
- Every modeling choice that is not forced by DaCe's documentation is recorded
  in `Dace/Decisions.md` and referenced from the source as `Decisions.md#Dn`.

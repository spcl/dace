# DaCe formal semantics (Lean 4)

This directory holds a Lean 4 formalization of DaCe's intermediate
representations:

- `Dace/Core`: shared data model (symbolic expressions, subsets, data
  descriptors, memlets, memory, access logs and their parallel merge, tasklets).
- `Dace/SDFG`: syntax, well-formedness, semantics, interpreter and JSON import
  of stateful dataflow multigraphs.
- `Dace/STree`: the same for schedule trees (`Syntax` with DaCe's
  `as_string` pretty-printer, `Json` importer for `.stree.json`, `WF`,
  relational `Semantics`); `tests/ImportStreeCorpus.lean` imports the
  corpus trees, checks them and compares the pretty-printer with DaCe.
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

## Differential testing

`scripts/difftest.py` cross-checks the Lean interpreter against DaCe on the
corpus under `tests/corpus`:

- `--record` loads each `<base>[.variant].sdfg.json` with `dace.SDFG.from_file`,
  takes inputs from the base program's factory in `scripts/corpus_programs.py`,
  compiles and runs the SDFG, and writes `<name>.inputs.json` and
  `<name>.expected.json`. A few programs (axpy, matmul, reduce_rows) are also
  cross-checked against a NumPy reference.
- `--check` (alias `--lean`) runs `lake exe dace-interp sdfg <sdfg.json>
  <inputs.json>` per file, parses its stdout (`{"containers": …}` on success,
  `{"error": …}` on a semantic error) and compares to `<name>.expected.json`
  with dtype-aware tolerances (float32 rtol 1e-5, float64 1e-9, exact for
  int/bool). Exit 2 from the tool means "not implemented" → SKIP; a missing
  `lake` skips everything. The run exits non-zero if any file FAILs.

With no mode flag it records when any expected file is missing, else checks.
Options: `--only name…`, `--corpus DIR`, `--tol-scale`.

The input/expected JSON format (the Lean reader is written against it):

```json
{
  "program": "matmul",
  "sdfg": "matmul.tiled.sdfg.json",
  "symbols": {"M": 4, "K": 6, "N": 5},
  "containers": {
    "A": {"dtype": "float64", "shape": [4, 6], "data": [ "…row-major flat…" ]},
    "a": {"dtype": "float64", "shape": [], "data": [2.0]}
  }
}
```

`containers` holds every non-transient array plus any `__return*`. DaCe `Scalar`
descriptors use `shape: []` with a one-element `data`; dtype strings are NumPy
names. `symbols` is the SDFG's free symbols plus explicit symbol args.
`expected.json` has the same shape but only `program`/`sdfg`/`containers`.

## Schedule tree JSON

Alongside each `<name>.sdfg.json`, `scripts/dump_corpus.py` writes the schedule
tree of the same SDFG as `<name>.stree.json` (machine-readable) and
`<name>.stree.txt` (`as_string()`, for human review and Lean pretty-printer
checks). The JSON is produced by `dace.sdfg.analysis.schedule_tree.serialize`
(`to_json`/`from_json`/`save`/`load`).

Every node is `{"type": "<ClassName>", …fields…}`; scopes carry a `"children"`
list. Dataclass fields are serialized by name, and embedded DaCe objects
(memlets, data descriptors, `Map`/`Consume`/`Tasklet`/`LibraryNode`, `CodeBlock`,
loop metadata, symbolic expressions and dtypes) use the same encodings as
`.sdfg` files, so the Lean importer can reuse its parsers. The root
(`ScheduleTreeRoot`) additionally carries `name`, `containers` (`{name:
descriptor}`), `symbols` (`{name: dtype}`), `constants`, `arg_names`,
`callback_mapping` and `global`/`init`/`exit_code`. Parent pointers are not
stored; they are restored on load. Files are written with sorted keys, two-space
indentation and a trailing newline. (As with `.sdfg.json`, the per-element
`guid`s are random, so regenerating the corpus is not byte-reproducible.)

`difftest.py --check --ir stree` runs `lake exe dace-interp stree
<name>.stree.json <inputs.json>` and compares against the same
`<name>.expected.json` (the schedule tree of an SDFG must compute the same
outputs); `--ir sdfg` (the default) is unchanged. The Lean schedule-tree
interpreter does not exist yet, so exit 2 → SKIP for now.

## Conventions

- Only `Dace/Prelude.lean` imports Mathlib.
- No `sorry` in committed code; `scripts/check_axioms.sh` is run in CI.
- Every modeling choice that is not forced by DaCe's documentation is recorded
  in `Dace/Decisions.md` and referenced from the source as `Decisions.md#Dn`.

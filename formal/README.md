# DaCe formal semantics (Lean 4)

This directory holds a Lean 4 formalization of DaCe's intermediate
representations:

- `Dace/Core`: shared data model (symbolic expressions, subsets, data
  descriptors, memlets, memory, access logs and their parallel merge, tasklets).
- `Dace/SDFG`: syntax, well-formedness, semantics, interpreter and JSON import
  of stateful dataflow multigraphs.
- `Dace/STree`: the same for schedule trees (`Syntax` with DaCe's
  `as_string` pretty-printer, `Json` importer for `.stree.json`, `WF`,
  relational `Semantics`, the executable `Interp` and its `Soundness`
  proof); `tests/ImportStreeCorpus.lean` imports the
  corpus trees, checks them and compares the pretty-printer with DaCe.
- `Dace/Equiv`: observational equivalence, refinement and congruence, and
  verified checks (`lake exe dace-equiv`): `dace-equiv racefree <sdfg.json>...`
  runs the static race check `raceFree?`, proven sound (`raceFree?_sound`);
  `tests/RaceFreeCorpus.lean` pins its corpus verdicts. Congruence:
  replacing a control-flow block (`Congr.block`) or the body of a map
  (`Congr.scopeBody`, `Congr.mapBody`; decidable side condition
  `sameOutside?`, exercised on every corpus map by
  `tests/ScopeBodyCheck.lean`) by one whose runs it matches refines the SDFG.
  Determinism (`Equiv/Det`): on the fragment `seqPure?` with convex map
  scopes (`scopesOk?`), one normal run of a state makes every run normal
  with the same memory, whatever topological orders the scopes take
  (`StateExec.det`).  Control flow is deterministic (several true interstate
  edges are undefined behavior), so this lifts to programs (`Equiv/DetProg`):
  for a lawful WCR semantics, a program passing `SDFG.det?` (every state in
  the fragment) with a normal outcome has no other outcome, in particular no
  race (`SDFG.Exec.det`, `SDFG.Exec.no_err`); `tests/DetCheck.lean` checks
  the hypotheses on the corpus.
  SDFG / schedule-tree pairs (`Equiv/Corr*`, `lake exe dace-equiv stree
  <sdfg.json> <stree.json>`): the checker `Corr.corr?` is proven sound
  (`Corr.corr?_exec`: every run of the tree matches a run of the SDFG, with
  observationally equal final memory, or the SDFG may be undefined;
  `Corr.corr?_refines`: `Refines t g` for a supported SDFG).
  Structured control flow (`Equiv/CorrCtl`): chains of blocks joined by
  unconditional edges whose assignments become `AssignNode`s, loop regions,
  conditional blocks (`if`/`elif`/`else`), break, continue and return.
  Nested SDFGs (`SegCorr.nestedMap`): a map scope whose body is one nested
  SDFG matches the tree's inlined inner region; the per-call transients are
  related to the tree's names by renaming (`SDFG/Sim.lean`, the simulation
  theorem `SDFGExec.sim` over a name permutation, `Corr.flatCtx`).
  `tests/EquivCorpus.lean` prints the corpus verdicts.
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
corpus under `tests/corpus`.

The corpus is generated, not committed (`tests/corpus/` is git-ignored); CI
regenerates it on every run with a fixed seed. To (re)generate it locally, with
DaCe importable, from the repository root:

```bash
python formal/scripts/dump_corpus.py               # <name>.sdfg.json, .stree.json, .stree.txt
python formal/scripts/difftest.py --record --seed 0  # compiles with DaCe: .inputs.json, .expected.json
cd formal
for t in tests/*.lean; do lake env lean "$t"; done  # Lean corpus tests (fail on any problem)
python scripts/difftest.py --check --ir sdfg
python scripts/difftest.py --check --ir stree
```

The Lean corpus tests read `tests/corpus` relative to `formal/`; set
`DACE_FORMAL_CORPUS=<dir>` to point them elsewhere (`--corpus`/`--out` do the
same for the Python scripts).

- `--record` loads each `<base>[.variant].sdfg.json` with `dace.SDFG.from_file`,
  takes inputs from the base program's factory in `scripts/corpus_programs.py`,
  compiles and runs the SDFG, and writes `<name>.inputs.json` and
  `<name>.expected.json`. Inputs are drawn from NumPy generators seeded per
  program; `--seed S` (default 0) is combined with that per-program seed, and
  seed 0 gives the per-program streams alone, so the same seed always
  reproduces the same inputs. A few programs (axpy, matmul, reduce_rows) are
  also cross-checked against a NumPy reference.
- `--check` (alias `--lean`) runs `lake exe dace-interp sdfg <sdfg.json>
  <inputs.json>` per file, parses its stdout (`{"containers": …}` on success,
  `{"error": …}` on a semantic error) and compares to `<name>.expected.json`
  with dtype-aware tolerances (float32 rtol 1e-5, float64 1e-9, exact for
  int/bool). Exit 2 from the tool means "not implemented" → SKIP, as does an
  `unsupported library node` error (a library node without a model, e.g.
  `matmul_lib`'s `MatMul`); a missing `lake` skips everything. The run exits
  non-zero if any file FAILs.
- Expected errors: a program whose formal semantics is an error although DaCe
  runs it (e.g. `unordered_writes`, a happens-before race, `Decisions.md#D61`)
  is listed in `EXPECTED_ERRORS` of `scripts/corpus_programs.py` as
  `{"<ir>": "<reason>"}`. `--record` copies the entry into
  `<name>.expected.json` as `"expect_error"`; `--check --ir <ir>` then passes
  exactly when `dace-interp` reports an error with that `reason` (the tool
  prints `{"error": …, "kind": …, "reason": …}`, `reason` = `ErrReason.tag`),
  while IRs not listed are compared by value as usual.

With no mode flag it records when any expected file is missing, else checks.
Options: `--only name…`, `--corpus DIR`, `--seed S` (record), `--tol-scale` (check).

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
`expected.json` has the same shape but only `program`/`sdfg`/`containers` (and
`expect_error` for the programs listed in `EXPECTED_ERRORS`).

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
`guid`s are random, so the regenerated SDFG and tree files are not
byte-reproducible; the recorded inputs and expected outputs are.)

`difftest.py --check --ir stree` runs `lake exe dace-interp stree
<name>.stree.json <inputs.json>` and compares against the same
`<name>.expected.json` (the schedule tree of an SDFG must compute the same
outputs); `--ir sdfg` (the default) is unchanged. The schedule-tree
interpreter (`Dace/STree/Interp.lean`, `STree.run`) mirrors the relational
semantics function by function and is proven sound against it
(`Dace/STree/Soundness.lean`, `STree.run_sound`); the containers are laid out
over the tree's own descriptor repository (`TCtx.ofTree`).

## Conventions

- Only `Dace/Prelude.lean` imports Mathlib.
- No `sorry` in committed code; `scripts/check_axioms.sh` is run in CI.
- Every modeling choice that is not forced by DaCe's documentation is recorded
  in `Dace/Decisions.md` and referenced from the source as `Decisions.md#Dn`.

# Tilification — Design (index)

This page is the entry point to the K-dim CPU vectorizer design. The original
single-file spec has been split into two concise documents; the detailed draft
(audit / gap list / delivery schedule / progress notes) lives in git history.

- **[VECTORIZATION_MODEL.md](VECTORIZATION_MODEL.md)** — the theory: iteration
  domain, tiling, the iteration mask, access expansion + the per-dim access
  lattice, the tile function (read/write sets), and the pass pipeline that brings
  a loop nest to that form.
- **[TILE_LOWERING.md](TILE_LOWERING.md)** — the IR contract: the tile lib-node
  set, why `TileLoad` ≠ `TileStore`, the connector grammar (`_src` / `_dst` /
  `_mask` / `_idx_<k>` / …), operand kinds + broadcasting, the per-ISA dispatch
  (pure / scalar / avx512 / avx2 / neon / sve / cutile), and the validation locks.

Companion focused docs: `BROADCAST_DESIGN.md`, `CUTILE_EXPANSION_DESIGN.md`,
`STAGE_GLOBAL_THROUGH_SCALARS_SPEC.md`. The lib-node `validate()` methods are the
machine-checked source of truth for the contract.

## Section-anchor map (for `design section N.M` references in code)

The code docstrings cite the old section numbers; they now resolve as:

| old § | topic | now in |
|---|---|---|
| 2.x | iteration space, tiled body | MODEL §1–2, §6 |
| 2.3 | layout / stride invariants | MODEL §4 |
| 3.x | inside-body staging, scalar exception | MODEL §5 |
| 3.5 | WCR memlets | LOWERING §2 |
| 3.8 | per-lane index materialisation | MODEL §4, LOWERING §3 |
| 4.x | per-dim access lattice | MODEL §4 |
| 5.x | cross-dim composition, diagonal/transpose | MODEL §4 |
| 6.1–6.3 | node set, operand contract, broadcast | LOWERING §1, §4 |
| 6.4–6.5 | codegen dispatch / inner-dim strategy | LOWERING §5 |
| 6.7 | implementation phasing | LOWERING §5 (now: avx512 load/gather landed) |
| 7.x | masking | MODEL §3, LOWERING §6 |
| 8.x | remainder loops / strategies | MODEL §3 |
| 9.1–9.4 | index encoding, connector grammar | LOWERING §3 |
| 10.1–10.6 | validation locks | LOWERING §6 |

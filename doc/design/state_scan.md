# State scans: parallel loops that carry a value

A loop whose iterations chain through one scalar, `s = f(s, d[i])`, has a carried dependence on every iteration
pair, so no subset test parallelizes it. It is still a scan, because composing transitions is associative. The
composition is cheap in two cases, and canonicalization lifts both to a `StateScan` library node.

| case | example | proof | block summary | compose |
|---|---|---|---|---|
| finite state | DFA, KMP, Huffman, UTF-8, CSV, Aho-Corasick | `s` indexes a dimension of extent `E` before it is assigned; z3 shows the in-bounds read confines `s` to `E` values | table of `E` end states, one per start state | lookup |
| GF(2)-affine | CRC, LFSR | z3 shows `f(s, d, p) = c(d, p) ^ XOR_j s_j A_j(p)` for every state, datum and parameter | the block run from zero | `A^l` by repeated squaring |

## Pieces

- `passes/analysis/finite_state.py`: the step slice (definitions the state depends on, the guards around them, the
  tables they read) and the finite-state bound.
- `passes/analysis/smt_region.py`: symbolic execution of a loop-body region (tasklets, interstate assignments,
  branches, short loops) over bitvectors with C semantics on the containers' types.
- `passes/analysis/carried_state.py`: the GF(2)-affine proof on the step region.
- `libraries/standard/nodes/state_scan.py`: `StateScan`. One SDFG expansion for every device: summarize blocks
  in parallel, compose on one thread, run the blocks again from their start states. `CPU` uses `CPU_Multicore`
  maps, `CUDA` uses `GPU_Device` maps (CUDA and HIP), `pure` uses sequential maps.
- `passes/loop_to_state_scan.py`: `LoopToStateScan`, in the `lift_semantics` stage after `LoopToScan`. It puts the
  scan in front of the loop and has every iteration read its start state, so `LoopToMap` parallelizes the loop.

## Rules

- The step reads only containers the loop does not write, and no other carried value.
- The other values the loop carries must be sum or product reductions; otherwise the loop stays sequential.
- An element read indexed by a computed value (`trans[s, c]`) is a function of that value, never a free input of
  the GF(2) proof.
- A finite scan does `(E + 1) N` steps and wins only when the parallel units outnumber the states; the node records
  that trade in its `specialization_hint`.

# If-converting per-element branches in a parallel map (TSVC `s2710`)

`s2710.py` runs TSVC `s2710` through the CPU canonical form twice, with and without
`IfConvertParallelMapBodies`, checks both results against NumPy, and writes both CPF renderings to the
working directory (`s2710_branchy.cpp`, `s2710_select.cpp`) so the generated loops can be diffed.

```bash
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores PYTHONHASHSEED=0 python samples/optimization/if_conversion/s2710.py
```

## What changes

`s2710` is one parallel loop whose body branches on the element's own data (`a[i] > b[i]`), with nested
branches and unequal write sets (`b` is written only in the else arm). The canonical form keeps it as one
`omp parallel for` over `i` with the original `if`/`else` inside. On random data the branch mispredicts about
half the time, and gcc neither if-converts nor vectorizes that body ("not vectorized: not profitable").

`IfConvertParallelMapBodies` (`dace/transformation/passes/cpu_specialization/if_convert_map_bodies.py`, run in
`cpu_specialize` after the schedules are final) rewrites the body into selects. It evaluates both arms and
writes `a`, `b` and `c` unconditionally, writing back the old value where an arm does not write. Codegen
emits each select as `((c) ? (t) : (e))`, and gcc vectorizes the loop.

The rewrite is only sound when every arm reads and writes nothing but the lane's own element (or element 0),
and writes only containers the body also reads. A gather behind an index guard, a shifted read, a library
node or a loop keeps the branch (see `tests/passes/cpu_specialization_if_convert_test.py`).

## Measured

One MI300A node, gcc, `LEN_1D = 43397065` (hpcagent-bench preset M), median of 7 calls, both outputs
checked against NumPy:

| `OMP_NUM_THREADS` | branchy (ms) | select (ms) | speedup |
|---|---|---|---|
| 1 | 266.4 | 47.5 | 5.6x |
| 8 | 45.3 | 33.3 | 1.4x |
| 24 | 24.1 | 15.5 | 1.6x |

The gap is largest when the cores are not bandwidth-bound. As threads are added, both forms approach the
memory bandwidth limit (five reads and three writes per element). Before the pass, the canonical form ran at
0.81x of a numba-parallel baseline on the 24-core grading slot; LLVM vectorizes numba's version with masked
selects.

## For codegen work

- The selects come from the vectorizer's branch lowering (`SameWriteSetIfElseToITECFG`, then
  `BranchNormalization`) applied to the map's nested body; `ITE` is printed as `?:` on every target.
- The always-write-back form costs a store per element even where the original arm did not write. A
  masked store (AVX-512 `vmaskmov`, SVE predicates) would avoid it; nothing emits one today.
- On GPUs the branchy form already diverges per warp rather than mispredicting, so the pass runs only in CPU
  specialization.

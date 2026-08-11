# `np.linalg.solve` with a vector RHS crashes with `IndexError: list index out of range`

`Solve.validate()` always reads `shape_out[1]` to get the number of right-hand sides.
That is fine for `np.linalg.solve(A, B)` with a matrix `B`, but `np.linalg.solve(A, b)`
with a 1-D vector `b` squeezes to a shape list of length 1, and the index is out of
range.

Verified on `main` `d7efcef0c` and `extended` `ac9398525` (the `validate` method is
byte-identical between the two, so both hit the same line).

Found via the `raman_fitting` kernel in our generated NumPy-to-DaCe corpus: it solves a
`(3K+1)x(3K+1)` Levenberg-Marquardt normal-equations system against a `(3K+1,)`
gradient vector, `np.linalg.solve(__lm0_atad, __lm0_rhs)`.

## Which path triggers it

`to_sdfg(simplify=False)` alone does **not** trigger it. The frontend replacement
(`dace/frontend/python/replacements/linalg.py:263`) just drops a `Solve` library node
into the SDFG, no validation yet. The crash needs `expand_library_nodes()`, which
picks an implementation (OpenBLAS is installed on this machine, so
`ExpandSolveOpenBLAS` wins by default) and that expansion calls `node.validate()`.
Same shape as the gemv `out`-connector issue: parse-only gates miss this.

## Repro

```python
import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def solve_vec(A: dc.float64[N, N], b: dc.float64[N]):
    return np.linalg.solve(A, b)


sdfg = solve_vec.to_sdfg(simplify=False)
sdfg.expand_library_nodes()
```

## Traceback

```
Traceback (most recent call last):
  File "repro_solve.py", line 15, in <module>
    sdfg.expand_library_nodes()
  File "dace/sdfg/sdfg.py", line 3165, in expand_library_nodes
    impl_name = node.expand(state)
  File "dace/sdfg/nodes.py", line 1586, in expand
    transformation.apply(actual_state, sdfg, **expansion_kwargs)
  File "dace/transformation/transformation.py", line 743, in apply
    expansion = type(self).expansion(node, state, sdfg, *args, **kwargs)
  File "dace/libraries/linalg/nodes/solve.py", line 116, in expansion
    return _make_sdfg_getrs(node, parent_state, parent_sdfg, "OpenBLAS")
  File "dace/libraries/linalg/nodes/solve.py", line 21, in _make_sdfg_getrs
    arr_desc = node.validate(parent_sdfg, parent_state)
  File "dace/libraries/linalg/nodes/solve.py", line 224, in validate
    desc_out.dtype, strides_out, shape_out[0], shape_out[1], desc_ain.storage)
                                               ~~~~~~~~~^^^
IndexError: list index out of range
```

The raman_fitting kernel produces the identical traceback, same file, same line 224.

## Root cause

`dace/libraries/linalg/nodes/solve.py:191-224`. `_bin`/`_bout` memlets get squeezed:

```python
squeezed_out = copy.deepcopy(out_memlet.subset)
dims_out = squeezed_out.squeeze()
...
shape_out = squeezed_out.size()
...
return (..., shape_out[0], shape_out[1], desc_ain.storage)
```

For a vector RHS, `out_arr` has shape `[N]`, squeeze leaves `shape_out == [N]`
(length 1). `shape_out[1]` (line 224) assumes a matrix RHS with shape `[N, rhs]`
(length 2) and reads past the end. `shape_out[0]` (`n`, line 224) is fine either way;
only the `rhs`-count read at index 1 is the bug. The earlier size check at
`solve.py:199` (`len(squeezed_bin.size()) > 2`) explicitly allows the 1-D case to
reach here, so the vector path is meant to work, it just isn't finished.

`shape_bin` has the same shape as `shape_out` (checked by `np.array_equal` at line
211), so the same crash is reachable from `shape_bin[1]` if that path is read first
in a future edit; right now only `shape_out[1]` is evaluated before the return.

## Repro script

`repro/linalg_solve_index_error.py` runs a vector-RHS case (expected to crash) and a
matrix-RHS case (expected to expand cleanly, control). Needs OpenBLAS or MKL findable
by DaCe to reach expansion; no C++ compiler is invoked. Exits 0 when both still match
what this file documents:

```
PYTHONPATH=/path/to/dace python3 repro/linalg_solve_index_error.py
```

## Suggested fix

Treat `rhs` as `shape_out[1] if len(shape_out) > 1 else 1` (and build `_binout`/
`_bout` shapes accordingly for the vector case), instead of assuming `shape_out`
always has two entries.

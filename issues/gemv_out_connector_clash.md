# BLAS pure expansions mint a bare `out` tasklet connector, which clashes with a user array named `out`

The pure expansions of `gemv`, `gemm`, `batched_matmul`, `csrmv` and `csrmm` build their
zero-init map with a tasklet whose connector is literally `out`. Every other tasklet in
those same expansions uses a dunder-guarded name (`__A`, `__x`, `__out`, `__y_in`, `__c`).
`out` is safe only inside the expansion's own nested SDFG. As soon as that nested SDFG is
inlined into a parent that has an array named `out`, validation rejects the graph and the
program cannot be compiled. The name is a plain, common output name, so this is not exotic.

Verified on `main` `d7efcef0c`. Also fails on the downstream `extended` branch that our CI
pins (`a3b3d9a23894cc68800e3bcbb843c7897bab2ec3`; local clone HEAD `11004dab0`).

## Repro

```python
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def gemv_out(A: dc.float64[M, N], x: dc.float64[N], out: dc.float64[M]):
    out[:] = A @ x


sdfg = gemv_out.to_sdfg(simplify=False)
sdfg.expand_library_nodes()
sdfg.simplify()
sdfg.validate()
```

## Error

```
dace.sdfg.validation.InvalidSDFGNodeError: Connector name 'out' is already used as a
symbol, constant, or array name (at state _MatMult_gemv_initstate_0, node gemv_init)
Originating from source code at File "dace/libraries/blas/nodes/gemv.py", line 77
```

Replacing `A @ x` with `A @ B` gives the same error from `gemm_init`, originating at
`dace/libraries/blas/nodes/gemm.py` line 100.

## Root cause

The offending connector, one line per site, `main` `d7efcef0c`:

- `dace/libraries/blas/nodes/gemv.py:82` -- `"out = 0", {"out": dace.Memlet(...)}`
- `dace/libraries/blas/nodes/gemm.py:105`
- `dace/libraries/blas/nodes/batched_matmul.py:66`
- `dace/libraries/sparse/nodes/csrmv.py:119`
- `dace/libraries/sparse/nodes/csrmm.py:119`

Rejected by `dace/sdfg/validation.py:594`, which tests every non-nested-SDFG connector
against `sdfg.constants_prop | sdfg.symbols | sdfg.arrays`. Pre-inline the enclosing SDFG
is the expansion, whose arrays are `_A`, `_x`, `_y`, so the check passes. Inlining moves
the tasklet into the caller, where `out` is a real array, and the check fires.

`dace/transformation/helpers.py:194` has the same bare `out` connector in
`__did_ret_init`, but that tasklet is added straight to the target SDFG, so it can clash
without any inlining.

## Rename A/B

Renaming the output parameter is the only change, and it decides the outcome.

| output parameter | `gemv` | `gemm` |
| --- | --- | --- |
| `out` | InvalidSDFGNodeError | InvalidSDFGNodeError |
| `res` | expands, simplifies, validates | expands, simplifies, validates |

`repro/gemv_out_connector_clash.py` runs all four cases, needs no C++ compiler, and exits
0 while each still behaves as documented. It reports 4/4 on `main` `d7efcef0c` and on
`extended` `11004dab0`.

## Downstream

Four kernels in our generated NumPy-to-DaCe corpus fail to compile on this alone: `atax`
and `gesummv` via `gemv`, `covariance2` and `k3mm` via `gemm`. All four pass an array named
`out`. It is invisible to a parse-only gate: `to_sdfg` succeeds, expansion is what breaks.

## Suggested fix

Rename the connector to `__out` at the five sites, matching the dunder convention the rest
of each expansion already follows.

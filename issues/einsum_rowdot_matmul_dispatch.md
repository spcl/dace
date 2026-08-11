# Row-wise einsum (`xyzk,xyzk->xyz`) dies in MatMul expansion after simplify: `NotImplementedError: Matrix multiplication not implemented for shapes: [Lb**3, k] and [Lb**3, k]`

`np.einsum('xyzk,xyzk->xyz', a, a)` is a per-element dot over the last axis (elementwise
multiply + reduce over `k`). The frontend lowers it to a `MatMul` library node. Before
simplify the operands keep their 4-D shapes and the batched path handles the expansion --
`to_sdfg(simplify=False)` followed by `expand_library_nodes()` (or a full `compile()`)
goes through cleanly. After simplify, the operand views collapse to `[Lb**3, k]` and
`[Lb**3, k]`, and `MatMul`'s specialization dispatch
(`dace/libraries/blas/nodes/matmul.py:296`) has no case for two same-shape 2-D operands
with a 1-D output: every branch expects a real matrix product (GEMM/GEMV/dot/batched),
so it raises.

Verified on `extended` `a4740d4e7` (pristine clone, no local edits).

Found via the `fragment_patch_density` kernel in our generated NumPy-to-DaCe corpus:
fragment density accumulation, `dens = np.einsum('xyzk,xyzk->xyz', psi_frag[f], psi_frag[f])`
over `psi_frag: float[nfrag, Lb, Lb, Lb, k]`.

## Which path triggers it

- `to_sdfg(simplify=False)` + `expand_library_nodes()`: OK.
- `to_sdfg(simplify=False)` + `compile()`: OK.
- `to_sdfg()` (simplify on) + `compile()`: raises during `expand_library_nodes` inside
  compile, at `MatMul` expansion. Simplify is what folds the `xyz` box into the flat
  `Lb**3` view the dispatch cannot classify.

Parse-only gates miss it, same as the gemv `out`-connector and vector-RHS `solve` issues.

## Repro

```python
import numpy as np
import dace as dc

nfrag, Lb, k, N = (dc.symbol(s, dtype=dc.int64) for s in ('nfrag', 'Lb', 'k', 'N'))


@dc.program
def kernel(offsets: dc.int64[nfrag, 3], alpha: dc.float32[nfrag], psi_frag: dc.float32[nfrag, Lb, Lb, Lb, k],
           rho: dc.float32[N, N, N]):
    box = np.arange(Lb)
    rho[:] = 0.0
    for f in range(nfrag):
        dens = np.einsum('xyzk,xyzk->xyz', psi_frag[f], psi_frag[f])
        xs = (offsets[f, 0] + box) % N
        ys = (offsets[f, 1] + box) % N
        zs = (offsets[f, 2] + box) % N
        __ix0_v = alpha[f] * dens
        for __ix0_i0 in range(xs.shape[0]):
            for __ix0_i1 in range(ys.shape[0]):
                for __ix0_i2 in range(zs.shape[0]):
                    rho[xs[__ix0_i0], ys[__ix0_i1], zs[__ix0_i2]] += __ix0_v[__ix0_i0, __ix0_i1, __ix0_i2]


sdfg = kernel.to_sdfg()   # simplify on -- required to trigger
sdfg.compile()            # NotImplementedError: ... shapes: [Lb**3, k] and [Lb**3, k]
```

## Expected

Either the einsum replacement recognizes `...k,...k->...` as multiply+reduce and never
mints a MatMul, or the MatMul dispatch grows a case for the row-wise dot form (same-shape
2-D inputs, 1-D output: `C[i] = sum_k A[i,k] * B[i,k]`).

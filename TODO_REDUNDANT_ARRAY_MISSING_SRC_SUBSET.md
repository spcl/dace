# TODO: `validate_subsets failed: Source subset is missing` on heat3d after LoopToMap + MapFusion

Reported by Elias against `extended` (a `pip`-installed dace in `FP-Arena/.venv`, python 3.12), right
after updating the branch. Not started -- queued behind the current tasks.

## Symptom

```
dace/transformation/dataflow/redundant_array.py:1072: UserWarning: validate_subsets failed:
Source subset is missing (dst_subset: 1:N - 1, 1:N - 1, 1:N - 1, src_shape: (1, 1, 1)
```

Warning only; the run completes. But it fires from the check whose own comment says
`# 2-c. Validate subsets in memlet tree` / `# (should not be needed for valid SDGs)`, so either the
SDFG is invalid or that comment is wrong. That is the question to settle.

## Reproducer

```python
import dace as dc
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def kernel(TSTEPS: dc.int64, A: dc.float64[N, N, N], B: dc.float64[N, N, N]):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1, 1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) +
                               0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) +
                               0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) +
                               A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1, 1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) +
                               0.125 * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) +
                               0.125 * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) +
                               B[1:-1, 1:-1, 1:-1])


sdfg = kernel.to_sdfg(simplify=True)
sdfg.apply_transformations_repeated(LoopToMap)
sdfg.apply_transformations_repeated(MapFusion)
sdfg.simplify()
```

This is polybench `heat_3d`, so `tests/corpus/polybench/stencils/heat_3d.py` is the in-tree copy.
Note the chain is the plain transformation chain, NOT `canonicalize`.

## Where it fires

`RedundantSecondArray.can_be_applied` (`redundant_array.py:833`), step **2-c** -- the memlet-tree
walk at `redundant_array.py:1066-1072`, not the earlier 2-a probe. It calls
`_validate_subsets(e3, sdfg.arrays, src_name=out_array.data)`.

The raise is `_validate_subsets` at `redundant_array.py:66-77`: the edge has **no** `src_subset`, and
because `edge.data.data == dst_name` it synthesizes `src_subset = Range.from_array(desc)`. With
`desc.shape == (1, 1, 1)` that is 1 element against an `(N-2)^3` destination, so the element-count
guard raises `ValueError`. `can_be_applied` then returns `False` -- the transformation is refused.

## The two readings (this is what needs deciding)

1. **Refusal is right, warning is too loud.** A 1-element source cannot cover `(N-2)^3`, so declining
   is correct, and `warnings.warn` inside a `can_be_applied` probe is simply the wrong channel -- it
   should be a debug-level message. Cheap fix, but only valid if the SDFG is genuinely well-formed.
2. **Real upstream defect.** Something in the chain produced a shape-`(1,1,1)` transient sitting on a
   memlet-tree edge that carries an `(N-2)^3` copy, and left that edge without a `src_subset`. Then
   the 2-c comment is accurate, the graph is malformed, and the fix belongs in whatever pass built it.

Do not fix the warning before settling which one it is. Reading 2 is a latent wrong-answer risk;
reading 1 is cosmetic.

## Leads

- **Prime suspect: `a1ebc74aa` "Keep the copy destination index when scalarizing a length-1 array"**
  (`dace/transformation/passes/length_one_array_scalar_conversion.py`, +46/-16 plus tests). It is on
  `extended`, it is about length-1 arrays and copy subsets, and the report arrived immediately after
  a branch update. First move: A/B the reproducer at `a1ebc74aa` and at its parent.
- Related prior work, same family: `project_dace_singleton_slice_squeeze` (SLICE-singleton squeeze),
  `project_s293_broadcast_copy_fix` (broadcast copy). A `(1,1,1)` descriptor against a strided
  destination is exactly the broadcast/squeeze shape both of those dealt with.
- Also on `extended` and worth ruling out: `937c59ac5` "Stop three transformations refusing legal,
  required rewrites" and `d91b23360` "Make pop_dims() handle Indices and index the subset it was
  handed" -- both touch subset handling.

## Plan

1. Reproduce, and print the offending `e3`: its `data`, `src_subset`, `dst_subset`, both endpoint
   descriptors and their shapes. Identify which container has shape `(1,1,1)` and who created it.
2. `sdfg.validate()` after each step of the chain (`to_sdfg` / LoopToMap / MapFusion / simplify) to
   find the first step whose output the validator rejects. That alone separates reading 1 from 2.
3. Bisect the chain against `origin/main` and against `a1ebc74aa^` to confirm whether `extended`
   introduced it.
4. Fix at the depth the answer dictates: the producing pass (reading 2) or the warning channel
   (reading 1). No patch over the symptom in `can_be_applied`.
5. Test: heat3d through this exact chain, asserting the SDFG validates and -- since heat3d has a
   numeric oracle in the corpus -- that it still computes the right answer. Verify by EXECUTION
   against numpy, never against a sympy identity.

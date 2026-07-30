# heat3d: `validate_subsets failed: Source subset is missing` — diagnosed, fix needs a decision

Reported by Elias against `extended` (a `pip`-installed dace in `FP-Arena/.venv`, python 3.12) right
after updating the branch.

```
dace/transformation/dataflow/redundant_array.py:1072: UserWarning: validate_subsets failed:
Source subset is missing (dst_subset: 1:N - 1, 1:N - 1, 1:N - 1, src_shape: (1, 1, 1)
```

## Verdict

**Not a miscompile, and not a malformed graph.** The SDFG validates after *every* step of the chain,
including the step that warns:

```
to_sdfg(simplify=False) VALID  warnings=0
simplify #1             VALID  warnings=0
LoopToMap               VALID  warnings=0
MapFusion               VALID  warnings=0
simplify #2             VALID  warnings=1
```

The transformation refuses (`can_be_applied` returns `False`) and the refusal is correct. What is
wrong is that a normal shape is reported as an anomaly, and that the refusal hides a buffer that
could be removed. The 2-c comment "should not be needed for valid SDFGs" is what is inaccurate.

## Reproducer

`tests/corpus/polybench/stencils/heat_3d.py` is the in-tree copy of this kernel. The chain is the
plain transformation chain, NOT `canonicalize`:

```python
sdfg = kernel.to_sdfg(simplify=True)
sdfg.apply_transformations_repeated(LoopToMap)
sdfg.apply_transformations_repeated(MapFusion)
sdfg.simplify()          # warns here
```

## Mechanism (traced, not inferred)

The warning comes from `RedundantSecondArray.can_be_applied` step **2-c**, the memlet-tree walk at
`redundant_array.py:1066-1072`, which calls `_validate_subsets(e3, sdfg.arrays, src_name=out_array.data)`
for every edge in `state.memlet_tree(e2)`.

Instrumented, the offending edge and the branch it takes:

```
2-c edge:  src=MapExit(_Add__map)  dst=AccessNode(B)  data=B
           src_subset=None  dst_subset=1:N-1, 1:N-1, 1:N-1   out_array=__map_fusion_B_0

_validate_subsets:  src_name=__map_fusion_B_0  desc=Array shape=(1,1,1)
                    edge.data.data=B   dst_name=B   isView=False
```

`dst_name` is not passed by 2-c; `_validate_subsets:48-49` derives it from the edge's destination
AccessNode, giving `B`. So `edge.data.data == dst_name` holds, and the branch at `:69` applies the
plain-copy convention "the memlet is written in the destination's terms, therefore the source subset
is the whole source array". It synthesizes `src_subset = Range.from_array(desc)` — **1** element —
compares it against the destination's `(N-2)^3`, and raises at `:76`.

That convention does not hold for this edge. Its source is a **MapExit**, not `out_array`: the
transient is written once per iteration and the map's range supplies the outer dimensions. 2-c passes
`src_name=out_array.data` for every edge in the tree, including edges whose endpoint is a map scope
node rather than the array.

## Why it appeared only now (bisected)

Not `a1ebc74aa` — the warning is present at that commit **and** at its parent. The prime suspect in
the first draft of this note was wrong.

The difference is `squeezed_shape()`, introduced on `extended` in `redundant_array.py`, used at
`:914` in this same `can_be_applied`:

| | `out_desc.shape == (1,1,1)` becomes | result |
|---|---|---|
| `main` | `[sz for sz in shape if sz != 1]` → `[]` | `_subset_has_shape` fails, early `return False` at `:917` |
| `extended` | `squeezed_shape(...)` → `[1]` | check passes, walk continues into 2-c |

Confirmed by substituting main's expression at `:914` on `extended`: `warnings=0`. Both branches
produce the *same graph* — 29 shape-`(1,1,1)` transients either way — so this is purely about how far
`can_be_applied` gets before refusing.

`squeezed_shape()` is correct and should stay: `Range.squeeze()` keeps one dimension rather than
producing a zero-dimensional subset, so an all-size-1 descriptor must keep one too. Main's version
made an all-size-1 descriptor compare against a 0-dimensional shape, which silently prevented
`RedundantSecondArray` from ever firing on a Scalar or a `(1,1)` array. Fixing that is what made the
pre-existing rough edge in 2-c reachable.

## Blast radius

- No wrong answer: the graph is valid at every step and the pass refuses.
- No optimization regression against `main`: `main` refuses too, just earlier and quietly.
- What is new is the noise, plus a newly-visible missed optimization — the redundant
  `__map_fusion_B_0` buffer survives on both branches.

## The decision this needs

**Option A — stop mis-reporting, keep refusing.** In 2-c, do not treat "cannot evaluate" as an
anomaly for an edge whose endpoint is a map scope node. Refusal behaviour is unchanged, so there is
zero correctness risk; the buffer still survives. Cosmetic.

**Option B — teach 2-c about map-boundary edges so the buffer can actually be removed.** Ask
`_validate_subsets` only about edges where `out_array` really is the endpoint, and let the map's range
account for the missing dimensions. This *changes refusal into application* in a core `simplify`
transformation, so it needs its own soundness argument and a numeric gate — this is the
"remove-redundant-second-array on heat3d" case, worth real performance if it holds.

Do not silence the warning without choosing. A is safe but leaves the buffer; B is the actual win and
carries the risk. B should be verified by EXECUTION against a numpy oracle, never against a sympy
identity.

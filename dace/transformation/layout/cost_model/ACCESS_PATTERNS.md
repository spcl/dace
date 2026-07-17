# Access patterns in the LogP model

What the model says about strided and unstructured access, and when to reorder.
Numbers are fp64 on 64-byte blocks (8 elements/block). Pinned by
`tests/transformations/layout/cost_model_access_patterns_test.py`.

## One number: cache efficiency

```
eps = distinct useful bytes / bytes actually moved        in (0, 1]
```

It enters **exactly one** LogP term:

```
t_nest = max( useful_bytes/eps * G ,  messages * L / concurrency )
                        ^ eps here          ^ eps CANNOT express this
```

Latency is paid per **message** (a distinct block touched), bandwidth per **byte**. Four GPU sectors
in one 128 B request and four spread over four requests move identical bytes — same `eps`, 1 vs 4
messages. So `elementwise LogP x eps` is not a simplification, it is a lost term.

`eps = eps_spatial * eps_write`, where `eps_write = 1/2` for a partial write (read-for-ownership:
the block is fetched before being merged) and `1` for a read or a block-covering write. Hence
`eps in [1/8, 1]` reading, `[1/16, 1]` writing.

## Strided

| stride | blocks/iter | eps |
|---|---|---|
| 1 | 1/8 | 1 |
| 2 | 1/4 | 1/2 |
| 4 | 1/2 | 1/4 |
| 8 | 1 | 1/8 |
| 16, 64, 800 | 1 | 1/8 |

Degrades as `1/stride`, then **saturates** at one block per access. Stride 8 and stride 800 cost the
same. So **8x is a hard ceiling** for what layout can buy on fp64 — a kernel reporting more than 8x
from a Permute is confounded, not fast.

## Unstructured

The model does not care that an index is "random", only how many blocks it lands in. Same 4096
random accesses:

| pattern | eps |
|---|---|
| random over `10^7` (scattered) | 1/8 |
| random over `4096` (clustered) | 1 |

**Spread is the cost, not irregularity.** A clustered gather is already perfect and has nothing to win.

## When to reorder

Break-even for reordering an array (relayout, or removing an indirection):

```
passes * (1/eps0 - 1/eps1)  >=  2 + overhead_passes
```

`G` and the array size **cancel** — this is a pure traffic ratio, independent of the machine, since
a faster machine speeds the reorder and the nest equally. The `2` is the reorder itself: one clean
read pass + one clean write pass.

**With `overhead = 0` and a perfect target, one pass pays as soon as `eps0 <= 1/3`.** A nest using
under a third of every block it moves is worth relaying out *for that nest alone* — no reuse
argument. (k16 measured 1.4 passes; `report3` §4.)

### Static replace

When `sigma` in `A[sigma[i]]` is known without running the program, reorder once into `A'` and
rewrite consumers to read `A'[i]`. That is `overhead_passes = 0`, so a scattered gather
(`eps0 = 1/8`) pays on the first pass.

`overhead_passes > 0` models a reorder that must first LEARN `sigma` at runtime — i.e.
inspector-executor. **We are not pursuing I/E**; the parameter exists only to quantify it as a threat
to validity (a reviewer will ask whether static replace is subsumed by it). The short version: at
`eps0 = 1/8` I/E's first pass wins too, so do not claim a speed advantage — claim that static replace
needs no reuse argument and pays nothing at runtime. Full analysis and the numbers:
`~/Downloads/layout/report1_cost_models_and_optimality_plan.md` § "Threats to state".

## Caveat

`logp_analysis` still computes `bytes = messages * block_bytes` with a **single** granularity, so the
line/sector split above is wired into `relayout.py` but **not yet into the nest analysis** — on GPU
that path keeps a 4x conflation until `sector_bytes` is threaded through `logp_analysis`,
`achievable_rate`, and `bandwidth_delay_product`.

## Open: three formulas, two of them disagree

`loggp.nest_memory_time` (formerly `relayout.nest_time`) is **algebraically identical** to `total_bytes / achievable_rate`
(`total_time_overlapped`) — verified at ratio 1.000000 — because `achievable_rate`'s `min` already
is that `max`. It adds only the sector generalization (`bytes != messages * line`).

But it disagrees with `LoopNestLogP.total_time()` by a factor of `concurrency` (~24x at
L=95ns, g=4ns) whenever the nest is latency-regime, because `total_time()` returns
`total_time_serialized` there — the zero-overlap upper bound. Unresolved: the regime test asks
whether concurrency suffices to *saturate the channels*; "no" means the rate is `conc*line/L`
(latency-bound but still overlapped), not that overlap collapses to one request. On that reading
`total_time_overlapped` is right in both regimes and `total_time()`'s latency branch is wrong.
Needs a decision plus a measurement — it changes existing numbers.

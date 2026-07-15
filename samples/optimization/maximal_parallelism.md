# Maximal parallelism is not maximal performance

A loop-carried reduction is the textbook case where you *can* expose more
parallelism than you *should*. This sample walks through one — a contour-style
array reduction — shows the transformation DaCe uses to parallelize its
reduction axis (`LiftLoopCarriedReduction`), measures when that helps and when it
hurts, and explains the rule DaCe uses to decide.

## The kernel

An outer loop `idx` accumulates a per-iteration array increment into a
whole-array output. The output index is **invariant** over `idx`, so `P` is a
reduction of the increment over `idx`:

```python
# BEFORE — outer reduction loop is SEQUENTIAL
@dace.program
def contour(B: dace.float64[KK, NR, NM], P: dace.float64[NR, NM]):
    P[:] = 0.0
    for idx in range(KK):                       # <- reduction axis, sequential
        X = np.zeros((NR, NM))
        for i, j in dace.map[0:NR, 0:NM]:       # <- element map, already parallel
            X[i, j] = B[idx, i, j] * B[idx, i, j]
        for i, j in dace.map[0:NR, 0:NM]:
            P[i, j] = P[i, j] + X[i, j]          # in-place accumulate -> loop-carried on P
```

`P` is read back and written every iteration, so `LoopToMap` sees a loop-carried
dependency on `P` and refuses to parallelize `idx`. The available parallelism is
just the inner `NR x NM` element map; the `KK` reduction axis runs serially.

## The transformation

`LiftLoopCarriedReduction` recognizes the pattern, **drops the read-back**, and
turns the in-place accumulation into a write-conflict resolution (WCR):

```
P[i, j] = P[i, j] + X[i, j]        ==>        P[i, j]  (wcr: +)=  X[i, j]
```

With `P` no longer read across `idx` iterations, `LoopToMap` now parallelizes
`idx` too. The result is equivalent to making **every** axis parallel and letting
the WCR resolve the cross-iteration accumulation (an OpenMP `reduction` clause /
atomic at code generation):

```python
# AFTER — reduction axis is ALSO parallel; the accumulation is a WCR reduction
@dace.program
def contour(B: dace.float64[KK, NR, NM], P: dace.float64[NR, NM]):
    P[:] = 0.0
    for idx, i, j in dace.map[0:KK, 0:NR, 0:NM]:   # all three axes parallel
        P[i, j] += B[idx, i, j] * B[idx, i, j]      # (+)= resolved by WCR / atomic
```

This is **more** parallelism: `KK * NR * NM` independent-ish units instead of
`NR * NM`. Maximal. But it is not free — the WCR adds a reduction (atomic or
privatize-and-combine) over the `KK` axis that the sequential form never paid.

## The performance diff

Contour reduction, 4 threads, total volume held fixed at ~16.7M elements, inner
map `= S x S`, `KK = 16.7M / (S*S)`. Median of 25 runs.

| inner map | pass ON (parallel `KK`) | pass OFF (serial `KK`) | verdict |
|----------:|------------------------:|-----------------------:|:--------|
| 8 x 8     | 73 ms                   | 468 ms                 | ON **6.4x faster** |
| 16 x 16   | 62 ms                   | 131 ms                 | ON **2.1x faster** |
| 32 x 32   | 57 ms                   | 41 ms                  | ON 1.4x slower — **crossover** |
| 64 x 64   | 65 ms                   | 23 ms                  | ON 2.8x slower |
| 128 x 128 | 46 ms                   | 18 ms                  | ON 2.6x slower |

Same shape at other thread counts and for `min` / `max` / `*` reductions.

## Why the crossover exists

The inner element map is the parallelism you already have. Two regimes:

- **Inner map too small to fill the cores** (top rows). With only 64 or 256
  elements, the inner map leaves most threads idle and pays loop/thread overhead
  `KK` times over. Parallelizing the `KK` axis fills the machine — a real 2-6x
  win. Here the extra parallelism is exactly what was missing.

- **Inner map already saturates the cores** (bottom rows). The threads are busy
  regardless of `KK`. Adding a parallel reduction over `KK` buys no occupancy and
  *adds cost*: atomic contention / privatization on `P[i, j]`, worse cache
  locality on the strided `B[idx, i, j]` access, and a combine step. Maximal
  parallelism here is pure overhead — 3x slower.

The lesson: you want **enough** parallelism to fill the machine, not the most the
program can express. Past saturation, extra parallel axes are a tax.

## The decision rule DaCe uses

The crossover depends on comparing the inner map size against the machine — a
comparison you can only make with **concrete** extents. Real kernels carry
symbolic sizes (`N`, `KLEV`, ...), where the inner size is unknown at compile
time and the lift is a blind gamble that, past the crossover, usually loses.

So `LiftLoopCarriedReduction` refuses to fire when any relevant extent — the
inner map's iteration space or the reduction loop's trip count — is symbolic. It
lifts only when the sizes are concrete (a specialized program, constants
substituted) and can later be cost-modeled. When in doubt, keep the smaller,
predictable parallelism rather than gamble on the maximal one.

See `dace/transformation/passes/canonicalize/lift_loop_carried_reduction.py` for
the pass, its dataflow surgery, and the pseudocode.

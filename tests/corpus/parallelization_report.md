# TSVC corpus parallelization report

A living per-kernel record of which TSVC kernels canonicalize parallelizes
today and which it does not, the reason, and the fix that would unlock each
missed one. Update this file when a fix lands (move the row from a ☐
section into the ✓ rollup, with the commit hash).

Re-generate the raw numbers any time with:

```
python -m tests.corpus.measure_parallelization > /tmp/l2m_report.txt
```

## Aggregate (151 kernels, peel_limit=4, break_anti_dep=True)

| Strategy        | Loops | Maps | Reduces |
|-----------------|------:|-----:|--------:|
| baseline (simplify only) | 178 | 2 | 0 |
| LoopToMap only  |   105 |   71 |       0 |
| canonicalize    |    92 |   81 |       2 |

| Outcome | Kernels | % of 151 |
|---|---:|---:|
| Already parallel after simplify (no L2M needed) | 2 | 1.3% |
| Both pipelines parallelize identically | 59 | 39.1% |
| Only canon parallelizes (canon wins) | 15 | 9.9% |
| Canon fuses sibling maps / collapses nested → 1 map (parallelism preserved) | 2 | 1.3% |
| **Regression** (canon left more loops sequential than L2M) | **1** | 0.7% |
| Neither parallelizes | 72 | 47.7% |

Canonicalize parallelizes **45.5%** of baseline loops (vs L2M-only **38.8%**) and turns final iteration to parallel for **47.4%** of constructs (vs **40.3%**). The 72 unparallelized kernels are the target of this document.

## Status legend

| | meaning |
|---|---|
| ✓ | parallel map landed |
| ✓R | parallel via `Reduce` libnode |
| ✓F | canon fused/collapsed sibling maps (parallelism preserved) |
| ✗ | genuinely sequential (true carried dependence, break, or unknown alias) |
| ⚠ | regression — L2M parallelizes it, canon leaves it sequential |
| ☐ | missed opportunity — fixable with the indicated work |

## Improvement priority (decided)

**B → C → D**, then SCATTER. Reductions first (small, isolated extensions to
`LoopToReduce`), then forward-read patterns (needs to coexist with
`UniqueLoopIterators` — see s172), then triangular 2D nests, then runtime-
guarded scatter.

## ⚠ Regressions (1)

| Kernel | Pattern | L2M | Canon | Root cause | Fix |
|---|---|---|---|---|---|
| **s172** | `for i in range(n1-1, LEN, n3): a[i] += b[i]` | ✓ map `range=n1-1:LEN:n3` | ✗ loop `_loop_it_0` ranging `0:(LEN-n1+1-1)//n3+1` | `UniqueLoopIterators` rewrites the stride loop into a normalised `_loop_it_0` form whose body uses `n1-1 + n3*_loop_it_0` to index `a`; L2M does not recognise this as a unique-write-per-iter pattern | Either (a) teach L2M's `_is_uniquely_indexed` to follow the symbolic substitution introduced by `UniqueLoopIterators`, or (b) gate `UniqueLoopIterators` to skip stride loops where the original iterator is itself the unique-index symbol |

The same root cause likely blocks **s175** (in C below) and possibly the s171/s174 family.

## ✓F Map fusion / collapse (2, verified not a regression)

| Kernel | Pattern | L2M | Canon |
|---|---|---|---|
| s1115 | 2D `aa[i,j] = aa[i,j]*cc[j,i] + bb[i,j]` | 2 separate 1-D maps (`0:LEN_2D`, `0:LEN_2D`) | 1 single 2-D map `(0:LEN_2D, 0:LEN_2D)` — MapCollapse |
| s152  | two independent 1-D loops over the same range | 2 sibling maps | 1 fused map — sibling-map fusion |

Both express the same parallel work; canon's form is preferable for downstream lowering. No action needed.

## ☐B Missed reductions (LoopToReduce extension) — priority 1

`LoopToReduce` today catches `s = 0.0; for i: s += a[i]; out[0] = s` (Python scalar accumulator written out at the end). It misses several common variants:

| Kernel | Pattern | Variant | Notes |
|---|---|---|---|
| s311  | `sum_out[0] = 0; for i: sum_out[0] += a[i]`                  | array-slot accumulator | trivial extension: detect when accumulator is a 1-element array slot, not a Python scalar |
| s313  | `dot[0] = 0; for i: dot[0] += a[i]*b[i]`                     | array-slot dot product | same fix as s311 |
| vdotr | `dot_out[0] = 0; for i: dot_out[0] += a[i]*b[i]`             | array-slot dot product | same fix as s311 |
| s317  | `q[0] = 1.0; for i in range(LEN/2): q[0] *= 0.99`            | constant-operand product reduction | same fix; check WCR identity is `*` |
| s314  | `x = a[0]; for i: if a[i] > x: x = a[i]; result[0] = x`      | max with branch         | branch-form max/min; emit `Reduce(max)` |
| s316  | `x = a[0]; for i: if a[i] < x: x = a[i]; result[0] = x`      | min with branch         | same as s314 |
| s319  | `s = 0; for i: a[i]=c+d; s += a[i]; b[i]=c+e; s += b[i]; b[0]=s` | reduction + per-element side writes | requires fission: split the parallel writes from the reduction |
| s352  | `dot = 0; for i in range(0, LEN-4, 5): dot += a[i]*b[i] + … + a[i+4]*b[i+4]` | manually-unrolled dot product | `RerollUnrolledLoops` must run *before* `LoopToReduce` for this; both passes exist but the ordering doesn't recognise this case |
| s4115 | `s = 0; for i: s += a[i] * b[ip[i]]; sum_out[0] = s`         | gather + sum reduction  | gather is fine inside a `Reduce` body (read-only indirection); extend to handle gather-in-reduce |
| s4116 | `s = 0; for i: s += a[off] * aa[j-1, ip[i]]; sum_out[0] = s` | gather + sum reduction (2D) | same as s4115 |
| s318  | `index=0; maxv=|a[0]|; for i: if |a[k]|>maxv: index=i; maxv=|a[k]|; k+=inc` | max-with-index + stride | complex: a max-tracking + argmax. Lower priority than s314/s316 |
| s332  | `for i: if a[i]>threshold: index=i; value=a[i]; break`       | conditional + break     | break makes this genuinely sequential; leave as ✗ |

**Expected impact:** ~10 kernels move from `(1,0,0)` to `(0,0,1)`. Final-parallelism rate would climb from 47.4% → ~54%.

## ☐C Missed forward-read parallel patterns — priority 2

Loops where the read offset is *forward* from the write (`a[i+k]` read, `a[i]` written, `k > 0`) have no carried dependence and are parallel. Canon already handles the linear case (s121/s131/s151 win above), but not these variants:

| Kernel | Pattern | Why missed | Fix |
|---|---|---|---|
| s112  | `for i in range(LEN-2, -1, -1): a[i+1] = a[i] + b[i]`        | reverse iteration not normalised | `NormalizeReverseLoops`: rewrite `range(end, -1, -1)` as `range(0, end+1)` with mirrored index in the body |
| s116  | `for i in range(0, LEN-4, 4): a[i]=a[i+1]*a[i]; a[i+1]=a[i+2]*a[i+1]; a[i+2]=a[i+3]*a[i+2]; a[i+3]=a[i+4]*a[i+3]` | 4×-unrolled chain of forward reads | `RerollUnrolledLoops` to fold to `for i: a[i] = a[i+1]*a[i]`, then L2M recognises forward-read parallel |
| s175  | `for i in range(0, LEN-inc, inc): a[i] = a[i+inc] + b[i]`    | symbolic stride blocks L2M — same root cause as **s172** | same fix as s172 |
| s1113 | `for i: a[i] = a[LEN//2] + b[i]`                             | reads `a[const]`; if `i == LEN//2` for some iter, RAW conflict. Conservatively refused. | requires "constant-index read with write at a function of `i`" carry-check: write subset `i` covers `LEN//2` only at one iter — no inter-iter conflict |

## ☐D Missed triangular / 2D-mixed-parallelism — priority 3

Nested 2D loops where at least one dimension is parallel. Need `MapFission` (or partial L2M) on the parallel dim while keeping the sequential dim sequential.

| Kernel | Pattern | Parallel dim | Sequential dim |
|---|---|---|---|
| s114  | `for i: for j in range(i*VLEN): aa[i,j] = aa[j,i] + bb[i,j]` | inner j | outer i (triangular bound) |
| s115  | `for j: for i in range(j+1, LEN): a[i] -= aa[j,i]*a[j]`     | inner i | outer j (reduction-like) |
| s118  | `for i: for j in range(i): a[i] += bb[j,i]*a[i-j-1]`        | neither (carries via a[i-…]) | both — actually likely ✗ |
| s3110, s3111, s3112, s3113 | various 2D | TBD on inspection | TBD |

## ☐E Missed conditional-parallel patterns

| Kernel | Pattern | Why missed | Fix |
|---|---|---|---|
| s161  | `for i: if b[i]<0: c[i+1]=…; else: a[i]=…` | write `c[i+1]` is output-distance 1 — conservatively refused | anti-dep recognises output-distance ≥1 with no in-iter read as parallel |
| s162  | `if k>0: for i: a[i]=a[i+k]+…`                            | outer `if` over a free symbol `k` prevents L2M from descending into the inner loop | gate-the-guard: lift the conditional, parallelize the inner loop, then re-guard |
| s277  | `for i: if a[i]<0: if b[i]<0: a[i]+=c*d; b[i+1]=c+d*e`     | output-distance 1 write to b in branch | same fix as s161 |

## ☐SCATTER (3 + 1) — needs runtime permutation check (see TODO below)

| Kernel | Pattern | Parallel iff |
|---|---|---|
| s491  | `a[ip[i]] = b[i] + c[i]*d[i]`            | `ip` is a permutation (no duplicates) |
| vas   | `a[ip[i]] = b[i]`                        | `ip` is a permutation |
| s4113 | `a[ip[i]] = b[ip[i]] + c[i]`             | `ip` is a permutation (write *and* read via ip) |
| s171  | `a[i*inc] = a[i*inc] + b[i]`             | `inc != 0`; equivalent to a permutation when `0 < inc < LEN/N_iters` |

The pattern is mechanical and identical: a once-per-call O(n) "is-permutation" check on the index array, gating the parallel map vs. a sequential fallback. See the TODO entry in memory.

## ✗ Genuinely sequential — correct refusal (39)

These have a true carried dependence, an early-exit break, or an unknown
alias that no local analysis can disprove. Listed for completeness so we
don't waste cycles re-visiting them.

| Kernel | Pattern shorthand | Reason |
|---|---|---|
| s1213 | `a[i]=b[i-1]+…; b[i]=a[i+1]*…`           | cross-array forward+backward carry |
| s1221 | `b[i] = b[i-4] + a[i]`                   | 4-step carry on b |
| s122  | `k+=j; a[i] += b[LEN-k]`                 | prefix-sum on k |
| s123  | `j+=1; a[j]=…; if c[i]>0: j+=1; a[j]=…`  | conditional pack on j |
| s124  | `if b[i]>0: j+=1; a[j]=…`                | conditional pack on j |
| s125  | `k+=1; flat[k] = aa+bb*cc`               | flat 2D pack on k |
| s128  | `k=j+1; a[i]=b[k]-d[i]; j=k+1; b[k]=…`   | dual pack on j/k |
| s132  | `aa[0,i] = aa[1,i-1] + …`                | forward carry on aa[1,*] |
| s141  | flat 2D pack on `k = (i+1)i/2 + …`       | triangular index recurrence |
| s13110 | 2D max-with-index + break (`break` if found) | sequential due to break + index carry |
| s171  | `a[i*inc] = a[i*inc] + b[i]`             | unknown stride alias — *see SCATTER above for runtime-check variant* |
| s174  | `a[i+M] = a[i] + b[i]`                   | alias depends on M ≥ LEN |
| s211  | `a[i]=b[i-1]+…; b[i]=b[i+1]-…`           | mutual b carry |
| s212  | `a[i]=a[i]*c[i]; b[i]=b[i]+a[i+1]*d[i]`  | WAR on a[i+1] |
| s221  | `b[i] = b[i-1] + a[i] + d[i]`            | true carry on b |
| s2251 | `s = b[i]+c[i]; a[i]=s*e; …`             | scalar `s` carry |
| s241  | `b[i] = a[i] * a[i+1] * d[i]`            | WAR on a[i+1] |
| s242  | `a[i] = a[i-1] + …`                      | true carry on a |
| s244  | `a[i+1] = b[i] + a[i+1]*d[i]`            | output dep on a[i+1] |
| s252  | scalar `t = s` carry                     | scalar carry |
| s254  | scalar `x = b[i]` carry                  | scalar carry (would need reverse pass to expand) |
| s255  | scalar `x, y` carry (2-deep)             | scalar carry |
| s256  | `a[j] = 1 - a[j-1]; aa[j,i] = …`         | inner j-carry |
| s257  | `a[i] = aa[j,i] - a[i-1]`                | outer i-carry inside j-loop |
| s258  | scalar `s` under branch                  | conditionally carried scalar |
| s261  | `a[i] = t + c[i-1]; c[i] = c[i]*d[i]`    | c-carry |
| s281  | reverse read of a + simul write          | partial aliasing |
| s321  | `a[i] = a[i] + a[i-1] * b[i]`            | true carry |
| s322  | 3-term recurrence                        | multi-step carry |
| s323  | `a[i]=b[i-1]+…; b[i]=a[i]+…`             | cross-array recurrence |
| s3251 | `a[i+1]=b[i]+c[i]; b[i]=c[i]*e[i]; d[i]=a[i]*e[i]` | a[i+1] forward output dep |
| s331  | `if a[i]<0: j = i`                       | scalar `j` carry |
| s341  | `if b[i]>0: j+=1; a[j]=b[i]`             | pack on j |
| s342  | `if a[i]>0: j+=1; a[i]=b[j]`             | pack with self-read |
| s343  | 2D pack on k                             | flat-index pack |
| s453  | `s += 2.0; a[i]=s*b[i]`                  | scalar `s` arithmetic carry |
| s481  | `if d[i]<0: break; a[i] += …`            | break (sequential by definition) |
| s482  | `a[i]+=…; if c[i]>b[i]: break`           | break |
| s2111 | `aa[j,i] = (aa[j,i-1] + aa[j-1,i])/1.9`  | both-dim recurrence |

## Per-kernel quick index

Generated by the script. Re-run after fixes to see deltas:

```
python -m tests.corpus.measure_parallelization | grep -E '^\s+s|^\s+v'
```

Status markers: `=` (both pipelines same parallel-construct count), `+N`
(canon has N more parallel constructs than L2M), `-N` (canon has N fewer),
`R+N` (canon emits N reduces), `!seq+N` (regression: canon left N more
loops sequential).

## Future ideas / TODOs

### Scatter with runtime permutation check

For `a[ip[i]] = f(b[i], c[i], …)` patterns (s491, vas, s4113), emit
**both** a parallel map and a sequential loop, dispatched by a one-time
permutation check on `ip`. The check is `O(n)` (mark/visit or sort+adjacent-
compare) and amortizes to nothing for long-running kernels or repeated
invocations with the same `ip`. Simpler variant: when the frontend can
prove `ip` is the same array between calls (write-once / constant input),
hoist the check out of the kernel call site.

See `~/.claude/projects/-home-primrose-Work/memory/project_scatter_runtime_permutation_check.md`
for the full design sketch and prerequisites.

### Reverse-loop normalization

Rewrite `for i in range(end, -1, -1): body(i)` into
`for i in range(0, end+1): body(end - i)` so L2M sees a forward-iterating
loop. Useful for **s112** and likely several others not yet identified.

### `UniqueLoopIterators` coexistence with L2M

Today `UniqueLoopIterators` rewrites stride/start loops to a normalised
`_loop_it_0` form that L2M does not recognise as a unique-index pattern
(see **s172** above). Either L2M follows the symbolic substitution or
`UniqueLoopIterators` skips loops whose original iterator is itself the
unique-index symbol of a body access.

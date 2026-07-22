# Mid-flight relayout: audit + cost model + related work

## The claim under audit

Layout papers say layout wins most of the performance. But GPU ablations tell a different story:
**schedule alone (tiling, fusion, parallelization, coalesced mapping) buys ~85%; picking the best
per-nest layout on top buys only ~5% more.** Read literally: per-nest layout barely matters.

That reading is right and misleading at the same time. It measures the wrong thing.

## What the 5% actually measures

The 5% is the value of the *single-layout-per-array* choice: fix one layout for an array and let the
schedule do the rest. When one array is used by several nests that all want the SAME orientation, the
schedule already coalesced it — layout adds almost nothing. That is the common case, hence 5%.

Layout stops being marginal exactly when two phases of the program want OPPOSITE layouts of the same
array. Then no single layout is good; the schedule cannot fix it (a loop permutation that helps one
phase hurts the other); and the real decision is whether to pay ONE layout-transformation phase — a
transpose — between the two phase groups. That is a mid-flight relayout, and it is where the missing
performance lives on big problems.

## Witness: `tests/transformations/layout/phased_relayout_test.py`

Eight nests over one input `A`: nests 0-3 read `A[i,j]` (row-major happy), nests 4-7 read `A[j,i]`
(column-major happy), with straight `P` + a reversed running array pinning the schedule to `(i,j)`
so `A`'s transposition is a genuine layout conflict, not a schedule artifact. Model cost table:

    array   conflict   chosen                         single-layout   global (mid-flight)
    A       yes        id·4 then perm10·4             1.89e-4         5.36e-5   (3.5x)

The per-array Viterbi DP inserts exactly ONE transpose at the 3->4 boundary; `brute_force_trajectories`
confirms it is optimal; the single-layout regime is 3.5x worse. This is the local, cost-model-placed
form of the OMEN transpose (`tests/library/mpi/mpi_omen_transpose_test.py`), where the same two-phases-
disagree decision is hand-placed across MPI ranks.

## The cost model (already in tree)

`cost_model/relayout.py` — when a mid-flight change pays for itself:

    break_even_uses = t_relayout / (t_nest_row - t_nest_col)      # consuming nests before it pays
    insert iff  (# consuming nests) >= break_even_uses

`global_assign.py::per_array_dp` — the global placement: a per-array Viterbi over the layout trajectory
across the line graph, pricing each `streaming_relayout_time` boundary edge against the node-cost
savings, with `brute_force_trajectories` as the enumeration oracle. `assignment_costs.py` fills the
table from either the LogGP model or measured medians. So the "cost model for a mid-flight layout
change" is built and validated; this audit is what it is FOR.

**Why "big problems".** The transpose is O(bytes); each consuming nest's saving is also O(bytes), so the
break-even count is scale-invariant in the ratio — but the amortizing quantity is the number of nests
that reuse the relaid array. Long phase groups (many consuming nests) clear break-even; short ones do
not. Bigger arrays raise the absolute saving and, on real machines, deepen the strided penalty, so the
threshold gets easier, not harder, with size.

**Why GPUs make layout matter more here — but only in the latency regime.** `EXAMPLE_GPU` (A100-class:
128B line, 32B coalescing sector, ~500 ns latency, ~1.55 TB/s) makes this precise. On the transposed
witness nest (N=256, `t(row-major, read transposed) / t(column-major)`):

    CPU  bandwidth (inf conc)       7.76x        CPU  latency (core_mlp=24)    7.76x
    GPU  bandwidth (inf conc)       3.94x        GPU  latency (core_mlp=500)  15.06x

Two facts fall out. (1) CPU is regime-invariant — `line == sector`, so bytes and messages scale together.
(2) On GPU the small 32B sector makes a strided read RELATIVELY CHEAPER than CPU's 64B line under pure
bandwidth (3.94x < 7.76x); the real coalescing penalty is the TRANSACTION COUNT (messages), which binds
only in the latency term `messages * L / concurrency`, i.e. under FINITE device MLP — there it explodes
to 15x. So the mid-flight transpose pays MOST on a latency/occupancy-bound GPU nest, and the model prices
that correctly ONLY when fed a finite concurrency (`n_cores * core_mlp`), never the parallel-schedule
`inf` default. Locked in `cost_model_gpu_relayout_test.py`. The 85/5 ablation and "layout matters" are
consistent: per-nest layout is 5%, but the phase-boundary transpose is a separate lever — up to 15x on a
latency-bound array — the ablation never varied.

## Related work

- **SmartMem** (ASPLOS'24, arXiv:2404.13528) — the dual approach: classify operators into 4 groups and
  choose layouts so producer-consumer chains SHARE one layout, ELIMINATING transforms. We instead PLACE
  the minimal transform when elimination is impossible (opposite-preference phases). Same objective —
  minimize transform traffic — attacked from the two ends. Their reported 2.8-7.9x on mobile DNNs is the
  size of this lever when it is not modelled.
- **Tensor Memory Engine** (arXiv:2604.13319) — the hardware analog: reorganizes tensor layout on the
  data path "when beneficial," decoupling compute from layout. Their "when beneficial" is our
  `break_even_uses` decision moved into hardware.
- **Hexcute** (arXiv:2504.16214) and **Linear Layouts** (arXiv:2505.23819) — layout *synthesis* / an
  F2 layout algebra for GPU codegen; the search/representation side of choosing the per-nest layout the
  ablation prices at 5%.
- **TVM / graph-compiler layout propagation** — anchor a layout at conv/matmul, propagate through
  layout-agnostic ops to minimize transposes, with a cost model weighing transpose traffic vs an
  alternate intrinsic. The per-array DP here is the same idea made optimal over a line graph rather than
  greedy propagation.

## Takeaways for the benchmark suite

1. Per-nest layout sweeps (k01-k18) measure the 5% lever — keep them, but they are not where layout wins.
2. The mid-flight lever needs opposite-preference phase groups: `phased_relayout_test` (synthetic 4+4),
   `conflict3` (1 producer + 2 transposed readers), and the OMEN transpose (distributed). These are the
   witnesses that actually move the needle.
3. `EXAMPLE_GPU` LogGP params landed; the model quantifies the GPU strided penalty (up to 15x on a
   latency-bound nest) once given a finite concurrency. Still open: a schedule x layout brute-force to
   reproduce the 85/5 split directly and show the phase-boundary transpose as the third axis the
   ablation omitted; and a measured microbenchmark fit (`fit_message_size` + `validate`) to replace the
   illustrative `EXAMPLE_GPU` numbers on a real device.

## How to make the LogGP model actually price a GPU relayout

The model already prices layout structurally: `count_loop_nest` derives per-array `(messages, sectors,
bytes)` from the access subset by counting UNIQUE blocks touched, so a strided/transposed read counts
more blocks -> higher cost, no measurement needed. What was missing was not modelling but instantiation:

1. **GPU parameters** — `EXAMPLE_GPU` in `assignment_costs.py` (line/sector split is load-bearing).
2. **Finite concurrency** — `exposed_concurrency` returns `inf` for a parallel schedule with
   `n_cores=None`, which collapses the latency term and hides coalescing. Pass `n_cores` (or a device MLP
   cap) so `messages * L / concurrency` engages; that term IS the coalescing penalty.
3. **Regime awareness** — report which term binds (`LoopNestLogP.regime`). A relayout that looks marginal
   under bandwidth can be decisive under latency; the decision must be made in the regime the nest runs in.

The break-even and per-array DP are unchanged — they consume whatever node/relayout costs the parameter
set produces. So "making it work" for GPU is parameters + finite MLP, not new machinery.

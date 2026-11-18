# Fusion audit (canonicalization and specialization)

Status 2026-11-18. Static audit: nothing here was compiled or run. Line numbers refer to branch `fix-1117`.

## Where fusion runs

| Pass | Runs in | Call site |
|---|---|---|
| `FuseConsecutiveLoops` | canon, `normalize_statements` | `canonicalize/pipeline.py:816` |
| `FuseChainedScalarReductions` | canon, `derive_parallelism` | `pipeline.py:954` |
| `FuseLoops` (`LoopFusion`), GPU after `ReorderStateForLoopFusion` | canon, `derive_parallelism` and every fusion round | `pipeline.py:997-998`, `:1021` |
| `FuseElementwiseIntoGemv` | canon, `recompose` | `pipeline.py:1089` |
| `FuseStates` (`StateFusionExtended`, `BlockFusion`) | canon structural cleanup, fusion rounds; finalize | `pipeline.py:310`, `:1019`; `finalize.py:879`, `:1027`, `:1099` |
| `FuseMaps` (`MapFusionVertical`, `MapFusionHorizontal`, `MapPartialReduceFusion`) | canon fusion rounds; finalize DOT gather; offload | `pipeline.py:1027`; `finalize.py:754`; `offloading/offload_to_accelerator.py:670` |
| `FuseMapsAtSharedAxes` | canon fusion rounds | `pipeline.py:1033` |
| `MapReduceFusion`, `MapWCRFusion` | canon fusion rounds | `pipeline.py:1036` |
| `MergeGuards`, `SelectBranchMaps`, `LiftTrivialIf` | canon outer fusion round | `pipeline.py:1048-1051` |
| `maximal_fusion` again, after `MoveMapInvariantIfUp` | CPU and GPU specialization (`hoist_invariant_guards`) | `interstate/move_map_invariant_if_up.py:950`; `cpu_specialization/pipeline.py:103`; `gpu_specialization/gpu_specialization_pipeline.py:36` |
| `RecomputeIntermediates` (`OTFMapFusion`) | CPU specialization | `cpu_specialization/pipeline.py:107` |
| `FuseScanChains` | CPU specialization | `cpu_specialization/pipeline.py:108` |
| `FuseCopyChains`, `recompute_fuse_for_gpu` (`OTFMapFusion`) | GPU offload | `finalize.py:850`, `:856` (`:783`) |

## Refusals

"Required" means a fusion past the check would compute wrong values. "Conservative" means it would be correct in some
of the refused cases.

| Pass | Refusal | Where | Required? | Status |
|---|---|---|---|---|
| MapFusionVertical | parameters paired by name, then by order: with two axes over one range, `T[i, j]` written and `T[j, i]` read were paired as two elements | `map_fusion_helper.py:19` | no | **fixed** (second review, below) |
| MapFusionVertical | different schedules | `dataflow/map_fusion_helper.py:1242` | no, but schedules are unset in canon | by design |
| MapFusionVertical | ranges not structurally equal (no offset, no symbol relation) | `map_fusion_helper.py:92`, `:112`; prefilter `passes/fuse_maps.py:95` | conservative | by design: origins are normalized upstream (`NormalizeLoopAndMapOrigin`), steps by `align_maps_to_unit_step`. Symbols compare by name (`symbolic.py:390`), so a dtype or assumption difference no longer splits a range |
| MapFusionVertical | maps in different scopes | `map_fusion_helper.py:1256` | yes for this rewrite | by design |
| MapFusionVertical | dynamic map ranges bind different values | `map_fusion_helper.py:1266` | yes | by design |
| MapFusionVertical | conflicting accesses inside the bodies / through the scope nodes / a read moved past a write | `dataflow/map_fusion_vertical.py:454`, `:469`, `:480` | yes | by design |
| MapFusionVertical | intermediate is a View, or neither Array nor Scalar | `map_fusion_vertical.py:971`, `:976` | conservative (reshapes) | by design; `RemoveViews` runs before |
| MapFusionVertical | reduced intermediate of symbolic size | `map_fusion_vertical.py:1019`, `:1066` | no | canon sets `allow_intermediates_with_symbolic_size=True` |
| MapFusionVertical | WCR producer edge | `map_fusion_vertical.py:1040` | yes for a partial result | by design; `MapWCRFusion`, `MapPartialReduceFusion` take the reduction cases |
| MapFusionVertical | conditional (dynamic) producer, unless reader runs under an equal guard (`guarded_alike`, `:2899`) | `map_fusion_vertical.py:1045` | no when fused in place (`in_place_intermediates`, `:2343`: a skipped element keeps its value) | by design: a performance guard (commit `6003805d8`, warpx 37 -> 1603 nested SDFGs) |
| MapFusionVertical | two producers of the intermediate may overlap | `map_fusion_vertical.py:1108` | yes | by design |
| MapFusionVertical | consumer read not covered by exactly one producer of the same iteration: gathers `t[idx[i]]` of the intermediate, data-dependent subsets, reads of another iteration | `map_fusion_vertical.py:1178` | yes (in place: unless no other iteration writes it) | by design |
| MapFusionVertical | shared intermediate read again downstream (strict dataflow, reduced) | `map_fusion_vertical.py:1240` | yes as implemented | by design. `relaxed_dataflow` does NOT lift it: it did until `71d0085fe`, which restored the refusal for a privatized (reduced) copy after fuzz seed 1093 read it stale. Only an in-place intermediate skips it |
| MapFusionVertical | nested SDFG reads the intermediate through an InOut connector, or its strides cannot be rewritten | `map_fusion_vertical.py:2781`, `:2746` | conservative | by design |
| OTFMapFusion | producer with more than one output | `dataflow/otf_map_fusion.py:69` | conservative (multiple outputs) | by design |
| OTFMapFusion | intermediate written by a second node / producer not its only writer | `otf_map_fusion.py:71`, `:116` | yes (masked writes after the producer) | by design |
| OTFMapFusion | dynamic producer write | `otf_map_fusion.py:160` | yes | by design |
| OTFMapFusion | producer parameter not solved from the consumer read (whole-range nested SDFG writes) | `otf_map_fusion.py:240` | yes | by design |
| FuseMapsAtSharedAxes | only top-level maps and transient intermediates | `canonicalize/fuse_maps_at_shared_axes.py:91`, `:97` | conservative (nested scopes) | by design |
| FuseMapsAtSharedAxes | shared axis picked by range alone: with two producer axes over one range the first one was tried, and a refused split was never retried with the other | `fuse_maps_at_shared_axes.py:23` | no | **fixed** (axis read off the intermediate's index, below) |
| FuseMapsAtSharedAxes | consumer split only when the shared axes are its outermost | `fuse_maps_at_shared_axes.py:111` | no (loop order only) | **proposed** (warpx, below) |
| StateFusionExtended | side-effect nodes, scoped writes the other state touches, reused transients with several producers | `interstate/state_fusion_with_happens_before.py:408`, `:460-487`, `:498` | yes | by design |
| StateFusion / maps across control flow | states in different `ConditionalBlock` arms or on either side of one never share a state | structural | yes without duplication | by design: fusing across a guard means pushing the guard in (`MoveIfIntoLoop`, `MoveIfIntoMap`) or duplicating the other side |
| LoopFusion | edge between the loops carries assignments or a condition; different iteration spaces; independently DOALL loop | `interstate/loop_fusion.py:104-116`, `:206` | yes / conservative | by design |
| MoveMapInvariantIfUp (spec.) | unswitching `map {B; if c: A}` copies `B` into both arms | `interstate/move_map_invariant_if_up.py` docstring | n/a | note: code duplication in specialization, not canon |

## Change made

`FuseMapsAtSharedAxes` matched each consumer axis to the first producer axis with an equal range. When two producer
axes share that range (`T[i, k]` over `0:8, 0:8`, read as `T[3, j]`), it split at `i`, the vertical fusion refused
(the iteration `i = j` writes row `j`, not column `j`), and the merge-back left the pair unfused. Now the pass reads
the intermediate's index on both sides first: a dimension that the producer's writes index by one parameter alone and
the consumer's reads index by one parameter alone pairs those two parameters (`index_pairs`, `bare_indices`). The
range rule fills in the axes without such a pair. Legality stays with `MapFusionVertical.can_be_applied`, so a wrong
pairing can only cost a fusion. A pair of maps with an equal number of axes is skipped before the walk.
Test: `tests/passes/canonicalize/fuse_maps_at_shared_axes_test.py`
(`test_the_split_axis_is_the_one_the_consumer_indexes_when_two_share_its_range`). Not run.

## warpx_field_gather trace

Kernel: `hpcagent_bench/benchmarks/scientific_computing/n_body_methods/field_gather/warpx_field_gather_dace.py`
(generated from the numpy file). `geom = 3` and `galerkin_interpolation = 1` are module constants; `depos_order` is a
runtime symbol. Numba (`warpx_field_gather_numba.py`) runs one `prange` over particles with the order branch inside
the loop body, all shape factors in per-particle registers.

Evidence: no current CPU canonical build is on disk. The newest canonical graph is the GPU prerender of 2026-11-10
(`.hpcagentbench-cache/.cpf-prerender/cache/59/5919c9e4.../canonical.sdfgz`). It predates `3630e92d7` (guarded producer
fusion, 11-12), so it shows the structure but not the current fusion verdicts. The plain DaCe CPU build in
`.hpcagentbench-cache/dace_numeric/warpx_field_gather_rank0/` is not canonicalized.

What the graph holds, per axis (x, y, z) and per field type selector:

1. `compute_shape_factor_into` writes `sx_node[k, :]` (literal rows `k`), `idx[:]`, in five arms guarded by
   `depos_order == 0..4`, nested in the `ey_type[0] == NODE` elif chain. Inside one arm the per-particle maps fuse (the
   prerender shows one map per arm with `__map_fusion_*` scalars). Across arms nothing fuses: each arm writes the rows
   its order has, and the consumer reads `sx[0:depos_order + 1, i]`. That the five arms together cover those rows
   holds only per arm (`order == depos_order` there). Proving coverage needs path-sensitive facts, and fusing into the
   arms duplicates the gather five times, which is the cost trade canon does not make. **By design.** This leaves 12
   shape-factor planes `(o + 1, n)` and 12 index vectors `(n,)` per call materialized, written in the shape phase and
   read in the gather phase. Numba keeps all of them in registers.
2. `j_node[:] = idx` copies the arm result out of the type guard. That is a transient-forwarding case
   (`RedundantArray` / `EliminateCopyBack`), owned by the transient-removal audit, not fusion.
3. `_select` is `np.where(type[0] == NODE, node, cell)` over `(k, i)`, a select on a data scalar. The int casts
   (`.astype(np.int64)`) are tasklet casts. Neither blocks fusion. `_copy_sel` is dead in the kernel (never called).
4. `_tap3`: `ix/iy/iz` index planes, `gathered`, `weight` iterate `(tx, ty, tz, i)` with the particle axis innermost
   (numpy broadcasting). They fuse among themselves (the prerender holds `__map_fusion_*_ix_b` as `(1,1,1,1)`
   registers). The per-particle producers feeding them, `lox + j_ex` over `(i,)` and `sx_ex` over `(k, i)`, have
   fewer axes. `FuseMapsAtSharedAxes` would split the consumer at `i`, but refuses (`:111`) because `i` is not the
   consumer's outermost axis. On CPU, `RecomputeIntermediates` (OTF) can still recompute both inside the gather (one
   reader, single writer, point reads). It cannot do that for the shape-factor planes of item 1, which have several
   guarded writers (`otf_map_fusion.py:116`).
5. The `sz_*` rows are read at `sz[tz, i]`, where `ty` and `tz` both range over `0:depos_order + 1`. Matching by
   range alone paired `sz`'s row axis with `ty`. That is fixed above, but item 4's outermost rule still refuses
   these.

Conclusion: the 1.84x is dominated by item 1, by design. The cheaper structural blocker is item 4, and it is a loop
order choice.

## Proposals (not implemented)

- **Split a consumer at inner shared axes (`fuse_maps_at_shared_axes.py:111`).** Allow the split-consumer case when
  the shared axes are not the consumer's outermost: reorder the consumer to shared-axes-first, as the producer split
  already does unconditionally (`try_fuse`, `reorder(split, axes + rest)`). This makes `_tap*` one particle map with
  the taps inside, the numba shape. The cost: a consumer access that is indexed by the remaining axes and stays
  materialized gets an outer stride. That makes it a loop-order decision, so it needs a user ruling. The check was
  added by an agent (commit `aa3c92afb`, "so its loop order stays"), not by a recorded user decision. Expected warpx
  effect: removes `lox + j_*` and the `sx_*` selections as per-particle planes in the canonical form. On CPU, OTF
  already removes them, so the CPU gain is small. The GPU gain is larger.
- **Dynamic producer fused in place (`map_fusion_vertical.py:1045`).** Correct as is. Leave it, because the refusal
  guards the 1603-nested-SDFG regression.

## Second review (2026-11-18)

Static, like the first: nothing compiled or run. Each "required" verdict above was re-derived from the code. One was
wrong: the downstream-access refusal is not lifted by `relaxed_dataflow` (row corrected).

### Change made: parameter pairing read off the intermediate

`find_parameter_remapping` (`map_fusion_helper.py:19`) pairs the second Map's parameters with the first's by name,
then by position among equal ranges. Its own docstring names the failure: with two axes over one range there are two
valid renamings and only one fuses. The shape is a square intermediate read transposed: `T[i, j] = f` over
`0:N, 0:N`, then `B[i, j] = T[j, i]` over the same range. Name matching keeps `i -> i`, the consumer then reads the
element another iteration writes, and the fusion is refused. The frontend names every elementwise map `__i0, __i1`, so
the name step fires exactly in the common case. A non-square transpose fused already, because there the ranges tell
the axes apart (`fusion_rename` in `map_fusion_vertical_test.py`).

Now `MapFusionVertical` passes `preferred` pairs, read by `intermediate_index_pairs` (`map_fusion_helper.py:222`). A
dimension of the intermediate that every producer write indexes by one parameter alone and every consumer read
indexes by one parameter alone pairs those two parameters. The pair is taken first when the two ranges are equal.
`can_be_applied` and `apply` compute it from the same unchanged graph (`map_fusion_vertical.py:425`, `:653`).
`bare_indices` and `index_pairs` moved from `fuse_maps_at_shared_axes.py` into the helper, so both passes share one
definition.

Soundness: any renaming between equal ranges is an alpha-renaming of the second Map over the same iteration set, so
the preference only chooses which valid renaming the unchanged legality checks judge. It also cannot lose a fusion
the old pairing found. For a dimension that both sides index by single parameters, coverage holds only when the
renamed consumer index equals the producer index, which is the preferred pair. Unequal ranges drop the preference.

Counterexample checked: `B[i, j] = T[i, j] - T[j, i]`. The two reads disagree per dimension, so `bare_indices` gives no
pair, the old pairing runs, and `T[j, i]` stays uncovered: refused, correctly. Test:
`tests/transformations/map_fusion_index_pairing_test.py` (not run). Corpus: the frontend's 2-D `.T` is a `Transpose`
library node that canon keeps, so the shape shows up where a lifted loop nest or an explicit map indexes `[j, i]` over
a square extent (`qe_cegterg` `_conj_transpose`, `C[i, j] = conj(A[j, i])`, square when `n == m`; the TSVC `LEN_2D`
kernels). No kernel was confirmed to hit it, because no canonical graph was built.

### Proposals (not implemented)

- **Range spelling.** `iteration_space` (`passes/fuse_maps.py:94`) and `find_parameter_remapping` compare range
  tuples with structural `==`. Symbols are their names, so dtype and assumption duplicates no longer split a range.
  `int_floor(x, 2)` from a parsed string and `floor(x/2)` from Python's `//` in a shape still compare unequal.
  `symbolic.comparable` relaxes both to one head. Applying it per map in the prefilter and on the remapping's
  mismatch path is sound, since the fused Map keeps the first Map's range. It is not done: no corpus kernel was shown
  to reach fusion with both spellings, and the mismatch path is the hot one.
- **Equal-rank partial sharing in `FuseMapsAtSharedAxes`.** `candidates` (`fuse_maps_at_shared_axes.py:72`) skips a
  pair with equal axis counts. A producer over `(i, k)` and a consumer over `(i, j)` that reads `T[i, 0:K]` share only
  `i`, and fusing them needs both maps split. That is the inner-axis split question again, so it waits for the same
  ruling.

### Refusals re-confirmed required

Producer overlap (`:1108`), WCR producer, dynamic map ranges that bind different values, gathers `T[idx[i]]` of the
intermediate (no producer iteration covers a data-dependent read), and reads of another iteration (stencils
`T[i + 1]`). The guarded-producer refusal (`:1045`) and the conditional-arm producer are out of scope by user decision.

## Third review (2026-11-18): safe sub-cases of the required refusals

Static, like the first two: nothing compiled or run, every test below is unrun. The question was which structurally
provable sub-cases of the five required refusals fuse without a cost trade. Two are implemented, one more existed
already, and two have none worth implementing. The extent-spelling proposal of the second review and two user rulings
on `FuseMapsAtSharedAxes` are implemented too.

### Intermediate accessed again downstream (`map_fusion_vertical.py`, `_is_data_accessed_downstream`)

What the refusal guards. A shared intermediate `T` fused through a privatized buffer is rebuilt behind the fused Map:
the fused exit writes the node `T_a`, and `T_a` keeps every out edge except the ones into the second Map. A direct
reader of `T_a` was never refused: the walk only counts another AccessNode of `T`. So the refusal is about a second
node `T_b` of `T` reachable from `T_a`.

Mechanism of `71d0085fe` / seed 1093 (inferred from the code: the commit carries no test and no reproducer is on
disk). A `T_b` reached from `T_a` only through the second Map loses its order with `T_a`. Before the fusion the path
`T_a -> second Map -> ... -> T_b` orders them. After it, both hang off the fused exit as siblings
(`fused exit -> T_a` and `fused exit -> O -> ... -> T_b`), and no path joins them. Graph analyses that order
accesses by paths then see two unordered accesses of `T`, which is where a stale read can come from. The existing test
`test_fusion_dataflow_intermediate_downstream` is this shape: `T_b` written by a Map that reads only the second Map's
output.

Safe sub-case, implemented under `relaxed_dataflow` (canon's setting). A `T_b` that `T_a` still reaches on a path that
avoids the second Map entry. Proof: the rewrite only replaces `first exit -> T_a` by `fused exit -> T_a` and drops
`T_a -> second entry`. A path from `T_a` that does not enter the second Map runs through nodes the rewrite does not
touch, since its scope can only be entered through its entry. So `fused Map < T_a < T_b` holds in the graph, and the
fused Map writes `T` exactly as the first did (the shared write-back preconditions: no WCR, no dynamic write, the NSDFG
coverage check). The second Map's reads of `T` now come from the buffer and are no longer ordered against `T_b`. If
`T_b` was also reached through the second Map, the order holds through `T_a` again. If it was not, the original graph
left the second Map and `T_b`'s writer unordered, and the fused one picks one valid order.
Code: `_is_data_accessed_downstream(..., avoid=second_map_entry)`. It refuses only an access that every path reaches
through `avoid`. Without `relaxed_dataflow` it behaves as before.
Tests: `tests/transformations/map_fusion_safe_subcases_test.py`
(`test_an_intermediate_written_again_past_another_reader_fuses_under_relaxed_dataflow`, numeric;
`test_an_intermediate_written_again_only_past_the_second_map_stays_refused`, the seed-1093 shape).

The write-through variant (no buffer, consumer reads in-iteration) already exists. A second-Map write, a conditional
write, or a non-point-wise read-write makes the intermediate in place, and an in-place intermediate skips this
refusal.

### Reducing (WCR) producer

What existed. `MapPartialReduceFusion` covers a `Reduce` library node with an identity: it builds
`Map[kept]{acc = identity; Map[reduced] acc (+)= ...; T = acc}`. A raw WCR Map (`T[i] += A[i, k]`, from `LoopToMap` or
`AccumulatorCopyChainToWCR`) did not fuse. `FuseMapsAtSharedAxes` splits it into `Map[i]{Map[k] T[i] += ...}`, which is
exactly the inner-reduction shape, but `MapFusionVertical` refused every WCR producer edge.

Why the general refusal holds. A privatized buffer has no start value: the WCR needs `T[i]`'s value before the
reduction (an earlier zero fill), and the buffer would start from garbage. Also, if another iteration contributes to
`T[i]`, no single fused iteration holds the finished sum.

Safe sub-case, implemented under `relaxed_dataflow`. The intermediate is fused in place (the container keeps the start
value), and no other iteration of the first Map reduces into the elements this iteration reduces into
(`other_iterations_miss(first_map, [wcr subset], [wcr subset])`, the same test the in-place path already uses). Proof:
in the fused body the consumer reads `T` from the node that the WCR edge enters, so it runs after every contribution
of this iteration (an inner Map's exit completes before its successors). No other iteration touches those elements,
so the value read is `start + sum over this iteration`, which is what the unfused consumer read. The rebuilt edge
behind the fused Map is a plain ordering copy, so its `wcr` is cleared; a copied WCR would add `T` onto itself.
Code: `in_place_intermediates` adds a WCR intermediate that the second Map reads. `partition_first_outputs` admits the
WCR edge only on that condition. `handle_intermediate_set` clears `wcr` on the in-place rebuild. `T` stays full size:
shrinking it would need a seed copy-in, which is a code change, not a fusion.
Tests: `tests/passes/canonicalize/fuse_maps_at_shared_axes_test.py`
(`test_a_reduction_over_the_unshared_axis_completes_inside_the_fused_iteration`, numeric;
`test_a_reduction_every_iteration_contributes_to_keeps_the_producer`, `T[0] += A[i, k]`).

### Overlapping producers (`:1108`)

The check compares the per-iteration subsets of two producer edges of one iteration, in the first Map's parameters.
An overlap is two writes of one element in one iteration of the first Map. Two cases:
- The two writers are unordered in the body. Then the first Map already races, and only "equal values" would make it
  defined. That is not a structural fact (one tasklet connector with two edges is the only provable form, and the
  frontend does not emit it). Refused, required.
- One writer reaches the other through the body (`T[i] = a; ...; T[i] = b` in sequence). The last writer is defined,
  and fusing keeps that order: both edges are retargeted to the buffer node, which the consumer reads after both.
  This is provably safe, but it needs three changes: `tiled_producer_subset` (equal volumes reject an overlap),
  coverage by "exactly one producer" (`:1178`), and a path test between the producers. **Not implemented.** No corpus
  shape was shown to reach it, and three changes to the core coverage logic with no reproducer go against the
  minimal-diff rule.

### Gathers and stencils (`:1178`)

Both stay refused: covering them means recomputing the producer per consumer read, which is the recompute-vs-store
trade (CPU specialization's `RecomputeIntermediates` makes it, canon does not).
- Gather `T[idx[i]]`: the safe sub-case is an index that is not data-dependent after simplification (`idx` a known
  identity or affine table). That is index forwarding upstream of fusion, not a fusion rule. Once forwarded, the read
  is `T[i]` or `T[a*i + b]` and the existing coverage test judges it.
- Stencil `T[i + 1]`: the only safe sub-case is an offset along an axis the producer writes whole per iteration (rows
  `T[i, 0:M]` read at `T[i, j + 1]`). The coverage test already accepts it.

### Extent spelling (second-review proposal, implemented)

`mfhelper.same_range` compares two `(begin, end, step)` triples structurally, then by
`symbolic.equal(a, b, is_length=False) is True`. `find_parameter_remapping` uses it at all three of its range
compares (preferred pairs, name matches, candidates), and `shared_axes` in `FuseMapsAtSharedAxes` uses it at both of
its compares. The `FuseMaps` prefilter `iteration_space` keys each non-atomic bound by `symbolic.comparable`, so
`int_floor(x, 2)` and `floor(x/2)` land on one key. Only key-equal pairs reach the remapping, and the remapping
accepts all of them. The fused Map keeps the first Map's range (`rename_map_parameters`). Open risk:
`symbolic.equal` already identifies `int_floor` with `floor` everywhere, but the emitted `int_floor` truncates on
negative numerators. For `0:x//2` both spellings give an empty map whenever `x < 0`. For `0:x//2 + 1` they differ at
`x = -1`: truncation gives one iteration and floor gives none. An extent symbol is nonnegative in canon
(`set_symbol_nonnegative_assumptions`), but nothing here proves `x >= 0` per range.
Tests: `map_fusion_safe_subcases_test.py` (`test_floor_and_int_floor_spell_one_range`,
`test_maps_over_one_extent_in_two_spellings_fuse` (numeric), `test_maps_over_different_floors_stay_apart`).

### FuseMapsAtSharedAxes user rulings (2026-11-18)

- **Inner shared axis (`:111`)**: canon does NOT reorder and fuse. `candidates` reports the pair to `inner_axis`.
  After the state's last walk, `note_inner_axis_pair` appends one line to both map entries' `specialization_hint`,
  once, below any existing hint:
  `fusion: shares (p) with <producer> at an inner axis; alt: reorder <consumer> to (p, t), fuse, w per iteration`
  (on the producer: `... with <consumer> at an inner axis of it; ...`). `AnnotateLoopKinds` keeps its class line
  first, since `fusion:` is not a class prefix. The graph is unchanged. This closes the "proposed" row above and the
  warpx item 4 proposal; the hint is what warpx's `_tap*` maps now carry.
- **Equal-rank partial sharing**: kept as is (not fused, no hint). This closes the second review's proposal.
Tests: `fuse_maps_at_shared_axes_test.py`
(`test_a_consumer_sharing_an_inner_axis_keeps_its_order_and_both_maps_get_a_hint`,
`test_an_existing_hint_keeps_its_line_and_the_fusion_note_follows`,
`test_maps_of_equal_rank_sharing_only_an_outer_axis_stay_apart_without_a_hint`).

### Integration

`tests/canonicalize/canon_fusion_fixes_integration_test.py` runs small numpy kernels through `canonicalize` and
`finalize_for_target(..., 'cpu')`, then compares against numpy. Each fix gets a positive case and a counterexample:
the shared-axes column pairing (`4aa8b9969`), the transposed-read pairing (`40e531a57`), the in-place reduction, the
downstream access, and the extent spelling. A full-size transient left behind means not fused. For a reduction, the
signal is a WCR edge or `Reduce` inside the map writing the result. The downstream counterexample asserts values only:
through the pipeline another fusion order (second and third Map first, then in place) can fuse it legally.

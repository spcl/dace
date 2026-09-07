# AGENTS.md -- working contract for this repository

For any coding agent working in a DaCe repository. Self-contained: nothing here
requires an external skill, plugin, or tool bundle to be installed.

Sections:

- **Part A -- DaCe contributor rules.** Repo layout, the utilities to use,
  symbolic/subset/memlet semantics, the gotchas that silently miscompile, lowering
  rules, how to write a test.
- **Part B -- Python quality gates.** Exact commands to run, and the style rules
  for new Python.

Markers used below:

- `RULE:` a hard rule. Each carries a one-line `WHY:`.
- `!! SILENT` a rule whose violation produces wrong code or wrong numbers with no
  error, no warning, and often no failing test.

Read Part A section "Gotchas" before touching any core file. Most silent
miscompiles in this project come from those rules.

Code is cited by file and symbol name, never by line number -- grep for the symbol,
because everything here drifts.

---

# Part A -- DaCe contributor rules

## A1. Where a fix goes

RULE: no patch-fixes. A special-case bypass, an elision, an "if this odd shape do X
instead" written around a design limit is a patch. An invariant CHECK that asserts
the design holds is the opposite of a patch and is wanted.
WHY: !! SILENT -- a patch masks the broken structure instead of fixing it, and the
structure comes back as a miscompile somewhere else. A band-aid that makes one
kernel pass leaves every other kernel on the same broken path.

RULE: structural fixes and features go in the LIBRARY -- `dace` itself, or the
bridge -- never inlined into a driver, repro script, benchmark harness, or test
helper. If a script needs something the library lacks, add it to the library and
call it.
WHY: an inlined copy forks the behavior, so the fix never reaches real users and
the two versions drift.

RULE: extend an existing pass or helper rather than adding a parallel detector.
Grep `dace/sdfg/utils.py`, `dace/sdfg/scope.py`, `dace/transformation/helpers.py`
and two or three sibling passes before writing a helper, and check whether
something later in the pipeline already handles it.
WHY: ~70 helpers exist, and a hand-rolled duplicate reproduces the subtle bugs the
existing one already fixed (empty-memlet ordering, WCR edges, symbol identity).
Trap: `sdutil.is_parallel(state, node)` dereferences the `None` its scope walk ends
on, so it raises on a top-level node -- use `is_in_scope`.

RULE: deprecated code is REMOVED, not shimmed. Delete the old knob, alias or entry
point and update every caller in the same change.
WHY: a delegating shim is dead weight and hides the real call graph; the next
reader cannot tell which path is live.

RULE: a new third-party import must also be declared in `pyproject.toml`
`dependencies`.
WHY: the local gates pass because the package happens to be installed already, and
CI collection then fails with `ModuleNotFoundError` across every test file.

RULE: after editing a `dace/` module, run `python -c "import dace"` and
`ruff check --no-cache <files touched>`.
WHY: a module that does not import breaks every consumer of the package, not just
your work, and `ruff.toml` selects only F401 -- an undefined name introduced by an
edit is not caught by the lint gate at all.

Tests live at `tests/**/*_test.py`; GPU tests at `tests/**/*_cudatest.py`.

## A2. Architecture in one breath

- An `SDFG` **is** a `ControlFlowRegion` (`dace/sdfg/sdfg.py`). Regions are
  directed graphs of `ControlFlowBlock`s connected by `InterstateEdge`s
  (condition + assignments).
- A block is either an `SDFGState` -- a dataflow multigraph
  (`OrderedMultiDiConnectorGraph`: nodes plus memlet edges with connectors) -- or a
  nested region: `ControlFlowRegion`, `LoopRegion` (init/cond/update),
  `ConditionalBlock` (list of (condition, region) branches). Break/continue/return
  are their own blocks.
- The hierarchy is recursive: regions inside regions, and `NestedSDFG` nodes inside
  states hold whole child SDFGs.

RULE: to reach if-branches and loop bodies use
`all_control_flow_regions(recursive=True)`, not `sdfg.nodes()`.
WHY: !! SILENT -- `sdfg.nodes()` sees only ONE region's blocks, so a pass written
against it silently ignores everything nested and reports success.

Classes: `SDFGState`, `AbstractControlFlowRegion`, `ControlFlowRegion`,
`LoopRegion` and `ConditionalBlock` in `dace/sdfg/state.py`; `NestedSDFG` in
`dace/sdfg/nodes.py`.

## A3. Graph library

Graphs are networkx, wrapped in DaCe-serializable classes in `dace/sdfg/graph.py`
(`Edge` / `MultiConnectorEdge`, `DiGraph`, `OrderedMultiDiConnectorGraph`, ...).
Edges carry typed `.data`; `state._nx` is the escape hatch into raw nx algorithms.

RULE: go through the wrapper classes; reach for `._nx` only for an algorithm the
wrapper does not expose, and do not store the raw handle on the graph.
WHY: the wrappers are what serialize. A raw nx object put back into the SDFG has no
`to_json` and the SDFG stops round-tripping.

## A4. Key utilities -- learn these before writing a pass

RULE: check these before writing a new helper.
WHY: ~70 helpers already exist; a duplicate helper diverges from the tested one.

`from dace.sdfg import utils as sdutil` -- `dace/sdfg/utils.py`:

- traversal: `dfs_topological_sort`, `scope_aware_topological_sort`,
  `find_upstream_nodes` / `find_downstream_nodes`, `weakly_connected_component`,
  `traverse_sdfg_with_defined_symbols`, `postdominators`
- edge/connector: `change_edge_src` / `change_edge_dest` (rewire, do not recreate),
  `remove_edge_and_dangling_path`, `consolidate_edges`, `in_edge_with_name`,
  `out_desc_with_name`
- memlets/scopes: `canonicalize_memlet_trees`, `dynamic_map_inputs`,
  `get_global_memlet_path_src` / `_dst`, `is_parallel`, `local_transients`,
  `trace_nested_access`
- data/symbols: `get_used_data`, `get_used_symbols`, `prune_symbols`
- structure: `fuse_states`, `inline_sdfgs`, `inline_control_flow_regions`,
  `set_nested_sdfg_parent_references`

`dace/transformation/helpers.py` -- `nest_state_subgraph`, `state_fission`, `tile`,
`permute_map`, `offset_map`, `redirect_edge`, `unsqueeze_memlet`,
`split_interstate_edges`, `modified_symbols_between`, `scope_tree_recursive`.

Transformation framework: `dace/transformation/transformation.py`
(`PatternTransformation`: `expressions()` / `can_be_applied()` / `apply()`),
`pass_pipeline.py` (`Pass`, `Pipeline`), `passes/pattern_matching.py`
(`PatternMatchAndApplyRepeated`); implementations under `dataflow/`, `interstate/`,
`subgraph/`, `passes/`.

`dace/symbolic.py` -- `symbol(name, dtype)` (default `DEFAULT_SYMBOL_TYPE` =
int32), `pystr_to_symbolic` (string to sympy), `symstr`, `issymbolic`, `evaluate`,
`simplify` (cached -- use this one), `SymExpr` (main + overapprox pair), `equal`
(tri-state: True/False/None), `equalize_symbols` / `inequal_symbols`,
`serialize_symbolic` / `deserialize_symbolic`,
`serialization_symbol_dtypes(authority)` context manager.

RULE: parse symbolic strings with `pystr_to_symbolic`, never `sympy.sympify`.
WHY: see gotcha G4 -- `//` semantics differ.

RULE: never call sympy directly -- everything goes through `dace.symbolic`. If a
sympy operation has no wrapper (`diff`, say), ADD the wrapper to `dace/symbolic.py`
rather than importing sympy at the call site.
WHY: !! SILENT -- a symbol reconstructed outside `symbolic` is not the instance
embedded in the subset, so the sympy operation answers for a different symbol.
Measured: `diff(r[0], symbolic.symbol(itervar))` returned 0 because the rebuilt
symbol carried different assumptions, every index read as non-injective, and three
legal transformations were refused.

RULE: when substituting or differentiating against an expression already in the
graph, take the symbol INSTANCE out of `expr.free_symbols`; never construct a new
one from the name.
WHY: same mechanism -- see G3.

RULE: symbolic expression to string is `symstr` (pass the container names via
`symbolic.arrays(expr)`), never `str()` or an f-string.
WHY: !! SILENT -- `str()` on an expression holding an array read prints the
internal `Subscript(g, i)` form, which codegen cannot compile.

RULE: build and rewrite TASKLET code with `dace/frontend/python/astutils.py`; build
and rewrite MEMLET subsets, shapes and ranges with `dace/symbolic.py`. Never
hand-build `ast.Call` / `ast.BinOp` trees, and never `re.sub` a name inside an
expression string.
WHY: the two are not interchangeable. `pystr_to_symbolic` fails on tasklet
constructs -- `dace.float64(c)` raises `TypeError: 'Attr' object is not callable`,
`a and b` raises `SympifyError`, `ITE(c,t,e)` becomes an opaque `Function` -- and
sympy REORDERS terms even where it parses. Regex renames miss word boundaries and
mishandle attributes and subscripts.

## A5. Gotchas -- the part that bites

### G1. Empty memlets are ordering edges, not degenerate data edges

`Memlet.is_empty()` (`dace/memlet.py`) -- data / subset / other_subset all
`None`. They enforce happens-before (tasklet to scope, control-flow ordering).

RULE: guard with `is_empty()` when iterating memlets; never field-test an empty
memlet; never delete one as "dead".
WHY: !! SILENT -- every field test on one lands in the wrong branch or raises, and
deleting one makes results `PYTHONHASHSEED`-dependent. ~50 `is_empty()` guards
already exist in transformations.

RULE: do not conflate `Memlet.is_empty()` with `SDFGState.is_empty()`.
WHY: the latter is a node count, an unrelated predicate.

### G2. WCR memlets are atomics: read-modify-write

`dace/runtime/include/dace/reduction.h` does `*ptr = wcr(*ptr, value)`.

RULE: dead-store and copy-forward passes must treat a WCR edge as a read as well
as a write.
WHY: !! SILENT -- treating it as a pure write drops the accumulator read and the
reduction returns partial results with no error.

### G3. Symbol identity is name-based; dtype is NOT part of identity on main

`_eval_subs` compares by name (`symbolic.py`). Two same-named symbols with
different dtypes alias silently -- the frontend mints map params as int64 via
`result_type_of`, consumers re-mint int32 via a bare `symbol('i')`. Putting dtype
in the hash instead makes 49 tests fail (measured; reverted).

RULE: never re-mint a symbol from a bare name -- resolve the instance from the
expression's `free_symbols`.
WHY: !! SILENT -- a re-minted symbol aliases the original at a different dtype.

RULE: compare via `equalize_symbols` / `inequal_symbols`, not raw `==`, across
dtype boundaries.
WHY: raw `==` cannot see the dtype difference.

RULE: wrap serialization in `serialization_symbol_dtypes({name: dtype})` when a
scope-authoritative dtype exists.
WHY: otherwise the dtype is lost on round-trip.

RULE: construct symbol-bearing DaCe function expressions uncached.
WHY: SymPy caches (`Function.__new__`, `eval()`) return equal-named wrong-dtype
expressions.

RULE: `DEFAULT_SYMBOL_TYPE` is `int32` (`dace/symbolic.py`) and stays `int32`.
Never widen it to make a dtype line up. Resolve the symbol from the subject
expression instead -- `symbolic.resolve_symbol(name, symbolic.symbols_in([...]))`.
WHY: !! SILENT -- widening the default retypes every bare `symbol('i')` in the
codebase at once, so an index that was int32 by contract silently becomes int64 in
signatures, in serialized SDFGs, and in every consumer that reads the symbol's
dtype. The defect being papered over is always local: one site minted a symbol from
a name instead of taking the instance already in the expression.

### G4. `//` in symbolic strings

RULE: emit `int_floor` / `int_ceil` in memlet, shape, and interstate strings.
WHY: !! SILENT -- `pystr_to_symbolic` maps `//` to `int_floor`; plain
`sympy.sympify` gives a rational, which changes the computed index. C `/` also
truncates toward zero, which differs from floor for negatives.

The rule splits by DESTINATION, not by syntax:

- symbolic -- interstate assignments and conditions, memlet subsets, shapes,
  strides, anything reaching `pystr_to_symbolic`: `int_floor(a, b)` /
  `int_ceil(a, b)`. `int_floor` is in the recognized call-target whitelist and
  works inside assignment strings.
- Python AST -- a `Language.Python` tasklet body, and a `@dace.program` body: plain
  `//`. Do not rewrite these; `int_floor` inside a `@dace.program` raises
  `DaceSyntaxError: Function "int_floor" is not registered with an SDFG
  implementation`.

RULE: an inverse must unwrap BOTH `int_floor` and `int_ceil`.
WHY: `floor(E/f)*f` does not fold back to `E` symbolically.

The real risk zone is a `//` string handed to `sympy.sympify` / `parse_expr`
instead of `pystr_to_symbolic`, and hand-written shape strings outside DaCe's
parser (YAML `derive:` entries and the like).

### G5. `reset_cfg_list()` / `reset_sdfg_list()` are expensive

They recompute the whole CFG tree and propagate to every region
(`state.py`).

RULE: after deepcopying graph parts, fix parents manually -- set `_parent`,
`_parent_sdfg`, `parent_nsdfg_node` from the memo. See `SDFG.__deepcopy__`
(`sdfg.py`) and `NestedSDFG.__deepcopy__` (`nodes.py`).
WHY: the reset call is O(whole tree) and runs on every invocation.

RULE: thread the SDFG explicitly through helpers that touch a detached copy.
WHY: a detached copy has `state.sdfg is None`, so helpers reading `.sdfg.arrays`
break.

### G6. Transformations are minimal graph operations

RULE: rewire edges (`change_edge_src` / `change_edge_dest`) instead of
remove-and-recreate.
WHY: !! SILENT -- constructors regenerate IDs while `__deepcopy__` preserves the
guid, so remove-and-recreate silently changes node identity.

RULE: a pass or transformation that does NOT apply must leave the SDFG
bit-identical. Decide on a deepcopy if unsure.
WHY: !! SILENT -- LoopFission once returned "nothing applied" after destroying
ordering.

### G7. Range/subset bounds must stay symbolic

`subsets.py` coerces in `__init__` and `__setitem__` (`tuple_to_symexpr`,
`symbolic_range_tuple`).

RULE: coerce a bound to symbolic at write time.
WHY: a raw Python int stored in a bound explodes much later in an unrelated pass as
`'int' object has no attribute 'match'`.

### G8. Never a plain `set`

RULE: use `OrderedSet` -- `from ordered_set import OrderedSet`, a declared
dependency in `pyproject.toml`. A `Set[...]` annotation is a smell -- write
`OrderedSet[...]`. This holds for membership-only sets too.
WHY: !! SILENT -- iteration order leaks into codegen, so the same SDFG compiles
differently per `PYTHONHASHSEED`. A membership-only set today gets iterated
tomorrow. Measured on a polybench kernel: correct code on one hash seed, wrong
code on five others, deterministically per seed.

RULE: the replacement for a plain `set` is `OrderedSet`, NOT `dict.fromkeys` and
NOT `{'a': None}`.
WHY: the dict idiom says "mapping" where the code means "set", and the container
then answers to neither API cleanly -- a whole sweep of it broke CI because callers
kept calling `.add()` and `.union()`. Set operators (`|`, `-`, `&`, `^`) return an
OrderedSet, while `dict.fromkeys(...).keys() | other` silently degrades back to an
unordered `set`. Existing `dict.fromkeys` mimicry is correct, just not preferred --
do not churn a file solely to convert it.

RULE: `OrderedSet` is a `Sequence`, so `==` compares element ORDER, not
membership. Do not write an invariant check that compares two OrderedSets unless
order is what you mean.
WHY: !! SILENT -- insertion order is a property of how a set was BUILT, not of what
it holds. An invariant check that compares two accumulation orders refuses correct
input as soon as a caller adds elements in a different sequence.

RULE: ordering is necessary, not sufficient. An order-dependent RESULT usually
means the graph is genuinely ambiguous (two accesses to one container in a state
with no edge ordering them) -- fix the missing dependency as well.
WHY: pinning the seed hides the ambiguity instead of removing it.

RULE: connector sets follow the same rule -- pass `{'a': None, 'b': None}` /
`{'c': None}`, not `{'a', 'b'}` / `{'c'}`.
WHY: `add_nested_sdfg` warns "Using sets for connectors is discouraged as it leads
to indeterministic behavior".

RULE: canonical order = topological sort, ties broken by insertion index. Never
assert on SDFG hashes.
WHY: hashes are not stable across runs or versions.

### G9. Never set `sdfg.start_block` manually

RULE: use `add_state(..., is_start_block=True)`.
WHY: !! SILENT -- the getter prefers the cached / unique-source block and silently
ignores the override, giving an orphaned entry and dominator `KeyError`s.

### G10. Never reuse a Memlet or Subset object across two edges

RULE: build fresh objects per edge.
WHY: validation rejects reuse with `Duplicate subset detected`.

### G11. Memlet `volume` is not subset size

`memlet.py` -- volume is a separate symbolic property (max if `dynamic`, 0 means
unbounded).

RULE: check which of the two a pass actually needs.
WHY: they diverge for dynamic and unbounded memlets.

### G12. `scope_dict()` restarts per NestedSDFG

RULE: thread an `in_kernel` flag down instead of trusting scope lookups.
WHY: !! SILENT -- a node at the top of an NSDFG state has no map scope even inside
a GPU kernel, so the lookup answers "not in a kernel" and the wrong code is emitted.

RULE (related codegen trap): `allocate_array`'s `dfg` argument is the first state
the data appears in, not the allocation scope.
WHY: reading it as the allocation scope places the allocation wrongly.

### G13. Interstate-edge scoping

Condition executes BEFORE assignments (`sdfg.py`, `used_symbols`); `i = i + 1`
keeps `i` free.

RULE: scope interstate symbols with condition-before-assignment order.
WHY: !! SILENT -- wrong scoping loses loop-carried symbols.

### G14. Performance foot-guns

RULE: do not rely on `sdfg.nodes()` for topological order.
WHY: it is insertion order.

RULE: never call `graph.node_id()` inside a loop.
WHY: it is O(V), so the loop goes quadratic.

RULE: do not treat `save()` as round-trip verification.
WHY: `save()` writes a hash that `from_json` ignores.

### G15. `dace.map` is data-parallel by definition

RULE: a cross-iteration dependence belongs in a sequential loop, not a map.
WHY: !! SILENT -- in a map it is a race, and the numbers are wrong
nondeterministically.

RULE: pass symbols at call time. The SDFG call set is program args plus free
symbols.
WHY: symbols are not in the `dace.program` signature but are required by the
compiled object.

### G16. `@lru_cache` always `typed=True`

RULE: always `typed=True`; never cache sympy or mutable objects.
WHY: !! SILENT -- untyped collapses `1` / `1.0` / `True` / dtype objects onto one
key and returns a value computed for the wrong type.

### G17. Squeeze semantics: a length-1 SLICE keeps its dim, an integer INDEX drops it

numpy: `A[:, 0:1]` gives `(N, 1)`; `A[:, 0]` gives `(N,)`. Ranges:
`[0..N, 0..1]` gives `(N, 1)`; `[0..N, 0]` gives `(N,)`.

`Range.squeeze()` drops EVERY size-1 dim and cannot tell a slice singleton from an
index singleton.

RULE: pass the provenance as `ignore_indices` at every squeeze site.
WHY: !! SILENT -- conflating the two is a shape miscompile; `x.T - x` collapsed
`(N,N)` to `(N,)` and produced zeros. The provenance exists only at parse time, in
`memlet_parser._fill_missing_slices`, and is gone by the time a pass sees the
Range -- so a squeeze site downstream cannot reconstruct it and must be handed it.
`newast.make_slice` squeezes blind; treat any squeeze without `ignore_indices` as
a suspect.

Related: G7, and Part B "numpy indexing changes RANK".

### G18. One counter per loop NAME, declared at function scope

DaCe mints ONE index variable per loop name and declares it at function scope. Two
nested Python loops both spelled `for _ in ...` therefore share it: the inner loop
resets the outer's counter, and if the inner breaks early the outer NEVER
terminates.

RULE: give every loop in a generated or hand-written `@dace.program` a distinct
target name; never reuse `_` for a nested loop.
WHY: !! SILENT -- it presents as a TIMEOUT, not a wrong answer, so it reads as a
slow kernel and invites a budget change. The C and Fortran emitters are unaffected
(they declare the index inside the `for`), so only the DaCe leg breaks.

RULE: a kernel that burns exactly its timeout every run, rather than varying, is a
HANG suspect before it is a slowness suspect.
WHY: a varying runtime is slow; a pinned one is stuck.

### G19. An advanced index into an EXTENT-1 dimension is a broadcast

A dimension consumed by a SCALAR INDEX disappears. A dimension of EXTENT 1
survives and can still be gathered from. They are two concepts, and code that
treats "collapses to a point" as one of them breaks: `table[idx]` on a
`float64[1]` raised `KeyError: '__i0'`.

RULE: fix this class in the FRONTEND, not in the validator and not in the memlet.
WHY: `data_dims()` is right and codegen types the connector by the same squeeze
rule, so relaxing the validator only defers the failure to a C++ compile error.
`subsets.Range` cannot tell `A[0]` from `A[0:1]` -- both are `(0, 0, 1)` -- so
expressing "size-1 but still indexable" would mean changing `Range`, a core IR
change. The frontend is the last place that still knows which dims went to an index
array.

### G20. `used_symbols(all_symbols=False)` vs `free_symbols`

`sdfg.used_symbols(all_symbols=False)` is the symbols the code actually NAMES.
`sdfg.free_symbols` is `used_symbols(all_symbols=True)`, a SUPERSET that also
counts symbols appearing only in descriptor shapes -- every symbol in scope.

RULE: signatures and declaration guards take `all_symbols=False`. Justify the wider
set explicitly wherever a call needs it.
WHY: !! SILENT -- with the wider set a merely DECLARED symbol reads as used, so an
emitted signature grows a parameter the code never names and every argument after
it lands in the wrong slot. Nothing raises; the call just reads the wrong values.

RULE: `SDFGState.used_symbols(with_contents=False)` returns an empty set -- ask
states WITH contents and regions WITHOUT (a region's contents arrive as its own
blocks).
WHY: the recursion is already done for regions and not for states.

### G21. Never round-trip a `SymExpr` through `str`

Anything `StripMining` / `MapTiling` produced is a `symbolic.SymExpr`, whose
`__str__` prints `main (approx)` -- e.g. `Min(LEN_1D - 2, t + 4095) (~t + 4095)`.

RULE: never write `pystr_to_symbolic(str(expr))` on a subset bound or map range.
Branch on the type: rewriting a bound, keep the pair --
`SymExpr(rewrite(e.expr), rewrite(e.approx))`; inspecting one, walk both halves; if
a fresh range needs a single value, take `.expr`.
WHY: sympy parses `Min(...) (...)` as a CALL and raises `'Min' object is not
callable`, far from the site that built it. The round-trip is lossy by
construction: the pair carries an exact expression AND an over-approximation, and
DaCe's parser has no syntax for the pair.

RULE: a `SymExpr` is NOT a `sympy.Basic`. Audit any helper opening with
`isinstance(x, sp.Basic)`.
WHY: !! SILENT -- that guard in `sdfg/replace.py` handed every strip-mined bound
back untouched, so a range tuple came out HALF-renamed and surfaced far away as
"Missing symbols on nested SDFG".

### G22. Interstate reads are invisible to AccessNode analysis

`if mask[i]:` lowers to an interstate ASSIGNMENT `mask_index = mask[i]` plus a
condition on the resulting symbol. There is NO AccessNode for that read.

RULE: any read/write analysis must also walk interstate edges
(`edge.data.free_symbols & sdfg.arrays.keys()`, what
`ControlFlowRegion.read_and_write_sets` already does) AND `ConditionalBlock.branches`
-- branch conditions live on the block, not on the edges.
WHY: !! SILENT -- a pass that walks AccessNodes concludes the container is unread.
One pass then privatized a mask into never-written copies and emitted a reference
to an undeclared array; another distributed a loop between the producer and the
consumer of a mask, so the consumer ran against the producer's LAST iteration --
silent wrong numbers.

RULE: a test for this needs a mask that CHANGES per iteration.
WHY: with a constant mask the split and the original agree and the bug is
INVISIBLE.

### G23. A hand-written `__deepcopy__` goes stale

`AccessNode.__deepcopy__` is a hot-path shortcut: `object.__new__` then a FIXED
LIST of `_`-prefixed field assignments. A `Property` added to `Node` afterwards is
not in that list, so the copy never gets the attribute.

RULE: after adding a `Property` to a base node class, backfill every hand-written
`__deepcopy__` from `type(self).__properties__`.
WHY: reading an unassigned Property raises `AttributeError` instead of returning
its default. Adding one Property broke 80 tests plus every pipeline that
deepcopies graph parts. A fresh node is fine and a `to_json`/`from_json` round-trip is fine --
only `copy.deepcopy` breaks -- so a cached or serialized SDFG MASKS it.

### G24. Library nodes size from the MEMLET, not the descriptor

RULE: extents come from `e.data.subset.size()`; strides come from the container;
dtype comes from the DATA DESCRIPTOR of the incoming memlet. Never from
`sdfg.arrays[e.data.data].shape`, and never from a connector.
WHY: !! SILENT -- descriptor and subset agree only when a whole array is moved, so
reading the descriptor rejects or miscompiles every node writing into or reading
out of a SLICE. A library-node connector describes the tasklet INTERFACE -- the
frame hands it a pointer -- so a connector type resolves to `dace.pointer(...)` and
an expansion that types its call from it emits the pointer type where the element
type belongs. There is no "use the connector when it is not a pointer" exception;
do not add one.

RULE: compare extents with `symbolic.inequal_symbols`, never raw `!=`. An expansion
that cannot express a stride array must REFUSE, not guess.
WHY: one extent arriving through two rewrites is two spellings of the same value,
and raw `!=` on two spellings rejects a legal graph.

Validate is not the only descriptor reader -- every EXPANSION sizes its call the
same way. Prior art for both halves: `dace/libraries/blas/nodes/axpy.py`.

### G25. A set of graph objects is unordered even with the hash seed pinned

RULE: pinning `PYTHONHASHSEED` does NOT make a `set` of nodes, states or blocks
deterministic. Use an `OrderedSet` there, exactly as for a set of names.
WHY: !! SILENT -- the seed pins `str` hashing only. An SDFG node has no `__hash__`
of its own, so it hashes by `id()` -- its allocation address -- which varies within
a single process as well as between runs. `dace/sdfg/analysis/cfg.py` builds ~20
plain sets of blocks and several are ITERATED to choose a region's entry and exit,
so the structure chosen for the emitted code can differ between two calls on one
finalized SDFG.

RULE: never assert on a hash of an SDFG or of generated code. Compare a graph
digest (`sdfg.to_json()` with the per-object identity fields -- `guid`,
`cfg_list_id`, `hash` -- dropped), or assert the ORDERED TRAVERSAL LIST.
WHY: a hash mismatch says "something moved" and nothing else, and it breaks on
benign formatting changes; a traversal list names which element moved.

### G26. Tasklets: no C++ outside a library-node expansion, no array offsets inside

RULE: an array offset never goes inside a tasklet body -- `A[i,i]` belongs on the
memlet subset, and the element arrives on a connector. A whole-array memlet feeding
a pointer-typed connector that the body then indexes is the same violation wearing
a connector; what matters is where the INDEXING happens.
WHY: it hides the access from every subset-based analysis and from propagation.

RULE: outside library-node expansion (a reduction or tile node), emit a Python
tasklet, never a C++ one.
WHY: a C++ tasklet's symbols and accesses are opaque to symbol analysis.

Two carve-outs. A side-effecting GUARD tasklet -- `side_effects=True`, no output
connector feeding real dataflow, a body that only traps -- may be `Language.CPP`
and may index an array, because `abort()` has no Python-tasklet spelling and a
guard over a whole array needs its loop inside the one node. So may a true
intrinsic with no Python form (`__syncthreads`, a warp reduction, a stream
synchronize, a timer). A tasklet that computes a VALUE is neither.

### G27. Symbols are nonnegative by contract, and unknown is not small

Array sizes, strides and offsets are nonnegative by contract, but DaCe symbols
carry no sign assumption (`is_nonnegative` is `None`).

RULE: to decide an offset's sign, substitute `sympy.Symbol(name, nonnegative=True)`
and query `is_nonnegative`. Never use an ordering-dependent heuristic like
`could_extract_minus_sign()`.
WHY: that heuristic answers False for `K-M` and True for `M-K` at the same
magnitude. A sum of nonnegatives is provably `>= 0`; a DIFFERENCE `K-M` stays
sign-undecidable even under the assumption -- refuse, do not emit a runtime guard.

RULE: unit tests bind nonnegative symbol values only.
WHY: a negative binding tests a program the contract does not describe.

RULE: a symbolic extent is assumed BIG ENOUGH, and every symbol is equivalent in
magnitude to every other. Never rank `N` against `M` against `LEN_1D`, and never
special-case a symbol by name. A size gate whose comparison returns unknown lands
on the parallel side; only a PROVABLY-below-threshold count takes the serial one.
WHY: !! SILENT -- unknown is not small. Treating a symbolic count as small
serialized every dynamically sized kernel, lowering a 9.4 MB copy to a
single-threaded memcpy for a 22x loss, purely because `int()` on it raised and the
except branch returned False. Decide statically; do not emit a runtime size guard.
Symbol-vs-symbol comparisons carry no information for a cost model.

### G28. Scalars for internal values, len-1 arrays only for outputs

RULE: use `dace.Scalar` transients freely for internal values. Emit a length-1
Array ONLY where the value must leave the SDFG (an output argument).
WHY: a Scalar transient reads correctly on all three codeblock contexts -- a
ConditionalBlock branch condition, a LoopRegion `condition_expr`, and an interstate
assignment. The older belief that it reads as free-symbol zero is outdated.

RULE: on a codeblock, reference a scalar by BARE NAME and any array -- including a
`(1,)` array -- with a SUBSCRIPT.
WHY: the two spellings are not interchangeable at that seam.

## A6. Python frontend: writing a case, and looking at the graph

Every spelling below was executed before being written down.

RULE: re-probe the frontend rather than trusting this list after a frontend change.
WHY: a documented form the parser rejects is worse than no documentation.

### Smallest reproducer, and its SDFG

```python
import dace, numpy as np
N = dace.symbol('N', dtype=dace.int64)

@dace.program
def k(a: dace.float64[N], out: dace.float64[N]):
    out[:] = a * 2.0

sdfg = k.to_sdfg(simplify=False)   # PARSE ONLY -- no C++ compiler runs, cheap enough for a sweep
sdfg.validate()                    # the check a hand-built graph usually fails
csdfg = k.compile()                # separate step; this is what invokes the toolchain
```

RULE: reproduce bugs with `simplify=False`; turn simplify on only to show the bug
survives it.
WHY: simplify can fuse the very state or memlet the report is about.

Getting the picture out:

```python
sdfg.save('k.sdfg')                # JSON; open in the VS Code extension or sdfv
sdfg.save('k.sdfg', readable=True) # diffable -- use in a bug report instead of a screenshot
sdfg.view()                        # opens it directly
k.to_sdfg(save=True)               # writes into the build folder as it parses
print(sdfg.to_json()[:2000])       # when a diff is what you want to inspect
```

RULE: in a sweep over many programs, run each parse in its own SUBPROCESS.
WHY: DaCe's parse state is process-global -- one program that wedges or corrupts it
makes every later verdict in that process untrustworthy. Measured: 24 kernels came
back as crashes that all parse fine alone.

### Frontend features worth knowing before hand-building an SDFG

RULE: reach for these before hand-building a graph.
WHY: they say the same thing and survive a frontend change.

```python
# Schedule hint on a map -- the @ operator, the one BinOp allowed in a for-iterator
for i in dace.map[0:N] @ dace.ScheduleType.Sequential: ...
for i in dace.map[0:N] @ dace.dtypes.ScheduleType.CPU_Multicore: ...
# (parsed in newast._parse_for_iterator; anything but ast.MatMult there is a DaceSyntaxError)

# Storage hint on a local -- an ANNOTATED assignment whose annotation is a data descriptor.
# visit_AnnAssign reads `.storage` off it and applies it to the resulting array.
buf: dace.data.Array(dace.float64, (N,), storage=dace.StorageType.CPU_ThreadLocal) = np.zeros((N,))

# Transients without an annotation
buf = dace.define_local([N], dace.float64)
acc = dace.define_local_scalar(dace.float64)

# Explicit dataflow when you need the exact tasklet/memlet shape
for i in dace.map[0:N]:
    with dace.tasklet:
        inp << a[i]
        res >> out[i]
        res = inp * 3.0
```

`StorageType` names the memory space (`CPU_Heap`, `CPU_ThreadLocal`, `GPU_Global`,
`GPU_Shared`, `Register`); `ScheduleType` names how a map executes (`Sequential`,
`CPU_Multicore`, `GPU_Device`, `GPU_ThreadBlock`). Both live in `dace.dtypes` and
are re-exported from `dace`.

## A7. Contribution guidelines (from CONTRIBUTING.md -- upstream, read-only)

CONTRIBUTING.md is upstream and read-only. Do not edit it.

RULE: Google Python Style Guide. Power features allowed. New functions need type
hints.
WHY: upstream style contract.

RULE: check the declared supported Python versions before using new syntax.
WHY: the package declares the floor; `pyproject.toml` currently says
`requires-python = ">=3.10, <3.15"` -- re-read it rather than trusting this line.

RULE: **no direct class/function imports**, except `SDFG`, `SDFGState`, `Memlet`,
`InterstateEdge`. No `import *`.
WHY: upstream import policy. See "Conflicts" -- this contradicts the general Python
rule in Part B.

RULE: imports at top; an inline import needs an adjacent reason comment (e.g.
`# Avoid import loop`).
WHY: an undocumented deferred import hides a cycle.

RULE: leave a blank line before `:param:` blocks in Sphinx docstrings.
WHY: docs build breaks otherwise.

RULE: formatting gate is yapf pep8 column_limit 120 plus ruff-check, via
pre-commit. CI enforces it.
WHY: merge is blocked on it.

## A8. Lowering and parallel maps

RULE: keep a device-neutral graph device-neutral. Do not bake a CPU-only artifact
into a form both targets consume -- a `Language.CPP` tasklet carrying an omp pragma,
an `omp_schedule` property, a thread-count assumption. If one form is not best for
both targets, add the other target's lowering instead of biasing the shared one.
WHY: a graph carrying a host-side scheduling decision cannot be offloaded, and the
decision is invisible until the GPU path fails or silently runs single-threaded.

RULE: never emit `#pragma omp critical` or `#pragma omp sections`. Parallelism
comes from `omp parallel for` over a real iteration domain.
WHY: both serialize. `sections` scales with the number of lexical sections;
`critical` is a lock -- measured ~9x slower than sequential at 1 thread and
180-255x at 8, with the penalty flat in trip count and linear in thread count. The
same reduction under the real CAS lowering runs 2x FASTER than sequential.

RULE: FP reassociation is allowed WITHIN a reduction's or a scan's combining
operation, and nowhere else. Prefer an OpenMP `reduction(op:var)` clause over
`reduce_atomic` wherever a clause fits. Everything else stays bit-exact: no
`-ffast-math`, no global associative math, element-wise and streaming code in
source order.
WHY: requiring bit-exact serial association on a reduction blocks the native
lowering and forces atomics, which are far slower. Outside the combining operation
there is no licence at all -- a reassociated element-wise expression changes results
with nothing to justify it. A tolerance rewrite on a REDUCTION result, commented as
citing this policy, is sanctioned rather than test-weakening.

Trap: a bare `if(...)` clause on a combined `parallel for simd` binds to `simd` too
on GCC, silently devectorizing the loop (measured 3-4.8x). Use `if(parallel: ...)`
or no clause.

RULE: GPU offloading uses the default stream (`nullptr`); pin
`compiler.cuda.max_concurrent_streams = -1` in the pass or driver itself, not only
through an environment variable.
WHY: deterministic launch order and no cross-stream sync bugs. D2H copies are
host-synced LAZILY at the first CPU consumer (the tasklet or map entry the
destination access node's memlet path reaches), not at the copy site; H2D needs no
sync when the consumer is on the same stream. Kernel-launch by-value arguments
count as host consumers -- they are resolved on the HOST at launch time, while
events order only device work.

RULE: an unexpected `#pragma omp parallel for` in emitted code is a BUG to trace,
not a free speedup. Never make the resulting numeric divergence go away by pinning
`OMP_NUM_THREADS=1` in the harness.
WHY: a map nobody scheduled got `ScheduleType.Default` and was then inferred to
`CPU_Multicore` -- usually inside a library-node expansion, where the fix belongs.
Pinning the thread count hides a real parallel miscompile behind a passing test.

RULE: do not splice an `_out=_in` copy tasklet onto an edge an analysis depends
on, and never onto a View's defining edge.
WHY: !! SILENT -- dataflow analyses pattern-match DIRECT copies (AccessNode to
AccessNode, MapEntry to AccessNode, a View's defining edge). A spliced tasklet makes
the copy invisible to them, so an analysis reports no copy where one exists; on a
View's defining edge `sdutil.get_view_edge` then raises "Ambiguous or invalid edge
to/from a View". When a sub-tasklet needs an anchor, give it an EMPTY
`Memlet(None)` from the scope entry rather than a copy tasklet.


## A9. How to write a test

RULE: never weaken, delete, `xfail`, loosen the tolerance of, narrow the
parametrize list of, or comment out a test to make a run pass. Only a test that is
GENUINELY WRONG may change -- it encodes a stale spec, a superseded design, or
asserts an implementation detail instead of the property -- and the reason goes in
the commit message. If the fix is out of scope, leave the test RED.
WHY: a red test is the cheapest bug report that exists. A suite going green after a
deletion looks identical to one going green after a fix, so the deletion converts a
known bug into an unknown one.

RULE: prefer STRENGTHENING. Replacing a syntactic assertion (a printed spelling, a
node shape) with a semantic one (concrete evaluated values) is the good move; a
"genuinely wrong" verdict is not a license to lower the bar.
WHY: the first replacement for a wrong assertion is usually too weak to fail if the
behavior flips again.

RULE: this applies to anything gate-shaped, not only `test_*.py` -- benchmark
manifests, pinned ratchet lists, golden and reference outputs.
WHY: deleting a benchmark whose ABI disagrees is the same move as deleting a red
test; the bug survives wherever that path is still reachable.

RULE: a structure-changing pass gets STRUCTURAL assertions in ADDITION to the
numeric check, never instead of it. Removing a tile level: assert it is gone --
nest depth, no leftover tile-index loop or remainder guard, recovered extents at
full size rather than tile size. Unrolling: assert the number of body copies, and
that no residual loop remains where full unrolling was claimed. Generalizes to
parallel region counts, map counts, fusion counts.
WHY: !! SILENT -- a value-only test passes when the pass did NOTHING. A tiled
stencil family stayed numerically green at baseline speed for weeks while the pass
that was meant to remove the hand-written tiling silently did nothing. Numerics
cannot see a missing transformation.

RULE: when the emitted TEXT is the product of the thing under test, asserting on
the text is REQUIRED. A property set on a map and the pragma codegen emits for it
are two different facts -- the property says the pass decided, the pragma says
codegen honored the decision IN THE RIGHT PLACE. Assert both.
WHY: no graph-level assertion can express clause placement: a leaf map takes the
combined clause, a nest takes the outer clause on the outer map and the inner one
on the inner map, and only the emitted text distinguishes them. The anti-pattern is
narrower than it looks -- it is a string grep restating a fact already asserted
semantically, not any assertion on generated code.

RULE: assert SEMANTIC predicates over parsed structure, not full-line equality
against rendered output. Byte-exact comparison only where bytes ARE the contract
(determinism: same input, same bytes).
WHY: an exact-string test breaks on every benign rewording and still misses the
property it meant to pin.

RULE: do not skip. A `pytest.skip` that can fire on a healthy working copy reports
green while verifying nothing. For a condition that cannot happen there, ASSERT.
For a known upstream limitation, `xfail(strict=True)` -- it
documents the gap and fails when it is fixed, so the marker cannot rot. For a
genuinely absent optional toolchain, mark the test and exclude it by MARK, not by a
runtime skip.
WHY: a skip is silent on a broken premise -- `skip("kernel not in the corpus")`
looks the same whether the corpus is fine or has lost half its kernels.

RULE: pair every hand-built SDFG fixture with at least one `@dace.program` fixture,
and dump the generated code once to confirm the feature actually fires there.
WHY: !! SILENT -- hand-built fixtures reproduce the mental model you just
implemented, not the SDFGs the frontend emits (map scopes, multi-state transients,
LoopRegions, nested SDFGs). A real multi-state miscompile was invisible because
every fixture was single-state, and a codegen knob passed all its tests while being
near-dead on real code.

RULE: verify every regression test FAILS without the fix -- reintroduce the bug and
watch it go red.
WHY: a test that cannot fail is not a test.

RULE: never reach for `norm_error` or a loosened tolerance to make a kernel pass. A
correct lowering is bit-exact with the reference, or within a tight rtol/atol. A
divergence is a REAL bug to localise.
WHY: !! SILENT -- tolerance masks a defect. A 30-step sim's 6.5e-4 relative error
was argued to be inherent FP; the real cause was a shape mis-inference that left
part of an array un-advanced every step. After the fix: bit-exact on every backend.

RULE: an incumbent implementation (sympy included) is a SPEED baseline, never a
correctness oracle. Assert against executable semantics -- substitute concrete
values over the declared ranges and compare with the reference operator definition.
WHY: !! SILENT -- `modulo(3*i - 1, 3)` folded to `-1` where the true truncated value
is `2`, and sympy produces the SAME wrong answer, so a differential run against it
would have scored agreement and shipped the miscompile.

RULE: warnings are errors, in hand-written AND in generated code. A generated
kernel that warns is a generator bug. Wire the gate (`-Wall -Wextra -Werror`,
clippy `deny`, ruff/yapf) rather than trusting discipline; silencing one needs a
targeted suppression with a reason.
WHY: warnings on emitted C/Fortran are the cheapest available signal for exactly
the defect class ctypes cannot catch -- type, arity, aliasing. Check that the
linter is WIRED before trusting a clean build; an unwired one accumulates real
defects silently (120 warnings once hid a hot-path 168-byte return for an 8-byte
payload).

RULE: a test is a legitimate consumer. Grep `tests/` before deleting a "dead"
helper, and keep it with a one-line note if a test is its only caller. A
`pytest -k` sweep is NOT full coverage -- it silently skips whole modules whose
names miss the keyword.
WHY: linters cannot see cross-module test imports. After a commit,
`git diff HEAD~1 HEAD -- '*.py' | grep -E "^-(def |class )"` lists every removed
top-level symbol -- cheap, and it catches exactly this.

RULE: tests must be meaningful and non-overlapping. Before adding one, read the
existing file and name what it does not already cover; extend rather than append a
near-duplicate. Bad tests added for coverage are worse than no tests.
WHY: an assertion that duplicates one already made above buys no coverage, adds
runtime, and adds a second thing to update on every refactor. Two tests differing in
which pragma lands where are NOT overlapping; an assertion restating a fact already
asserted is. Call-count and monkeypatch assertions pin the
implementation -- justified only as a named performance-regression guard.

RULE: a hand-built SDFG in a test needs a UNIQUE name; interpolate the parameters
for a parametrized test.
WHY: !! SILENT -- the build cache is keyed by SDFG name, so every case sharing
`dace.SDFG("testing")` compiles into the same directory and they overwrite each
other's library. It surfaces as `Could not load library` or as a bare wrong-output
assertion, neither of which points at the name.

RULE: a conditional assertion becomes a silent no-op exactly when the thing it
guards is removed.
WHY: `if earlier in names and later in names: assert ...` passed vacuously once
both passes moved elsewhere -- 34 green tests, two assertions asserting nothing.
Assert PRESENCE where a pass is required, then assert the ordering.

---

# Part B -- Python quality gates

`<file>.py` is the placeholder for the target throughout -- swap in the real path.
Every command is copy-pasteable.

## B0. Golden rule

RULE: all gates run. Warnings are errors. Type errors are errors. A clean pass =
zero diagnostics from yapf (`--diff` shows nothing), ruff, pyright (or mypy), and
the warnings-as-errors smoke, PLUS a clean `pre-commit run` and green pytest
consumers.
WHY: a partially-run ladder cannot distinguish "clean" from "unchecked".

RULE: do not report "looks good" until every gate is green.
WHY: same.

RULE: fix findings at the source -- never silence a warning, never `# type: ignore`,
never `# noqa` to pass. A targeted `# noqa: CODE` with a reason is allowed only for
a genuine third-party false positive.
WHY: a silenced diagnostic is an unfixed defect that no longer reports itself.

RULE: probe what is available before running (`ruff --version`,
`pyright --version`, ...) and adapt. If a tool is absent, run its gate where the
project provides it (repo config / CI) and report that gate as DEFERRED.
WHY: a silent skip reads as a pass.

RULE: do NOT `pip install` (or `npm install`) anything to make a gate pass.
WHY: it changes the environment the result was measured in.

RULE: use `python` (>= 3.10). If the project pins an interpreter (a pyenv venv, a
`.python-version`), use that one.
WHY: gate results must match what the project actually runs.

Tools: `yapf`, `ruff` (with `pyflakes` / `flake8` as fallbacks), `pyright` or
`mypy` for the type gate, `pre-commit`, `pytest`.

## B1. Gate 1 -- yapf, format first, in place (column 120)

yapf auto-discovers a project `.style.yapf` / `setup.cfg [yapf]` /
`pyproject.toml [tool.yapf]` at or above the file; the explicit `--style` below is
the fallback used only when none exists (house default: pep8 base, 120 columns).
This repo has `.style.yapf` at its root: `based_on_style = pep8`,
`column_limit = 120`.

```bash
cfg="$(dirname <file>.py)/.style.yapf"
[ -f "$cfg" ] || cfg="$(git -C "$(dirname <file>.py)" rev-parse --show-toplevel 2>/dev/null)/.style.yapf"
if [ -f "$cfg" ]; then
  yapf -i --style="$cfg" <file>.py
else
  yapf -i --style='{based_on_style: pep8, column_limit: 120}' <file>.py
fi
```

RULE: yapf is the formatter here -- do NOT switch to black or ruff-format.
WHY: either would reflow the whole tree to a different style.

Check without editing (the form the golden rule scores) -- exits non-zero if
anything would change:

```bash
yapf --diff --style='{based_on_style: pep8, column_limit: 120}' <file>.py
```

## B2. Gate 2 -- ruff lint

```bash
ruff check --line-length 120 <file>.py
```

Stronger explicit rule set, recommended when the repo has no ruff config of its
own (pyflakes + pycodestyle + bugbear + comprehensions + pyupgrade + simplify):

```bash
ruff check --select E,F,W,B,C4,UP,SIM --target-version py310 --line-length 120 <file>.py
```

RULE: always pass `--line-length 120` unless the repo's own ruff config sets it.
WHY: ruff defaults to 88 while gate 1 formats at 120, so every line yapf just
produced between 89 and 120 columns comes back as a wall of `E501`. That is a bug
in the invocation, not the file -- read the codes before reflowing anything; if
they are all `E501`, re-run at 120 first.

`--fix` applies the autofixable subset (re-run yapf after). If `ruff` is absent,
fall back to `flake8 --max-line-length 120 <file>.py`, or at minimum
`pyflakes <file>.py` -- these catch unused imports and undefined names but far less
than ruff. flake8 defaults to 79, tighter still, so the width caveat applies with
more force; `pyflakes` has no width check at all.

## B3. Gate 3 -- type check (pyright strict and/or mypy strict)

```bash
pyright <file>.py                 # honors pyrightconfig.json / [tool.pyright]; add --strict for full strict
mypy --strict <file>.py           # alternative / second opinion
```

RULE: treat every type error as a failure.
WHY: this is the strong correctness gate; it is what catches implicit `Any`,
incompatible assignments, and int/float/None mismatches.

RULE: if neither checker is on `PATH`, run the gate the way the project provides it
(a `pyrightconfig.json` / `[tool.pyright]` driven by the editor's bundled pyright or
a repo dev-dep, or mypy in CI) -- from inside the repo that provides it. If the
target repo configures neither, this gate is DEFERRED: say so loudly in the report.
WHY: a skipped type gate is the one most likely to be mistaken for a pass.

## B4. Gate 4 -- warnings-as-errors import / compile smoke

```bash
python -W error -m py_compile <file>.py                 # SyntaxWarning + byte-compile, no execution
python -W error -c "import package.module"              # import path -- runs module top-level, warnings fatal
```

RULE: surface `Deprecation` / `Syntax` / `Resource` warnings as hard errors.
WHY: they are the early form of a future breakage, and they are free to fix now.

Use the interpreter the module's dependencies require; prefer plain `python` and
switch only when a version-specific dependency forces it. `python -We <file>.py`
executes the file directly with warnings fatal -- use it when the file IS a
runnable script rather than an importable module.

## B5. Gate 5 -- pre-commit, on EVERY touched file

```bash
[ -f "$(git -C "$(dirname <file>.py)" rev-parse --show-toplevel 2>/dev/null)/.pre-commit-config.yaml" ] \
  && pre-commit run --files <file>.py
```

RULE (standing mandate): yapf + pre-commit on every file you touch, no exceptions.
A failing hook is a failing gate -- fix the code, do not `--no-verify`.
WHY: CI runs the same hooks and blocks the merge.

RULE: if a new import was added, ensure the dependency is declared in
`pyproject.toml`.
WHY: otherwise the hooks and CI cannot resolve it.

This repo's `.pre-commit-config.yaml` runs: ruff-check (`--fix`),
check-merge-conflict, check-yaml, end-of-file-fixer, trailing-whitespace, yapf.

## B6. Gate 6 -- tests, the file's pytest consumers

```bash
pytest path/to/test_<thing>.py           # the matching test module(s)
pytest -k "<thing>" path/to/tests/       # or select by keyword
```

RULE: tests are consumers, not dead code -- exercise whatever imports or covers
this file.
WHY: an unexercised change is an unverified change.

RULE: run from the repo root so the package prefix (`from pkg.sub import ...`)
resolves. Never `sys.path` hacks.
WHY: a path hack passes locally and fails in CI.

RULE: a new warning during the run is a failure too (zero-warning policy).
WHY: warnings are errors, everywhere.

## B7. Reporting

RULE: report each gate's status individually. Only "clean" when B1-B6 all pass with
zero output, and note explicitly if the type gate was DEFERRED for lack of an
in-repo checker.
WHY: an aggregate "green" hides a deferred gate.

---

# Part B, continued -- Writing modern Python (>= 3.10, no OO bloat)

Decision ladder first (KISS / YAGNI): does it need to exist? -> already in the
codebase? -> stdlib? -> native? -> installed dep? -> one line? -> else the minimum
that works.

RULE: prefer plain functions plus small dataclasses over class hierarchies,
factories, or indirection.
WHY: new code is a liability.

## Type hints

RULE: type hints ALWAYS -- every parameter, every return, every non-trivial local.
WHY: the type gate is only as strong as the annotations it has.

RULE: modern 3.10+ syntax -- `X | None` (PEP 604), not `Optional[X]`; `list[int]`,
`dict[str, int]`, `tuple[int, ...]`, not `typing.List` / `Dict` / `Tuple`. Reach
into `typing` only for what has no builtin form (`Callable`, `Protocol`, `TypeVar`,
`Iterable`, `Self`, `Literal`).
WHY: house style; the builtin generics are the current form.

## No implicit conversions

RULE: convert EXPLICITLY -- wrap with `int()` / `float()` / `str()` / `bool()` at
the point a type changes; use `//` (not `int(a / b)`) for integer division.
WHY: silent coercions are where int/float precision is lost unobserved.

RULE: never use `bool` and `int` interchangeably (`True + 1`). Prefer explicit
comparisons (`if n != 0:`, `if s is not None:`) over bare truthiness when the intent
is a specific check, not "is it falsy".
WHY: bare truthiness conflates `0`, `''`, `[]` and `None`.

RULE: keep numeric kinds consistent in hot loops -- no int/float churn.
WHY: each transition costs a conversion and can change results.

RULE: indices stay int64. An index array, a permutation or gather vector, a
compaction rank, an ownership tag -- anything that STORES an index -- is never
narrowed to int32 to save memory traffic.
WHY: !! SILENT -- a narrowed index wraps. Two writers whose indices agree modulo the
tag width read each other's tag back as their own, which is a MISSED duplicate in a
conflict check: the one failure mode that check exists to catch. When an
index-typed buffer costs too much, attack its EXISTENCE or its access pattern --
recompute the value inline, fuse the producing pass into the consuming one, drop the
buffer -- never its element width.

The strict type checker (gate B3) enforces this. Fix findings with an explicit
conversion or a corrected annotation, never a `# type: ignore`.

## numpy indexing changes RANK -- three rules, not one

- `a[0]` -- integer index -- **DROPS** the axis. `a[0:N, 0]` is `(N,)`.
- `a[0:1]` -- slice -- **KEEPS** the axis at length 1. `a[0:N, 0:1]` is `(N, 1)`.
- `a[None]` / `a[np.newaxis]` -- **INSERTS** a length-1 axis; treat it as a `0:1`
  slice on an axis the source does not have.

They are NOT interchangeable. A length-1 axis **broadcasts** -- every position along
it reads the same element -- whereas a dropped axis right-aligns against a different
axis of the other operand. So `a[:, 0:1] * b` and `a[:, 0] * b` compute different
things, and the second is an error unless the extents happen to match.

RULE: in a translator / emitter / scalarizer, index a length-1 source axis by its
slice **start**, never by the loop variable.
WHY: !! SILENT -- `out[:, :] = a[:, 0:1] + b` reads column 0 of `a` for every column
of the result; indexing it with the column iterator reads a whole row instead and
produces wrong numbers with no error.

RULE: never "normalise away" a size-1 dimension.
WHY: !! SILENT -- `squeeze()` removing them is a deliberate operation, not a shape
simplification you may apply. See A5 G17 for the DaCe-side version of this bug.

## Imports

RULE: all imports at module top, absolute and package-qualified
(`from pkg.sub.mod import fn`). Never relative (`from .x import y`,
`from ..pkg import z`).
WHY: relative imports break when a module moves or is run directly.

RULE: a function-local / deferred import is allowed ONLY to break a genuine import
cycle or defer a heavy optional dependency, and it carries a one-line comment
saying which.
WHY: an undocumented deferred import hides the cycle it was papering over.

RULE: do NOT run `python -c "<script>"` for real work -- write a `.py` file with
top-level imports.
WHY: an inline script is unreviewable, unformattable, and ungatable.

See "Conflicts" -- DaCe's CONTRIBUTING bans direct class/function imports and this
rule requires them.

## Classes

RULE: classes declare every attribute up front; no dynamically added or removed
attributes. Use `__slots__` -- for a dataclass, `@dataclass(slots=True)` (Py3.10+),
NOT a hand-written `__slots__`.
WHY: a hand-written `__slots__` breaks field defaults. Slots fix the attribute set
so typos become `AttributeError` at write time. The win is memory + cache locality,
not raw attribute speed on 3.11+ (that edge is about gone by 3.13).

RULE: skip `__slots__` (with a one-line reason) where it is unsafe -- a non-slotted
base reintroduces `__dict__`, the class is a mixin, monkey-patched, or
weakref'd/pickled in a way slots break.
WHY: slots there either do nothing or break the object.

RULE: no optional attributes. Every attribute is a declared slot ALWAYS assigned in
`__init__`. A logically-absent one gets a module-level `SENTINEL = object()` (or
`None`) default; code DETECTS the default (`if obj.attr is SENTINEL`) and never asks
whether the attribute exists.
WHY: "does this attribute exist" is a question a static schema should never have.

## No `getattr` / `hasattr`

RULE: `getattr` and `hasattr` are banned for control flow -- attributes are known
statically.
WHY: they defeat the type checker and hide typos.

Substitution ladder:

- static slotted schema -> direct access (`obj.attr`), sentinel to mark "unset";
- type/shape/capability probe -> `isinstance(x, np.ndarray)` then `.ndim` directly,
  never `hasattr(x, "shape")`;
- genuinely optional attr on an object with a real `__dict__` (e.g. dynamically-set
  AST-node attrs) -> `vars(obj).get("name", default)`;
- dynamic member of a module by name -> `vars(mod)["name"]`.

`vars()` sees only the instance `__dict__` -- unsafe for class attrs, properties,
slots, or base-class attrs, and raises on `__slots__` objects. The ONE kept
exception: C-extension objects with no `__dict__` (tree-sitter etc.) -- `getattr`
stays there.

Note the tension with EAFP: `try/except AttributeError` for a genuinely optional
external interface is a last resort only, because exceptions-as-control-flow is
itself discouraged -- prefer `isinstance` / `vars().get()` / a sentinel slot.

## `lru_cache`

RULE: `functools.lru_cache(maxsize=..., typed=True)` -- always `typed=True`. Never
bare `@lru_cache` / `@lru_cache()`, never `@functools.cache` (it is
`lru_cache(maxsize=None)` with no `typed`; for unbounded write
`@functools.lru_cache(maxsize=None, typed=True)`).
WHY: !! SILENT -- untyped collapses `1`, `1.0`, `True`, and `numpy.float32(x)` vs a
Python float onto one key. In dtype/symbol/sympy-keyed code that returns a value
computed for the wrong type: a miscompile, not a nit.

RULE: never `lru_cache` a sympy object or any mutable object.
WHY: equality/hash ignore dtype metadata, and a mutated object leaves a stale entry.

## Naming

RULE: no leading-underscore names -- never prefix a function, class, or module with
`_`. There is no "genuinely-private helper" carve-out; this house rule overrides
PEP 8's `_private` convention. Module-level DATA constants prefer public names too;
the hard rule is functions/classes/modules.
WHY: house style -- public names everywhere.

See "Conflicts" -- DaCe core itself uses `_parent`, `_parent_sdfg`, `_nx`,
`_eval_subs`.

## Cheap checks before expensive

RULE: order `and` / `or` chains and guard clauses cheapest-first so short-circuit
skips the costly term.
WHY: the cheap term is free; the expensive one is not.

Cheap = int/float compare, identity test, `len()`, small set/dict membership.
Expensive = a numpy call, regex, isinstance chain, attribute walk, `str` build,
anything touching filesystem / imports / re-parse.
`if name in KNOWN and expensive_probe(name)`, never the reverse.

RULE: reorder only where semantically equivalent -- a `None` check that makes a
later attribute read safe must stay first. When adding a guard to an existing chain,
find the cheapest position that is still correct; do not just append.
WHY: reordering past a safety guard turns a fast path into a crash.

## No hardcoded paths

RULE: no absolute literals (`/home/...`), no `sys.path.insert`, no `importlib`
file-loading.
WHY: they break on every other machine and in CI.

Current interpreter -> `sys.executable`; repo root ->
`Path(__file__).resolve().parents[N]` (or `git rev-parse --show-toplevel`, or an
env/config var); scratch -> `tempfile`; fixtures -> a package-relative import.

## Formatting

RULE: yapf, column 120, no nested f-strings -- never reuse the outer quote char
inside an f-string expression. `f"{d['k']}"`, never `f"{d["k"]}"`.
WHY: yapf's bundled parser can hard-crash on PEP 701 same-quote nesting, and
un-nested f-strings are clearer regardless of tooling.

## Perf patterns

RULE: apply these ONLY in a proven hot path (per-node / per-edge / per-item /
per-match) and only when behavior-preserving. Cold code stays readable.
WHY: premature optimization costs clarity for nothing.

- local-alias a bound method reused across thousands of iterations
  (`append = lst.append`); cache a deep attribute leaf before the loop
  (`fn = obj.a.b.c`), never `obj.a.b.c()` per iteration;
- comprehensions over manual `.append` loops (LIST_APPEND in C);
  `list(filter(None, data))` over `[x for x in data if x]` when dropping falsy;
- hoist loop-invariant work out of the loop; do not re-derive what the caller
  already has; avoid `list.index()` and linear identity scans (they go quadratic) --
  keep an `{id(obj): index}` map beside the list;
- membership: frozen-literal `x in {1, 2, 3}` / `x in (1, 2, 3)` is fine;
- **never** swap an ordered list/dict to a set to speed membership where iteration
  order is observed. WHY: !! SILENT -- that is a determinism bug;
- `OrderedSet` over a plain `set`, in EVERY project and for every collection,
  including a membership-only one -- and the replacement for a set is `OrderedSet`,
  never `dict.fromkeys` or `{'a': None}`. A `Set[...]` / `FrozenSet[...]`
  annotation is a smell. WHY: !! SILENT -- a set that is "membership only" today
  gets iterated tomorrow, and the order dependence is invisible until a hash-seed
  change moves it; the dict idiom then answers to neither API cleanly and its `|`
  degrades back to an unordered `set`;
- never probe a `defaultdict` with `d[k]` (it INSERTS) -- use `.get(k)` / `k in d`;
- `if seq:` over `if len(seq) != 0:`; `x.keys().isdisjoint(y)` over
  `len(x.keys() & y) > 0`;
- do not deepcopy a result the callers discard; give a hot class a custom
  `__deepcopy__`; reset a regenerable cache to its `__init__` state rather than
  deepcopying it;
- sympy: cheap structural `a == b` before `(a - b).simplify()`; use `.is_*`
  assumptions, `xreplace` over `subs`, the cached `symbolic.simplify`; never
  `simplify()` in a loop.

RULE: do NOT cargo-cult dead folklore on 3.11+. `__slots__`-for-speed, "cache
builtins into locals", and "`s += x` in a loop is always quadratic" are obsolete
(`+=` is linear on CPython -- reach for `"".join(...)` only when slicing the
accumulator defeats the in-place fast path).
WHY: those rewrites cost readability and buy nothing now.

RULE: do not let exceptions fire in normal control flow (creation cost); do not
override `__getattribute__` unless proxying.
WHY: both are real costs that remain real.

## KISS / YAGNI / no OO bloat

RULE: prefer functions plus small dataclasses (`@dataclass(slots=True)`, add
`frozen=True` where immutable) over deep hierarchies. Reuse existing utilities and
stdlib before writing new. No speculative generality, no config knobs "just in
case", no single-call-site abstractions. Delete dead code.
WHY: new code is a liability.

## Comments and docstrings

RULE: zero fluff. Explain only the non-obvious **why**, never the **what**. Never
restate signatures or types. Single line where possible.
WHY: a comment restating the code rots into a lie.

RULE: comment-to-code ratio stays at or below **0.2** -- at most 1 comment line per
5 code lines, per file and per hunk. DOCSTRINGS and `#:` annotations count. A test
docstring is one line and the intent goes in the test NAME; a module docstring is a
few lines, what and usage. Up to 0.5 is allowed only where the comment is
absolutely necessary -- a silent-miscompile trap, a why-not-the-obvious-thing, an
invariant the next reader would break. It is not a budget to spend.
WHY: the failure mode is writing the comment as a JUSTIFICATION aimed at a reviewer
instead of as INFORMATION for the next reader. Write for the reader who already
accepts the change and just needs the gotcha. If the comment is longer than the
code it explains, it is wrong. Counting only `#` lines while shipping a docstring
essay games the metric -- a reported 0.078 hid a 20-line module docstring, `#:`
annotations on every constant, and a paragraph docstring on every test.

RULE: ASCII only in comments and docstrings. No emoji, no warning or stop signs, no
em dash, no middle dot, no curly quotes, no multiplication sign, no ellipsis
character, no check or cross marks. Write `--`, `x`, `...`, and plain words
(`WARNING:`, `NOTE:`, `DO NOT`).
WHY: source has to survive tools, terminals, diffs and encodings that mangle it,
and a mangled comment is noise forever. Check with
`grep -nP '[^\x00-\x7F]' <file>`; sweep a change with
`git diff --name-only <base>..HEAD | xargs grep -nP '[^\x00-\x7F]'`. This is about
comments and docstrings; it does not by itself forbid non-ASCII in a runtime output
string (a multiplication sign in a speedup column is conventional).

RULE: in prose written into the repo -- commit messages, docstrings, docs --
avoid the AI tells. Banned above all: the "Y is not just X, but Z" antithesis, and
its cousins ("not only X but also Y", "not merely X", "more than a X"). Also:
aphoristic openers, em-dash pileups, opening by saying what something is NOT,
meta-scaffolding ("It is worth noting"), cutesy section titles, and restating the
same framing section after section.
WHY: that voice reads as machine-generated and gets rejected on sight. State the
point directly, one claim per sentence.

RULE: after writing or modernizing Python, run all gates in Part B on the result.
WHY: the style rules and the gates are one workflow, not two.

---

# Conflicts between the rule sets

Stated rather than silently resolved. Decide per case; when in doubt in this repo,
the DaCe-specific rule wins inside `dace/`, because CI enforces it.

1. **Direct imports.** DaCe CONTRIBUTING (A7): no direct class/function imports
   except `SDFG`, `SDFGState`, `Memlet`, `InterstateEdge`. Modern-Python rule
   ("Imports", Part B continued): absolute, package-qualified `from pkg.sub.mod import fn`, which IS a
   direct function import. Inside `dace/`, follow CONTRIBUTING -- import the module
   and qualify at the use site. The "absolute, never relative, top-level" half of
   the B rule is uncontested and applies everywhere.

2. **Leading underscore names.** Modern-Python rule: never prefix a function,
   class, or module with `_`; no private carve-out. DaCe core uses `_parent`,
   `_parent_sdfg`, `_nx`, `_eval_subs`, and the gotchas in A5 require touching them
   by name. Do not rename existing DaCe underscore attributes; apply the no-
   underscore rule to new code only.

3. **Python version floor.** The modern-Python rules target >= 3.10 and use 3.10+
   syntax unconditionally. DaCe's rule is "match what the packaging metadata
   declares, check before using new syntax". Here they agree: `pyproject.toml`
   declares `requires-python = ">=3.10, <3.15"`. Re-check the metadata before
   assuming that stays true.

4. **`__slots__`.** The Python rules mandate slots on new classes. A global slots
   sweep of DaCe was measured and SKIPPED -- the win on 3.11+ is memory and cache
   locality, not attribute speed, and DaCe's property/serialization machinery makes
   it invasive. Apply the rule to new standalone classes; do not sweep existing
   DaCe node/property classes for it.

# Out of scope

- `CONTRIBUTING.md` is upstream and READ-ONLY. Do not edit it.
- Process -- how to run the tests on a given machine, git and PR workflow, how to
  measure -- is deliberately absent. This file is about writing correct code in
  this repository.

# Design: TSVC kernel corpus + per-consumer fixtures (design only)

## Problem

Today the ~150 TSVC kernels are inline `@dace.program` defs scattered across
`tests/passes/vectorization/tsvc_1d/*.py` and `tsvc_2d/*.py`, each test file
carrying its own `run_vectorization_test` copy and `build_tsvc_matrix`
branch-dedup. We want one corpus, declared once, consumed by several transforms.

## Core idea — locked division of responsibility

**The corpus owns ONLY three things: the kernels (+ neutral `tags`/metadata),
`collect()`, and input generation. Nothing else.**

- The corpus owns **no transform**, **no reference oracle**, **no assertions**,
  and **no per-consumer knowledge**. It must not import or name canonicalize /
  loop2map / vectorize, and carries no consumer-specific expectations or skips.

**Each transform is its own consumer file** that:
1. tests its *own* hand-written examples first, THEN
2. `import`s the corpus, `collect()`s kernels, and runs the transform on them
   for e2e correctness;
3. owns its *own* config matrix (parametrization axes), its *own*
   tag→expectation interpretation, and *all* assertions — numerical equivalence
   (it builds the untransformed reference itself) **and** any structural checks.

Dependency is one-way: **consumers import the corpus; the corpus never imports a
consumer.** No kernel is re-declared in a consumer. This is the design for
canonicalize and loop2map, and **vectorization follows the identical pattern** —
its tile-path tests become just another corpus consumer.

## 1. Corpus layout — ONE kernels file (hard requirement)

```
tests/corpus/
  __init__.py
  tsvc/
    __init__.py            # re-exports: collect, TSVCKernel, make_inputs, tsvc_kernel
    _registry.py           # @tsvc_kernel decorator + global registry + collect()
    _inputs.py             # make_inputs(kernel, seed) — input generation only
    tsvc_kernels.py        # *** ALL *** TSVC kernels (1-D, 2-D, 3-D) in this ONE file
```

**All TSVC kernels live in a single `tsvc_kernels.py`** — NOT split by
dimensionality or topic. The metadata (`dims`, `tags`), not the file layout, is
the sole carrier of structure: a consumer selects its subset by `dims`/`tags`.
There is no `reference.py` and no consumer-aware module in the corpus — the
corpus stays consumer-agnostic. (The existing `tests/corpus/` already holds the
cloudsc fixtures; the same `collect()` mechanism can tag fixtures `tsvc` vs
`cloudsc`, but the corpus still owns only kernels + collect + inputs.)

## 2. Kernel descriptor — NEUTRAL metadata only

One frozen dataclass per kernel. It carries only kernel-intrinsic facts — never
a consumer's name, expectation, or skip:

```python
@dataclass(frozen=True)
class ArraySpec:
    shape: Tuple[str, ...]     # symbolic, e.g. ("LEN_1D",) or ("LEN_2D", "LEN_2D")
    dtype: dace.typeclass      # float64 / float32 / int32
    layout: str = "C"          # "C" | "F"
    role: str = "inout"        # "in" | "out" | "inout" (drives input init + which arrays a consumer compares)

@dataclass(frozen=True)
class TSVCKernel:
    name: str                          # "s1112"
    program: Callable                  # the @dace.program (NOT yet to_sdfg'd)
    arrays: Dict[str, ArraySpec]
    symbols: Dict[str, int]            # default concrete sizes: {"LEN_1D": 256}
    dims: int                          # K = 1 | 2 | 3
    tags: FrozenSet[str]               # NEUTRAL kernel properties only:
                                       #   {"branch","reduction","gather","scatter",
                                       #    "strided","break","recurrence","transcendental",...}
```

`tags` describe the KERNEL, not any transform's opinion of it. A consumer maps
tags → its own expectation (e.g. the loop2map consumer decides a `recurrence`
kernel stays sequential; the vectorize consumer decides a `branch` kernel needs
both branch_mode arms). That interpretation lives in the consumer, never here.

`tags` are derived where possible (AST scan of `program.f`) and overridable
explicitly. The existing
[tsvc_matrix.py:kernel_has_branch](tests/passes/vectorization/helpers/tsvc_matrix.py)
predicate becomes the `branch` tag-deriver — one of several.

## 3. Registration + collection (the whole corpus API)

```python
# _registry.py
_REGISTRY: Dict[str, TSVCKernel] = {}

def tsvc_kernel(**meta):
    """Transparent, side-effect-only registrar (see §3b)."""
    def deco(program):
        k = _build_descriptor(program, meta)
        _REGISTRY.setdefault(k.name, k)     # idempotent: dedup by name
        return program                       # IDENTITY — zero behavior change
    return deco

def collect(*, dims=None, tags=None, name=None) -> List[TSVCKernel]:
    """Filtered view of the corpus. `tags` = require-all-of. The ONLY entry
    point a consumer uses."""
    _ensure_loaded()          # import tsvc_kernels.py once (populates _REGISTRY)
    out = list(_REGISTRY.values())
    if dims is not None: out = [k for k in out if k.dims == dims]
    if tags:             out = [k for k in out if tags <= k.tags]
    if name:             out = [k for k in out if re.search(name, k.name)]
    return out
```

Plus input generation (the third and last corpus responsibility):

```python
# _inputs.py
def make_inputs(kernel: TSVCKernel, seed: int = 0) -> Tuple[Dict, Dict]:
    """Build (arrays, params) for a compiled-SDFG call: one seeded
    numpy.random.default_rng(seed) array per ArraySpec at the concrete `symbols`,
    honoring dtype + C/F layout; output-only arrays zero-initialized. params =
    the symbol->size map. Pure data generation — no transform, no oracle."""
```

That is the complete corpus surface: `collect`, `make_inputs`, `TSVCKernel`,
`tsvc_kernel`. No correctness logic, no consumer hooks.

## 3b. Decorator safety — MUST NOT change behavior or collide with pytest

The kernels arrive via the **yakup/dev merge**. If yakup/dev already ships a
kernel decorator, adopt it and hang the metadata off it — don't add a competing
one. The contract either way:

- **`@tsvc_kernel` returns its argument unchanged** (identity); it only appends
  to `_REGISTRY`. Decorator order is `@tsvc_kernel(...)` *outside* `@dace.program`
  (innermost first); the name still binds the same `DaceProgram`, and
  `.to_sdfg()` / `.f` / `.name` / the SDFG are untouched. **Existing
  transformation tests behave byte-identically** — the corpus layer is additive.
- **No pytest collision.** pytest collects `test_*` / `Test*`; kernels are
  `dace_sNNNN`, matching neither, and the decorator adds no `pytest.mark` and
  touches no pytest attribute (`pytestmark`, `__test__`). A real pytest marker,
  if ever needed, is applied in the *consumer* via `parametrize`, never on the
  corpus def.
- **Idempotent** (dedup by name; `_ensure_loaded` rides the module cache).
- **Tags derived read-only** from `program.f`; never rewrites the program.

## 4. Correctness is CONSUMER-side (the corpus provides only inputs)

The corpus deliberately ships no reference oracle. A consumer establishes
correctness itself, using the corpus's `make_inputs`, with the
untransformed-SDFG-as-reference convention:

```python
# consumer-side shared helper (lives with the consumers, NOT in the corpus)
def assert_preserves_numerics(kernel, transform, *, seed=0, rtol=1e-12):
    arrays, params = make_inputs(kernel, seed)             # corpus input gen
    ref = kernel.program.to_sdfg(simplify=True)            # untransformed reference
    ra = deepcopy(arrays); ref.compile()(**ra, **params)
    got = kernel.program.to_sdfg(simplify=True)
    transform(got)                                         # the consumer's transform+config
    got.validate()
    ga = deepcopy(arrays); got.compile()(**ga, **params)
    for n, spec in kernel.arrays.items():
        if spec.role in ("out", "inout"):
            assert numpy.allclose(ra[n], ga[n], rtol=rtol)
```

This helper is shared *among consumers* (e.g. `tests/.../corpus_consumers/_runner.py`)
but is **not** part of the corpus — it imports the corpus, not vice versa. A
consumer may use it for the numerical leg and add its own structural assertions
on top; consumers that prefer a numpy/fortran ground truth supply their own.

## 5. Per-consumer files (each: own examples THEN corpus run)

Each consumer is a self-contained test file. It owns its config axes, its
tag→expectation mapping, its skips, and its assertions. Shape:

```python
# test_<transform>.py
from tests.corpus.tsvc import collect
from tests...corpus_consumers._runner import assert_preserves_numerics

# 1) the consumer's OWN hand-written examples
def test_<transform>_example_foo(): ...      # bespoke SDFGs, full assertions

# 2) then the corpus, parametrized over THIS consumer's axes
def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        metafunc.parametrize("case", _build_cases(), ids=_ids)

def test_<transform>_on_corpus(case):
    kernel, cfg = case
    assert_preserves_numerics(kernel, lambda s: <transform>(cfg).apply_pass(s, {}))
    # + consumer-specific structural checks keyed on kernel.tags
```

### a. Canonicalize consumer
- Own examples (recurrence-refusal, mixed-parallelism, …) tested first.
- Corpus axes: canonicalize knobs (`peel_limit`, `assign_loop_iterator_post_value`, …).
- Asserts canonicalize is value-preserving (numerics) + structural invariants
  (e.g. no `ConditionalBlock` left when branch-normalizing).

### b. LoopToMap consumer
- Own examples first.
- Corpus axes: `permissive {True, False}`.
- Asserts numerics preserved **AND** a structural check the consumer derives
  from tags: a parallel kernel (no `recurrence`/`break` tag) must yield ≥1 `Map`;
  a `recurrence`/`break` kernel must stay a `LoopRegion`.

### c. Vectorization consumer (follows the identical pattern)
- Own examples first.
- Corpus axes: `branch_mode {merge,fp_factor} × remainder {scalar,masked} × emission_style × widths/ISA`.
- The consumer's own pruner collapses the branch arms for tag-`branch`-free
  kernels (today's `build_tsvc_matrix` dedup, now living in the vectorize
  consumer) and skips its WIP combos — all keyed on `kernel.tags`, never on
  corpus-stored consumer state.

## 6. Skips / expectations — consumer-owned, tag-driven

Because the corpus carries no consumer state, each consumer derives its own
skips/expectations from neutral `tags` (and its own local table for genuine
per-kernel WIP exceptions), deselecting at collection (mirrors the current
[conftest.pytest_collection_modifyitems](tests/passes/vectorization/conftest.py)).
Test id = `<kernel>-<axis1>-<axis2>...` (e.g. `s1112-merge-masked-W8`). One
place per consumer to flip when that consumer's feature lands.

## 7. Migration (incremental; kernels move once)

1. Land `tests/corpus/tsvc/` = `_registry.py` (decorator + `collect`) +
   `_inputs.py` (`make_inputs`) + empty `tsvc_kernels.py`. Corpus-only, no
   consumer code.
2. Move 1-D kernels into `tsvc_kernels.py` with `@tsvc_kernel(...)` metadata;
   rewrite `tsvc_1d/test_selected.py` etc. as vectorize-consumer files that
   `collect(dims=1)`. Sweep green at each batch.
3. Same for 2-D / 3-D (same single file).
4. Delete the per-file `run_vectorization_test` copies; the consumer-shared
   `assert_preserves_numerics` + the canonical harness replace them.
5. Add the canonicalize + loop2map consumer files (new coverage, same corpus).

## Key properties

- **Corpus = kernels + `collect()` + `make_inputs` only**; consumer-agnostic,
  imports no consumer, stores no consumer expectation.
- **One declaration per kernel** in one file; appears in every consumer that
  `collect()`s its dims/tags.
- **Each transform is a consumer file**: own examples first, then the corpus;
  owns config + tag-interpretation + assertions (numerical + structural).
- **Correctness is consumer-side** via untransformed-SDFG reference + corpus
  `make_inputs`; numpy/fortran ground truth is a consumer's own opt-in.
- **Decorator is transparent** — no behavior change for existing transforms, no
  pytest collision.
- Reuses today's `kernel_has_branch` (→ the `branch` tag-deriver); the
  `build_tsvc_matrix` dedup moves into the vectorize consumer, not the corpus.

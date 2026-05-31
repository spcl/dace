# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pytest configuration for the topical vectorization test directory.

Provides the ``branch_mode`` fixture, parameterized over the two branch
lowering paths the M3.2 work exposes on ``VectorizeCPU``:

- ``"fp_factor"``: today's path, ``EliminateBranches`` collapses if/else
  to ``a = c*x + (1-c)*y``.
- ``"merge"``: the new M3 path, ``SameWriteSetIfElseToMergeCFG`` plus
  ``BranchNormalization`` rewrite arms into ``merge(cond, ..., ...)``
  tasklets that the vectorizer will later lower to a SIMD blend.

Tests that exercise conditionals consume ``branch_mode`` and forward it
to ``run_vectorization_test``. Tests with no branches do not need it.
Both modes must produce numerically identical results against the
unvectorized scalar reference, otherwise the two lowerings have drifted.
"""
import os

import pytest

# Note: tests that pass legacy knobs (``lower_to_intrinsics=True``,
# ``collapse_laneid_index_loads=True``, ``filter_map=``) are NOT deselected
# at collection time — the legacy non-NSDFG ``VectorizeCPU`` path is a
# permanent first-class knob, exercised end-to-end. The harness routes such
# tests to the legacy path automatically (see ``run_vectorization_test``'s
# legacy-knob detection); the tile path's per-knob skips only fire when a
# test forces the tile arm explicitly via ``vectorize_config``.


@pytest.fixture(params=["fp_factor", "merge"])
def branch_mode(request) -> str:
    """Branch lowering variant the K=1 tile path must support:

    - ``"merge"`` — same-write-set if/else -> per-lane ``TileMerge`` select.
    - ``"fp_factor"`` — ``c*x + (1-c)*y`` arithmetic (the legacy
      ``EliminateBranches`` form), lowered on the tile path via tile binops.

    Both must produce numerically identical results against the unvectorized
    scalar reference."""
    return request.param


@pytest.fixture(params=["flat", "nested", "nested_copies"])
def tile_emit_mode(request):
    """Tile-arm emit-path × boundary-copy combination, opt-in.

    Combines ``nest_map_bodies`` × ``insert_copies`` into three canonical
    configurations (vs the 4-cell independent cross-product). Tests opt in
    by taking ``tile_emit_mode`` in their signature; tests that don't take
    it inherit the harness defaults (``nest_map_bodies=False``,
    ``insert_copies=False``).

    - ``"flat"`` — ``nest_map_bodies=False, insert_copies=False`` (default
      tile arm, hybrid emit path: flat body -> ``EmitTileOps``; already-NSDFG
      body -> ``PromoteNSDFGBodyToTiles`` descent).
    - ``"nested"`` — ``nest_map_bodies=True, insert_copies=False`` (descent
      is the single emit path; every innermost map body is wrapped into a
      NestedSDFG so ``Promote`` handles it).
    - ``"nested_copies"`` — ``nest_map_bodies=True, insert_copies=True``
      (descent + boundary copy emission — legacy-arm structural-test
      contract; required by stencil tests that assert on the
      ``<base>_vec`` shared union buffer).

    The ``(insert_copies=True, nest_map_bodies=False)`` cell is omitted
    because boundary copies without body nesting rarely affect the tile
    arm and would inflate the combo count without adding distinct coverage.

    :returns: ``(nest_map_bodies, insert_copies)`` tuple — unpack in tests
        and forward to :func:`run_vectorization_test`.
    """
    return {
        "flat": (False, False),
        "nested": (True, False),
        "nested_copies": (True, True),
    }[request.param]


@pytest.fixture(params=["default", "sve_style"])
def emission_style(request) -> str:
    """Emission model the K=1 tile path must support:

    - ``"default"`` — fixed-width tile maps + remainder per ``remainder_strategy``.
    - ``"sve_style"`` — SVE-style always-mask emission: the per-core block runs
      as a masked while-loop, the tail handled by the iteration mask (no
      remainder split).

    Both must produce numerically identical results against the unvectorized
    scalar reference."""
    return request.param


def pytest_configure(config):
    """Register the ``tile_nodes`` marker (kept for back-compat; the tile-op
    config is now the only arm, so the marker is a no-op selector).

    :param config: The pytest config object.
    """
    config.addinivalue_line(
        "markers",
        "tile_nodes: legacy marker; the K-dim tile-op config "
        "(VectorizeCPUMultiDim) is now the sole vectorize_config arm.",
    )
    config.addinivalue_line(
        "markers",
        "simple: redundant / trivial test (duplicates lib_nodes coverage or "
        "a more comprehensive sibling). Skipped by default; opt in with "
        "``--run-simple`` to include them in the sweep.",
    )


def pytest_addoption(parser):
    """Add ``--tile-nest-bodies`` to select the descent-only emit arm.

    :param parser: The pytest option parser.
    """
    parser.addoption(
        "--run-simple",
        action="store_true",
        default=False,
        help="Include tests marked ``@pytest.mark.simple`` (redundant / "
        "trivial duplicates of a more comprehensive sibling test or of "
        "``lib_nodes/`` coverage). Default sweep skips these to keep the "
        "feedback loop fast; the hardening sweep / CI includes them.",
    )
    parser.addoption(
        "--run-full-matrix",
        action="store_true",
        default=False,
        help="Hyper-thorough sweep mode: every test that goes through "
        "``run_vectorization_test`` is parametrised over the full knob "
        "matrix (branch_mode x tile_emit_mode x emission_style x "
        "remainder_strategy), even when the test signature only declares "
        "a subset. The harness picks up the missing knobs from "
        "``request.getfixturevalue`` so tests don't need code changes. "
        "Expect ~3-10x more test cases than the default sweep — opt-in "
        "only because most tests pin specific knob shapes by design.",
    )


def pytest_generate_tests(metafunc):
    """Parametrise ``vectorize_config`` over the tile-op + legacy paths.

    Two first-class arms — each is a permanent optimisation knob; every
    harness test runs through both (subject to per-arm skip predicates):

    - ``"tile_nodes"`` — the K-dim tile-op path (``VectorizeCPUMultiDim``)
      in hybrid mode (flat -> ``EmitTileOps``; already-NSDFG bodies ->
      descent ``PromoteNSDFGBodyToTiles``). The single tile arm; the
      previously-separate ``tile_nodes_nested`` arm is now expressed as a
      per-test fixture (``tile_emit_mode`` or explicit ``nest_map_bodies``
      kwarg to ``run_vectorization_test``).
    - ``"legacy_cpu"`` — the legacy non-NSDFG ``VectorizeCPU`` path
      (``scalar_postamble``, K=1 / VLEN=8). A permanent first-class knob;
      tests that pass legacy-only knobs (``lower_to_intrinsics=True`` etc.)
      route here automatically, and EVERY harness test also runs once on
      this arm so regressions on the legacy path surface early.

    :param metafunc: The pytest metafunc for the test being collected.
    """
    full_matrix = metafunc.config.getoption("--run-full-matrix")
    if "vectorize_config" in metafunc.fixturenames:
        metafunc.parametrize("vectorize_config", ["tile_nodes", "legacy_cpu"], indirect=True)
    elif full_matrix:
        # ``--run-full-matrix``: also parametrise ``vectorize_config`` for
        # tests that don't declare it. The harness routes the missing knob
        # through ``request.getfixturevalue`` so the test code doesn't have
        # to change.
        metafunc.fixturenames.append("vectorize_config")
        metafunc.parametrize("vectorize_config", ["tile_nodes", "legacy_cpu"], indirect=True)
    # ``--run-full-matrix``: inject parametrisation for every knob fixture
    # the test signature does NOT already declare. We use ``indirect=True``
    # so values flow through the existing fixtures (whose definitions read
    # ``request.param``); the harness picks them up from
    # ``request.getfixturevalue`` when the caller didn't pass an explicit
    # kwarg.
    if not full_matrix:
        return
    knob_params = {
        "branch_mode": ["fp_factor", "merge"],
        "tile_emit_mode": ["flat", "nested", "nested_copies"],
        "emission_style": ["default", "sve_style"],
        "remainder_strategy": ["scalar", "masked"],
    }
    for knob, vals in knob_params.items():
        if knob in metafunc.fixturenames:
            continue  # already declared by the test; pytest will parametrise via the fixture itself
        # The fixture isn't on the test signature but we want this test to
        # run once per value. ``metafunc.parametrize`` requires the argname
        # to be a fixturename, so append it first (no value reaches the
        # test function — the harness reads it via ``request``).
        metafunc.fixturenames.append(knob)
        metafunc.parametrize(knob, vals, indirect=True)


#: Test files that exercise ONLY the legacy 1D ``VectorizeCPU`` / ``VectorizeSVE``
#: / ``VectorizeBreak`` path (they directly instantiate those orchestrators and
#: never route through the tile harness ``run_vectorization_test`` /
#: ``VectorizeCPUMultiDim``). The branch is migrating to the K-dim tile-op path,
#: so these are disabled for now (user directive). Re-enable by removing an entry
#: as its kernels gain tile-path coverage. Paths are relative to this directory.
_LEGACY_ONLY_FILES = frozenset({
    "kernels/test_disjoint_chain.py",
    "kernels/test_gather_scatter_knob.py",
    "kernels/test_int_floor.py",
    "kernels/test_inter_lane_stride.py",
    "kernels/test_multi_element_strided.py",
    "kernels/test_remainder_required.py",
    "passes/test_detect_multi_dim_strided.py",
    "passes/test_force_op_variant.py",
    "sve/test_sve_style.py",
    "sve/test_sve_style_parity.py",
    "sve/test_sve_variable_probe.py",
})


def _is_legacy_only(item) -> bool:
    """Whether ``item`` belongs to a legacy-only (non-tile-path) test file.

    Tests in :data:`_LEGACY_ONLY_FILES` are deselected at collection time —
    they exercise dataflow / orchestrator paths the tile arm does not own.
    A test that merely passes a legacy *knob* (``lower_to_intrinsics=True``
    etc.) is NOT legacy-only — the harness routes it to the legacy
    ``VectorizeCPU`` arm automatically, so it stays collected and runs.

    :param item: A collected pytest item.
    :returns: ``True`` when ``item``'s file is in :data:`_LEGACY_ONLY_FILES`.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rel = os.path.relpath(os.fspath(item.path), here)
    except (AttributeError, ValueError):
        return False
    return rel.replace(os.sep, "/") in _LEGACY_ONLY_FILES


def _knob_combo_supported(params: dict) -> bool:
    """Whether a (branch_mode, remainder_strategy, emission_style) combo is a
    valid tile-pipeline configuration — independent of the kernel.

    Only the knobs present in ``params`` constrain the result (a test that
    declares just ``remainder_strategy`` is never deselected on the others).
    The remaining rule:

    - ``emission_style='sve_style'`` (always-masked, no remainder split) forces
      ``branch_mode='merge'`` — ``fp_factor``'s ``c*x + (1-c)*y`` arithmetic
      cannot share the ``_iter_mask`` predicate the SVE chain emits everywhere.
      The remainder knob is now free (``remainder='masked'`` simply re-uses
      ``full_mask`` since the SVE chain is itself always-masked).

    ``fp_factor`` + ``remainder='masked'`` is allowed on the tile path: the
    orchestrator combines the per-lane bool mask with the float arithmetic by
    a single bool->float cast on the merge result. Previously deselected as a
    locked-plan rule; now exercised end-to-end by the harness.

    :param params: The item's parametrization (callspec ``params``).
    :returns: ``False`` for a globally-invalid combo (deselected at collection
        so it is never reported as a skip), ``True`` otherwise.
    """
    branch = params.get("branch_mode")
    emission = params.get("emission_style")
    remainder = params.get("remainder_strategy")
    if emission == "sve_style" and branch not in (None, "merge"):
        return False
    # ``vectorize_config=legacy_cpu`` ignores ``tile_emit_mode`` (it has no
    # ``nest_map_bodies`` concept — legacy keeps bodies flat). The combo
    # would run with the tile_emit_mode value simply being ignored, but
    # exercising 3 redundant variants under legacy adds nothing. Keep only
    # the ``"flat"`` arm of ``tile_emit_mode`` on the legacy path.
    tile_emit = params.get("tile_emit_mode")
    vec_config = params.get("vectorize_config")
    if vec_config == "legacy_cpu" and tile_emit is not None and tile_emit != "flat":
        return False
    # ``legacy_cpu`` cannot do ``use_fp_factor=True + masked_remainder`` (its
    # ``VectorizeCPU`` constructor raises ValueError) and ``sve_style`` has
    # no remainder loop (the global ``_iter_mask`` covers the tail).
    # Deselect at collection so these don't surface as runtime skips.
    if vec_config == "legacy_cpu":
        if branch == "fp_factor" and remainder == "masked":
            return False
        if emission == "sve_style" and remainder is not None and remainder != "scalar":
            return False
    return True


def pytest_collection_modifyitems(config, items):
    """Deselect globally-invalid knob combinations at collection time.

    The ``branch_mode`` x ``remainder_strategy`` x ``emission_style`` cross
    product contains combinations no tile pipeline supports (e.g.
    ``fp_factor`` + ``masked``). Previously each ran and SKIPPED, inflating the
    skip count with pure knob-incompatibility noise. We now drop them before
    collection so the remaining skips reflect genuine feature gaps only.

    :param config: The pytest config object.
    :param items: The collected test items (filtered in place).
    """
    kept, deselected = [], []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if _is_legacy_only(item) or not _knob_combo_supported(params):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
    # Skip ``@pytest.mark.simple`` items unless ``--run-simple`` is given.
    # A skip (rather than a deselect) keeps the test visible in collection
    # reports so the gate is observable.
    if not config.getoption("--run-simple"):
        skip_simple = pytest.mark.skip(reason="@pytest.mark.simple — pass --run-simple to include")
        for item in items:
            if "simple" in item.keywords:
                item.add_marker(skip_simple)


@pytest.fixture
def vectorize_config(request) -> str:
    """Backend pipeline under test (parametrised by ``pytest_generate_tests``).

    - ``"tile_nodes"`` — the K-dim tile-op routing through
      ``VectorizeCPUMultiDim``: K=1 and K>=2 both emit the tile lib nodes
      (TileBinop / TileLoad / TileStore / TileMerge / TileGather /
      TileScatter), expanded to the per-ISA backend (scalar reference in
      the harness). Body-nesting is controlled per-test via the
      ``tile_emit_mode`` fixture (or an explicit ``nest_map_bodies``
      kwarg to :func:`run_vectorization_test`).
    - ``"legacy_cpu"`` — the legacy ``VectorizeCPU`` 1D path.

    Both arms must match the unvectorized scalar reference."""
    return request.param


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request) -> str:
    """Parametrise tests over the remainder-handling strategies wired into
    ``VectorizeCPU``.

    There is no ``"divides_evenly"`` strategy: P2
    (``SplitMapForVectorRemainder``) always runs and skips the split
    itself when the trip count is *provably* a multiple of ``W`` (so a
    provably-divisible map carries no remainder regardless of the
    strategy below). The strategy only selects the remainder *shape*
    when divisibility cannot be proven:

    - ``"scalar"`` — P2(mode='scalar'): step-1 ``Sequential`` postamble.
    - ``"masked"`` — P2(mode='masked') + P3 (iter_mask attach) + the
      mask-aware emitter, for full SIMD-width execution of the trailing
      ``R<W`` elements.

    ``"full_loop_mask"`` is queued (R3) and not yet exercised.

    Tests that go through ``run_vectorization_test`` and want to cover
    every strategy declare a ``remainder_strategy`` parameter. Tests
    that pin a specific strategy can ignore the fixture and pass the
    knob directly to ``run_vectorization_test``."""
    return request.param

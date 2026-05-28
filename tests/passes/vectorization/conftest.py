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


@pytest.fixture(params=["fp_factor", "merge"])
def branch_mode(request) -> str:
    """Branch lowering variant the K=1 tile path must support:

    - ``"merge"`` — same-write-set if/else -> per-lane ``TileMerge`` select.
    - ``"fp_factor"`` — ``c*x + (1-c)*y`` arithmetic (the legacy
      ``EliminateBranches`` form), lowered on the tile path via tile binops.

    Both must produce numerically identical results against the unvectorized
    scalar reference."""
    return request.param


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
    # ``--tile-nest-bodies`` is a GLOBAL override: it flips every harness test
    # (whether or not it takes the ``vectorize_config`` fixture) to the
    # single-descent path, so the whole suite can be run under nest_map_bodies
    # =True. Stored on the harness module so ``run_vectorization_test`` reads it.
    from tests.passes.vectorization.helpers import harness as _harness
    _harness.FORCE_NEST_MAP_BODIES = bool(config.getoption("--tile-nest-bodies"))


def pytest_addoption(parser):
    """Add ``--tile-nest-bodies`` to select the descent-only emit arm.

    :param parser: The pytest option parser.
    """
    parser.addoption(
        "--tile-nest-bodies",
        action="store_true",
        default=False,
        help="Run the 'tile_nodes_nested' vectorize_config arm "
        "(VectorizeCPUMultiDim(nest_map_bodies=True): every map body nested "
        "into a NestedSDFG so PromoteNSDFGBodyToTiles is the single emit path) "
        "instead of the default hybrid 'tile_nodes' arm.",
    )


def pytest_generate_tests(metafunc):
    """Parametrise ``vectorize_config`` over the tile-op path.

    The legacy 1D ``VectorizeCPU`` (``scalar_postamble``, K=1 / VLEN=8) arm
    has been dropped: the K-dim tile-op path (``VectorizeCPUMultiDim``) now
    covers both K=1 and K>=2, so the full sweep runs through it alone. The arm
    is the default hybrid ``"tile_nodes"`` unless ``--tile-nest-bodies`` selects
    the single-descent ``"tile_nodes_nested"`` arm (the in-progress emit path
    being brought to parity); both must match the unvectorized scalar reference.

    :param metafunc: The pytest metafunc for the test being collected.
    """
    if "vectorize_config" not in metafunc.fixturenames:
        return
    arm = "tile_nodes_nested" if metafunc.config.getoption("--tile-nest-bodies") else "tile_nodes"
    metafunc.parametrize("vectorize_config", [arm], indirect=True)


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
    "tsvc_1d/test_bulk.py",
    "tsvc_1d/test_misc.py",
    "tsvc_1d/test_vector_ops.py",
    "tsvc_2d/test_misc_2d.py",
})


def _is_legacy_only(item) -> bool:
    """Whether ``item`` belongs to a legacy-only (non-tile-path) test file.

    :param item: A collected pytest item.
    :returns: ``True`` if the item's file is in :data:`_LEGACY_ONLY_FILES`.
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
    if emission == "sve_style" and branch not in (None, "merge"):
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


@pytest.fixture
def vectorize_config(request) -> str:
    """Backend pipeline under test (parametrised by ``pytest_generate_tests``).

    - ``"tile_nodes"`` — the K-dim tile-op routing through
      ``VectorizeCPUMultiDim``: K=1 and K>=2 both emit the tile lib nodes
      (TileBinop / TileLoad / TileStore / TileMerge / TileGather /
      TileScatter), expanded to the per-ISA backend (scalar reference in
      the harness). The default arm now that the legacy ``scalar_postamble``
      1D path has been dropped.
    - ``"tile_nodes_nested"`` — the same path with
      ``VectorizeCPUMultiDim(nest_map_bodies=True)``: every innermost map body
      is nested into a NestedSDFG so the descent
      (``PromoteNSDFGBodyToTiles``) is the single emit path and ``EmitTileOps``
      is a no-op. Selected by ``--tile-nest-bodies``.

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

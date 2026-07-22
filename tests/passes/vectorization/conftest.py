# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pytest configuration for the topical vectorization test directory.

Provides the ``branch_mode`` fixture, parameterized over the two branch
lowering paths ``VectorizeCPUMultiDim`` supports:

- ``"fp_factor"``: ``EliminateBranches`` collapses if/else to
  ``a = c*x + (1-c)*y``, lowered on the tile path via tile binops.
- ``"merge"``: ``SameWriteSetIfElseToITECFG`` plus ``BranchNormalization``
  rewrite arms into ``ITE(cond, ..., ...)`` tasklets that the tile path
  lowers to a per-lane ``TileITE`` select. The fixture name is kept as
  ``"merge"`` (the routing label used by many downstream test files); the
  emitted ternary form is ``ITE``.

Tests that exercise conditionals consume ``branch_mode`` and forward it
to ``run_vectorization_test``. Tests with no branches do not need it.
Both modes must produce numerically identical results against the
unvectorized scalar reference, otherwise the two lowerings have drifted.
"""
import zlib

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_global_numpy_seed(request):
    """Seed NumPy's GLOBAL RNG per test, deterministically from the test node id,
    so any unseeded ``np.random.*`` call is reproducible across processes. A fresh
    global RNG (or a ``hash()``-derived seed) otherwise varies per run -> flaky,
    and a lucky draw can mask a real OOB (as it long did for s4116). Tests that
    build their own ``np.random.default_rng(seed=...)`` are unaffected (separate
    stream); the per-node-id seed also keeps distinct tests from sharing data.

    (Audit note: this crc32 matches ``tsvc.stable_seed``; reuse that single source
    once verified -- left inline here to avoid importing tsvc into conftest, which
    loads at collection and has whole-suite blast radius, without a test run.)"""
    np.random.seed(zlib.crc32(request.node.nodeid.encode()) & 0xFFFFFFFF)


@pytest.fixture(params=["fp_factor", "merge"])
def branch_mode(request) -> str:
    """Branch lowering variant the K=1 tile path must support:

    - ``"merge"`` — same-write-set if/else -> per-lane ``TileITE`` select.
    - ``"fp_factor"`` — ``c*x + (1-c)*y`` arithmetic (the
      ``EliminateBranches`` form), lowered on the tile path via tile binops.

    Both must produce numerically identical results against the unvectorized
    scalar reference."""
    return request.param


@pytest.fixture(params=["no_copies", "copies"])
def tile_emit_mode(request):
    """Tile-arm boundary-copy emission toggle, opt-in.

    Selects whether the tile arm emits boundary copies (``insert_copies``).
    Tests opt in by taking ``tile_emit_mode`` in their signature; tests that
    don't take it inherit the harness default (``insert_copies=False``).

    - ``"no_copies"`` — ``insert_copies=False`` (default tile arm, no
      boundary-copy emission).
    - ``"copies"`` — ``insert_copies=True`` (boundary copy emission), used by
      stencil tests that exercise the boundary-copy path.

    :returns: the ``insert_copies`` bool — forward to
        :func:`run_vectorization_test`.
    """
    return {
        "no_copies": False,
        "copies": True,
    }[request.param]


@pytest.fixture(params=["default"])
def emission_style(request) -> str:
    """Emission model the K=1 tile path must support:

    - ``"default"`` — fixed-width tile maps + remainder per ``remainder_strategy``.

    Must produce numerically identical results against the unvectorized
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
    """Parametrise ``vectorize_config`` over the tile-op path.

    - ``"tile_nodes"`` — the K-dim tile-op path (``VectorizeCPUMultiDim``)
      in hybrid mode (flat -> ``EmitTileOps``; already-NSDFG bodies ->
      descent ``PromoteNSDFGBodyToTiles``). The single tile arm.

    :param metafunc: The pytest metafunc for the test being collected.
    """
    full_matrix = metafunc.config.getoption("--run-full-matrix")
    if "vectorize_config" in metafunc.fixturenames:
        metafunc.parametrize("vectorize_config", ["tile_nodes"], indirect=True)
    elif full_matrix:
        # ``--run-full-matrix``: also parametrise ``vectorize_config`` for
        # tests that don't declare it. The harness routes the missing knob
        # through ``request.getfixturevalue`` so the test code doesn't have
        # to change.
        metafunc.fixturenames.append("vectorize_config")
        metafunc.parametrize("vectorize_config", ["tile_nodes"], indirect=True)
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
        "emission_style": ["default"],
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


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.simple`` items unless ``--run-simple`` is given.

    :param config: The pytest config object.
    :param items: The collected test items (filtered in place).
    """
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
      (TileBinop / TileLoad / TileStore / TileITE / TileLoad (gather) /
      TileStore (scatter)), expanded to the per-ISA backend (scalar reference in
      the harness). Boundary-copy emission is controlled per-test via the
      ``tile_emit_mode`` fixture (or an explicit ``insert_copies``
      kwarg to :func:`run_vectorization_test`).

    The tile arm must match the unvectorized scalar reference."""
    return request.param


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request) -> str:
    """Parametrise tests over the remainder-handling strategies of the tile
    path (``VectorizeCPUMultiDim`` via ``VectorizeConfig``).

    There is no ``"divides_evenly"`` strategy: the tile-remainder split
    always runs and skips the split itself when the trip count is *provably*
    a multiple of ``W`` (so a provably-divisible map carries no remainder
    regardless of the strategy below). The strategy only selects the
    remainder *shape* when divisibility cannot be proven; the harness maps:

    - ``"scalar"`` -> ``scalar_postamble``: W-strided interior + a step-1
      scalar tail.
    - ``"masked"`` -> ``masked_tail``: mask-free interior + a masked
      boundary region, for full SIMD-width execution of the trailing
      ``R<W`` elements.

    Tests that go through ``run_vectorization_test`` and want to cover
    every strategy declare a ``remainder_strategy`` parameter. Tests
    that pin a specific strategy can ignore the fixture and pass the
    knob directly to ``run_vectorization_test``."""
    return request.param

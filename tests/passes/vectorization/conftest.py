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
import pytest


@pytest.fixture(params=["fp_factor", "merge"])
def branch_mode(request) -> str:
    return request.param


@pytest.fixture(params=["default", "sve_style"])
def emission_style(request) -> str:
    """Parametrise tests over the vectorizer emission model.

    - ``"default"`` — today's pipeline (``sve_style=None``); the
      ``branch_mode`` / ``remainder_strategy`` fixtures still apply.
    - ``"sve_style"`` — SVE-style always-mask emission (``sve_style=
      "fixed"``): the per-core block runs as a masked while-loop, the
      tail is handled by the global ``_iter_mask`` (no remainder split).
      ``branch_mode`` is forced to ``merge`` and ``remainder_strategy``
      is N/A under this style (the harness skips incompatible param
      combos rather than passing them through).

    Both must produce numerically identical results against the
    unvectorized scalar reference."""
    return request.param


def pytest_configure(config):
    """Register the ``tile_nodes`` marker (opt-in tile-op config arm).

    :param config: The pytest config object.
    """
    config.addinivalue_line(
        "markers",
        "tile_nodes: also run this test under the K-dim tile-op config "
        "(VectorizeCPUMultiDim), in addition to the default scalar_postamble.",
    )


def pytest_generate_tests(metafunc):
    """Parametrise ``vectorize_config`` per the ``tile_nodes`` marker.

    Running *every* vectorization test under the tile-op config is
    wasteful; the tile path is validated on a curated subset that
    maximises distinct-pattern coverage (strided load/store, scatter,
    gather, buffer-reuse, long cloudsc kernels, ...). A test (or a whole
    module, via ``pytestmark``) opts that subset in with
    ``@pytest.mark.tile_nodes``; unmarked tests run only
    ``scalar_postamble``.

    :param metafunc: The pytest metafunc for the test being collected.
    """
    if "vectorize_config" not in metafunc.fixturenames:
        return
    configs = ["scalar_postamble"]
    if metafunc.definition.get_closest_marker("tile_nodes") is not None:
        configs.append("tile_nodes")
    metafunc.parametrize("vectorize_config", configs, indirect=True)


@pytest.fixture
def vectorize_config(request) -> str:
    """Backend pipeline under test (parametrised by ``pytest_generate_tests``):

    - ``"scalar_postamble"`` — today's ``VectorizeCPU`` (1D path, scalar
      postamble for the remainder tail). The default for every test that
      takes this fixture.
    - ``"tile_nodes"`` — the K-dim routing: K=1 falls back to
      ``VectorizeCPU`` (direct tasklet swap), K>=2 uses the
      ``VectorizeCPUMultiDim`` tile-op libnodes. Added only for tests
      carrying the ``tile_nodes`` marker.

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

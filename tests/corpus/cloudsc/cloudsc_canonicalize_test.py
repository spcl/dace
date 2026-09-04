# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end integration test: ``canonicalize`` on CloudSC stays numerically faithful.

One ``canonicalize`` call replaces the hand-rolled stage list of
``cloudsc_parallelize_chain_test`` and is checked against the same oracle: the
un-transformed SDFG (``simplify=False``) run on identical physical inputs under an IEEE
build. Two arms exercise the result:

* ``sequential`` -- every Map forced to a sequential schedule, so the only difference
  from the reference is canonicalize's own reassociation (``Reduce`` / ``Scan`` lifting,
  WCR accumulation order). Deterministic run-to-run.
* ``parallel`` -- the schedules canonicalize actually produced (OpenMP multicore), which
  additionally reorders parallel reductions. This is the arm that proves the emitted
  parallelism is numerically sound, not just that it compiles.

Both arms build at ``-O0`` with ``-fno-fast-math -ffp-contract=off`` so the C++ compiler
adds no reassociation of its own; every observed difference is attributable to the SDFG.
``-ffast-math`` is deliberately never used on CloudSC -- it rewrites the transcendentals
and the flux prefix sums, producing drift no tolerance can bound meaningfully.

Placement: this file lives with the CloudSC corpus rather than under ``tests/canonicalize``
because ``tests/conftest.py`` auto-marks that directory ``canonicalization``, which would
pull a multi-minute CloudSC parse and two full compiles into the canonicalization
workflow's 600s per-test timeout. As an ``integration`` test it runs in the dedicated
integration workflow next to the other CloudSC chain tests, and is excluded from general CI.

Cost: the parse and the ``canonicalize`` call dominate and are each paid once for the whole
file (module-scoped fixtures, and the parse is additionally cached on disk across runs).
Measured on a loaded 16-core dev box: parse 1245s, canonicalize 4433s -- hence the raised
``--timeout`` in ``integration-ci.yml``.

Manual run::

    pytest tests/corpus/cloudsc/cloudsc_canonicalize_test.py -v -s -m integration
"""
import collections
import contextlib
import copy
import gc
import os

import pytest

import dace
from dace import dtypes
from dace.codegen.codegen import generate_code
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            generate_cloudsc_inputs, make_sequential)

#: CloudSC species PARAMETER constants (Fortran NCLV=5, NCLDQL=1..NCLDQV=5), baked in so the
#: species/LU loops become constant-trip. klev/klon/kidia/kfdia stay symbolic.
_SPECIES_CONSTANTS = {'nclv': 5, 'ncldql': 1, 'ncldqi': 2, 'ncldqr': 3, 'ncldqs': 4, 'ncldqv': 5}

#: ``(force_sequential, rtol/atol)`` per arm.
#:
#: ``sequential``: canonicalize lifts accumulations into ``Reduce`` / ``Scan`` nodes and
#: WCR edges, which reassociates sums; a few ULP of drift is expected and bounded.
#: ``parallel``: adds OpenMP reduction order on top, which is thread-count dependent.
_ARMS = {
    'sequential': (True, 1e-12),
    'parallel': (False, 1e-10),
}

#: Structural end-state of ``canonicalize`` on CloudSC, pinned exactly so both a coverage
#: regression and an unreviewed improvement are a test failure. Every Map is outermost in
#: its state and every one reaches the backend as a ``#pragma omp parallel for``, so the
#: three numbers move together; a change means the parallelization decision changed.
#:
#: The 27 loops that stay sequential are NOT arbitrary: one is the vertical ``jk`` sweep
#: carrying ``zcovptot[0:klon]`` down the column, one is the ``pfsq*`` / ``pfcq*`` flux
#: prefix sum (a genuine scan), and the other 25 are column ``jl`` loops that
#: ``LoopToMap`` refuses on
#: an unprivatized per-iteration scalar (``zevap`` / ``zmelt`` / ``zzratio`` /
#: ``zexplicit_1_2`` written at ``dst_subset=0``).
_EXPECTED_MAPS = 234
_EXPECTED_SEQUENTIAL_LOOPS = 27
_EXPECTED_OMP_PARALLEL_FOR = 234


def _map_entries(sdfg: dace.SDFG):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _loop_regions(sdfg: dace.SDFG):
    """Control-flow loops that survived parallelization (still carry a loop variable)."""
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]


def _omp_parallel_for_count(sdfg: dace.SDFG) -> int:
    """``#pragma omp parallel for`` occurrences in the generated host code."""
    return sum(obj.code.count('#pragma omp parallel for') for obj in generate_code(sdfg) if obj.language == 'cpp')


def _run(sdfg: dace.SDFG, inputs, sequential: bool, tag: str):
    """Run ``sdfg`` once on a private copy of ``inputs`` under the IEEE build, returning the
    mutated buffers."""
    sdfg.name = f'cloudsc_canon_{tag}'
    if sequential:
        make_sequential(sdfg)
    # Specialization erases the species symbols; drop inputs the SDFG no longer takes.
    needed = set(sdfg.arglist().keys()) | {str(s) for s in sdfg.free_symbols}
    args = {k: v for k, v in copy.deepcopy(inputs).items() if k in needed}
    saved_args = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        sdfg(**args)
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved_args)
    return args


@pytest.fixture(scope='module')
def reference_sdfg_file(tmp_path_factory):
    """The un-transformed CloudSC SDFG, built once (the parse is minutes) and persisted."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    return path


@pytest.fixture(scope='module')
def reference_outputs(reference_sdfg_file):
    """Oracle: the un-transformed SDFG run sequentially under IEEE flags. Shared by both arms --
    it is the mathematical answer, independent of how the candidate is scheduled."""
    ref = dace.SDFG.from_file(reference_sdfg_file)
    inputs = generate_cloudsc_inputs(ref, seed=0)
    out = _run(ref, inputs, sequential=True, tag='ref')
    del ref
    gc.collect()
    return inputs, out


@pytest.fixture(scope='module')
def canonical_sdfg_file(reference_sdfg_file, tmp_path_factory):
    """CloudSC canonicalized once and persisted; each arm reloads it and schedules it its own way."""
    sdfg = dace.SDFG.from_file(reference_sdfg_file)
    # The loop transforms log every refused loop; keep the test output readable.
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        canonicalize(sdfg, validate=True, validate_all=False, specialize_constants=_SPECIES_CONSTANTS)
    sdfg.validate()
    path = str(tmp_path_factory.mktemp('cloudsc_canon') / 'cloudsc_canonical.sdfgz')
    sdfg.save(path, compress=True)
    return path


@pytest.mark.integration
def test_cloudsc_canonicalize_structure(canonical_sdfg_file):
    """Canonicalize expressed the kernel as parallel Maps, and said so in the generated code."""
    sdfg = dace.SDFG.from_file(canonical_sdfg_file)
    maps = _map_entries(sdfg)
    loops = _loop_regions(sdfg)
    schedules = collections.Counter(m.map.schedule for m in maps)
    n_omp = _omp_parallel_for_count(sdfg)
    print(f'canonicalize: maps={len(maps)} sequential_loops={len(loops)} omp_parallel_for={n_omp}')
    print(f'canonicalize: map schedules={dict(schedules)}')

    assert len(maps) == _EXPECTED_MAPS, f'{len(maps)} maps, expected {_EXPECTED_MAPS}'
    assert len(loops) == _EXPECTED_SEQUENTIAL_LOOPS, (f'{len(loops)} loops stayed sequential, '
                                                      f'expected {_EXPECTED_SEQUENTIAL_LOOPS}')
    # Canonicalize leaves every Map on ``ScheduleType.Default``; codegen's default-schedule
    # inference is what turns an outermost Map into a multicore one. Pinning it here keeps a
    # future pipeline that starts assigning schedules from doing so silently.
    assert set(schedules) == {dtypes.ScheduleType.Default}, f'unexpected map schedules: {dict(schedules)}'
    # The Map count alone does not prove parallelism reached the backend -- a Map that codegen
    # declines to emit a pragma for is a silent serialization.
    assert n_omp == _EXPECTED_OMP_PARALLEL_FOR, (f'{n_omp} "#pragma omp parallel for" in the generated code, '
                                                 f'expected {_EXPECTED_OMP_PARALLEL_FOR}')


@pytest.mark.integration
@pytest.mark.parametrize('arm', list(_ARMS))
def test_cloudsc_canonicalize_numerics(reference_outputs, canonical_sdfg_file, arm):
    force_sequential, tol = _ARMS[arm]
    inputs, reference_out = reference_outputs

    candidate = dace.SDFG.from_file(canonical_sdfg_file)
    out = _run(candidate, inputs, sequential=force_sequential, tag=arm)

    report = compare_outputs(out, reference_out, rtol=tol, atol=tol)
    worst = max(((ma, mr) for ma, mr, _ in report.values()), default=(0.0, 0.0))
    print(f'canonicalize/{arm}: worst |abs|={worst[0]:.3e} |rel|={worst[1]:.3e} (tol={tol:.0e})')
    bad = {name: (ma, mr) for name, (ma, mr, ok) in report.items() if not ok}
    assert not bad, (f'canonicalize/{arm}: outputs diverge from the un-transformed reference '
                     f'(tol={tol:.0e}): {bad}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

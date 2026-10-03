# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC through the default parallelization pipeline, on the host and on the device.

The pipeline under test is :class:`~dace.transformation.passes.parallelize.ParallelizePipeline`::

    specialize (nclv, ncldq*) -> ShortLoopUnroll -> UniqueLoopIterators -> simplify -> LoopToMap
        -> 2 x (StateFusionExtended -> FuseMaps)   [-> offload_to_gpu]

This is the dwarf-scale test of that pipeline: ``canonicalize`` does not fit a CI budget here, and
the device leg only needs the graph to reach fusible maps. The two legs share every phase; the
device leg appends the offload, so a divergence between them is the offload's and nothing else's.

Correctness is checked at EVERY phase boundary by ``run_pipeline``: the transformed graph is run on
the same physical inputs and every output array is compared against the un-transformed reference --
bit-exact through ``simplify``, relaxed from ``LoopToMap`` on (that phase is in ``_REASSOC_PHASES``,
so parallel reductions may reassociate). The device leg's final phase RUNS ON THE DEVICE.

Manual run::

    pytest tests/corpus/cloudsc/cloudsc_parallel_pipeline_test.py -v -s -m "integration and not gpu"
    pytest tests/corpus/cloudsc/cloudsc_parallel_pipeline_test.py -v -s -m "integration and gpu"
"""
import gc

import pytest

import dace
from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg
from dace.transformation.passes.parallelization_prep import DEFAULT_UNROLL_LIMIT, _constant_trip_count, _loops
from tests.corpus.cloudsc.pipelines import (build_reference_outputs, gpu_is_runnable, is_device_scheduled,
                                            load_checkpoint, map_entries, numeric_check_from, omp_parallel_for_count,
                                            run_pipeline, uniquely_named)

#: CloudSC species PARAMETER constants (Fortran NCLV=5, NCLDQL=1..NCLDQV=5). Baking them in is the
#: ``specialize`` phase: the species and LU loops become constant-trip, which is what gives
#: ``ShortLoopUnroll`` anything to unroll. klev / klon / kidia / kfdia stay symbolic.
SPECIES_CONSTANTS = {'nclv': 5, 'ncldql': 1, 'ncldqi': 2, 'ncldqr': 3, 'ncldqs': 4, 'ncldqv': 5}

#: Constant-trip loops the specialization is expected to expose, measured on the shipped dwarf: 29
#: at trip 5 (the ``nclv`` species loops) and 8 at trip 4, every one of them inside
#: ``DEFAULT_UNROLL_LIMIT``. Specializing is what creates them -- NONE of the 150 loops has a
#: constant trip count before it -- so a regression that stops baking the species constants shows up
#: here as ShortLoopUnroll silently having nothing to do. A lower bound, not the exact count: a
#: frontend change may legitimately produce more.
MIN_UNROLLABLE_AFTER_SPECIALIZE = 37

#: Maps ``LoopToMap`` lifts on the shipped dwarf. LOWER BOUND: 314 of 360 loops measured at HEAD,
#: floored at 300 so ordinary drift does not fail CI while a collapse does. Two pipeline stages hold
#: that number up, and removing either is a silent parallelism loss this assert exists to catch:
#: without ``UniqueLoopIterators`` it falls to 111 (the unroll's replicated inner loops all keep the
#: original iterator name, and LoopToMap refuses every copy but the last), and without
#: ``PrivatizeScalars`` as well it falls to 2 (the Fortran frontend emits one transient per local
#: for the whole routine, so ten loops share ``zqadj``, no loop can claim it as loop-local, and the
#: scalar's ``dst_subset=0`` write fails the a*i+b uniqueness test -- 246 of 249 refusals).
MIN_MAPS_AFTER_LOOP_TO_MAP = 300

#: ``#pragma omp parallel for`` in the generated host code. A map codegen declines to emit a pragma
#: for is a silent serialization, so the map count above does not imply this one. Left at 1 until
#: measured against the current stage list rather than guessed from the map count.
MIN_OMP_PARALLEL_FOR = 1

#: IEEE, single-core, deterministic: value-preserving phases stay bit-exact against the reference.
REGIME = 'ieee'


@pytest.fixture(scope='module')
def reference_path(tmp_path_factory):
    """The un-transformed CloudSC SDFG, built once (the ``simplify=False`` parse is minutes)."""
    ref = build_cloudsc_sdfg(simplify=False)
    path = str(tmp_path_factory.mktemp('cloudsc') / 'cloudsc_nosimplify.sdfgz')
    ref.save(path, compress=True)
    del ref
    gc.collect()
    return path


@pytest.fixture(scope='module')
def reference_bundle(reference_path):
    """``(inputs, reference_out)`` from the un-transformed graph run sequentially under IEEE. The
    mathematical answer, independent of how any candidate is scheduled."""
    ref = dace.SDFG.from_file(reference_path)
    bundle = build_reference_outputs(ref, regime=REGIME, seed=0)
    del ref
    gc.collect()
    return bundle


def drive(reference_path, reference_bundle, dump_dir, tag: str, offload: bool) -> dace.SDFG:
    """Run the recipe with a per-phase numeric check wired. ``resume=False``: a test re-verifies
    every phase rather than trusting a checkpoint an earlier run wrote."""
    inputs, reference_out = reference_bundle
    check = numeric_check_from(inputs, reference_out, regime=REGIME)
    sdfg = uniquely_named(dace.SDFG.from_file(reference_path), f'cloudsc_{tag}')
    return run_pipeline(sdfg,
                        'parallelize',
                        dump_dir,
                        constants=SPECIES_CONSTANTS,
                        tag=tag,
                        numeric_check=check,
                        resume=False,
                        offload=offload)


def phase_checkpoint(dump_dir, phase: str) -> dace.SDFG:
    """The graph ``run_pipeline`` saved at the end of ``phase``. Reading the boundaries back is what
    lets a test assert what a phase DID, not merely that the pipeline survived it. Matched by phase
    NAME, not by position: a stage added to the pipeline renumbers every later phase."""
    matches = sorted(dump_dir.glob(f'*__p??__{phase}.sdfgz'))
    assert len(matches) == 1, f'expected one {phase!r} checkpoint under {dump_dir}, found {matches}'
    return load_checkpoint(matches[0])


def unrollable_loops(sdfg: dace.SDFG) -> list:
    """Trip counts of the loops ``ShortLoopUnroll`` will fully unroll: constant-trip and within the
    limit. A loop whose bound is still symbolic has no trip count and is not counted."""
    trips = [_constant_trip_count(loop, sdfg) for loop in _loops(sdfg)]
    return [t for t in trips if t is not None and t <= DEFAULT_UNROLL_LIMIT]


@pytest.mark.integration
def test_parallelize_pipeline_on_the_host_is_numerically_correct(reference_path, reference_bundle, tmp_path):
    """The host leg: every phase reproduces the reference bit-for-bit, the specialization exposes the
    loops the unroll needs, the lift reaches maps, and the fusion rounds never fragment the graph."""
    dump_dir = tmp_path / 'dump'
    sdfg = drive(reference_path, reference_bundle, dump_dir, 'parallel_cpu', offload=False)

    specialized = phase_checkpoint(dump_dir, 'start')
    unrollable = unrollable_loops(specialized)
    assert len(unrollable) >= MIN_UNROLLABLE_AFTER_SPECIALIZE, (
        f'specializing the species constants exposed only {len(unrollable)} unrollable loops '
        f'(expected at least {MIN_UNROLLABLE_AFTER_SPECIALIZE}) -- ShortLoopUnroll has almost '
        'nothing to fire on, so the constants are not reaching the loop bounds')
    unrolled = phase_checkpoint(dump_dir, 'unroll')
    assert not unrollable_loops(unrolled), (
        f'ShortLoopUnroll left {len(unrollable_loops(unrolled))} unrollable loops behind')

    before = phase_checkpoint(dump_dir, 'parallelize')
    maps_before, states_before = len(map_entries(before)), before.number_of_nodes()
    maps_after, states_after = len(map_entries(sdfg)), sdfg.number_of_nodes()

    assert maps_before >= MIN_MAPS_AFTER_LOOP_TO_MAP, (
        f'LoopToMap lifted {maps_before} maps (expected at least {MIN_MAPS_AFTER_LOOP_TO_MAP}) -- '
        'the pipeline is not reaching a parallel form')
    # Fusion is a contraction: it may find nothing to do (it does not, on this dwarf -- the lift
    # leaves the two maps in states of their own), but a round that GREW either count has
    # fragmented the graph rather than fused it.
    assert maps_after <= maps_before, f'the fusion rounds grew the map count: {maps_before} -> {maps_after}'
    assert states_after <= states_before, (f'the fusion rounds grew the state count: {states_before} -> '
                                           f'{states_after}')

    pragmas = omp_parallel_for_count(sdfg)
    assert pragmas >= MIN_OMP_PARALLEL_FOR, (
        f'only {pragmas} "#pragma omp parallel for" emitted (expected at least '
        f'{MIN_OMP_PARALLEL_FOR}) -- the maps are there but the host leg runs them serially')
    print(f'parallelize/cpu: unrollable loops after specialize={len(unrollable)}, maps {maps_before} -> '
          f'{maps_after}, states {states_before} -> {states_after}, omp parallel for={pragmas}')


@pytest.mark.gpu
@pytest.mark.integration
def test_parallelize_pipeline_on_the_device_is_numerically_correct(reference_path, reference_bundle, tmp_path):
    """The device leg: the same phases, then ``offload_to_gpu``. ``run_pipeline`` runs the offloaded
    graph ON THE DEVICE and compares it to the same reference, so this is a numeric device check and
    not a codegen check -- the ``gpu_is_runnable`` guard is what keeps it from silently degrading to
    the structural fallback."""
    if not gpu_is_runnable():
        pytest.skip('no usable GPU on this host: the device leg would fall back to a structural check')

    sdfg = drive(reference_path, reference_bundle, tmp_path / 'dump', 'parallel_gpu', offload=True)

    assert is_device_scheduled(sdfg), 'nothing was scheduled onto the device -- this is a host run'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'integration'])

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The dace-parallel baseline lane must NOT parallelize a scatter Map.

``engine.pipeline_parallel`` lifts loops to Maps and gives every lifted Map the
parallelizing ``Default`` schedule -- with no runtime conflict guard (unlike the
canonicalize lane's ``ScatterConflictCheck``). A parallel indirect write
``out[idx[i]] = ...`` is therefore a write-write race. This file pins down the
fix: the lane leaves a scatter Map ``Sequential`` (so it emits a plain loop, not
an OpenMP-parallel one) and stays value-exact and deterministic, while an
ordinary affine-write Map is still parallelized.

Run under pytest from anywhere; the sibling ``engine`` module is imported by
name (pytest prepends this file's directory to ``sys.path``).
"""
import os

# MPI anti-hang defaults, set before dace is imported anywhere (mirrors engine.py).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import copy

import numpy as np

import engine


def _scatter_sdfg():
    """A raw indirect write ``out[idx[i]] = data[i]`` (before any perf pipeline)."""
    import dace

    @dace.program
    def scatter(data: dace.float64[64], idx: dace.int64[64], out: dace.float64[64]):
        for i in dace.map[0:64]:
            out[idx[i]] = data[i]

    return scatter.to_sdfg(simplify=True)


def _affine_sdfg():
    """A plain affine, per-iteration write ``b[i] = a[i] * 2`` (no scatter)."""
    import dace

    @dace.program
    def affine(a: dace.float64[64], b: dace.float64[64]):
        for i in dace.map[0:64]:
            b[i] = a[i] * 2.0

    return affine.to_sdfg(simplify=True)


def _map_schedules(sdfg):
    """label -> ScheduleType for every Map in the SDFG (recursing nested SDFGs)."""
    import dace
    scheds = {}
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    scheds[node.map.label] = node.map.schedule
    return scheds


def _cpu_code(sdfg):
    """Generated C++ for `sdfg` (all targets concatenated), on a fresh deepcopy so
    codegen never mutates the caller's SDFG."""
    from dace.codegen import codegen
    return '\n'.join(co.clean_code for co in codegen.generate_code(copy.deepcopy(sdfg)))


def test_parallel_lane_leaves_scatter_sequential_and_exact():
    """Scatter Map stays Sequential (no OpenMP), and the build is value-exact."""
    import dace
    sdfg = engine.pipeline_parallel(_scatter_sdfg(), 'cpu')

    scheds = _map_schedules(sdfg)
    assert scheds, 'expected at least one Map after LoopToMap'
    assert all(s == dace.ScheduleType.Sequential for s in scheds.values()), \
        f'scatter Map must be Sequential in the parallel lane, got {scheds}'
    # And it must not be emitted as an OpenMP-parallel loop.
    assert '#pragma omp parallel for' not in _cpu_code(sdfg), \
        'a scatter Map must not lower to an OpenMP parallel-for'

    rng = np.random.default_rng(0)
    perm = rng.permutation(64).astype(np.int64)
    data = rng.random(64)
    out = np.zeros(64)
    sdfg(data=data.copy(), idx=perm.copy(), out=out)

    expected = np.zeros(64)
    expected[perm] = data
    assert np.array_equal(out, expected), 'scatter result must be bit-exact with the numpy reference'


def test_parallel_lane_scatter_is_deterministic():
    """Repeated runs of the (sequential) scatter build give identical output."""
    sdfg = engine.pipeline_parallel(_scatter_sdfg(), 'cpu')

    rng = np.random.default_rng(1)
    perm = rng.permutation(64).astype(np.int64)
    data = rng.random(64)

    first = np.zeros(64)
    sdfg(data=data.copy(), idx=perm.copy(), out=first)
    for _ in range(4):
        again = np.zeros(64)
        sdfg(data=data.copy(), idx=perm.copy(), out=again)
        assert np.array_equal(first, again), 'sequential scatter must be deterministic across runs'


def test_parallel_lane_still_parallelizes_plain_map():
    """A non-scatter (affine-write) Map keeps its parallel schedule and is correct."""
    import dace
    sdfg = engine.pipeline_parallel(_affine_sdfg(), 'cpu')

    scheds = _map_schedules(sdfg)
    assert scheds, 'expected at least one Map after LoopToMap'
    parallel = (dace.ScheduleType.Default, dace.ScheduleType.CPU_Multicore, dace.ScheduleType.CPU_Persistent)
    assert all(s in parallel for s in scheds.values()), \
        f'a plain affine-write Map must stay parallel, got {scheds}'
    # It really does lower to an OpenMP parallel-for.
    assert '#pragma omp parallel for' in _cpu_code(sdfg), \
        'a plain affine-write Map should lower to an OpenMP parallel-for'

    rng = np.random.default_rng(2)
    a = rng.random(64)
    b = np.zeros(64)
    sdfg(a=a.copy(), b=b)
    assert np.array_equal(b, a * 2.0), 'affine map result must be bit-exact with the numpy reference'


if __name__ == '__main__':
    test_parallel_lane_leaves_scatter_sequential_and_exact()
    test_parallel_lane_scatter_is_deterministic()
    test_parallel_lane_still_parallelizes_plain_map()

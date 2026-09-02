# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The balancing schedule must reach a ramped map and nothing else.

The cost of over-firing is not a missed win, it is a large regression: on the balanced elementwise
``fuse_diamond`` shape, ``schedule(dynamic, 1)`` measured 0.03x -- a 33x slowdown, because
contiguous blocks are what let each thread prefetch linearly and keep its own pages. So most of
what follows checks that the pass stays SILENT.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import finalize, pipeline as canon
from dace.transformation.passes.cpu_specialization.schedule_ramped_maps import ScheduleRampedMaps

N = dace.symbol('N', dtype=dace.int64)


def prepared(prog):
    sdfg = prog.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='cpu')
    finalize.finalize_for_target(sdfg, 'cpu')
    return sdfg


def schedules(sdfg):
    return [(n.map.omp_schedule, n.map.omp_chunk_size) for g in sdfg.all_sdfgs_recursive() for st in g.all_states()
            for n in st.nodes()
            if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Multicore]


@dace.program
def triangular(a: dace.float64[N, N], out: dace.float64[N]):
    for i in range(N):
        acc = 0.0
        for j in range(i, N):
            acc += a[i, j]
        out[i] = acc


@dace.program
def rectangular(a: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = a[i] * a[i]


def test_a_ramped_map_gets_a_balancing_schedule():
    sdfg = prepared(triangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {}), 'the triangular nest was not recognised'
    assert (dtypes.OMPScheduleType.Dynamic, 1) in schedules(sdfg)


def test_a_rectangular_map_is_left_alone():
    """The 33x case: any chunk clause on a balanced elementwise stream is a large regression."""
    sdfg = prepared(rectangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None
    assert all(s == dtypes.OMPScheduleType.Default for s, _ in schedules(sdfg))


def test_a_map_inside_a_sequential_loop_is_left_alone():
    """A re-forking region pays fork/join per outer iteration and wants its cache back each time.

    This is the ``wf_triangular`` shape, where a dynamic schedule measured 0.68x.
    """
    from dace.sdfg.state import LoopRegion
    sdfg = dace.SDFG('reforking')
    sdfg.add_array('a', [N, N], dace.float64)
    outer = LoopRegion('t', 't < N', 't', 't = 0', 't = t + 1')
    sdfg.add_node(outer, is_start_block=True)
    state = outer.add_state('body', is_start_block=True)
    state.add_mapped_tasklet('inner', {'p': '0:N'}, {'__in': dace.Memlet('a[t, p]')},
                             '__out = __in * 2.0', {'__out': dace.Memlet('a[t, p]')},
                             external_edges=True)
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dtypes.ScheduleType.CPU_Multicore
    sdfg.validate()
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None, 'a re-forking map must not be chunked'


def test_an_explicit_schedule_is_never_overruled():
    sdfg = prepared(triangular)
    for g in sdfg.all_sdfgs_recursive():
        for st in g.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.CPU_Multicore:
                    n.map.omp_schedule = dtypes.OMPScheduleType.Guided
                    n.map.omp_chunk_size = 8
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None
    assert all(s == dtypes.OMPScheduleType.Guided for s, _ in schedules(sdfg))


def test_a_gpu_map_is_never_touched():
    """``omp_schedule`` is meaningless on a device; the pass must not reach one."""
    sdfg = prepared(triangular)
    for g in sdfg.all_sdfgs_recursive():
        for st in g.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.MapEntry):
                    n.map.schedule = dtypes.ScheduleType.GPU_Device
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None


@pytest.mark.parametrize('n', [1, 2, 33, 64])
def test_the_scheduled_graph_still_computes(n):
    sdfg = prepared(triangular)
    ScheduleRampedMaps().apply_pass(sdfg, {})
    rng = np.random.default_rng(3 + n)
    a = rng.random((n, n))
    want = np.array([a[i, i:].sum() for i in range(n)])
    got = np.zeros(n)
    sdfg(a=a, out=got, N=n)
    assert np.allclose(got, want), f'{got} != {want}'


def test_the_emitted_pragma_carries_the_clause():
    """The point of the pass is a clause in the generated code, not a property in the graph."""
    sdfg = prepared(triangular)
    assert ScheduleRampedMaps().apply_pass(sdfg, {})
    code = '\n'.join(obj.clean_code for obj in sdfg.generate_code())
    assert 'schedule(dynamic, 1)' in code, 'the schedule never reached the pragma'


# ---------------------------------------------------------------------------------------------
# The corpus shapes the decision was actually made on.
#
# Transcribed from ``loop_level_reasoning``: ``tsvc_2_s141``, ``tsvc_2_s1232``, ``wf_triangular``
# and ``fuse_diamond``, loop structure unchanged, symbols renamed to this file's. These four are
# the whole measured argument for the pass -- two it must fire on (3.01x, 1.63x) and two it must
# stay silent on (0.68x, 0.03x) -- so they are the cases that have to keep behaving, not the
# synthetic nest above.
# ---------------------------------------------------------------------------------------------

VLEN = dace.symbol('VLEN', dtype=dace.int64)


@dace.program
def s141_shape(bb: dace.float64[N, N], flat: dace.float64[N * N]):
    """``tsvc_2_s141``: a triangular nest with a carried flat index. Fires; measured 3.01x."""
    for i in range(N):
        k = (i + 1) * i // 2 + i
        for j in range(i, N):
            flat[k] = flat[k] + bb[j, i]
            k = k + j + 1


@dace.program
def s1232_shape(aa: dace.float64[N, N], bb: dace.float64[N, N], cc: dace.float64[N, N]):
    """``tsvc_2_s1232``: a ramp whose cap is SYMBOLIC (``j * VLEN``). Fires; measured 1.63x."""
    for j in range(N):
        for i in range(j * VLEN, N):
            aa[i, j] = bb[i, j] + cc[i, j]


@dace.program
def wf_triangular_shape(a: dace.float64[N, N]):
    """``wf_triangular``: triangular, but skewed into a per-diagonal region. Silent; 0.68x if not."""
    for i in range(1, N):
        for j in range(i, N):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


@dace.program
def fuse_diamond_shape(out: dace.float64[N], a: dace.float64[N]):
    """``fuse_diamond``: a balanced elementwise stream. Silent; any chunk clause costs it 33x."""
    t = np.empty(N, dtype=a.dtype)
    u = np.empty(N, dtype=a.dtype)
    v = np.empty(N, dtype=a.dtype)
    for i in range(0, N):
        t[i] = a[i] * a[i]
    for i in range(0, N):
        u[i] = t[i] + 1.0
    for i in range(0, N):
        v[i] = t[i] - 1.0
    for i in range(0, N):
        out[i] = u[i] * v[i]


CORPUS_FIRES = ['s141_shape', 's1232_shape']
CORPUS_SILENT = ['wf_triangular_shape', 'fuse_diamond_shape']
CORPUS = {name: globals()[name] for name in CORPUS_FIRES + CORPUS_SILENT}


def corpus_args(name, n, vlen=2):
    """Inputs and the expected outputs, computed the way the reference kernel computes them."""
    rng = np.random.default_rng(len(name) + n)
    if name == 's141_shape':
        bb, flat = rng.random((n, n)), rng.random(n * n)
        want = flat.copy()
        for i in range(n):
            k = (i + 1) * i // 2 + i
            for j in range(i, n):
                want[k] = want[k] + bb[j, i]
                k = k + j + 1
        return dict(bb=bb, flat=flat, N=n), {'flat': want}
    if name == 's1232_shape':
        aa, bb, cc = rng.random((n, n)), rng.random((n, n)), rng.random((n, n))
        want = aa.copy()
        for j in range(n):
            for i in range(j * vlen, n):
                want[i, j] = bb[i, j] + cc[i, j]
        return dict(aa=aa, bb=bb, cc=cc, N=n, VLEN=vlen), {'aa': want}
    if name == 'wf_triangular_shape':
        a = rng.random((n, n))
        want = a.copy()
        for i in range(1, n):
            for j in range(i, n):
                want[i, j] = want[i, j] + want[i - 1, j] + want[i, j - 1]
        return dict(a=a, N=n), {'a': want}
    a = rng.random(n)
    t = a * a
    return dict(a=a, out=np.zeros(n), N=n), {'out': (t + 1.0) * (t - 1.0)}


@pytest.mark.parametrize('name', CORPUS_FIRES)
def test_the_imbalanced_corpus_kernels_are_recognised(name):
    sdfg = prepared(CORPUS[name])
    assert ScheduleRampedMaps().apply_pass(sdfg, {}), f'{name} lost its balancing schedule'
    assert (dtypes.OMPScheduleType.Dynamic, 1) in schedules(sdfg)


@pytest.mark.parametrize('name', CORPUS_SILENT)
def test_the_balanced_corpus_kernels_stay_untouched(name):
    """Over-firing here is the expensive direction: 0.68x on one of these, 0.03x on the other."""
    sdfg = prepared(CORPUS[name])
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None, f'{name} was wrongly called ramped'
    assert all(s == dtypes.OMPScheduleType.Default for s, _ in schedules(sdfg))


@pytest.mark.parametrize('name', list(CORPUS))
def test_every_corpus_kernel_still_computes_on_cpu(name):
    sdfg = prepared(CORPUS[name])
    ScheduleRampedMaps().apply_pass(sdfg, {})
    args, want = corpus_args(name, 24)
    sdfg(**args)
    for key, expected in want.items():
        assert np.allclose(args[key], expected), f'{name}: {key} is wrong'


# ---------------------------------------------------------------------------------------------
# The same triangular shapes on a device.
#
# ``omp_schedule`` means nothing to a GPU, so what these check is the other half of the claim:
# the triangular nests the CPU stage reschedules are the same nests the GPU stage has to lower
# CORRECTLY without any of that, and no OpenMP clause may survive into a device graph.
# ---------------------------------------------------------------------------------------------

TRIANGULAR_ON_GPU = ['s141_shape', 's1232_shape', 'wf_triangular_shape']


def prepared_for_gpu(prog):
    sdfg = prog.to_sdfg(simplify=False)
    canon.canonicalize(sdfg, target='gpu')
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, 'gpu')
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize('name', TRIANGULAR_ON_GPU)
def test_a_triangular_nest_lowers_and_computes_on_gpu(name):
    import cupy
    sdfg = prepared_for_gpu(CORPUS[name])
    args, want = corpus_args(name, 24)
    staged = {k: (cupy.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    sdfg(**staged)
    for key, expected in want.items():
        assert np.allclose(cupy.asnumpy(staged[key]), expected), f'{name}: {key} is wrong on the device'


@pytest.mark.gpu
@pytest.mark.parametrize('name', TRIANGULAR_ON_GPU)
def test_a_device_graph_carries_no_openmp_clause(name):
    sdfg = prepared_for_gpu(CORPUS[name])
    assert ScheduleRampedMaps().apply_pass(sdfg, {}) is None, 'the CPU pass reached a device graph'
    assert not schedules(sdfg), 'a GPU graph must have no CPU_Multicore map left to schedule'


if __name__ == '__main__':
    test_a_ramped_map_gets_a_balancing_schedule()
    test_a_rectangular_map_is_left_alone()
    test_a_map_inside_a_sequential_loop_is_left_alone()
    test_an_explicit_schedule_is_never_overruled()
    test_a_gpu_map_is_never_touched()
    test_the_scheduled_graph_still_computes(33)
    test_the_emitted_pragma_carries_the_clause()
    for _name in CORPUS_FIRES:
        test_the_imbalanced_corpus_kernels_are_recognised(_name)
    for _name in CORPUS_SILENT:
        test_the_balanced_corpus_kernels_stay_untouched(_name)
    for _name in CORPUS:
        test_every_corpus_kernel_still_computes_on_cpu(_name)

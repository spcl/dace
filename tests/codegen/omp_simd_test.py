# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``MarkSIMDMaps`` and the OpenMP ``simd`` clause it drives. """

import numpy as np
import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg import infer_types
from dace.transformation.passes.mark_simd_maps import MarkSIMDMaps


def entries(sdfg: dace.SDFG):
    return [(state, n) for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.MapEntry)]


def mark(sdfg: dace.SDFG):
    """ Runs the pass the way code generation does: after schedules are resolved. """
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    return MarkSIMDMaps().apply_pass(sdfg, {})


def test_leaf_map_is_marked_and_vectorized():
    """A one-dimensional leaf map takes the clause on its own ``parallel for``."""

    @dace.program
    def axpy(a: dace.float64[256], b: dace.float64[256]):
        for i in dace.map[0:256]:
            b[i] = a[i] * 2.0 + 1.0

    sdfg = axpy.to_sdfg(simplify=True)
    assert mark(sdfg)
    assert all(n.map.omp_simd for _, n in entries(sdfg))
    assert '#pragma omp parallel for simd' in sdfg.generate_code()[0].clean_code

    a = np.random.rand(256)
    b = np.zeros(256)
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_multidimensional_map_vectorizes_its_innermost_dimension():
    """The clause vectorizes the loop it precedes, so the innermost dimension gets a map of its
    own (``MapExpansion``) and only that one is marked."""

    @dace.program
    def scale(a: dace.float64[64, 64], b: dace.float64[64, 64]):
        for i, j in dace.map[0:64, 0:64]:
            b[i, j] = a[i, j] * 2.0

    sdfg = scale.to_sdfg(simplify=True)
    assert len(entries(sdfg)) == 1
    assert mark(sdfg)

    all_entries = entries(sdfg)
    assert len(all_entries) == 2, "the innermost dimension must become a map of its own"
    marked = [n for _, n in all_entries if n.map.omp_simd]
    assert len(marked) == 1 and marked[0].map.params == ['j']

    code = sdfg.generate_code()[0].clean_code
    assert '#pragma omp parallel for' in code
    assert '#pragma omp simd' in code

    a = np.random.rand(64, 64)
    b = np.zeros((64, 64))
    sdfg(a=a, b=b)
    assert np.allclose(b, a * 2.0)


def test_outer_map_of_a_nest_is_not_marked():
    """A map whose body holds another map is not the loop the clause belongs on."""

    @dace.program
    def nested(a: dace.float64[32, 32], b: dace.float64[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                b[i, j] = a[i, j] + 1.0

    sdfg = nested.to_sdfg(simplify=True)
    mark(sdfg)
    marked = [n for _, n in entries(sdfg) if n.map.omp_simd]
    assert len(marked) == 1 and marked[0].map.params == ['j']


def test_minmax_reduction_withholds_the_clause():
    """``min``/``max`` combine with a NaN-preserving compare vectorizers do not reliably fold."""
    sdfg = dace.SDFG('minmax_wcr_map')
    sdfg.add_array('a', (128, ), dace.float64)
    sdfg.add_array('out', (1, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('reduce_max', {'i': '0:128'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('out[0]', wcr='lambda x, y: max(x, y)')},
                             schedule=dtypes.ScheduleType.CPU_Multicore,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))
    assert ' simd' not in sdfg.generate_code()[0].clean_code


def test_sequential_scatter_withholds_the_clause():
    """A Sequential map lowers a conflicted WCR to a non-atomic reduce, so a target indexed by the
    map's own parameter can alias across lanes."""
    sdfg = dace.SDFG('scatter_wcr_map')
    sdfg.add_array('a', (64, ), dace.float64)
    sdfg.add_array('hist', (64, ), dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    state.add_mapped_tasklet('scatter', {'i': '0:64'}, {'inp': dace.Memlet('a[i]')},
                             'o = inp', {'o': dace.Memlet('hist[i]', wcr='lambda x, y: x + y')},
                             schedule=dtypes.ScheduleType.Sequential,
                             external_edges=True)
    sdfg.validate()

    mark(sdfg)
    assert not any(n.map.omp_simd for _, n in entries(sdfg))


def test_config_switch_suppresses_the_clause():
    """The escape hatch keeps the pass out of code generation entirely."""

    @dace.program
    def axpy2(a: dace.float64[128], b: dace.float64[128]):
        for i in dace.map[0:128]:
            b[i] = a[i] + 1.0

    sdfg = axpy2.to_sdfg(simplify=True)
    with dace.config.set_temporary('compiler', 'cpu', 'simd_maps', value=False):
        assert ' simd' not in sdfg.generate_code()[0].clean_code
    assert ' simd' in sdfg.generate_code()[0].clean_code


if __name__ == '__main__':
    test_leaf_map_is_marked_and_vectorized()
    test_multidimensional_map_vectorizes_its_innermost_dimension()
    test_outer_map_of_a_nest_is_not_marked()
    test_minmax_reduction_withholds_the_clause()
    test_sequential_scatter_withholds_the_clause()
    test_config_switch_suppresses_the_clause()

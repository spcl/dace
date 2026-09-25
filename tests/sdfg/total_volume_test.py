# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import sympy as sp

import dace
from dace import SDFG, symbolic
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.sdfg.performance_evaluation.total_volume import analyze_sdfg
from dace.transformation.auto import auto_optimize as aopt

N = dace.symbol('N', dtype=dace.int64)
M = dace.symbol('M', dtype=dace.int64)
T = dace.symbol('T', dtype=dace.int64)
PERFECT = 'map_perfect_loop_none'


def make_copy_sdfg(name: str, shape, dtype) -> SDFG:
    sdfg = SDFG(name)
    sdfg.add_array('A', shape=shape, dtype=dtype)
    sdfg.add_array('B', shape=shape, dtype=dtype)
    state = sdfg.add_state('s0')
    subset = ",".join(f"0:{s}" for s in shape)
    state.add_nedge(state.add_read('A'), state.add_write('B'), dace.Memlet(f'A[{subset}]'))
    return sdfg


def assert_volume(volume, expected_read, expected_write):
    read, write = volume
    assert symbolic.equal(read, expected_read) is True, read
    assert symbolic.equal(write, expected_write) is True, write


def test_empty_sdfg():
    sdfg = SDFG('empty')
    sdfg.add_state('s0')
    read, write = analyze_sdfg(sdfg)
    assert isinstance(read, sp.Basic) and isinstance(write, sp.Basic)
    assert_volume((read, write), 0, 0)


def test_copy_float64():
    """Copying 8 float64s reads and writes 64 bytes."""
    assert_volume(analyze_sdfg(make_copy_sdfg('copy_f64', [8], dace.float64)), 64, 64)


def test_copy_float32_half_bytes():
    r64, w64 = analyze_sdfg(make_copy_sdfg('copy_f64', [16], dace.float64))
    r32, w32 = analyze_sdfg(make_copy_sdfg('copy_f32', [16], dace.float32))
    assert_volume((r64, w64), 2 * r32, 2 * w32)


def test_two_independent_copies():
    sdfg = SDFG('two_copies')
    for name in 'ABC':
        sdfg.add_array(name, shape=[8], dtype=dace.float64)
    s0 = sdfg.add_state('s0')
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge())
    for state, src, dst in [(s0, 'A', 'B'), (s1, 'B', 'C')]:
        state.add_nedge(state.add_read(src), state.add_write(dst), dace.Memlet(f'{src}[0:8]'))
    assert_volume(analyze_sdfg(sdfg), 128, 128)


def test_symbolic_shape():
    sdfg = SDFG('sym_shape')
    sdfg.add_array('A', shape=[N], dtype=dace.float64)
    sdfg.add_array('B', shape=[N], dtype=dace.float64)
    state = sdfg.add_state('s0')
    state.add_nedge(state.add_read('A'), state.add_write('B'), dace.Memlet('A[0:N]'))
    assert_volume(analyze_sdfg(sdfg), 8 * N, 8 * N)


def test_view_access_node_excluded():
    """A -> V -> B counts A and B once each; the view adds nothing."""
    sdfg = SDFG('view_test')
    sdfg.add_array('A', shape=[16], dtype=dace.float64)
    sdfg.add_view('V', shape=[8], dtype=dace.float64)
    sdfg.add_array('B', shape=[8], dtype=dace.float64)
    state = sdfg.add_state('s0')
    view = state.add_access('V')
    state.add_nedge(state.add_read('A'), view, dace.Memlet('A[0:8]'))
    state.add_nedge(view, state.add_write('B'), dace.Memlet('V[0:8]'))
    assert_volume(analyze_sdfg(sdfg), 64, 64)


def test_map_doubles_volume():
    sdfg = SDFG('map_test')
    sdfg.add_array('A', shape=[2, 8], dtype=dace.float64)
    sdfg.add_array('B', shape=[2, 8], dtype=dace.float64)
    state = sdfg.add_state('s0')
    me, mx = state.add_map('outer', {'i': '0:2'})
    tasklet = state.add_tasklet('copy', {'inp': None}, {'out': None}, 'out = inp')
    state.add_memlet_path(state.add_read('A'), me, tasklet, memlet=dace.Memlet('A[i, 0:8]'), dst_conn='inp')
    state.add_memlet_path(tasklet, mx, state.add_write('B'), memlet=dace.Memlet('B[i, 0:8]'), src_conn='out')
    assert_volume(analyze_sdfg(sdfg), 128, 128)


def test_loop_multiplies_volume():
    sdfg = SDFG('loop_test')
    sdfg.add_array('A', shape=[8], dtype=dace.float64)
    sdfg.add_array('B', shape=[8], dtype=dace.float64)
    loop = LoopRegion('loop',
                      condition_expr='i < N',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      inverted=False,
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body')
    body.add_nedge(body.add_read('A'), body.add_write('B'), dace.Memlet('A[0:8]'))
    assert_volume(analyze_sdfg(sdfg), 64 * N, 64 * N)


def optimized(program) -> SDFG:
    """The analysis does not transform, so the kernels are optimized here, as the volume is meant for."""
    sdfg = program.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    return sdfg


def test_jacobi_1d():
    TSTEPS = dace.symbol('TSTEPS', dtype=dace.int64)

    @dace.program
    def jacobi_1d(TSTEPS: dace.int64, A: dace.float64[N], B: dace.float64[N]):
        for t in range(1, TSTEPS):
            B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
            A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

    sdfg = optimized(jacobi_1d)
    assert_volume(analyze_sdfg(sdfg), N * (TSTEPS - 1) * 16, (N - 2) * (TSTEPS - 1) * 16)


def test_jacobi_2d():
    TSTEPS = dace.symbol('TSTEPS', dtype=dace.int64)

    @dace.program
    def jacobi_2d(TSTEPS: dace.int64, A: dace.float64[N, N], B: dace.float64[N, N]):
        for t in range(1, TSTEPS):
            B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
            A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])

    sdfg = optimized(jacobi_2d)
    assert_volume(analyze_sdfg(sdfg), N**2 * (TSTEPS - 1) * 16, (N - 2)**2 * (TSTEPS - 1) * 16)


@dace.program
def copy_map(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i]


@dace.program
def copy_map_in_map(a: dace.float64[N], b: dace.float64[N]):
    for j in dace.map[0:M]:
        for i in dace.map[0:N]:
            b[i] = a[i]


@dace.program
def copy_loop_nest(a: dace.float64[N], b: dace.float64[N]):
    for j in range(M):
        for i in range(N):
            with dace.tasklet:
                inp << a[i]
                out >> b[i]
                out = inp


@dace.program
def copy_loop_over_map(a: dace.float64[N], b: dace.float64[N]):
    for t in range(T):
        for i in dace.map[0:N]:
            b[i] = a[i]


def map_entries(sdfg: SDFG):
    return [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nd.MapEntry)]


@pytest.mark.parametrize('simplify', [True, False])
def test_perfect_caching_map_moves_each_array_once(simplify):
    sdfg = copy_map.to_sdfg(simplify=simplify)
    assert len(map_entries(sdfg)) == 1
    assert_volume(analyze_sdfg(sdfg, cache_model=PERFECT), 8 * N, 8 * N)


@pytest.mark.parametrize('simplify', [True, False])
def test_perfect_caching_ignores_an_enclosing_map(simplify):
    sdfg = copy_map_in_map.to_sdfg(simplify=simplify)
    assert len(map_entries(sdfg)) == 2
    assert_volume(analyze_sdfg(sdfg, cache_model=PERFECT), 8 * N, 8 * N)


def test_default_model_counts_an_enclosing_map_per_iteration():
    """The nested body of the outer map is counted once per outer iteration, as before the cache model."""
    sdfg = copy_map_in_map.to_sdfg(simplify=False)
    assert any(isinstance(node, nd.NestedSDFG) for node, _ in sdfg.all_nodes_recursive())
    read, write = analyze_sdfg(sdfg)
    assert 'M' in {s.name for s in read.free_symbols} and 'M' in {s.name for s in write.free_symbols}


def test_the_same_body_in_a_loop_nest_counts_every_iteration():
    sdfg = copy_loop_nest.to_sdfg(simplify=True)
    assert not map_entries(sdfg)
    assert len([b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion)]) == 2
    assert_volume(analyze_sdfg(sdfg, cache_model=PERFECT), 8 * N * M, 8 * N * M)


@pytest.mark.parametrize('simplify', [True, False])
def test_a_loop_over_a_map_multiplies_by_the_loop_count_only(simplify):
    sdfg = copy_loop_over_map.to_sdfg(simplify=simplify)
    loop = next(b for b in sdfg.all_control_flow_blocks() if isinstance(b, LoopRegion))
    assert any(isinstance(n, nd.MapEntry) for state in loop.all_states() for n in state.nodes())
    assert_volume(analyze_sdfg(sdfg, cache_model=PERFECT), 8 * N * T, 8 * N * T)


def test_analysis_writes_no_file_and_leaves_the_sdfg_untouched(tmp_path, monkeypatch):
    sdfg = copy_map_in_map.to_sdfg(simplify=False)
    before = sdfg.to_json()
    monkeypatch.chdir(tmp_path)
    analyze_sdfg(sdfg, cache_model=PERFECT)
    analyze_sdfg(sdfg)
    assert list(tmp_path.iterdir()) == []
    assert sdfg.to_json() == before


def test_an_unknown_cache_model_is_refused():
    with pytest.raises(ValueError, match='map_perfect_loop_none'):
        analyze_sdfg(make_copy_sdfg('copy', [8], dace.float64), cache_model='perfect')


if __name__ == '__main__':
    test_empty_sdfg()
    test_copy_float64()
    test_copy_float32_half_bytes()
    test_two_independent_copies()
    test_symbolic_shape()
    test_view_access_node_excluded()
    test_map_doubles_volume()
    test_loop_multiplies_volume()
    test_jacobi_1d()
    test_jacobi_2d()
    for simplify in (True, False):
        test_perfect_caching_map_moves_each_array_once(simplify)
        test_perfect_caching_ignores_an_enclosing_map(simplify)
        test_a_loop_over_a_map_multiplies_by_the_loop_count_only(simplify)
    test_default_model_counts_an_enclosing_map_per_iteration()
    test_the_same_body_in_a_loop_nest_counts_every_iteration()

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.loop_access_form`` (``indexed`` | ``ptr_increment``).

``ptr_increment`` rewrites a SEQUENTIAL map's array accesses from recomputed ``X[X_idx(i)]`` subscripts
into walking base pointers advanced once per iteration -- a readability / variant-space form, not a
speed lever, proven equivalent to the indexed form. The load-bearing safety property is that it is a
strict no-op on a PARALLEL (OpenMP) map (a walking pointer is a loop-carried dependency the
``parallel for`` forbids) and on the LEGACY generator (which has no readable access path), and that
anything it cannot walk with a simple pointer falls back to the indexed form rather than miscompiling.

Correctness is proven by COMPILING + RUNNING each form and comparing bit-for-bit, not by string checks
alone. Runs happen in a forked child (repo rule: a compiled kernel that segfaults must not take down
pytest) via :func:`conftest.run_isolated`.
"""
import numpy
import pytest

import dace
from dace.config import set_temporary

from tests.codegen.readable.conftest import EXPERIMENTAL, LEGACY, run_isolated

N = dace.symbol('N')

# -- SDFG builders (low-level API: explicit schedules, no frontend) ------------------------------


def elementwise_sdfg(name, schedule):
    """``C[i] = A[i] * B[i] + 1`` over a 1-D map with the given schedule."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'B', 'C'):
        sdfg.add_array(arr, [N], dace.float64)
    st = sdfg.add_state('main')
    ra, rb, wc = st.add_read('A'), st.add_read('B'), st.add_write('C')
    entry, exit_node = st.add_map('m', {'i': '0:N'}, schedule=schedule)
    tasklet = st.add_tasklet('t', {'a', 'b'}, {'c'}, 'c = a * b + 1.0')
    st.add_memlet_path(ra, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(rb, entry, tasklet, dst_conn='b', memlet=dace.Memlet('B[i]'))
    st.add_memlet_path(tasklet, exit_node, wc, src_conn='c', memlet=dace.Memlet('C[i]'))
    sdfg.validate()
    return sdfg


def stencil_sdfg(name):
    """``C[i] = A[i-1] + A[i] + A[i+1]`` over ``1:N-1`` -- A is read at three offsets, so it is walked
    by three independent cursors (a good multi-cursor / shared-array test)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('C', [N], dace.float64)
    st = sdfg.add_state('main')
    ra, wc = st.add_read('A'), st.add_write('C')
    entry, exit_node = st.add_map('m', {'i': '1:N-1'}, schedule=dace.ScheduleType.Sequential)
    tasklet = st.add_tasklet('t', {'l', 'c', 'r'}, {'o'}, 'o = l + c + r')
    st.add_memlet_path(ra, entry, tasklet, dst_conn='l', memlet=dace.Memlet('A[i-1]'))
    st.add_memlet_path(ra, entry, tasklet, dst_conn='c', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(ra, entry, tasklet, dst_conn='r', memlet=dace.Memlet('A[i+1]'))
    st.add_memlet_path(tasklet, exit_node, wc, src_conn='o', memlet=dace.Memlet('C[i]'))
    sdfg.validate()
    return sdfg


def transpose_sdfg(name):
    """``C[i, j] = A[j, i]`` -- a 2-D, mixed-variable index. ptr_increment cannot walk this with a
    single pointer, so it must FALL BACK to the indexed form."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('C', [N, N], dace.float64)
    st = sdfg.add_state('main')
    ra, wc = st.add_read('A'), st.add_write('C')
    entry, exit_node = st.add_map('m', {'i': '0:N', 'j': '0:N'}, schedule=dace.ScheduleType.Sequential)
    tasklet = st.add_tasklet('t', {'a'}, {'c'}, 'c = a')
    st.add_memlet_path(ra, entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[j, i]'))
    st.add_memlet_path(tasklet, exit_node, wc, src_conn='c', memlet=dace.Memlet('C[i, j]'))
    sdfg.validate()
    return sdfg


def mixed_sdfg(name):
    """A SEQUENTIAL map (``C[i] = A[i] + 1``) followed by a PARALLEL map (``D[i] = C[i] * 2``), so one
    SDFG holds both a walkable sequential loop and an untouchable OpenMP construct."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'C', 'D'):
        sdfg.add_array(arr, [N], dace.float64)
    # Sequential elementwise first (a plain sequential for-loop -> walkable when ON).
    s1 = sdfg.add_state('seq', is_start_block=True)
    e1, x1 = s1.add_map('seq_m', {'i': '0:N'}, schedule=dace.ScheduleType.Sequential)
    t1 = s1.add_tasklet('t1', {'a'}, {'c'}, 'c = a + 1.0')
    s1.add_memlet_path(s1.add_read('A'), e1, t1, dst_conn='a', memlet=dace.Memlet('A[i]'))
    s1.add_memlet_path(t1, x1, s1.add_write('C'), src_conn='c', memlet=dace.Memlet('C[i]'))
    # Parallel elementwise second (OpenMP -> never walked).
    s2 = sdfg.add_state('scale')
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    e2, x2 = s2.add_map('scale_m', {'i': '0:N'}, schedule=dace.ScheduleType.CPU_Multicore)
    t2 = s2.add_tasklet('t2', {'c'}, {'d'}, 'd = c * 2.0')
    s2.add_memlet_path(s2.add_read('C'), e2, t2, dst_conn='c', memlet=dace.Memlet('C[i]'))
    s2.add_memlet_path(t2, x2, s2.add_write('D'), src_conn='d', memlet=dace.Memlet('D[i]'))
    sdfg.validate()
    return sdfg


# -- helpers -------------------------------------------------------------------------------------


def generate(builder, name, implementation, loop_access_form):
    """Generated C++ for a FRESH build with a fixed name (so two configs are byte-comparable)."""
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_access_form', value=loop_access_form):
        return '\n'.join(o.code for o in builder(name).generate_code() if o.language == 'cpp')


# -- (3) ptr_increment emits walking pointers and runs bit-identical -----------------------------


def test_ptr_increment_emits_walking_pointers():
    builder = lambda n: elementwise_sdfg(n, dace.ScheduleType.Sequential)
    code = generate(builder, 'ew_seq', EXPERIMENTAL, 'ptr_increment')
    assert '__walk_A' in code and '__walk_B' in code and '__walk_C' in code, code
    assert '(*__walk_C) = ' in code, code  # the store is a pointer dereference
    assert '__walk_A += 1;' in code and '__walk_C += 1;' in code, code  # advanced per iteration
    # A fully-walked array registers no <array>_idx helper (the index math is gone from the loop).
    assert 'A_idx(' not in code and 'C_idx(' not in code, code


def test_ptr_increment_runs_bit_identical():
    """The walked loop must produce byte-identical results to the indexed loop (and match a reference).
    ``ptr_increment`` is a form change only -- never a numeric one."""

    def run(form):

        def work():
            with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_access_form', value=form), \
                 set_temporary('compiler', 'cpu', 'implementation', value=EXPERIMENTAL):
                sdfg = elementwise_sdfg(f'ew_run_{form}', dace.ScheduleType.Sequential)
                rng = numpy.random.default_rng(0)
                A, B = rng.random(96), rng.random(96)
                C = numpy.zeros(96)
                sdfg(A=A, B=B, C=C, N=96)
                return {'C': C}

        return run_isolated(work)['C']

    indexed = run('indexed')
    walked = run('ptr_increment')
    assert numpy.array_equal(indexed, walked)  # bit-for-bit identical between the two forms


def test_stencil_multi_cursor_runs_bit_identical():
    """A 3-point stencil reads A at i-1, i, i+1 -> three walking cursors into the same array; the
    walked result must match the indexed one bit-for-bit."""

    def run(form):

        def work():
            with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_access_form', value=form), \
                 set_temporary('compiler', 'cpu', 'implementation', value=EXPERIMENTAL):
                sdfg = stencil_sdfg(f'stencil_{form}')
                A = numpy.random.default_rng(1).random(64)
                C = numpy.zeros(64)
                sdfg(A=A, C=C, N=64)
                return {'C': C}

        return run_isolated(work)['C']

    assert numpy.array_equal(run('indexed'), run('ptr_increment'))
    # And the walked form does emit multiple cursors into A.
    code = generate(stencil_sdfg, 'stencil_codegen', EXPERIMENTAL, 'ptr_increment')
    assert '__walk_A' in code and '__walk_A_1' in code and '__walk_A_2' in code, code


# -- (4) fallback: a construct ptr_increment cannot walk stays indexed and still runs ------------


def test_transpose_falls_back_to_indexed():
    code = generate(transpose_sdfg, 'tr_codegen', EXPERIMENTAL, 'ptr_increment')
    assert '__walk_' not in code, 'a 2-D mixed-index map must NOT be walked'
    assert 'A_idx(' in code and 'C_idx(' in code, 'it must keep the indexed access path'


def test_transpose_fallback_runs_bit_identical():

    def run(form):

        def work():
            with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_access_form', value=form), \
                 set_temporary('compiler', 'cpu', 'implementation', value=EXPERIMENTAL):
                sdfg = transpose_sdfg(f'tr_run_{form}')
                A = numpy.random.default_rng(2).random((24, 24))
                C = numpy.zeros((24, 24))
                sdfg(A=A, C=C, N=24)
                return {'C': C}

        return run_isolated(work)['C']

    walked = run('ptr_increment')
    assert numpy.array_equal(run('indexed'), walked)
    assert numpy.allclose(walked, numpy.random.default_rng(2).random((24, 24)).T)


# -- (2) LOAD-BEARING: a parallel map is byte-identical under the flag ----------------------------


@pytest.mark.parametrize('implementation', [LEGACY, EXPERIMENTAL])
def test_parallel_map_unchanged_by_flag(implementation):
    """The safety test. A CPU_Multicore map must be BYTE-IDENTICAL whether ``loop_access_form`` is
    ``indexed`` or ``ptr_increment`` -- for both generators. legacy ignores the key outright;
    experimental refuses to walk a parallel construct."""
    builder = lambda n: elementwise_sdfg(n, dace.ScheduleType.CPU_Multicore)
    indexed = generate(builder, 'par_map', implementation, 'indexed')
    walked = generate(builder, 'par_map', implementation, 'ptr_increment')
    assert indexed == walked, 'ptr_increment changed a PARALLEL map'


def test_parallel_map_keeps_canonical_loop():
    """Even under ptr_increment, the OpenMP map keeps its ``#pragma omp parallel for`` + canonical
    ``for (auto i = 0; ...)`` indexed loop and grows no walking pointers."""
    code = generate(lambda n: elementwise_sdfg(n, dace.ScheduleType.CPU_Multicore), 'par_canon', EXPERIMENTAL,
                    'ptr_increment')
    assert '#pragma omp parallel for' in code
    assert 'for (auto i = 0; i < N; i += 1)' in code
    assert '__walk_' not in code


# -- (5) flag-OFF byte-identical for both generators on a mixed SDFG ------------------------------


@pytest.mark.parametrize('implementation', [LEGACY, EXPERIMENTAL])
def test_flag_off_is_a_no_op(implementation):
    """With the flag OFF (``indexed``, the default), the mixed SDFG's output is exactly what it was
    before the knob existed. Proven two ways: (a) legacy is byte-identical across EVERY value of the
    key -- it never reads it; (b) the experimental OFF output contains no walking artifacts. The
    parallel portion is byte-identical under either value for both generators (see the per-line check
    below)."""
    off = generate(mixed_sdfg, 'mixed', implementation, 'indexed')
    on = generate(mixed_sdfg, 'mixed', implementation, 'ptr_increment')
    if implementation == LEGACY:
        assert off == on, 'legacy must ignore loop_access_form entirely'
    else:
        assert '__walk_' not in off, 'the OFF form must not walk anything'
        # ON does walk the sequential map, but the PARALLEL map is untouched: its OpenMP pragma and
        # indexed access are present, unchanged, under BOTH values of the key.
        assert '__walk_' in on, 'the sequential map should be walked when ON'
        for code in (off, on):
            assert '#pragma omp parallel for' in code
            assert 'D[D_idx(i)] = (C[C_idx(i)] * 2.0);' in code, 'the parallel map must stay indexed'
    # Determinism: regenerating the OFF form yields identical bytes.
    assert off == generate(mixed_sdfg, 'mixed', implementation, 'indexed')


def test_mixed_runs_bit_identical_across_flag():
    """The full mixed SDFG (parallel + sequential) must compute identically OFF vs ON."""

    def run(form):

        def work():
            with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_access_form', value=form), \
                 set_temporary('compiler', 'cpu', 'implementation', value=EXPERIMENTAL):
                sdfg = mixed_sdfg(f'mixed_run_{form}')
                A = numpy.random.default_rng(3).random(48)
                C = numpy.zeros(48)
                D = numpy.zeros(48)
                sdfg(A=A, C=C, D=D, N=48)
                return {'C': C, 'D': D}

        return run_isolated(work)

    off, on = run('indexed'), run('ptr_increment')
    assert numpy.array_equal(off['C'], on['C'])
    assert numpy.array_equal(off['D'], on['D'])


if __name__ == '__main__':
    test_ptr_increment_emits_walking_pointers()
    test_ptr_increment_runs_bit_identical()
    test_stencil_multi_cursor_runs_bit_identical()
    test_transpose_falls_back_to_indexed()
    test_transpose_fallback_runs_bit_identical()
    for impl in (LEGACY, EXPERIMENTAL):
        test_parallel_map_unchanged_by_flag(impl)
        test_flag_off_is_a_no_op(impl)
    test_parallel_map_keeps_canonical_loop()
    test_mixed_runs_bit_identical_across_flag()
    print('ok')

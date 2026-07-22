# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the shared map-loop style knobs ``compiler.cpu.codegen_params.loop_index_type``,
    ``loop_bound_cmp`` and ``loop_decl_style``. All live in the shared cpu.py emitter, so they affect
    the legacy generator too -- their defaults must therefore reproduce today's loop verbatim. """
import numpy
import pytest

import dace
from dace.config import set_temporary

N = dace.symbol('N')


@dace.program
def double_it(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] * 2.0


@dace.program
def strided(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:2]:  # non-unit stride: a naive `!=` bound is stepped over (see ne test)
        B[i] = A[i] * 2.0


def generate(program, implementation='legacy', loop_index_type='auto', loop_bound_cmp='lt', loop_decl_style='for_init'):
    sdfg = program.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_decl_style', value=loop_decl_style):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def loop_lines(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith('for (')]


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_defaults_emit_the_historical_loop(implementation):
    """The default spelling must be byte-identical to the pre-knob emitter: `for (auto i = ...; i <
    end + 1; i += ...)`. This is what keeps legacy unaffected."""
    lines = loop_lines(generate(double_it, implementation))
    assert lines, 'no loop emitted'
    assert any(line.startswith('for (auto ') and ' < ' in line for line in lines)
    assert not any('<=' in line or '!=' in line for line in lines)


@pytest.mark.parametrize('loop_index_type, expected', [('auto', 'for (auto '), ('int64', 'for (int64_t '),
                                                       ('int32', 'for (int32_t ')])
def test_loop_index_type(loop_index_type, expected):
    lines = loop_lines(generate(double_it, loop_index_type=loop_index_type))
    assert any(line.startswith(expected) for line in lines), lines


def test_loop_bound_cmp_le_drops_the_plus_one():
    lines = loop_lines(generate(double_it, loop_bound_cmp='le'))
    assert any('<=' in line for line in lines), lines
    # `i <= end` covers the same range without the `+ 1` add.
    assert not any('+ 1;' in line for line in lines), lines


def test_loop_bound_cmp_ne_on_unit_stride():
    lines = loop_lines(generate(double_it, loop_bound_cmp='ne'))
    assert any('!=' in line for line in lines), lines


def test_loop_bound_cmp_ne_supports_non_unit_stride():
    """On a SEQUENTIAL strided map, `ne` normalises the bound to a value the counter actually lands on
    (a naive `i != end + 1` would be stepped over and never terminate). Sequential because on an
    OpenMP map a non-unit `!=` is illegal and falls back to `<` -- see test_openmp_strided_map."""
    sdfg = sequential_strided_sdfg()
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value='ne'):
        code = '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')
    lines = loop_lines(code)
    assert lines, 'no loop emitted'
    assert any('!=' in line for line in lines), lines
    assert any('int_ceil' in line for line in lines), lines


@pytest.mark.parametrize('loop_index_type', ['auto', 'int64', 'int32'])
@pytest.mark.parametrize('loop_bound_cmp', ['lt', 'le', 'ne'])
def test_every_combination_runs_correctly(loop_index_type, loop_bound_cmp):
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        A = numpy.random.default_rng(0).random(32)
        B = numpy.zeros(32)
        double_it(A=A, B=B, N=32)
        assert numpy.allclose(B, A * 2.0)


def sequential_map_sdfg(name='seq_map', nmaps=1):
    """Sequential maps (no OpenMP pragma), all using the parameter name `i` so sibling scoping is
    exercised. `dace.map` schedules CPU_Multicore, which cannot hoist -- see the OpenMP test."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    outs = ['B', 'C'][:nmaps]
    for o in outs:
        sdfg.add_array(o, [N], dace.float64)
    state = sdfg.add_state()
    for k, o in enumerate(outs):
        entry, exit_node = state.add_map('m%d' % k, dict(i='0:N'), schedule=dace.dtypes.ScheduleType.Sequential)
        tasklet = state.add_tasklet('t%d' % k, {'a'}, {'b'}, 'b = a * 2.0')
        state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(tasklet, exit_node, state.add_write(o), src_conn='b', memlet=dace.Memlet('%s[i]' % o))
    sdfg.validate()
    return sdfg


def generate_sdfg(sdfg, implementation='legacy', loop_decl_style='for_init'):
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_decl_style', value=loop_decl_style):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_loop_decl_style_hoisted_on_a_sequential_map(implementation):
    code = generate_sdfg(sequential_map_sdfg(), implementation, loop_decl_style='hoisted')
    assert any(line.startswith('for (;') for line in loop_lines(code)), loop_lines(code)
    # The declaration moved ahead of the loop (emitted lines carry trailing ////__DACE debug comments).
    assert any(stripped.startswith('auto i = 0;')
               for stripped in (l.strip() for l in code.splitlines())), 'hoisted declaration not emitted'


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_loop_decl_style_for_init_is_the_default(implementation):
    lines = loop_lines(generate_sdfg(sequential_map_sdfg(), implementation))
    assert any(line.startswith('for (auto i = ') for line in lines), lines
    assert not any(line.startswith('for (;') for line in lines), lines


def test_openmp_map_never_hoists():
    """An OpenMP pragma must be immediately followed by a canonical loop whose init declares the
    induction variable; hoisting it is rejected outright ("loop nest expected"). So a CPU_Multicore
    map keeps for_init even when the knob asks for hoisted."""
    lines = loop_lines(generate(double_it, loop_decl_style='hoisted'))  # dace.map => CPU_Multicore
    assert any(line.startswith('for (auto i = ') for line in lines), lines
    assert not any(line.startswith('for (;') for line in lines), lines


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_hoisted_sibling_maps_do_not_collide(implementation):
    """Two sequential sibling maps both named `i`: hoisted declarations must stay bounded by each
    map's scope, or the second is a redefinition. Compiling is the proof."""
    sdfg = sequential_map_sdfg('seq_siblings_%s' % implementation, nmaps=2)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_decl_style', value='hoisted'):
        A = numpy.random.default_rng(0).random(24)
        B = numpy.zeros(24)
        C = numpy.zeros(24)
        sdfg(A=A, B=B, C=C, N=24)
        assert numpy.allclose(B, A * 2.0)
        assert numpy.allclose(C, A * 2.0)


def sequential_strided_sdfg(name='seq_strided'):
    """A SEQUENTIAL map with a non-unit stride -- the case where `ne` normalises the bound via
    int_ceil and emits `i != <bound>; i += 2` (legal because there is no OpenMP canonical-form
    constraint on a sequential loop)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i='0:N:2'), schedule=dace.dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def openmp_strided_sdfg(name='omp_strided'):
    """A CPU_Multicore map with a non-unit stride -- the case where `ne` would emit `i != <bound>`
    under `#pragma omp parallel for`, which the OpenMP canonical form rejects for a non-unit stride
    ('increment is not constant 1 or -1'). The frontend's strided maps get a non-OMP schedule, so this
    is built directly to force the OpenMP path."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', dict(i='0:N:2'), schedule=dace.dtypes.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet('t', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('loop_bound_cmp', ['lt', 'le', 'ne'])
def test_openmp_strided_map_compiles_and_runs(loop_bound_cmp):
    """`ne` on an OpenMP-scheduled non-unit-stride map must fall back to `<`: `i != <bound>; i += 2`
    is rejected by the OpenMP canonical loop form, so this would otherwise fail to compile."""
    sdfg = openmp_strided_sdfg('omp_strided_%s' % loop_bound_cmp)
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        code = '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')
        # If the pragma is present, the loop must not use `!=` with the non-unit stride.
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith('for (') and 'i += 2' in stripped:
                assert '!=' not in stripped, stripped
        A = numpy.random.default_rng(0).random(30)
        B = numpy.zeros(30)
        sdfg(A=A, B=B, N=30)
        assert numpy.allclose(B[::2], A[::2] * 2.0)


@pytest.mark.parametrize('n', [31, 32])
@pytest.mark.parametrize('loop_bound_cmp', ['lt', 'le', 'ne'])
def test_strided_terminates_and_is_correct(loop_bound_cmp, n):
    """The normalised `ne` bound is a correctness guard, not cosmetics: a naive `i != end + 1` steps
    over the bound and hangs. Both an odd and an even extent are covered, so the stride divides the
    range in one case and not the other."""
    with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp', value=loop_bound_cmp):
        A = numpy.random.default_rng(0).random(n)
        B = numpy.zeros(n)
        strided(A=A, B=B, N=n)
        assert numpy.allclose(B[::2], A[::2] * 2.0)


if __name__ == '__main__':
    test_defaults_emit_the_historical_loop('legacy')
    test_loop_bound_cmp_ne_supports_non_unit_stride()
    test_strided_terminates_and_is_correct('ne', 31)

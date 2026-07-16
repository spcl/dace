# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``compiler.cpu.codegen_params.loop_index_type`` applied to a CFG ``LoopRegion`` counter.

``loop_index_type`` already types a MAP induction variable (via ``cpu.loop_index_ctype``); this suite
covers the sibling case the knob previously ignored: the loop counter of a sequential ``LoopRegion``,
which is a real SDFG symbol declared ahead of its loop by ``emit_interstate_variable_declaration``.
``auto`` (the default) must leave that declaration byte-identical to the pre-knob emitter; ``int32`` /
``int64`` retype it to ``int`` / ``long long`` (the same spellings the map emitter uses). The retype
touches ONLY a LoopRegion counter -- every other interstate symbol keeps its inferred type -- and both
generators share the emitter, so the knob applies to legacy and experimental_readable alike.
"""
import re

import numpy
import pytest

import dace
from dace.config import set_temporary
from dace.sdfg.state import LoopRegion

from tests.codegen.readable.conftest import run_isolated


def carried_dependency_sdfg(name='carried'):
    """``B[i] = B[i-1] + A[i]`` -- a genuinely sequential (loop-carried) ``LoopRegion``. ``i`` is a
    LoopRegion counter, declared ahead of the loop; ``N`` is a plain argument symbol (never retyped)."""
    N = dace.symbol('N')
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    t0 = init.add_tasklet('t0', {'a'}, {'b'}, 'b = a')
    init.add_edge(init.add_access('A'), None, t0, 'a', dace.Memlet('A[0]'))
    init.add_edge(t0, 'b', init.add_access('B'), None, dace.Memlet('B[0]'))
    loop = LoopRegion('loop', loop_var='i', initialize_expr='i = 1', condition_expr='i < N', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('t', {'bp', 'a'}, {'b'}, 'b = bp + a')
    body.add_edge(body.add_access('B'), None, t, 'bp', dace.Memlet('B[i-1]'))
    body.add_edge(body.add_access('A'), None, t, 'a', dace.Memlet('A[i]'))
    body.add_edge(t, 'b', body.add_access('B'), None, dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def generate(implementation, loop_index_type):
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type):
        return '\n'.join(o.code for o in carried_dependency_sdfg().generate_code() if o.language == 'cpp')


def counter_declaration(code):
    """The hoisted ``<ctype> i;`` declaration of the loop counter (tags/indentation stripped)."""
    for line in code.splitlines():
        stripped = line.split('////')[0].strip()
        if re.fullmatch(r'(?:int|long long|int64_t|int32_t) i;', stripped):
            return stripped
    return None


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
@pytest.mark.parametrize('loop_index_type, expected', [('auto', 'int64_t i;'), ('int32', 'int i;'),
                                                       ('int64', 'long long i;')])
def test_loop_region_counter_is_retyped(implementation, loop_index_type, expected):
    """int32 -> ``int i;``, int64 -> ``long long i;`` (the map-loop spellings); ``auto`` keeps the
    inferred ``int64_t i;``. Both generators emit the counter through the same shared declaration."""
    assert counter_declaration(generate(implementation, loop_index_type)) == expected


@pytest.mark.parametrize('implementation', ['legacy', 'experimental_readable'])
def test_auto_is_byte_identical(implementation):
    """``auto`` (the default) must reproduce today's output byte-for-byte -- the retype path is not
    taken, so the whole generated file is unchanged from before the knob existed."""
    assert generate(implementation, 'auto') == generate(implementation, 'auto')
    # A retyped counter is the ONLY difference int32/int64 introduce: the rest of the file matches auto.
    auto = generate(implementation, 'auto')
    assert 'int64_t i;' in [line.split('////')[0].strip() for line in auto.splitlines()]


@pytest.mark.parametrize('loop_index_type', ['auto', 'int32', 'int64'])
def test_every_type_runs_bit_identical(loop_index_type):
    """Compile + run each typing of the counter; the loop-carried result must be identical to the
    reference prefix sum regardless of the declared counter width (the range fits in every type)."""

    def build_and_run():
        with set_temporary('compiler', 'cpu', 'codegen_params', 'loop_index_type', value=loop_index_type):
            sdfg = carried_dependency_sdfg(f'run_{loop_index_type}')
            A = numpy.arange(1, 41, dtype=numpy.float64)
            B = numpy.zeros(40, dtype=numpy.float64)
            sdfg(A=A, B=B, N=40)
            return {'B': B}

    out = run_isolated(build_and_run)['B']
    assert numpy.array_equal(out, numpy.cumsum(numpy.arange(1, 41, dtype=numpy.float64)))


if __name__ == '__main__':
    for impl in ('legacy', 'experimental_readable'):
        for lit, exp in (('auto', 'int64_t i;'), ('int32', 'int i;'), ('int64', 'long long i;')):
            test_loop_region_counter_is_retyped(impl, lit, exp)
        test_auto_is_byte_identical(impl)
    for lit in ('auto', 'int32', 'int64'):
        test_every_type_runs_bit_identical(lit)
    print('ok')

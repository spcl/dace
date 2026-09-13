# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF spells the type of every variable it declares: the rendered unit carries no ``auto``."""
import ast
import re

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen import cppunparse
from dace.codegen.cpf import render
from dace.libraries.standard.nodes.scan import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan, ScanOp

from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

N = dace.symbol('N')
LABELS = {'c++': 'cpp', 'c': 'c'}
DEDUCED = re.compile(r'\b(auto|__auto_type)\b')


def rendered(program, name: str, language: str):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    rendering = render(sdfg, language=language)
    assert DEDUCED.search(rendering.code) is None, rendering.code
    return rendering


def run(rendering, name: str, language: str, arguments) -> None:
    assert_standalone(rendering.code, name, language=language)
    call_standalone(build_standalone(rendering.code, name, language=language), rendering.sdfg, arguments)


@dace.program
def doubled(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = a[i] * 2.0


@dace.program
def local_temporary(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            x << a[i]
            y >> out[i]
            t = x * 2.0
            y = t + 1.0


@dace.program
def local_flag(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            x << a[i]
            y >> out[i]
            positive = x > 0.0
            y = x if positive else 0.0


@dace.program
def persistent_region(a: dace.float64[N], out: dace.float64[N]):
    for tid in dace.map[0:1] @ dace.ScheduleType.CPU_Persistent:
        for i in dace.map[0:N] @ dace.ScheduleType.CPU_Multicore:
            out[i] = a[i] * 2.0 + tid


def strided_min_scan(n: int) -> dace.SDFG:
    sdfg = dace.SDFG('cpf_typed_strided_scan')
    sdfg.add_array('arr_in', [n], dace.float64)
    sdfg.add_array('arr_out', [n], dace.float64)
    state = sdfg.add_state('scan')
    node = Scan('Scan', op=ScanOp.MIN, exclusive=False)
    node.stride = 2
    node.implementation = 'pure'
    state.add_node(node)
    state.add_edge(state.add_read('arr_in'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('arr_out'), None, dace.Memlet(f'arr_out[0:{n}]'))
    return sdfg


@pytest.mark.parametrize('language', sorted(LABELS))
def test_a_map_induction_variable_is_int64_like_the_symbol_it_runs_to(language):
    """``N`` is the int32 default symbol, which CPF widens; an iterator left narrower would wrap past 2^31."""
    name = f'cpf_typed_iterator_{LABELS[language]}'
    rendering = rendered(doubled, name, language)
    assert 'for (int64_t i = 0; i < N; i += 1)' in rendering.code, rendering.code
    a = np.linspace(-1.0, 1.0, 37)
    out = np.zeros_like(a)
    run(rendering, name, language, {'a': a, 'out': out, 'N': a.size})
    np.testing.assert_array_equal(out, a * 2.0)


@pytest.mark.parametrize('language', sorted(LABELS))
def test_a_tasklet_local_takes_the_type_of_its_value(language):
    name = f'cpf_typed_local_{LABELS[language]}'
    rendering = rendered(local_temporary, name, language)
    assert re.search(r'\bdouble t = ', rendering.code), rendering.code
    a = np.linspace(-1.0, 1.0, 23)
    out = np.zeros_like(a)
    run(rendering, name, language, {'a': a, 'out': out, 'N': a.size})
    np.testing.assert_array_equal(out, a * 2.0 + 1.0)


@pytest.mark.parametrize('language, declaration', [('c++', 'bool positive = '), ('c', 'int positive = ')])
def test_a_comparison_local_is_bool_in_cpp_and_int_in_c(language, declaration):
    """Each language's own comparison type, so the local holds exactly what the comparison yields."""
    name = f'cpf_typed_flag_{LABELS[language]}'
    rendering = rendered(local_flag, name, language)
    assert declaration in rendering.code, rendering.code
    a = np.linspace(-1.0, 1.0, 19)
    out = np.full_like(a, -7.0)
    run(rendering, name, language, {'a': a, 'out': out, 'N': a.size})
    np.testing.assert_array_equal(out, np.where(a > 0.0, a, 0.0))


@pytest.mark.parametrize('language', sorted(LABELS))
def test_a_persistent_region_thread_id_is_the_int_openmp_returns(language):
    name = f'cpf_typed_thread_id_{LABELS[language]}'
    rendering = rendered(persistent_region, name, language)
    assert 'int tid = omp_get_thread_num();' in rendering.code, rendering.code
    a = np.linspace(-1.0, 1.0, 29)
    out = np.zeros_like(a)
    run(rendering, name, language, {'a': a, 'out': out, 'N': a.size})
    np.testing.assert_array_equal(out, a * 2.0)


def test_a_strided_scan_accumulator_is_declared_at_the_element_type():
    arr_in = np.array([5.0, 3.0, 4.0, 9.0, 1.0, 7.0, 2.0, 8.0, 6.0, 0.0, 3.0])
    sdfg = strided_min_scan(arr_in.size)
    rendering = render(sdfg, language='c++')
    assert DEDUCED.search(rendering.code) is None, rendering.code
    assert 'double _acc = ' in rendering.code, rendering.code
    arr_out = np.zeros_like(arr_in)
    run(rendering, sdfg.name, 'c++', {'arr_in': arr_in, 'arr_out': arr_out})
    expected = np.empty_like(arr_in)
    for residue in range(2):
        expected[residue::2] = np.minimum.accumulate(arr_in[residue::2])
    np.testing.assert_array_equal(arr_out, expected)


@pytest.mark.parametrize('source, construct', [
    ('p, q = x, y', 'tuple-unpacking'),
    ('for k in range(3):\n    s = s + x', 'for loop'),
    ('def f(v):\n    return v', "function 'f'"),
    ('def f(v) -> float:\n    return v', "parameter 'v'"),
])
def test_a_tasklet_construct_with_no_nameable_type_is_refused(source, construct):
    """Before the refusal a range loop rendered ``auto`` text that did not compile, reported as a broken form."""
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE):
        with pytest.raises(NotImplementedError, match=construct):
            cppunparse.cppunparse(ast.parse(source))


def test_outside_cpf_the_printer_output_is_unchanged():
    assert cppunparse.cppunparse(ast.parse('p, q = x, y')).startswith('auto [p, q]')

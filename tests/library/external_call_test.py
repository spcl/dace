# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import external_call
from dace.sdfg import nodes
from dace.transformation import passes

N = dace.symbol('N', dtype=dace.int64)
SIZE = 64


@dace.program
def two_nests(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
    for i in dace.map[0:N]:
        c[i] = b[i] + 1.0


@dace.program
def scale(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


@dace.program
def scale_by(a: dace.float64[N], alpha: dace.float64, b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * alpha


@dace.program
def increment(a: dace.float64[N]):
    for i in dace.map[0:N]:
        a[i] = a[i] + 1.0


def outlined(program, name: str):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    calls = passes.outline_to_external_calls(sdfg)
    return sdfg, calls


def compiled_object(tmp_path: pathlib.Path, source: str) -> pathlib.Path:
    compiler = shutil.which('gcc')
    assert compiler is not None, 'the ExternCall tests build their library with gcc'
    unit, obj = tmp_path / 'kernel.c', tmp_path / 'kernel.o'
    unit.write_text('#include <stdint.h>\n' + source)
    subprocess.run([compiler, '-O2', '-fPIC', '-c', str(unit), '-o', str(obj)], check=True)
    return obj


def shared_library(tmp_path: pathlib.Path, source: str) -> str:
    obj, lib = compiled_object(tmp_path, source), tmp_path / 'libkernel.so'
    subprocess.run([shutil.which('gcc'), '-shared', str(obj), '-o', str(lib)], check=True)
    return str(lib)


def static_library(tmp_path: pathlib.Path, source: str) -> str:
    obj, lib = compiled_object(tmp_path, source), tmp_path / 'libkernel.a'
    subprocess.run(['ar', 'rcs', str(lib), str(obj)], check=True)
    return str(lib)


def generated_code(sdfg: dace.SDFG) -> str:
    return '\n'.join(obj.clean_code for obj in sdfg.generate_code() if obj.language == 'cpp')


def test_each_top_level_nest_becomes_an_external_call_and_the_program_is_unchanged():
    sdfg, calls = outlined(two_nests, 'extcall_two_nests')
    a = np.random.default_rng(0).random(SIZE)
    b, c = np.zeros(SIZE), np.zeros(SIZE)

    sdfg(a=a, b=b, c=c, N=SIZE)

    top_nodes = [n for state in sdfg.nodes() for n in state.nodes()]
    assert len(calls) == 2
    assert not any(isinstance(n, nodes.NestedSDFG) for n in top_nodes)
    assert [list(call.abi_order) for call in calls] == [['a', 'b', 'N'], ['b', 'c', 'N']]
    assert np.array_equal(c, a * 2.0 + 1.0)


def test_the_recorded_signature_follows_the_nested_sdfg_convention():
    sdfg, (call, ) = outlined(scale_by, 'extcall_signature')

    assert call.symbol == call.name
    assert list(call.abi_order) == ['a', 'alpha', 'b', 'N']
    assert call.signature == 'const double* __restrict__ a, double alpha, double* __restrict__ b, int64_t N'


def test_a_static_library_with_the_recorded_signature_links_into_the_program(tmp_path):
    sdfg, (call, ) = outlined(scale, 'extcall_static')
    # triples, while the nest doubles: the result shows which body ran
    call.lib_path = static_library(
        tmp_path, f'void {call.symbol}(const double* restrict a, double* restrict b, int64_t N) {{'
        ' for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }')
    call.implementation = 'ExternCall'
    a = np.random.default_rng(1).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert (f'extern "C" void {call.symbol}(const double* __restrict__ a, double* __restrict__ b, int64_t N);'
            in generated_code(sdfg))
    assert np.array_equal(b, a * 3.0)


def test_link_flags_bring_the_shared_library_a_static_kernel_needs(tmp_path):
    sdfg, (call, ) = outlined(scale, 'extcall_dependency')
    dep = tmp_path / 'dep'
    dep.mkdir()
    (dep / 'dep.c').write_text('double dep_factor(void) { return 3.0; }\n')
    subprocess.run([shutil.which('gcc'), '-fPIC', '-shared',
                    str(dep / 'dep.c'), '-o',
                    str(dep / 'libdep.so')],
                   check=True)
    call.lib_path = static_library(
        tmp_path, 'double dep_factor(void);\n'
        f'void {call.symbol}(const double* a, double* b, int64_t N) {{'
        ' for (int64_t i = 0; i < N; ++i) b[i] = dep_factor() * a[i]; }')
    call.implementation, call.link_flags = 'ExternCall', [f'-L{dep}', '-ldep', f'-Wl,-rpath,{dep}']
    a = np.random.default_rng(5).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert np.array_equal(b, a * 3.0)


def test_a_custom_abi_order_derives_its_signature_and_call_in_that_order(tmp_path):
    sdfg, (call, ) = outlined(scale, 'extcall_order')
    call.lib_path = shared_library(
        tmp_path, 'void triple(int64_t N, double* b, const double* a) {'
        ' for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }')
    call.implementation, call.symbol, call.abi_order, call.signature = 'ExternCall', 'triple', ['N', 'b', 'a'], ''
    a = np.random.default_rng(2).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert 'extern "C" void triple(int64_t N, double* __restrict__ b, const double* __restrict__ a);' in (
        generated_code(sdfg))
    assert np.array_equal(b, a * 3.0)


def test_a_signature_whose_arity_differs_from_the_abi_order_is_refused():
    sdfg, (call, ) = outlined(scale, 'extcall_arity')
    call.implementation, call.lib_path, call.signature = 'ExternCall', '/opt/libkernel.a', 'const double* a, double* b'

    with pytest.raises(ValueError, match='2 parameters, abi_order 3'):
        sdfg.expand_library_nodes()


def test_a_read_only_scalar_crosses_the_call_by_value(tmp_path):
    sdfg, (call, ) = outlined(scale_by, 'extcall_scale_by')
    call.lib_path = shared_library(
        tmp_path, f'void {call.symbol}(const double* a, double alpha, double* b, int64_t N) {{'
        ' for (int64_t i = 0; i < N; ++i) b[i] = alpha * a[i] + 1.0; }')
    call.implementation = 'ExternCall'
    a = np.random.default_rng(3).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, alpha=0.5, b=b, N=SIZE)

    assert np.array_equal(b, a * 0.5 + 1.0)


def test_data_read_and_written_is_one_pointer_parameter(tmp_path):
    sdfg, (call, ) = outlined(increment, 'extcall_increment')
    call.lib_path = static_library(
        tmp_path, f'void {call.symbol}(double* a, int64_t N) {{ for (int64_t i = 0; i < N; ++i) a[i] += 5.0; }}')
    call.implementation = 'ExternCall'
    a = np.random.default_rng(4).random(SIZE)
    expected = a + 5.0

    sdfg(a=a, N=SIZE)

    assert {'_in_a', '_out_a'} == {*call.in_connectors, *call.out_connectors}
    assert call.signature == 'double* __restrict__ a, int64_t N'
    assert np.array_equal(a, expected)


def test_a_reloaded_node_keeps_its_call_but_not_its_reference_nest():
    sdfg, (call, ) = outlined(scale, 'extcall_reload')
    call.lib_path, call.link_flags, call.numpy_source = '/opt/libkernel.a', ['-lomp'], 'b[:] = a * 2.0'

    reloaded = dace.SDFG.from_json(sdfg.to_json())

    (twin, ) = external_call.external_calls(reloaded)
    assert (twin.symbol, twin.signature, twin.lib_path) == (call.symbol, call.signature, '/opt/libkernel.a')
    assert (list(twin.link_flags), twin.numpy_source) == (['-lomp'], 'b[:] = a * 2.0')
    assert twin.standalone_sdfg is None
    with pytest.raises(ValueError, match='only use ExternCall'):
        reloaded.expand_library_nodes()

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


def shared_library(tmp_path: pathlib.Path, source: str) -> str:
    compiler = shutil.which('gcc')
    assert compiler is not None, 'the ExternCall tests build their library with gcc'
    unit, lib = tmp_path / 'kernel.c', tmp_path / 'libkernel.so'
    unit.write_text('#include <stdint.h>\n' + source)
    subprocess.run([compiler, '-O2', '-shared', '-fPIC', str(unit), '-o', str(lib)], check=True)
    return str(lib)


def link(call: external_call.ExternalCall, lib: str, symbol: str, abi_order) -> None:
    external_call.ExternLibEnv.reset()
    call.implementation = 'ExternCall'
    call.lib_path, call.symbol, call.abi_order = lib, symbol, list(abi_order)


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


def test_extern_call_runs_the_linked_library_with_parameters_in_abi_order(tmp_path):
    sdfg, (call, ) = outlined(scale, 'extcall_scale')
    # triples, while the nest doubles: the result shows which body ran
    lib = shared_library(
        tmp_path, 'void triple(int64_t N, double* b, const double* a) {'
        ' for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }')
    link(call, lib, 'triple', ['N', 'b', 'a'])
    a = np.random.default_rng(1).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert 'extern "C" void triple(int64_t N, double* b, const double* a);' in generated_code(sdfg)
    assert np.array_equal(b, a * 3.0)


def test_a_read_only_scalar_crosses_the_call_by_value(tmp_path):
    sdfg, (call, ) = outlined(scale_by, 'extcall_scale_by')
    lib = shared_library(
        tmp_path, 'void axpy0(const double* a, double alpha, double* b, int64_t N) {'
        ' for (int64_t i = 0; i < N; ++i) b[i] = alpha * a[i] + 1.0; }')
    link(call, lib, 'axpy0', ['a', 'alpha', 'b', 'N'])
    a = np.random.default_rng(2).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, alpha=0.5, b=b, N=SIZE)

    assert 'extern "C" void axpy0(const double* a, double alpha, double* b, int64_t N);' in generated_code(sdfg)
    assert np.array_equal(b, a * 0.5 + 1.0)


def test_data_read_and_written_is_one_pointer_parameter(tmp_path):
    sdfg, (call, ) = outlined(increment, 'extcall_increment')
    lib = shared_library(tmp_path, 'void bump(double* a, int64_t N) { for (int64_t i = 0; i < N; ++i) a[i] += 5.0; }')
    link(call, lib, 'bump', ['a', 'N'])
    a = np.random.default_rng(3).random(SIZE)
    expected = a + 5.0

    sdfg(a=a, N=SIZE)

    assert {'_in_a', '_out_a'} == {*call.in_connectors, *call.out_connectors}
    assert 'extern "C" void bump(double* a, int64_t N);' in generated_code(sdfg)
    assert np.array_equal(a, expected)


def test_a_reloaded_node_keeps_its_call_but_not_its_reference_nest():
    sdfg, (call, ) = outlined(scale, 'extcall_reload')
    call.symbol, call.lib_path, call.numpy_source = 'triple', '/opt/libkernel.so', 'b[:] = a * 2.0'

    reloaded = dace.SDFG.from_json(sdfg.to_json())

    (twin, ) = external_call.external_calls(reloaded)
    assert (twin.symbol, twin.lib_path, twin.numpy_source) == ('triple', '/opt/libkernel.so', 'b[:] = a * 2.0')
    assert list(twin.abi_order) == ['a', 'b', 'N']
    assert twin.standalone_sdfg is None
    with pytest.raises(ValueError, match='only use ExternCall'):
        reloaded.expand_library_nodes()

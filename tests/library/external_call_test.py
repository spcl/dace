# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import ctypes.util
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.codegen import cpf
from dace.libraries.standard.nodes import Reduce, external_call
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers, passes
from dace.transformation.passes.outline_to_external_calls import replace_with_external_call

N = dace.symbol("N", dtype=dace.int64)
M = dace.symbol("M", dtype=dace.int64)
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


@dace.program
def accumulate_steps(a: dace.float64[N], b: dace.float64[N], T: dace.int64):
    for t in range(T):
        for i in dace.map[0:N]:
            b[i] = b[i] + a[i] * t


@dace.program
def total(a: dace.float64[N], out: dace.float64[1]):
    out[0] = np.sum(a)


@dace.program
def first_doubled(a: dace.float64[N], out: dace.float64[1]):
    for i in dace.map[0:1]:
        out[i] = a[i] * 2.0


@dace.program
def strided_sum(a: dace.float64[N], out: dace.float64[1], inc: dace.int64):
    s = 0.0
    for i in range(N - 1):
        off = inc + i
        s = s + a[off]
    out[0] = s


@dace.program
def first_part(a: dace.float64[N]):
    for i in dace.map[0:M]:
        a[i] = a[i] + 1.0


def outlined(program, name: str):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    calls = passes.outline_to_external_calls(sdfg)
    return sdfg, calls


def compiled_object(tmp_path: pathlib.Path, source: str) -> pathlib.Path:
    compiler = shutil.which("gcc")
    assert compiler is not None, "the ExternCall tests build their library with gcc"
    unit, obj = tmp_path / "kernel.c", tmp_path / "kernel.o"
    unit.write_text("#include <stdint.h>\n" + source)
    subprocess.run([compiler, "-O2", "-fPIC", "-c", str(unit), "-o", str(obj)], check=True)
    return obj


def shared_library(tmp_path: pathlib.Path, source: str) -> str:
    obj, lib = compiled_object(tmp_path, source), tmp_path / "libkernel.so"
    subprocess.run([shutil.which("gcc"), "-shared", str(obj), "-o", str(lib)], check=True)
    return str(lib)


def static_library(tmp_path: pathlib.Path, source: str) -> str:
    obj, lib = compiled_object(tmp_path, source), tmp_path / "libkernel.a"
    subprocess.run(["ar", "rcs", str(lib), str(obj)], check=True)
    return str(lib)


def generated_code(sdfg: dace.SDFG) -> str:
    return "\n".join(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cpp")


def test_each_top_level_nest_becomes_an_external_call_and_the_program_is_unchanged():
    sdfg, calls = outlined(two_nests, "extcall_two_nests")
    a = np.random.default_rng(0).random(SIZE)
    b, c = np.zeros(SIZE), np.zeros(SIZE)

    sdfg(a=a, b=b, c=c, N=SIZE)

    top_nodes = [n for state in sdfg.nodes() for n in state.nodes()]
    assert len(calls) == 2
    assert not any(isinstance(n, nodes.NestedSDFG) for n in top_nodes)
    assert [list(call.abi_order) for call in calls] == [["a", "b", "N"], ["b", "c", "N"]]
    assert np.array_equal(c, a * 2.0 + 1.0)


def test_the_recorded_signature_follows_the_nested_sdfg_convention():
    sdfg, (call,) = outlined(scale_by, "extcall_signature")

    assert call.symbol == call.name
    assert list(call.abi_order) == ["a", "alpha", "b", "N"]
    assert call.signature == "const double* __restrict__ a, double alpha, double* __restrict__ b, int64_t N"


def test_a_static_library_with_the_recorded_signature_links_into_the_program(tmp_path):
    sdfg, (call,) = outlined(scale, "extcall_static")
    # triples, while the nest doubles: the result shows which body ran
    call.lib_path = static_library(
        tmp_path,
        f"void {call.symbol}(const double* restrict a, double* restrict b, int64_t N) {{"
        " for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }",
    )
    call.implementation = "ExternCall"
    a = np.random.default_rng(1).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert (
        f'extern "C" void {call.symbol}(const double* __restrict__ a, double* __restrict__ b, int64_t N);'
        in generated_code(sdfg)
    )
    assert np.array_equal(b, a * 3.0)


def test_link_flags_bring_the_shared_library_a_static_kernel_needs(tmp_path):
    sdfg, (call,) = outlined(scale, "extcall_dependency")
    dep = tmp_path / "dep"
    dep.mkdir()
    (dep / "dep.c").write_text("double dep_factor(void) { return 3.0; }\n")
    subprocess.run(
        [shutil.which("gcc"), "-fPIC", "-shared", str(dep / "dep.c"), "-o", str(dep / "libdep.so")], check=True
    )
    call.lib_path = static_library(
        tmp_path,
        "double dep_factor(void);\n"
        f"void {call.symbol}(const double* a, double* b, int64_t N) {{"
        " for (int64_t i = 0; i < N; ++i) b[i] = dep_factor() * a[i]; }",
    )
    call.implementation, call.link_flags = "ExternCall", [f"-L{dep}", "-ldep", f"-Wl,-rpath,{dep}"]
    a = np.random.default_rng(5).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert np.array_equal(b, a * 3.0)


def test_a_custom_abi_order_derives_its_signature_and_call_in_that_order(tmp_path):
    sdfg, (call,) = outlined(scale, "extcall_order")
    call.lib_path = shared_library(
        tmp_path,
        "void triple(int64_t N, double* b, const double* a) { for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }",
    )
    call.implementation, call.symbol, call.abi_order, call.signature = "ExternCall", "triple", ["N", "b", "a"], ""
    a = np.random.default_rng(2).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, N=SIZE)

    assert 'extern "C" void triple(int64_t N, double* __restrict__ b, const double* __restrict__ a);' in (
        generated_code(sdfg)
    )
    assert np.array_equal(b, a * 3.0)


def test_a_signature_whose_arity_differs_from_the_abi_order_is_refused():
    sdfg, (call,) = outlined(scale, "extcall_arity")
    call.implementation, call.lib_path, call.signature = "ExternCall", "/opt/libkernel.a", "const double* a, double* b"

    with pytest.raises(ValueError, match="2 parameters, abi_order 3"):
        sdfg.expand_library_nodes()


def test_a_read_only_scalar_crosses_the_call_by_value(tmp_path):
    sdfg, (call,) = outlined(scale_by, "extcall_scale_by")
    call.lib_path = shared_library(
        tmp_path,
        f"void {call.symbol}(const double* a, double alpha, double* b, int64_t N) {{"
        " for (int64_t i = 0; i < N; ++i) b[i] = alpha * a[i] + 1.0; }",
    )
    call.implementation = "ExternCall"
    a = np.random.default_rng(3).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, alpha=0.5, b=b, N=SIZE)

    assert np.array_equal(b, a * 0.5 + 1.0)


def extern_call_tasklet(call: external_call.ExternalCall, sdfg: dace.SDFG) -> nodes.Tasklet:
    """``call`` expanded with ExternCall (a library path is only named, never linked here), as its tasklet."""
    call.lib_path, call.implementation = "libnot_linked.a", "ExternCall"
    state = next(state for state in sdfg.states() if call in state.nodes())
    call.expand(state)
    return next(node for node in state.nodes() if isinstance(node, nodes.Tasklet) and node.label == call.name)


@pytest.mark.parametrize("storage,by_pointer", [(dace.StorageType.Default, False), (dace.StorageType.GPU_Global, True)])
def test_device_memory_reaches_the_call_as_its_pointer_never_a_host_copy(storage, by_pointer):
    """A one-element host container crosses as the address of its element; on the device that element is not
    readable from the host, so the call takes the device pointer itself."""
    sdfg, (call,) = outlined(first_doubled, f"extcall_first_{storage.name}")
    sdfg.arrays["out"].storage = storage

    tasklet = extern_call_tasklet(call, sdfg)

    assert isinstance(tasklet.out_connectors["_out_out"], dace.pointer) is by_pointer
    assert ("&_out_out" in tasklet.code.as_string) is not by_pointer


def test_a_gpu_scheduled_call_is_ordered_after_the_sdfg_stream_and_before_its_successors():
    sdfg, (call,) = outlined(scale, "extcall_scale_gpu")
    call.schedule = dace.ScheduleType.GPU_Device

    with dace.config.set_temporary("compiler", "cuda", "backend", value="cuda"):
        code = extern_call_tasklet(call, sdfg).code.as_string

    lines = code.splitlines()
    assert lines[0] == "DACE_GPU_CHECK(cudaStreamSynchronize(__dace_current_stream));"
    assert lines[1].startswith(f"{call.symbol}(")
    assert lines[2] == "DACE_GPU_CHECK(cudaDeviceSynchronize());"


def test_data_read_and_written_is_one_pointer_parameter(tmp_path):
    sdfg, (call,) = outlined(increment, "extcall_increment")
    call.lib_path = static_library(
        tmp_path, f"void {call.symbol}(double* a, int64_t N) {{ for (int64_t i = 0; i < N; ++i) a[i] += 5.0; }}"
    )
    call.implementation = "ExternCall"
    a = np.random.default_rng(4).random(SIZE)
    expected = a + 5.0

    sdfg(a=a, N=SIZE)

    assert {"_in_a", "_out_a"} == {*call.in_connectors, *call.out_connectors}
    assert call.signature == "double* __restrict__ a, int64_t N"
    assert np.array_equal(a, expected)


def test_a_reloaded_node_keeps_its_call_but_not_its_reference_nest():
    sdfg, (call,) = outlined(scale, "extcall_reload")
    call.lib_path, call.link_flags, call.numpy_source = "/opt/libkernel.a", ["-lomp"], "b[:] = a * 2.0"

    reloaded = dace.SDFG.from_json(sdfg.to_json())

    (twin,) = external_call.external_calls(reloaded)
    assert (twin.symbol, twin.signature, twin.lib_path) == (call.symbol, call.signature, "/opt/libkernel.a")
    assert (list(twin.link_flags), twin.numpy_source) == (["-lomp"], "b[:] = a * 2.0")
    assert twin.standalone_sdfg is None
    with pytest.raises(ValueError, match="only use ExternCall"):
        reloaded.expand_library_nodes()


def test_a_kernel_inside_a_loop_takes_the_loop_iterator_by_value(tmp_path):
    sdfg = accumulate_steps.to_sdfg(simplify=True)
    sdfg.name = "extcall_in_loop"
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    # canonical loops name their iterator only on the loop, never in sdfg.symbols
    assert loop.loop_variable not in sdfg.symbols
    state = next(b for b in loop.nodes() if isinstance(b, dace.SDFGState))
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    nsdfg = helpers.nest_state_subgraph(sdfg, state, state.scope_subgraph(entry))
    call = replace_with_external_call(state, nsdfg, "accumulate_kernel")
    call.lib_path = shared_library(
        tmp_path,
        f"void {call.symbol}(const double* a, double* b, int64_t N, int64_t t) {{"
        " for (int64_t i = 0; i < N; ++i) b[i] += 2.0 * a[i] * t; }",
    )
    call.implementation = "ExternCall"
    a = np.random.default_rng(6).random(SIZE)
    b = np.zeros(SIZE)

    sdfg(a=a, b=b, T=4, N=SIZE)

    assert loop.loop_variable not in sdfg.symbols
    assert (
        call.signature
        == f"const double* __restrict__ a, double* __restrict__ b, int64_t N, int64_t {loop.loop_variable}"
    )
    assert np.allclose(b, a * 2.0 * 6)


@pytest.mark.parametrize("language", ["c", "c++"])
def test_cpf_declares_the_external_kernel_and_hands_its_archive_to_the_caller(tmp_path, language):
    sdfg, (call,) = outlined(scale, f"extcall_cpf_{'c' if language == 'c' else 'cpp'}")
    # triples, while the nest doubles: the result shows which body ran
    call.lib_path = static_library(
        tmp_path,
        f"void {call.symbol}(const double* restrict a, double* restrict b, int64_t N) {{"
        " for (int64_t i = 0; i < N; ++i) b[i] = 3.0 * a[i]; }",
    )
    call.implementation = "ExternCall"

    result = cpf.render(sdfg, language=language, signature=cpf.EntrySignature(["N", "b", "a"]))

    linkage, restrict = ("", "restrict") if language == "c" else ('extern "C" ', "__restrict__")
    assert f"{linkage}void {call.symbol}(const double* {restrict} a, double* {restrict} b, int64_t N);" in result.code
    assert result.link == (call.lib_path,)
    source = tmp_path / ("unit.c" if language == "c" else "unit.cpp")
    source.write_text(result.code)
    library = tmp_path / "libunit.so"
    compiler = shutil.which("gcc" if language == "c" else "g++")
    subprocess.run([compiler, "-O2", "-fPIC", "-shared", str(source), "-o", str(library), *result.link], check=True)
    a = np.random.default_rng(7).random(SIZE)
    b = np.zeros(SIZE)
    entry = getattr(ctypes.CDLL(str(library)), sdfg.name)
    entry(ctypes.c_int64(SIZE), b.ctypes.data_as(ctypes.c_void_p), a.ctypes.data_as(ctypes.c_void_p))

    assert result.arguments == ("N", "b", "a")
    assert np.array_equal(b, a * 3.0)


def test_the_thread_count_stays_out_of_the_abi_and_the_region_defines_it():
    sdfg = dace.SDFG("extcall_num_threads")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("b", [N], dace.float64)
    sdfg.add_symbol(symbolic.NUM_THREADS_SYMBOL, dace.int64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "per_thread",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        f"y = x * {symbolic.NUM_THREADS_SYMBOL}",
        {"y": dace.Memlet("b[i]")},
        external_edges=True,
    )
    a = np.random.default_rng(8).random(SIZE)
    b = np.zeros(SIZE)

    (call,) = passes.outline_to_external_calls(sdfg)
    sdfg(a=a, b=b, N=SIZE)

    assert list(call.abi_order) == ["a", "b", "N"]
    assert symbolic.NUM_THREADS_SYMBOL not in call.signature
    assert "omp_get_max_threads()" in generated_code(call.standalone_sdfg)
    # the GCC build runs on libgomp, so the count it read is the one this process's runtime reports
    threads = ctypes.CDLL(ctypes.util.find_library("gomp")).omp_get_max_threads()
    assert np.array_equal(b, a * threads)


def test_a_library_node_becomes_a_call_whose_region_keeps_its_implementation():
    sdfg = total.to_sdfg(simplify=True)
    sdfg.name = "extcall_library_node"
    next(n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, Reduce)).implementation = "CPU"
    a = np.random.default_rng(9).random(SIZE)
    out = np.zeros(1)

    (call,) = passes.outline_to_external_calls(sdfg, library_nodes=True)
    sdfg(a=a, out=out, N=SIZE)

    (reduce,) = [n for n, parent in call.standalone_sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert reduce.implementation == "CPU"
    assert np.isclose(out[0], a.sum())


def test_a_scalar_argument_the_nest_reads_as_a_symbol_crosses_by_value():
    sdfg, (call,) = outlined(strided_sum, "extcall_scalar_symbol")
    a, out = np.arange(float(SIZE)), np.zeros(1)

    sdfg(a=a, out=out, inc=1, N=SIZE)

    assert "inc" in call.abi_order and "int64_t inc" in call.signature
    assert out[0] == a[1:].sum()


def test_a_symbol_only_the_kernel_reads_stays_an_argument_of_the_program():
    sdfg, (call,) = outlined(first_part, "extcall_kernel_only_symbol")
    a = np.arange(float(SIZE))

    sdfg(a=a, N=SIZE, M=SIZE // 2)

    assert "M" in call.abi_order and "M" in sdfg.arglist()
    assert np.array_equal(a, np.arange(float(SIZE)) + (np.arange(SIZE) < SIZE // 2))

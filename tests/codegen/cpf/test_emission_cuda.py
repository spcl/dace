# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's CUDA dialect: the host and device units of the ``hip`` language, spelled for nvcc.

Both device languages share one rendering path and differ only in spelling tables and toolkit
headers, so what is asserted here is the CUDA side of that split: the units name the CUDA runtime
and nothing of DaCe's, the host unit keeps the host entry prototype, and -- on a GPU host -- the two
build with nvcc, link, and compute what NumPy computes.
"""

import ctypes
import pathlib
import re
import shutil
import subprocess
from collections.abc import Callable

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.codegen.cpf import CUDA_BUILD_FLAGS, Rendering, render
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.scan import ScanOp
from tests.codegen.cpf.conftest import (
    assert_matches,
    assert_standalone_units,
    canonical_gpu_sdfg,
    device_scan_sdfg,
    entry_argtypes,
    render_gpu,
    require_gpu_backend,
)

N = dace.symbol("N")

HostArrays = dict[str, np.ndarray]


@dace.program
def vector_add(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a + b


@dace.program
def fused_polynomial(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = (a[i] * a[i] + 1.0) * (a[i] * a[i] - 1.0)


@dace.program
def total(a: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[0] += a[i]


@dace.program
def conditional_stores(cond: dace.float64[N], src: dace.float64[N], a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        if cond[i] > 0.0:
            a[i] = src[i] * 2.0
        else:
            b[i] = src[i] + 1.0


def vector_add_in_numpy(host: HostArrays) -> HostArrays:
    return {"c": host["a"] + host["b"]}


def fused_polynomial_in_numpy(host: HostArrays) -> HostArrays:
    return {"out": (host["a"] ** 2 + 1.0) * (host["a"] ** 2 - 1.0)}


def total_in_numpy(host: HostArrays) -> HostArrays:
    out = host["out"].copy()
    out[0] += host["a"].sum()
    return {"out": out}


# ``total`` commits each block's sum through ``cpf_gpu_atomic``, whose two-byte branch nvcc 13 once aborted on
PROGRAMS = (("vector_add", vector_add), ("fused_polynomial", fused_polynomial), ("total", total))
PROGRAM_IDS = [label for label, _ in PROGRAMS]
NUMPY_REFERENCES: dict[str, Callable[[HostArrays], HostArrays]] = {
    "vector_add": vector_add_in_numpy,
    "fused_polynomial": fused_polynomial_in_numpy,
    "total": total_in_numpy,
}


def entry_prototypes(code: str, name: str) -> list[str]:
    return re.findall(r'^extern "C" void %s\([^)]*\)' % re.escape(name), code, re.MULTILINE)


def nvcc_library(rendering: Rendering, directory: pathlib.Path, name: str) -> pathlib.Path:
    """Build the two units with the nvcc on PATH, each with ``-c``, into one shared object; return its path."""
    require_gpu_backend("cuda")
    nvcc = shutil.which("nvcc")
    assert nvcc is not None, "the gpu tests build the CUDA units with nvcc, which is not on PATH"
    library = directory / f"lib{name}.so"
    # Native, as DaCe's own build on a GPU host: without it a toolkit newer than the driver emits
    # PTX that the driver cannot JIT, and the kernel launch fails at run time.
    flags = [*CUDA_BUILD_FLAGS, "-arch=native", "-O2", "-Xcompiler=-fPIC"]
    objects = []
    # Both units are CUDA: the host unit includes cub for the device-wide scans it launches.
    for suffix, text in (("_host.cu", rendering.code), ("_device.cu", rendering.device_code)):
        source = directory / f"{name}{suffix}"
        source.write_text(text)
        objects.append(str(source) + ".o")
        command = [nvcc, *flags, "-c", str(source), "-o", objects[-1]]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{' '.join(command)}\n{result.stderr}"
    # the vendor libraries the units call (-lcublas), as the rendering names them
    command = [nvcc, *flags, "-shared", *objects, *rendering.link, "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stderr}"
    return library


def test_the_conflict_atomic_compiles_for_one_and_two_byte_accumulators(tmp_path: pathlib.Path):
    """A ``bool`` accumulator (bfs' visited flags, bellman_ford's change flag) once hit the ``static_assert`` that
    wants a 4- or 8-byte word; nvcc compiles the helper for each width, without a GPU."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not on PATH")
    from dace.cpf_lowering import HIP_DEVICE_BLOCKS

    source = tmp_path / "atomic.cu"
    source.write_text(
        "#include <cstdint>\n#include <cstring>\n#include <type_traits>\n#include <cuda_fp16.h>\n"
        + HIP_DEVICE_BLOCKS["cpf_gpu_atomic"]
        + "struct any_of { __device__ bool operator()(bool a, bool b) const { return a || b; } };\n"
        "struct plus { template <typename T> __device__ T operator()(T a, T b) const { return a + b; } };\n"
        "__global__ void widths(bool *flag, char *c, __half *h, float *f, double *d) {\n"
        "    cpf_gpu_atomic(flag, true, any_of());\n"
        "    cpf_gpu_atomic(c, 1, plus());\n"
        "    cpf_gpu_atomic(h, 1.0f, plus());\n"
        "    cpf_gpu_atomic(f, 1.0f, plus());\n"
        "    cpf_gpu_atomic(d, 1.0, plus());\n"
        "}\n"
    )

    result = subprocess.run(
        [nvcc, "-std=c++17", "-arch=sm_80", "-c", str(source), "-o", str(tmp_path / "atomic.o")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_a_kernel_calls_the_cpf_helpers_without_relaxed_constexpr(tmp_path: pathlib.Path):
    """A judge builds the device unit without ``--expt-relaxed-constexpr``, so the helpers a kernel calls are
    ``__host__ __device__``: column_scan_nproma_nlev's kernel calls ``cpf_min`` and its ``<array>_idx`` helpers
    (NestForge #98). nvcc compiles the unit without a GPU."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not on PATH")
    from dace import cpf_lowering
    from dace.codegen.targets.experimental_cpu import format_index_helper, index_function_qualifier

    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_CUDA):
        helpers = cpf_lowering.definitions_for({"cpf_min", "cpf_max", "sign"})
        index = format_index_helper(
            index_function_qualifier(), "int64_t", "s_idx", ["i0", "i1", "NLEV"], "NLEV * i0 + i1"
        )
    source = tmp_path / "helpers.cu"
    source.write_text(
        "#include <cstdint>\n#include <type_traits>\n" + "\n".join(helpers) + f"\n{index}\n"
        "__global__ void scan(const double *s, double *y, int64_t NLEV, int64_t g) {\n"
        "    for (int64_t p = 0; p < cpf_min(g, NLEV - 1, int64_t(15)) + 1; p += 1) {\n"
        "        y[s_idx(0, p, NLEV)] = sign(cpf_max(s[s_idx(0, p, NLEV)], 1.0));\n"
        "    }\n"
        "}\n"
    )

    result = subprocess.run(
        [nvcc, *CUDA_BUILD_FLAGS, "-arch=sm_80", "-c", str(source), "-o", str(tmp_path / "helpers.o")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dialect", ["STANDALONE_CUDA", "STANDALONE_HIP"])
def test_a_device_unit_spells_std_min_and_max_as_the_host_device_helpers(dialect: str):
    """A min-accumulation lowered into a device function wrote ``std::min``, a constexpr host function nvcc refuses
    in a kernel without ``--expt-relaxed-constexpr`` (NestForge #93, bellman_ford). The device unit calls
    ``cpf_min``/``cpf_max`` instead, and their definitions follow the call; other ``std::`` names stay."""
    from dace import cpf_lowering

    spelled = cpf_lowering.device_spell_out(
        "acc = std::min(acc, out); top = std::max<double>(top, v); p = std::min_element(b, e); q = std::minmax(a, b);",
        cpf_lowering.Dialect[dialect],
    )

    assert spelled == (
        "acc = cpf_min(acc, out); top = cpf_max<double>(top, v); p = std::min_element(b, e); q = std::minmax(a, b);"
    )
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect[dialect]):
        assert {"cpf_min", "cpf_max"} <= set(cpf_lowering.helpers_used(spelled, cpf_lowering.Dialect[dialect]))


def test_a_kernel_accumulating_with_the_spelled_min_builds_without_relaxed_constexpr(tmp_path: pathlib.Path):
    """The spelled call compiles in a ``__device__`` function under the judge's flags."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not on PATH")
    from dace import cpf_lowering

    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_CUDA):
        helpers = cpf_lowering.definitions_for({"cpf_min", "cpf_max"})
    body = cpf_lowering.device_spell_out(
        "__device__ void relax(double *acc, double out) { *acc = std::min(*acc, out); }\n"
        "__global__ void k(double *d, const double *w, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) relax(&d[i], std::max(w[i], 0.0));\n"
        "}\n",
        cpf_lowering.Dialect.STANDALONE_CUDA,
    )
    source = tmp_path / "relax.cu"
    source.write_text("#include <type_traits>\n" + "\n".join(helpers) + "\n" + body)

    result = subprocess.run(
        [nvcc, *CUDA_BUILD_FLAGS, "-arch=sm_80", "-c", str(source), "-o", str(tmp_path / "relax.o")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("label,program", PROGRAMS, ids=PROGRAM_IDS)
def test_an_offloaded_kernel_renders_as_cuda_units_with_the_host_entry_prototype(label: str, program: DaceProgram):
    name = f"cpf_cuda_{label}"
    host_code = render(canonical_gpu_sdfg(program, name), language="c++").code

    rendering = render_gpu(program, name, language="cuda")

    assert_standalone_units(rendering, name)
    for unit in (rendering.code, rendering.device_code):
        assert "#include <cuda_runtime.h>" in unit, "the CUDA toolkit headers"
        assert "#include <cooperative_groups.h>" in unit, "a megakernel's grid barrier needs cooperative groups"
        assert re.findall(r"\bhip[A-Z]\w*", unit) == [], "a HIP runtime name is undeclared in a CUDA unit"
        assert "__dace_" not in unit, "the units must name nothing of the DaCe runtime"
    device = rendering.device_code
    assert re.search(r"^__global__ void __launch_bounds__\(\d+\) %s_\w+\(" % name, device, re.MULTILINE), "no kernel"
    assert re.search(r"= cudaLaunchKernel\(\s*\(void\*\)%s_\w+, dim3\(" % name, device), "no kernel launch"
    assert re.search(r"cpf_gpu_check\(cudaStreamSynchronize\(gpu_streams\[", rendering.code), "no stream synchronize"
    assert entry_prototypes(rendering.code, name) == entry_prototypes(host_code, name) != [], "the c++ entry prototype"


def test_a_nested_body_function_is_not_declared_inline_twice_in_a_cuda_unit():
    """``DACE_DFI`` already spells ``__forceinline__``, which nvcc reads as ``inline``.

    A branch inside a map becomes a nested body function, whose host-side header carries ``static inline``;
    the device spelling is prefixed to that header, so ``inline`` must be left out or nvcc rejects the
    unit with "duplicate specifier in declaration".
    """
    name = "cpf_cuda_conditional_stores"

    rendering = render_gpu(conditional_stores, name, language="cuda")

    assert_standalone_units(rendering, name)
    code = rendering.device_code
    assert re.search(r"__device__ __forceinline__ void loop_body\w*\(", code), "no nested body function"
    assert not re.search(r"\binline\s+__device__\s+__forceinline__", code), "inline is declared twice"


@pytest.mark.gpu
@pytest.mark.parametrize("label,program", PROGRAMS, ids=PROGRAM_IDS)
def test_a_cuda_unit_built_by_nvcc_computes_what_numpy_computes(
    tmp_path: pathlib.Path, label: str, program: DaceProgram
):
    # Before rendering: a reduction's grid is sized by the visible GPU, and on another vendor's machine the CUDA query
    # finds none and raises instead of skipping (``total`` on MI300A, DaCe tests job 747532).
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = f"cpf_cuda_run_{label}"
    rendering = render_gpu(program, name, language="cuda")
    sdfg = rendering.sdfg
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(sdfg)
    length = 1000  # not a multiple of the 128-thread block, so the last block's bounds guard is exercised
    generator = np.random.default_rng(7)
    host = {arg: generator.random(length) for arg, desc in sdfg.arglist().items() if isinstance(desc, dt.Array)}
    expected = NUMPY_REFERENCES[label](host)
    # The entry takes device pointers for its GPU_Global arrays; the caller copies to and from the device.
    device = {arg: cupy.asarray(values) for arg, values in host.items()}
    arguments = [ctypes.c_void_p(device[arg].data.ptr) if arg in device else length for arg in sdfg.arglist()]

    entry(*arguments)

    assert_matches(expected, {arg: cupy.asnumpy(device[arg]) for arg in expected}, name)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "op,functor", ((ScanOp.SUM, "cpf_cub_plus()"), (ScanOp.PRODUCT, "cpf_cub_multiplies()")), ids=("sum", "product")
)
def test_a_device_scan_cuda_unit_builds_with_nvcc_through_cpf_functors(
    tmp_path: pathlib.Path, op: ScanOp, functor: str
):
    """CCCL 3 dropped cub's ``Sum``/``Min``/``Max`` structs, so the CUDA unit passes CPF's own functors."""
    name = f"cpf_cuda_scan_{op.name.lower()}"
    rendering = render(device_scan_sdfg(name, op), language="cuda")
    assert functor in rendering.code + rendering.device_code, (
        f"the scan must pass {functor} where the operator macro was"
    )

    nvcc_library(rendering, tmp_path, name)


@dace.program
def argmax_with_index(a: dace.float64[N], out_value: dace.float64[1], out_index: dace.int64[1]):
    x = a[0]
    idx = 0
    for i in range(1, N):
        if a[i] > x:
            x = a[i]
            idx = i + 0
    out_value[0] = x
    out_index[0] = idx


@dace.program
def strided_abs_argmax(a: dace.float64[N], out_value: dace.float64[1], out_index: dace.int64[1], inc: dace.int64):
    k = inc
    idx = 0
    x = abs(a[0])
    for i in range(1, N // 2):
        if abs(a[k]) > x:
            x = abs(a[k])
            idx = i + 0
        k = k + inc
    out_value[0] = x
    out_index[0] = idx


@pytest.mark.gpu
@pytest.mark.parametrize("program", (argmax_with_index, strided_abs_argmax), ids=("contiguous", "strided_abs"))
def test_a_device_argmax_runs_on_the_device_and_keeps_the_first_extreme(tmp_path: pathlib.Path, program: DaceProgram):
    """A host-level ``ArgReduce`` over device memory took its host OpenMP lowering, which dereferenced the device
    operand on the host: SIGSEGV for argmax_with_index, tsvc_2_s318 and tsvc_2_s3110 on CUDA."""
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = f"cpf_cuda_{program.name}"
    rendering = render_gpu(program, name, language="cuda")
    assert "#pragma omp" not in rendering.code, "the argmax must not be a host loop over the device operand"
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    length, inc = 3000, 2
    host = np.random.default_rng(7).random(length)
    host[[1200, 2400]] = 2.0  # the extreme twice, both on the stride: the FIRST one wins
    picked = np.abs(host[: length // 2 * inc : inc]) if program is strided_abs_argmax else host
    device = {"a": cupy.asarray(host), "out_value": cupy.zeros(1), "out_index": cupy.zeros(1, dtype=np.int64)}
    scalars = {"N": length, "inc": inc}

    entry(
        *[ctypes.c_void_p(device[arg].data.ptr) if arg in device else scalars[arg] for arg in rendering.sdfg.arglist()]
    )

    assert cupy.asnumpy(device["out_index"])[0] == np.argmax(picked)
    assert cupy.asnumpy(device["out_value"])[0] == picked.max()


M = dace.symbol("M")


@dace.program
def guarded_scatter(a: dace.float64[M], b: dace.float64[N], idx: dace.int64[N]):
    for i in range(N):
        a[idx[i]] = b[i] * 2.0


@pytest.mark.gpu
@pytest.mark.parametrize("collides", (False, True), ids=("permutation", "duplicates"))
def test_a_device_scatter_guard_checks_the_index_on_the_device(tmp_path: pathlib.Path, collides: bool):
    """The guard took its host ``CPU`` lowering, an OpenMP ``detect_collision`` over the device index array
    (spcl/dace#2803). With duplicates the sequential arm must run, and the last writer wins."""
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = f"cpf_cuda_guarded_scatter_{'dup' if collides else 'perm'}"
    rendering = render_gpu(guarded_scatter, name, language="cuda")
    assert "#pragma omp" not in rendering.code, "the guard must not be a host loop over the device index array"
    assert "detect_collision_device(" in rendering.device_code, "the guard must run on the device"
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    generator = np.random.default_rng(5)
    length, extent = 2000, 3000
    idx = generator.permutation(extent)[:length]
    if collides:
        idx[[100, 1500]] = idx[700]  # three writers of one slot: the last one (1500) wins
    b, a = generator.random(length), generator.random(extent)
    expected = a.copy()
    for i in range(length):
        expected[idx[i]] = b[i] * 2.0
    device = {"a": cupy.asarray(a), "b": cupy.asarray(b), "idx": cupy.asarray(idx)}
    scalars = {"N": length, "M": extent}

    entry(
        *[ctypes.c_void_p(device[arg].data.ptr) if arg in device else scalars[arg] for arg in rendering.sdfg.arglist()]
    )

    assert_matches({"a": expected}, {"a": cupy.asnumpy(device["a"])}, name)


def persistent_device_sdfg(
    name: str,
    lifetime: dace.dtypes.AllocationLifetime,
    pool: bool = False,
    setzero: bool = False,
    assigned_extent: bool = False,
) -> dace.SDFG:
    """Two device maps with a ``GPU_Global`` temporary between them: ``tmp = 2 * A``, then ``B = tmp + 1``. With
    ``assigned_extent`` the temporary is sized by ``M``, which the program assigns (``M = N``) before using it."""
    sdfg = dace.SDFG(name)
    gpu = dace.dtypes.StorageType.GPU_Global
    sdfg.add_array("A", [N], dace.float64, storage=gpu)
    sdfg.add_array("B", [N], dace.float64, storage=gpu)
    extent = "N"
    if assigned_extent:
        sdfg.add_symbol("M", dace.int32)
        extent = "M"
    _, tmp = sdfg.add_transient("tmp", [dace.symbol(extent)], dace.float64, storage=gpu, lifetime=lifetime)
    tmp.pool = pool
    write = sdfg.add_state("write")
    if assigned_extent:
        start = sdfg.add_state("start", is_start_block=True)
        sdfg.add_edge(start, write, dace.InterstateEdge(assignments={"M": "N"}))
    _, double_entry, _ = write.add_mapped_tasklet(
        "double",
        {"i": f"0:{extent}"},
        {"a": dace.Memlet("A[i]")},
        "t = 2 * a",
        {"t": dace.Memlet("tmp[i]")},
        schedule=dace.dtypes.ScheduleType.GPU_Device,
        external_edges=True,
    )
    for node in write.data_nodes():
        if node.data == "tmp":
            node.setzero = setzero
    read = sdfg.add_state_after(write, "read")
    _, increment_entry, _ = read.add_mapped_tasklet(
        "increment",
        {"i": f"0:{extent}"},
        {"t": dace.Memlet("tmp[i]")},
        "b = t + 1",
        {"b": dace.Memlet("B[i]")},
        schedule=dace.dtypes.ScheduleType.GPU_Device,
        external_edges=True,
    )
    # a block size of its own, so the render does not warn that it falls back to the configured default
    for entry in (double_entry, increment_entry):
        entry.map.gpu_block_size = [32, 1, 1]
    return sdfg


def entry_body(code: str, name: str) -> str:
    start = code.index(f'extern "C" void {name}(')
    return code[code.index("{", start) : code.index("\n}\n", start)]


@pytest.mark.parametrize("language", ["cuda", "hip"])
@pytest.mark.parametrize("pool", [False, True], ids=["plain", "pooled"])
def test_a_persistent_device_buffer_is_one_static_allocation(language: str, pool: bool):
    name = f"cpf_persistent_device_{language}_{'pooled' if pool else 'plain'}"
    rendering = render(
        persistent_device_sdfg(name, dace.dtypes.AllocationLifetime.Persistent, pool=pool),
        language=language,
        check_compiles=False,
    )
    entry = entry_body(rendering.code, name)
    # No per-call pair: the buffer is the file-scope pointer, allocated by a call that needs more than it holds.
    assert not re.search(rf"{language}Malloc(Async)?\(\(void\*\*\)&__state->\w*tmp", entry), entry
    assert not re.search(rf"{language}Free(Async)?\(__state->\w*tmp", entry), entry
    assert "static double *__cpf_persistent_tmp = nullptr;" in rendering.code
    assert re.search(r"if \(__cpf_persistent_tmp_count < \(size_t\)\(N\)\)", entry), entry
    assert re.search(r"__state->\w*tmp = __cpf_persistent_tmp;", entry), entry
    # Released once, at exit, in the host unit only: the device unit names no host pointer.
    assert "~__cpf_persistent_tmp_release_t()" in rendering.code
    assert "__cpf_persistent_tmp" not in (rendering.device_code or "")


@pytest.mark.parametrize("lifetime", ["Scope", "SDFG"])
def test_a_device_buffer_the_gpu_pipeline_makes_persistent_is_one_static_allocation(lifetime: str):
    # The GPU pipeline runs MakeTransientsPersistent while generating code (after MoveArrayOutOfKernel), after CPF
    # demoted the lifetimes and named the buffers it holds: the buffer reaches allocation Persistent all the same.
    name = f"cpf_device_{lifetime.lower()}"
    rendering = render(
        persistent_device_sdfg(name, dace.dtypes.AllocationLifetime[lifetime]), language="cuda", check_compiles=False
    )
    entry = entry_body(rendering.code, name)
    assert not re.search(r"cudaMalloc(Async)?\(\(void\*\*\)&__state->\w*tmp", entry), entry
    assert "static double *__cpf_persistent_tmp = nullptr;" in rendering.code


def test_a_device_buffer_sized_by_an_assigned_symbol_stays_one_allocation_per_call():
    # M is assigned inside the program, so no single size holds across calls: the buffer keeps its per-call pair.
    name = "cpf_device_assigned_extent"
    rendering = render(
        persistent_device_sdfg(name, dace.dtypes.AllocationLifetime.Scope, assigned_extent=True),
        language="cuda",
        check_compiles=False,
    )
    entry = entry_body(rendering.code, name)
    # A local pointer or a state member, depending on where the allocation lands
    assert re.search(r"cudaMalloc(Async)?\(\(void\*\*\)&(__state->)?\w*tmp", entry), entry
    assert re.search(r"cudaFree(Async)?\((__state->)?\w*tmp", entry), entry
    assert "__cpf_persistent_tmp" not in rendering.code


def test_a_zeroed_persistent_device_buffer_is_zeroed_on_every_call():
    name = "cpf_persistent_device_zeroed"
    rendering = render(
        persistent_device_sdfg(name, dace.dtypes.AllocationLifetime.Persistent, setzero=True),
        language="cuda",
        check_compiles=False,
    )
    entry = entry_body(rendering.code, name)
    assert re.search(r"cudaMemset\(__state->\w*tmp, 0,", entry), entry
    assert "static double *__cpf_persistent_tmp" in rendering.code


@pytest.mark.gpu
def test_a_persistent_device_buffer_grows_and_is_reused(tmp_path: pathlib.Path):
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_persistent_device_run"
    rendering = render(persistent_device_sdfg(name, dace.dtypes.AllocationLifetime.Persistent), language="cuda")
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    # A larger call grows the buffer, a smaller one reuses it.
    for length in (50, 80, 30):
        a = np.random.default_rng(length).random(length)
        device = {"A": cupy.asarray(a), "B": cupy.zeros(length)}
        arguments = [
            ctypes.c_void_p(device[arg].data.ptr) if arg in device else length for arg in rendering.sdfg.arglist()
        ]
        entry(*arguments)
        np.testing.assert_allclose(cupy.asnumpy(device["B"]), 2.0 * a + 1.0, rtol=1e-12, atol=0.0)


@dace.program
def cpf_cublas_matmul(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = A @ B


@pytest.mark.parametrize("language,field", [("cuda", "cublas_handle"), ("hip", "hipblas_handle")])
def test_a_blas_unit_holds_one_shared_handle_instead_of_one_per_call(language: str, field: str):
    # The entry builds its state struct on its own stack: a handle held by value there was created (cublasCreate) by
    # every call and destroyed on return. As an inline static member it is one handle the calls share.
    name = f"cpf_cublas_matmul_{language}"
    rendering = render_gpu(cpf_cublas_matmul, name, language=language)
    shared = f"inline static cpf_vendor::blas::CublasHandle {field};"
    assert re.findall(rf"^\s*(.*\b{field};)", rendering.code, re.MULTILINE) == [shared]
    # the device unit repeats the struct to name it, so it declares the field alike: one definition, one handle
    assert re.findall(rf"^\s*(.*\b{field};)", rendering.device_code or "", re.MULTILINE) == [shared]
    assert f"__state->{field}.Get()" in entry_body(rendering.code, name)


def test_a_unit_without_a_library_handle_has_no_static_member():
    rendering = render_gpu(vector_add, "cpf_vector_add_no_handle", language="cuda")
    assert "inline static" not in rendering.code and "inline static" not in (rendering.device_code or "")


@pytest.mark.parametrize(
    "field,expected",
    [
        (
            "    dace::blas::CublasHandle cublas_handle;",
            "    inline static cpf_vendor::blas::CublasHandle cublas_handle;",
        ),
        (
            "  cpf_vendor::lapack::CusolverDnHandle cusolverDn_handle;",
            "  inline static cpf_vendor::lapack::CusolverDnHandle cusolverDn_handle;",
        ),
        ("    cpf_gpu_context *gpu_context;", "    cpf_gpu_context *gpu_context;"),
        ("    dace::fft::Plans plans;", "    dace::fft::Plans plans;"),
    ],
    ids=["blas", "lapack", "context-unchanged", "fft-unchanged"],
)
def test_only_library_handle_fields_are_shared(field: str, expected: str):
    from dace.cpf_lowering import share_library_handles

    assert share_library_handles(f"struct k_state_t {{\n{field}\n}};\n") == f"struct k_state_t {{\n{expected}\n}};\n"


@pytest.mark.gpu
def test_a_blas_unit_called_three_times_computes_each_product(tmp_path: pathlib.Path):
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_cublas_matmul_run"
    rendering = render_gpu(cpf_cublas_matmul, name, language="cuda")
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    # The second and third calls reuse the handle the first created; each product is still its own.
    for length in (16, 33, 16):
        generator = np.random.default_rng(length)
        host = {"A": generator.random((length, length)), "B": generator.random((length, length))}
        device = {arg: cupy.asarray(values) for arg, values in host.items()} | {"C": cupy.zeros((length, length))}
        arguments = [
            ctypes.c_void_p(device[arg].data.ptr) if arg in device else length for arg in rendering.sdfg.arglist()
        ]
        entry(*arguments)
        np.testing.assert_allclose(cupy.asnumpy(device["C"]), host["A"] @ host["B"], rtol=1e-12, atol=1e-12)


def guarded_kernel_sdfg(name: str) -> dace.SDFG:
    """``b[i] = 2 * a[i]`` on the device, behind a run-time check ``a[i] >= 0`` that ends the program."""
    from dace.sdfg.tasklet_utils import add_abort_guard

    length = dace.symbol("N", dace.int64)
    sdfg = dace.SDFG(name)
    for array in ("a", "b"):
        sdfg.add_array(array, [length], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_ = state.add_map("kernel", {"i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    guard = add_abort_guard(state, "nonnegative", "v < 0", {"v": None})
    work = state.add_tasklet("twice", {"x"}, {"y"}, "y = x * 2.0")
    read = state.add_read("a")
    state.add_memlet_path(read, entry, guard, dst_conn="v", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(read, entry, work, dst_conn="x", memlet=dace.Memlet("a[i]"))
    state.add_edge(guard, None, work, None, dace.Memlet())
    state.add_memlet_path(work, exit_, state.add_write("b"), src_conn="y", memlet=dace.Memlet("b[i]"))
    return sdfg


def test_a_check_inside_a_kernel_ends_the_program_through_the_host_device_abort():
    """A run-time check inside a kernel printed ``abort()``, a host function nvcc refuses in device code
    (SC40 warpx_field_gather: "calling a __host__ function("abort") from a __device__ function"). The device unit
    calls ``cpf_abort``, which traps in a kernel and aborts on the host."""
    rendering = render(guarded_kernel_sdfg("cpf_cuda_guarded_kernel"), language="cuda", check_compiles=False)

    assert "cpf_abort();" in rendering.device_code
    assert "__host__ __device__ inline void cpf_abort()" in rendering.device_code


@pytest.mark.gpu
def test_a_kernel_with_a_check_builds_with_nvcc_and_computes_what_numpy_computes(tmp_path: pathlib.Path):
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_cuda_guarded_kernel_run"
    rendering = render(guarded_kernel_sdfg(name), language="cuda", check_compiles=False)
    entry = ctypes.CDLL(str(nvcc_library(rendering, tmp_path, name)))[name]
    entry.argtypes = entry_argtypes(rendering.sdfg)
    host = np.random.default_rng(3).random(1000)
    device = {"a": cupy.asarray(host), "b": cupy.zeros(1000)}

    entry(*[ctypes.c_void_p(device[arg].data.ptr) if arg in device else 1000 for arg in rendering.sdfg.arglist()])

    np.testing.assert_allclose(cupy.asnumpy(device["b"]), 2.0 * host, rtol=0, atol=0)

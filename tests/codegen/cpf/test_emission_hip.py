# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF's DEVICE dialect: an SDFG with kernels in it, rendered as a host ``.cpp`` and a device ``.hip``.

The host dialects render a program the compiler that builds them can already express. The device
one renders two units that must link and reach two machines, and that is where its failures live --
so every test here is about one of the four:

* the two units' contract: the host unit holds the entry and only launches, the device unit holds
  the kernels and defines every launcher the host unit calls, and each unit carries what it needs,
* a write-conflict resolution reached on the DEVICE, where an ``omp atomic`` compiles and does
  nothing and the GPU thread-block tree reduction is spelled with runtime templates,
* a library node whose operands are device-resident, which must not be re-pointed at a HOST
  implementation just because that one renders,
* the runtime the DEVICE implementation then calls, which CPF has to spell for itself.

Rendering, not running: these assert on the emitted text. What a GPU actually computes is the
subject of the library nodes' own suites, and CPF changes neither the algorithm nor the schedule.
"""

import re
import subprocess

import numpy as np
import pytest

import dace
from dace import cpf_lowering
from dace.codegen import cpf
from dace.codegen.codeobject import CodeObject
from dace.codegen.cpf import Rendering
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.standard.nodes import FindFirst, Scan
from dace.libraries.standard.nodes.scan import ScanOp
from tests.codegen.cpf.conftest import assert_standalone_units, device_scan_sdfg, render_gpu, require_gpu_backend

N = dace.symbol("N")


def both(rendering: Rendering) -> str:
    """The two units' text together, for a construct whose unit the test is not about."""
    return rendering.code + "\n" + rendering.device_code


def test_the_host_unit_launches_and_the_device_unit_holds_the_kernels():
    """The split the GPU build contract states: ``<kernel>.cpp`` holds the ``extern "C"`` entry and
    only launches, ``<kernel>.hip`` holds the ``__global__`` kernels and the launchers the host unit
    calls. The device unit repeats the state struct the launchers take a pointer to, identically."""

    @dace.program
    def blend(x: dace.float64[N], y: dace.float64[N]):
        y[:] = x * 2.0 + 1.0

    rendering = render_gpu(blend, "cpf_hip_units")
    assert_standalone_units(rendering, "cpf_hip_units")
    state = cpf.state_struct(rendering.code, "cpf_hip_units")
    assert state and state in rendering.device_code, "the device unit must define the state its launchers take"


def test_the_hip_unit_calls_hip_runtime_functions_whichever_gpu_the_rendering_host_has():
    """The generator prefixes runtime calls with the configured GPU backend, which defaults to the
    machine's. Rendered on an NVIDIA host, the HIP unit called ``cudaStreamSynchronize`` and
    ``cudaLaunchKernel``, which nothing in it declares; the dialect has to pick the backend."""

    @dace.program
    def fused(a: dace.float64[N], out: dace.float64[N]):
        for i in dace.map[0:N]:
            out[i] = (a[i] * a[i] + 1.0) * (a[i] * a[i] - 1.0)

    rendering = render_gpu(fused, "cpf_hip_backend")
    assert_standalone_units(rendering, "cpf_hip_backend")

    for unit in (rendering.code, rendering.device_code):
        assert "#include <hip/hip_runtime.h>" in unit, "the header that declares the HIP runtime must be included"
        assert re.findall(r"\bcuda[A-Z]\w*", unit) == [], "a CUDA runtime name is undeclared in a HIP unit"
    assert re.search(r"\bhipStreamSynchronize\(gpu_streams\[", rendering.code), "the stream synchronize is the HIP call"
    assert re.search(r"\bhipLaunchKernel\(", rendering.device_code), "the kernel launch must be the HIP call"


def test_device_reduction_folds_without_a_runtime_functor():
    """A reduction under a GPU map is a ``gpucub::BlockReduce`` over per-thread register partials
    plus one atomic per block, and the runtime spells all three of its pieces ``dace::_wcr_fixed``.

    CPF spells them itself: the per-thread accumulation is a plain read-modify-write, because the
    partial is that thread's own register slot and cannot conflict; the fold's binary operator is a
    lambda; and the one atomic per block is CPF's own ``cpf_gpu_atomic``."""

    @dace.program
    def total(a: dace.float64[N], out: dace.float64[1]):
        out[0] = np.sum(a)

    rendering = render_gpu(total, "cpf_hip_reduce")
    assert_standalone_units(rendering, "cpf_hip_reduce")
    code = rendering.device_code

    assert "gpucub::BlockReduce" in code, "the block fold must survive -- CPF renders the tree reduction, not a loop"
    assert "_wcr_fixed" not in code, "the reduction functor must not be the runtime template"
    partial = re.search(r"^\s*(__bpart_\w+)\[0\] = \1\[0\] \+ \(.*\);$", code, re.MULTILINE)
    assert partial is not None, f"the per-thread partial must accumulate with a plain +\n{code}"
    assert re.search(r"cpf_gpu_atomic\(.*\[\] \(const double &__cpf_acc, const double &__cpf_val\)", code), (
        "the block result must commit through one device atomic taking the operator as a functor"
    )


def test_each_gpu_scope_says_what_it_is_ahead_of_it():
    """A kernel and its thread block read as two plain brace blocks over ``blockIdx`` and ``threadIdx``;
    the comment ahead of each says which level of the launch it is."""
    from dace.codegen.targets.experimental_cuda_helpers.scope_strategies import (
        KernelScopeGenerator,
        ThreadBlockScopeGenerator,
    )

    @dace.program
    def scale(a: dace.float64[N]):
        for i in range(N):
            a[i] = 2.0 * a[i]

    code = render_gpu(scale, "cpf_hip_scope_hints").device_code
    lines = [line.strip() for line in code.splitlines()]
    kernel = next(index for index, line in enumerate(lines) if line.startswith("__global__"))
    kernel_hint = ["// " + line for line in KernelScopeGenerator.SCOPE_HINT.splitlines()]
    assert lines[kernel - len(kernel_hint) : kernel] == kernel_hint, code
    assert "// " + ThreadBlockScopeGenerator.SCOPE_HINT.splitlines()[0] in lines[kernel:], code


def test_a_block_tiled_reduction_folds_without_a_runtime_functor():
    """A kernel whose rows reduce is block-tiled: its lanes fold through ``gpucub::BlockReduce`` with a lambda, so the
    unit needs no ``dace::_wcr_fixed``."""
    M = dace.symbol("M")

    @dace.program
    def matvec(A: dace.float64[M, N], x: dace.float64[N], y: dace.float64[M]):
        for i in range(M):
            s = 0.0
            for j in range(N):
                s += A[i, j] * x[j]
            y[i] = s

    rendering = render_gpu(matvec, "cpf_hip_block_tiled_reduce")
    assert_standalone_units(rendering, "cpf_hip_block_tiled_reduce")
    assert "gpucub::BlockReduce" in rendering.device_code, rendering.device_code


def test_conflicting_device_wcr_is_an_atomic_and_never_an_omp_pragma():
    """A scatter whose index array may repeat is the WCR that stays conflicting. On the host CPF
    writes ``#pragma omp atomic update`` for it; inside a ``__global__`` function that pragma is
    ignored by the device compiler, so the rendering would compile, run, and race. The device
    spelling has to be an actual device atomic."""

    @dace.program
    def scatter(idx: dace.int32[N], src: dace.float64[N], bins: dace.float64[N]):
        for i in dace.map[0:N]:
            bins[idx[i]] += src[i]

    rendering = render_gpu(scatter, "cpf_hip_scatter")
    assert_standalone_units(rendering, "cpf_hip_scatter")
    code = both(rendering)

    assert "omp atomic" not in code and "omp critical" not in code, (
        "an OpenMP atomic in device code is ignored by the device compiler, so the accumulation would race"
    )
    assert re.search(
        r"cpf_gpu_atomic\(.*\[\] \(const double &__cpf_acc, const double &__cpf_val\) "
        r"\{ return __cpf_acc \+ \(__cpf_val\); \}\)",
        code,
    ), "the conflicting accumulation must take a device atomic carrying the resolution as a functor"


def device_find_first_sdfg(name: str) -> dace.SDFG:
    """A ``FindFirst`` over DEVICE memory at host level -- the shape a lifted early-exit loop has
    on a GPU graph. Built by hand because what is under test is the IMPLEMENTATION CHOICE, and a
    frontend program would reach it through the whole lift."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("a", [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("out", [1], dace.int64)
    state = sdfg.add_state()
    node = FindFirst("ff", predicate="_a[__i] > 0.5", begin=0, end=N)
    node.implementation = "CPU"
    node.schedule = dace.dtypes.ScheduleType.GPU_Device
    node.add_in_connector("_a", dace.pointer(dace.float64))
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_a", dace.Memlet.from_array("a", sdfg.arrays["a"]))
    state.add_edge(node, "_out_idx", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_a_library_node_over_device_memory_takes_its_device_implementation():
    """``FindFirst``'s host expansions refuse to lower over device memory -- correctly, a host
    search would dereference a device pointer -- so CPF must not re-point the node at one just
    because ``pure`` renders. The device implementation is renderable too: CPF spells the search
    (``find_first_index_device``) the same way it spells the host one."""
    sdfg = device_find_first_sdfg("cpf_hip_findfirst")
    rendering = cpf.render(sdfg, language="hip")
    code = both(rendering)
    assert "find_first_index_device" in code, "the device search must be rendered, not the host one"
    assert "__global__ void find_first_kernel" in code, "the search kernel is part of the rendering"


def test_the_host_implementation_still_wins_over_host_memory():
    """The other side of the same rule, so it is a CHOICE and not a device-dialect override: the
    same node over HOST memory keeps the host implementation, which is the parallel search."""
    sdfg = device_find_first_sdfg("cpf_hip_findfirst_host")
    sdfg.arrays["a"].storage = dace.dtypes.StorageType.CPU_Heap
    node, state = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, FindFirst))
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        assert cpf.renderable_implementations(node, state)[0] != "CUDA"


def host_level_gemm_sdfg(name: str) -> dace.SDFG:
    """A ``Gemm`` at HOST level over device memory, in the shape ``finalize_for_target`` leaves it.

    That shape is the whole point: the node was pointed at a device LIBRARY CALL, which needs no
    schedule of its own, so ``Sequential`` is what a GPU-finalized graph carries here.
    """
    sdfg = dace.SDFG(name)
    storage = dace.dtypes.StorageType.GPU_Global
    sdfg.add_array("a", [N, N], dace.float64, storage=storage)
    sdfg.add_array("b", [N, N], dace.float64, storage=storage)
    sdfg.add_array("c", [N, N], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = Gemm("gemm", alpha=1.0, beta=0.0)
    node.schedule = dace.dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_edge(state.add_read("a"), None, node, "_a", dace.Memlet.from_array("a", sdfg.arrays["a"]))
    state.add_edge(state.add_read("b"), None, node, "_b", dace.Memlet.from_array("b", sdfg.arrays["b"]))
    state.add_edge(node, "_c", state.add_write("c"), None, dace.Memlet.from_array("c", sdfg.arrays["c"]))
    return sdfg


def test_a_host_level_node_over_device_memory_expands_into_a_kernel():
    """A ``Gemm`` / ``Dot`` / ``Reduce`` at host level over ``GPU_Global`` operands has no
    renderable DEVICE expansion -- cuBLAS and CUB are library calls a standalone unit cannot make --
    so CPF re-points it at ``pure``, whose expansion is maps. Those maps inherit ``node.schedule``,
    which a GPU-finalized graph left ``Sequential`` because the node was going to be a library call.

    Left there the expansion is a HOST map indexing device pointers, and validation refuses the
    whole render: ``Data container "_c" is stored as StorageType.GPU_Global but accessed on host``.
    Every scientific_computing kernel carrying a host-level matmul or dot product failed exactly
    this way (lulesh, cholesky, minife, quatrex_rgf, channel_flow, ls3df_scf,
    warpx_esirkepov_deposition), so the schedule is corrected where the implementation is chosen."""
    sdfg = host_level_gemm_sdfg("cpf_hip_host_gemm")
    rendering = cpf.render(sdfg, language="hip")
    assert_standalone_units(rendering, "cpf_hip_host_gemm")


def test_the_schedule_correction_is_confined_to_host_level_device_memory():
    """The correction is a CHOICE, not a device-dialect override. A node over HOST memory keeps its
    schedule (its maps are host code, correctly), and so does one already inside a kernel, which has
    no launch to issue."""
    host = host_level_gemm_sdfg("cpf_hip_host_gemm_hostmem")
    for array in ("a", "b", "c"):
        host.arrays[array].storage = dace.dtypes.StorageType.CPU_Heap
    node, state = next((n, s) for n, s in host.all_nodes_recursive() if isinstance(n, Gemm))
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        cpf.schedule_host_level_device_node(node, state)
    assert node.schedule == dace.dtypes.ScheduleType.Sequential, "host memory must not be rescheduled"

    device = host_level_gemm_sdfg("cpf_hip_host_gemm_device")
    node, state = next((n, s) for n, s in device.all_nodes_recursive() if isinstance(n, Gemm))
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_C):
        cpf.schedule_host_level_device_node(node, state)
    assert node.schedule == dace.dtypes.ScheduleType.Sequential, "a host dialect must not be rescheduled"
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        cpf.schedule_host_level_device_node(node, state)
    assert node.schedule == dace.dtypes.ScheduleType.GPU_Device


def test_a_device_library_node_inside_a_kernel_keeps_the_device_code_implementation():
    """A node ALREADY inside a kernel has no launch to issue, so the device library call -- which
    is host code -- is not available to it. Both halves of
    :func:`~dace.codegen.cpf.on_device_at_host_level` have to hold, not just the storage one."""
    sdfg = dace.SDFG("cpf_hip_inner_scan")
    sdfg.add_array("a", [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("b", [N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Scan("scan")
    entry, exit_node = state.add_map("outer", {"i": "0:N"}, schedule=dace.dtypes.ScheduleType.GPU_Device)
    read, write = state.add_read("a"), state.add_write("b")
    state.add_memlet_path(read, entry, node, dst_conn="_scan_in", memlet=dace.Memlet.from_array("a", sdfg.arrays["a"]))
    state.add_memlet_path(
        node, exit_node, write, src_conn="_scan_out", memlet=dace.Memlet.from_array("b", sdfg.arrays["b"])
    )
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        assert not cpf.on_device_at_host_level(node, state)
        assert cpf.renderable_implementations(node, state)[0] != "CUDA"


def test_a_device_scan_renders_the_device_scan_and_its_scratch():
    """The scan's device expansion issues ``gpucub::DeviceScan`` over a workspace it takes from the
    CUB scratch pool -- and the pool is normally an ENVIRONMENT, allocated by the init entry point
    and released by the exit one, neither of which a single self-contained call has. CPF defines
    the pool itself, over a buffer that allocates on first use and frees at static destruction, so
    the environment has nothing left to run and the rendering is admitted rather than refused."""
    rendering = cpf.render(device_scan_sdfg("cpf_hip_scan", ScanOp.SUM), language="hip")
    code = both(rendering)
    assert "::gpucub::DeviceScan::ExclusiveScan" in code, "the device scan must survive as the device scan"
    assert "get_scratch<ScanTag>" in code, "the workspace must come from CPF's own pool"
    for unit in (rendering.code, rendering.device_code):
        assert unit.count("static inline void *get_scratch(") <= 1, "the pool is defined once per unit"


def test_a_device_resident_scan_seed_is_read_where_it_lives():
    """The seed of an affine scan is an ARRAY of one element per residue class, and on a GPU graph
    it is device-resident like everything else. Re-pointing the node at a host expansion made host
    code dereference it, which validation rejects outright -- so the device expansion is the one to
    keep, and it reads the seed on the device where it lives."""
    sdfg = device_scan_sdfg("cpf_hip_affine", ScanOp.AFFINE, coefficients=True, seed=True)
    code = both(cpf.render(sdfg, language="hip"))
    assert "inclusive_affine" in code, "the affine recurrence must render as the device scan over its affine maps"
    assert "__global__ void cpf_affine_pack_kernel" in code, "the map-packing kernel is part of the unit"


def test_a_device_scan_seed_is_read_at_its_type_inside_the_kernel():
    """The scan's input iterator reads the device seed at the element type inside the kernel, the
    same on both backends: no host staging, no ``FutureValue``, and no ``InclusiveScanInit``, which
    hipCUB on ROCm 6.3 does not have."""
    code = both(cpf.render(device_scan_sdfg("cpf_hip_seeded_scan", ScanOp.SUM, seed=True), language="hip"))
    assert "const double* seed;" in code, code
    assert "static_cast<double>(*seed)" in code, code
    assert "InclusiveScanInit" not in code and "FutureValue" not in code, code


def test_a_device_product_scan_multiplies_through_a_typed_functor():
    code = both(cpf.render(device_scan_sdfg("cpf_hip_product_scan", ScanOp.PRODUCT), language="hip"))
    assert "DACE_CUB_MUL_OP" not in code and "#define" not in code, code
    assert "cpf_cub_multiplies()" in code, "the scan must pass the functor where the operator name was"
    assert "__host__ __device__ T operator()(const T& a, const T& b) const" in code, code


def build_units(rendering: Rendering, tmp_path, name: str) -> None:
    """Build the two units as the judge builds a GPU submission: each compiled with ``-c``, then one
    shared library."""
    require_gpu_backend("hip")
    objects = []
    for suffix, text in ((".cpp", rendering.code), (".hip", rendering.device_code)):
        source = tmp_path / f"{name}{suffix}"
        source.write_text(text)
        objects.append(str(source) + ".o")
        command = ["hipcc", "-std=c++20", "-fPIC", "-c", str(source), "-o", objects[-1]]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run(
        ["hipcc", "-shared", *objects, "-o", str(tmp_path / "unit.so")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu
def test_a_device_product_scan_unit_builds_with_hipcc(tmp_path):
    """The functor is what ``gpucub::DeviceScan`` instantiates, so the units have to build and link."""
    build_units(cpf.render(device_scan_sdfg("cpf_hip_product_build", ScanOp.PRODUCT), language="hip"), tmp_path, "unit")


@dace.program
def s255(a: dace.float64[N], b: dace.float64[N]):
    x = b[N - 1]
    y = b[N - 2]
    for i in range(N):
        a[i] = (b[i] + x + y) * 0.333
        y = x + 0.0
        x = b[i]


def test_a_read_only_operand_behind_an_ordering_edge_is_const_in_both_units():
    """tsvc s255: the frontend orders ``y = x`` before ``x = b[i]`` with an empty memlet into ``b``. That
    edge writes nothing, so ``b`` stays ``const`` in the launcher the host unit declares and the device
    unit defines -- one of them non-const and the host call no longer matched any declaration."""
    assert_standalone_units(render_gpu(s255, "cpf_hip_s255"), "cpf_hip_s255")


@dace.program
def gramschmidt(A: dace.float64[N, N], Q: dace.float64[N, N], R: dace.float64[N, N]):
    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        R[k, k + 1 :] = Q[:, k] @ A[:, k + 1 :]
        A[:, k + 1 :] = A[:, k + 1 :] - Q[:, k][:, None] * R[k, k + 1 :][None, :]


def test_a_nested_function_passes_the_state_to_its_launchers():
    """A library node's expansion renders as a function of its own whose launches take ``__state``, so the
    function takes it too (gramschmidt's Dot: ``use of undeclared identifier '__state'``)."""
    rendering = render_gpu(gramschmidt, "cpf_hip_nested_state")
    assert_standalone_units(rendering, "cpf_hip_nested_state")
    assert re.search(
        r"^static inline void \w+\(cpf_hip_nested_state_state_t \*__state", rendering.code, re.MULTILINE
    ), rendering.code


@dace.program
def cholesky_plus_upper(A: dace.float64[N, N]):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


def test_a_sequential_factorization_over_device_memory_touches_it_only_in_kernels():
    """cholesky2: the renderable Cholesky is a loop nest; at host level over device memory every statement
    of it has to be a kernel, or its scratch factor is device memory the host loop reads."""
    assert_standalone_units(render_gpu(cholesky_plus_upper, "cpf_hip_cholesky"), "cpf_hip_cholesky")


@dace.program
def fft_round_trip(x: dace.complex128[N], y: dace.complex128[N], z: dace.complex128[N]):
    y[:] = np.fft.fft(x)
    z[:] = np.fft.ifft(y)


NR, NQ, NP = (dace.symbol(name) for name in ("NR", "NQ", "NP"))


@dace.program
def doitgen(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
    out = np.zeros((NR, NQ, NP), A.dtype)
    for r in range(NR):
        for q in range(NQ):
            for p in range(NP):
                for s in range(NP):
                    out[r, q, p] = out[r, q, p] + A[r, q, s] * C4[s, p]
    A[:] = out


def test_a_matmul_an_offloaded_contraction_expands_to_is_a_kernel():
    """doitgen: the contraction's expansion leaves its matmul at host level carrying the thread-block
    schedule of the device scope it came from; its loops must still launch as a kernel."""
    assert_standalone_units(render_gpu(doitgen, "cpf_hip_doitgen"), "cpf_hip_doitgen")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("program", "name"),
    [
        (s255, "cpf_hip_s255_build"),
        (gramschmidt, "cpf_hip_nested_build"),
        (cholesky_plus_upper, "cpf_hip_chol_build"),
        (fft_round_trip, "cpf_hip_fft_build"),
    ],
)
def test_the_units_build_with_hipcc(program, name, tmp_path):
    build_units(render_gpu(program, name), tmp_path, name)


@pytest.mark.parametrize("language", ("c++", "c"))
def test_the_host_dialects_do_not_see_the_device_selection(language):
    """The device rule is gated on the dialect, so a host rendering of the same graph is unchanged
    -- which is what says this cannot regress the two dialects that were already complete."""
    sdfg = device_find_first_sdfg(f"cpf_hip_host_{language.replace('+', 'p')}")
    sdfg.arrays["a"].storage = dace.dtypes.StorageType.CPU_Heap
    node, state = next((n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, FindFirst))
    with cpf_lowering.dialect_scope(cpf.LANGUAGES[language]):
        assert not cpf.on_device_at_host_level(node, state)
        assert cpf.renderable_implementations(node, state) == (
            cpf.RENDERABLE_BY_NODE["FindFirst"] + cpf.RENDERABLE_IMPLEMENTATIONS
        )


def code_object(name: str, language: str, target_type: str, linkable: bool = True) -> CodeObject:
    """A CodeObject carrying only the labels :func:`~dace.codegen.cpf.split_units` reads."""
    return CodeObject(name, "", language, None, "title", target_type=target_type, linkable=linkable)


@pytest.mark.parametrize(
    "device_language,device_target_type",
    (
        ("cpp", "hip"),  # MEASURED on gfx942: hipcc compiles .cpp, so the device unit is not 'hip'
        ("cu", "cuda"),
        ("cu", ""),  # a target that labels only the language
        ("cpp", "cuda"),
    ),
)
def test_the_device_object_is_found_however_the_backend_labels_it(device_language, device_target_type):
    """A GPU rendering arrives as frame + device object, and the two are rendered as its two units.

    Selecting the device object by LANGUAGE alone refused every HIP rendering: hipcc compiles
    ``.cpp``, so both objects came back ``language='cpp'``, both looked like the frame, and no
    single frame was found. Selecting by ``target_type`` alone fails the mirrored way on a target
    that leaves it at the default. Either label has to be enough.
    """
    frame = code_object("kern", "cpp", "")
    device = code_object("kern_cuda", device_language, device_target_type)
    with cpf_lowering.dialect_scope(cpf.LANGUAGES["hip"]):
        assert cpf.split_units([frame, device], "kern") == (frame, device)


def test_a_split_that_is_not_a_device_object_is_still_refused():
    """The split is for the frame and its device object only. Two host units are the
    split-translation-unit case CPF cannot express, and a device object under a host dialect is a
    kernel no unit would hold; both must keep saying so."""
    units = [code_object("kern", "cpp", ""), code_object("kern_part2", "cpp", "")]
    with cpf_lowering.dialect_scope(cpf.LANGUAGES["hip"]):
        with pytest.raises(NotImplementedError, match="at most one device unit"):
            cpf.split_units(units, "kern")
    with cpf_lowering.dialect_scope(cpf.LANGUAGES["c++"]):
        with pytest.raises(NotImplementedError, match="at most one device unit"):
            cpf.split_units([code_object("kern", "cpp", ""), code_object("kern_cuda", "cpp", "hip")], "kern")


def test_non_linkable_objects_do_not_count_as_a_split():
    """The header and the sample ``main`` are generated for every SDFG and are not built, so a
    lone frame beside them is one translation unit, not three."""
    frame = code_object("kern", "cpp", "")
    extras = [
        code_object("kern", "h", "../../include", linkable=False),
        code_object("kern_main", "cpp", "../../sample", linkable=False),
    ]
    assert cpf.split_units([frame] + extras, "kern") == (frame, None)


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_the_host_unit_launches_and_the_device_unit_holds_the_kernels()
    test_the_hip_unit_calls_hip_runtime_functions_whichever_gpu_the_rendering_host_has()
    test_device_reduction_folds_without_a_runtime_functor()
    test_each_gpu_scope_says_what_it_is_ahead_of_it()
    test_a_block_tiled_reduction_folds_without_a_runtime_functor()
    test_conflicting_device_wcr_is_an_atomic_and_never_an_omp_pragma()
    test_a_library_node_over_device_memory_takes_its_device_implementation()
    test_the_host_implementation_still_wins_over_host_memory()
    test_a_host_level_node_over_device_memory_expands_into_a_kernel()
    test_the_schedule_correction_is_confined_to_host_level_device_memory()
    test_a_device_library_node_inside_a_kernel_keeps_the_device_code_implementation()
    test_a_device_scan_renders_the_device_scan_and_its_scratch()
    test_a_device_resident_scan_seed_is_read_where_it_lives()
    test_a_device_scan_seed_is_read_at_its_type_inside_the_kernel()
    test_a_device_product_scan_multiplies_through_a_typed_functor()
    test_a_device_product_scan_unit_builds_with_hipcc(pathlib.Path(tempfile.mkdtemp()))
    test_a_read_only_operand_behind_an_ordering_edge_is_const_in_both_units()
    test_a_nested_function_passes_the_state_to_its_launchers()
    test_a_sequential_factorization_over_device_memory_touches_it_only_in_kernels()
    test_a_matmul_an_offloaded_contraction_expands_to_is_a_kernel()
    test_the_units_build_with_hipcc(s255, "cpf_hip_s255_build", pathlib.Path(tempfile.mkdtemp()))
    test_the_units_build_with_hipcc(gramschmidt, "cpf_hip_nested_build", pathlib.Path(tempfile.mkdtemp()))
    test_the_units_build_with_hipcc(cholesky_plus_upper, "cpf_hip_chol_build", pathlib.Path(tempfile.mkdtemp()))
    test_the_units_build_with_hipcc(fft_round_trip, "cpf_hip_fft_build", pathlib.Path(tempfile.mkdtemp()))
    test_the_host_dialects_do_not_see_the_device_selection("c++")
    test_the_host_dialects_do_not_see_the_device_selection("c")
    test_the_device_object_is_found_however_the_backend_labels_it("cpp", "hip")
    test_the_device_object_is_found_however_the_backend_labels_it("cu", "cuda")
    test_the_device_object_is_found_however_the_backend_labels_it("cu", "")
    test_the_device_object_is_found_however_the_backend_labels_it("cpp", "cuda")
    test_a_split_that_is_not_a_device_object_is_still_refused()
    test_non_linkable_objects_do_not_count_as_a_split()

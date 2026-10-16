# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
GPU tests of generated code in different places: live tests marked ``ai`` and offline tests of the compile
check and of early expansion.
"""

import pathlib
import tempfile

import numpy as np
import pytest

import dace
from dace import dtypes, nodes
from dace.libraries.ai import verify
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.context import collect_context
from dace.libraries.ai.expansion import ExpandAI
from dace.libraries.ai.nodes import AINode
from tests.library.ai.ai_node_contexts_test import nested_gpu_scalar_sdfg
from tests.library.ai.ai_test_utils import offline, stub_provider

M = 32

DESCRIPTION = f"""
Compute a {M}x{M} single-precision matrix product _c = _a * _b (row-major, contiguous). All three
pointers cover the whole {M}x{M} matrix.
""".strip()


def _matmul_sdfg(name: str, storage: dtypes.StorageType, inside_kernel: bool) -> dace.SDFG:
    """
    Builds a matrix multiplication around an :class:`AINode` in one specific slot.

    :param name: Name of the SDFG.
    :param storage: Storage type of the three matrices.
    :param inside_kernel: If True, the node is placed inside a ``GPU_Device`` map.
    :return: The SDFG.
    """
    sdfg = dace.SDFG(name)
    for array in ("A", "B", "C"):
        sdfg.add_array(array, [M, M], dace.float32, storage=storage)

    state = sdfg.add_state()
    node = AINode("matmul", DESCRIPTION, inputs={"_a", "_b"}, outputs={"_c"})
    state.add_node(node)
    subset = f"0:{M}, 0:{M}"

    if inside_kernel:
        entry, exit_node = state.add_map("grid", {"b": "0:1"}, schedule=dtypes.ScheduleType.GPU_Device)
        state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="_a", memlet=dace.Memlet(f"A[{subset}]"))
        state.add_memlet_path(state.add_read("B"), entry, node, dst_conn="_b", memlet=dace.Memlet(f"B[{subset}]"))
        state.add_memlet_path(node, exit_node, state.add_write("C"), src_conn="_c", memlet=dace.Memlet(f"C[{subset}]"))
    else:
        state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet(f"A[{subset}]"))
        state.add_edge(state.add_read("B"), None, node, "_b", dace.Memlet(f"B[{subset}]"))
        state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet(f"C[{subset}]"))
    return sdfg


def _expand(sdfg: dace.SDFG) -> nodes.Tasklet:
    """
    Expands the SDFG's single AI node.

    :param sdfg: The SDFG to expand.
    :return: The generated tasklet.
    """
    state = sdfg.states()[0]
    node = next(n for n in state.nodes() if isinstance(n, AINode))
    node.expand(state, "ai")
    return next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))


def _run(sdfg: dace.SDFG, on_gpu: bool):
    """
    Runs the SDFG on random inputs and returns the result.

    :param sdfg: The compiled-and-run SDFG.
    :param on_gpu: If True, arguments are passed as GPU arrays.
    :return: A tuple of (a, b, c) as host arrays.
    """
    rng = np.random.default_rng(0)
    a = rng.random((M, M), dtype=np.float32)
    b = rng.random((M, M), dtype=np.float32)
    c = np.zeros((M, M), dtype=np.float32)
    if not on_gpu:
        sdfg(A=a, B=b, C=c)
        return a, b, c

    import cupy

    ga, gb, gc = cupy.asarray(a), cupy.asarray(b), cupy.zeros((M, M), dtype=np.float32)
    sdfg(A=ga, B=gb, C=gc)
    return a, b, cupy.asnumpy(gc)


@pytest.mark.ai
def test_host_cpu_context():
    sdfg = _matmul_sdfg("ai_ctx_host_cpu", dtypes.StorageType.CPU_Heap, inside_kernel=False)
    tasklet = _expand(sdfg)

    # Nothing GPU-related belongs in a pure host slot
    assert "__dace_current_stream" not in tasklet.code.as_string
    assert not tasklet.environments or all("cuda" not in e.lower() for e in tasklet.environments)

    a, b, c = _run(sdfg, on_gpu=False)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.ai
def test_host_slot_over_gpu_arrays():
    sdfg = _matmul_sdfg("ai_ctx_host_gpu", dtypes.StorageType.GPU_Global, inside_kernel=False)
    tasklet = _expand(sdfg)
    body = tasklet.code.as_string

    # Host code over device pointers: it must hand them to a library or a kernel launch, and
    # order the work on the SDFG's stream, rather than dereferencing them directly
    assert "__dace_current_stream" in body or "<<<" in body or "Launch" in body or "blas" in body.lower()

    a, b, c = _run(sdfg, on_gpu=True)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.ai
def test_device_slot_inside_a_kernel():
    sdfg = _matmul_sdfg("ai_ctx_device", dtypes.StorageType.GPU_Global, inside_kernel=True)
    tasklet = _expand(sdfg)
    body = tasklet.code.as_string

    # Inside a kernel there is no state struct, no stream variable, and no host library to call
    assert "__state" not in body
    assert "__dace_current_stream" not in body
    assert "<<<" not in body
    assert not tasklet.state_fields

    a, b, c = _run(sdfg, on_gpu=True)
    assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_a_nested_gpu_scalar_expanded_early_runs():
    """
    The end-to-end form of ``test_a_nested_container_takes_the_outer_storage``: a by-value connector
    here would make the host load the input straight out of device memory.
    """
    sdfg, state, node = nested_gpu_scalar_sdfg(dtypes.StorageType.Default)
    spec = TaskletSpec(
        code_global="#include <cuda_runtime.h>",
        code=(
            "double value;\n"
            "cudaDeviceSynchronize();\n"
            "cudaMemcpy(&value, _in, sizeof(double), cudaMemcpyDeviceToHost);\n"
            "value *= 2;\n"
            "cudaMemcpy(_out, &value, sizeof(double), cudaMemcpyHostToDevice);"
        ),
    )
    with dace.config.set_temporary("ai", "compile_check", value=False):
        with dace.config.set_temporary("ai", "sessions", value=False):
            with dace.config.set_temporary("ai", "cache", value=False):
                with stub_provider(spec):
                    node.expand(state, "ai")

    # Host arrays around the GPU ones, copied in and out
    sdfg.arrays["G"].transient = True
    sdfg.arrays["H"].transient = True
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    copy_in = sdfg.add_state_before(sdfg.start_state)
    copy_in.add_nedge(copy_in.add_read("A"), copy_in.add_write("G"), dace.Memlet("A[0]"))
    copy_out = sdfg.add_state_after(state.sdfg.parent)
    copy_out.add_nedge(copy_out.add_read("H"), copy_out.add_write("B"), dace.Memlet("H[0]"))

    a = np.array([21.0])
    b = np.zeros(1)
    sdfg(A=a, B=b)
    assert b[0] == 42.0


@pytest.mark.gpu
def test_the_compile_check_builds_device_code(tmp_path):
    """A slot inside a kernel is checked as device code: host-only code is rejected there."""
    sdfg = dace.SDFG("ai_check_device")
    sdfg.add_array("A", [64], dace.float32, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [64], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("kernel", {"i": "0:64"}, schedule=dtypes.ScheduleType.GPU_Device)
    node = AINode("twice", "Write twice _in into _out.", inputs={"_in"}, outputs={"_out"})
    state.add_memlet_path(state.add_read("A"), entry, node, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(node, exit_node, state.add_write("B"), src_conn="_out", memlet=dace.Memlet("B[i]"))
    ctx = collect_context(node, state, sdfg)
    assert ctx.capabilities.device_level

    def check(code: str, folder: str) -> verify.CheckResult:
        tasklet = ExpandAI._make_tasklet(node, TaskletSpec(code=code), ctx)
        return verify.compile_check(tasklet, node, state, sdfg, ctx, [], str(tmp_path / folder))

    # A single element of GPU memory is a pointer, as for every library node
    good = check("_out[0] = __fmul_rn(_in[0], 2.0f);", "good")
    assert good.ok and not good.skipped, good.skipped
    assert any(name.endswith(".cu") for name in good.sources)

    host_only = check("_out[0] = _in[0]; __state->nothing = 0;", "host_only")
    assert not host_only.ok


if __name__ == "__main__":
    test_host_cpu_context()
    test_host_slot_over_gpu_arrays()
    test_device_slot_inside_a_kernel()
    with offline():
        test_a_nested_gpu_scalar_expanded_early_runs()
        test_the_compile_check_builds_device_code(pathlib.Path(tempfile.mkdtemp()))

# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`SplitStateByGPUClass`.

Builds a small zoo of hand-written SDFGs that exercise the split pass's classification +
rewrite rules:

* pure CPU / pure GPU       -> no-op
* multiple independent CPU+GPU WCCs in one state    -> Case A (lift CPU before)
* mixed WCC ``CPU -> GPU``       -> Case B1 (lift CPU prefix before)
* mixed WCC ``GPU -> CPU``       -> Case B2 (lift CPU suffix after, via lifting the GPU middle)
* mixed WCC ``CPU -> GPU -> CPU``    -> Case B3 (two fissions: prefix before, then GPU middle)
* mixed WCC ``GPU -> CPU -> GPU`` or cycles  -> refuse, state untouched
"""

import dace
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NodeKind, classify_state_top_level
from dace.transformation.passes.gpu_specialization.split_state_by_gpu_class import SplitStateByGPUClass


def _state_kinds(sdfg):
    return [classify_state_top_level(s) for s in sdfg.states()]


def _state_kinds_in_topological_order(sdfg):
    return [classify_state_top_level(s) for s in dace.sdfg.utils.dfs_topological_sort(sdfg)]


def _gpu_kernel_then_cpu_finalize():
    """``GPU -> CPU`` chain inside one state: GPU reduces into scalar, CPU consumes it."""
    sdfg = dace.SDFG("gpu_to_cpu")
    sdfg.add_array("A", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("s", dace.float32, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_scalar("result", dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("mixed")

    # GPU: write a scalar via a single-iteration GPU map (the kernel body writes the scalar).
    me, mx = state.add_map("gpu_writer", dict(_="0:1"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = state.add_tasklet("write_s", {"_a": dace.float32}, {"_s": dace.float32}, "_s = _a + 1.0")
    a_r = state.add_read("A")
    s_w = state.add_access("s")
    state.add_memlet_path(a_r, me, kt, dst_conn="_a", memlet=dace.Memlet("A[0]"))
    state.add_memlet_path(kt, mx, s_w, src_conn="_s", memlet=dace.Memlet("s[0]"))

    # CPU finalize that consumes ``s``.
    s_r = state.add_access("s")
    state.add_edge(s_w, None, s_r, None, dace.Memlet("s[0]"))
    fin = state.add_tasklet(
        "finalize",
        {"_s": dace.pointer(dace.float32)},
        {"__out": dace.pointer(dace.float32)},
        "__out = _s * 2.0f;",
        language=dace.Language.CPP,
    )
    r_w = state.add_write("result")
    state.add_edge(s_r, None, fin, "_s", dace.Memlet("s[0]"))
    state.add_edge(fin, "__out", r_w, None, dace.Memlet("result[0]"))
    return sdfg


def _cpu_init_gpu_kernel_cpu_finalize():
    """``CPU -> GPU -> CPU`` chain inside one state."""
    sdfg = dace.SDFG("cpu_gpu_cpu")
    sdfg.add_array("A", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("s_in", dace.float32, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_scalar("s_out", dace.float32, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    sdfg.add_scalar("result", dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("mixed")

    init = state.add_tasklet(
        "init", {}, {"__out": dace.pointer(dace.float32)}, "__out = 3.0f;", language=dace.Language.CPP
    )
    s_in_w = state.add_access("s_in")
    state.add_edge(init, "__out", s_in_w, None, dace.Memlet("s_in[0]"))

    s_in_r = state.add_access("s_in")
    state.add_edge(s_in_w, None, s_in_r, None, dace.Memlet("s_in[0]"))

    a_r = state.add_read("A")
    b_w = state.add_write("B")
    s_out_w = state.add_access("s_out")
    me, mx = state.add_map("gpu_map", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    kt = state.add_tasklet(
        "kernel",
        {"_a": dace.float32, "_si": dace.float32},
        {"_b": dace.float32, "_so": dace.float32},
        "_b = _a * _si\n_so = _a + _si",
    )
    state.add_memlet_path(a_r, me, kt, dst_conn="_a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(s_in_r, me, kt, dst_conn="_si", memlet=dace.Memlet("s_in[0]"))
    state.add_memlet_path(kt, mx, b_w, src_conn="_b", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(kt, mx, s_out_w, src_conn="_so", memlet=dace.Memlet("s_out[0]"))

    s_out_r = state.add_access("s_out")
    state.add_edge(s_out_w, None, s_out_r, None, dace.Memlet("s_out[0]"))
    fin = state.add_tasklet(
        "fin",
        {"_s": dace.pointer(dace.float32)},
        {"__out": dace.pointer(dace.float32)},
        "__out = _s + 1.0f;",
        language=dace.Language.CPP,
    )
    r_w = state.add_write("result")
    state.add_edge(s_out_r, None, fin, "_s", dace.Memlet("s_out[0]"))
    state.add_edge(fin, "__out", r_w, None, dace.Memlet("result[0]"))
    return sdfg


def _pure_gpu_state():
    sdfg = dace.SDFG("pure_gpu")
    sdfg.add_array("A", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("main")
    me, mx = state.add_map("m", dict(i="0:16"), schedule=dace.dtypes.ScheduleType.GPU_Device)
    t = state.add_tasklet("t", {"_a": dace.float32}, {"_b": dace.float32}, "_b = _a + 1.0")
    state.add_memlet_path(state.add_read("A"), me, t, dst_conn="_a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, state.add_write("B"), src_conn="_b", memlet=dace.Memlet("B[i]"))
    return sdfg


def _pure_cpu_state():
    sdfg = dace.SDFG("pure_cpu")
    sdfg.add_array("A", [4], dace.float32)
    sdfg.add_array("B", [4], dace.float32)
    state = sdfg.add_state("main")
    me, mx = state.add_map("m", dict(i="0:4"), schedule=dace.dtypes.ScheduleType.Sequential)
    t = state.add_tasklet("t", {"_a": dace.float32}, {"_b": dace.float32}, "_b = _a + 1.0")
    state.add_memlet_path(state.add_read("A"), me, t, dst_conn="_a", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, state.add_write("B"), src_conn="_b", memlet=dace.Memlet("B[i]"))
    return sdfg


# Tests.


def test_cpu_suffix_chain_splits_into_two_states():
    sdfg = _gpu_kernel_then_cpu_finalize()
    assert _state_kinds(sdfg) == [NodeKind.MIXED]
    SplitStateByGPUClass().apply_pass(sdfg, {})
    kinds = _state_kinds_in_topological_order(sdfg)
    assert kinds == [NodeKind.GPU, NodeKind.CPU], f"Expected [GPU, CPU], got {kinds}"


def test_cpu_prefix_and_suffix_chain_splits_into_three_states():
    sdfg = _cpu_init_gpu_kernel_cpu_finalize()
    assert _state_kinds(sdfg) == [NodeKind.MIXED]
    SplitStateByGPUClass().apply_pass(sdfg, {})
    kinds = _state_kinds_in_topological_order(sdfg)
    assert kinds == [NodeKind.CPU, NodeKind.GPU, NodeKind.CPU], f"Expected [CPU, GPU, CPU], got {kinds}"


def test_pure_gpu_state_is_unchanged():
    sdfg = _pure_gpu_state()
    before = len(list(sdfg.states()))
    SplitStateByGPUClass().apply_pass(sdfg, {})
    after = len(list(sdfg.states()))
    assert before == after


def test_pure_cpu_state_is_unchanged():
    sdfg = _pure_cpu_state()
    before = len(list(sdfg.states()))
    SplitStateByGPUClass().apply_pass(sdfg, {})
    after = len(list(sdfg.states()))
    assert before == after


if __name__ == "__main__":
    test_cpu_suffix_chain_splits_into_two_states()
    test_cpu_prefix_and_suffix_chain_splits_into_three_states()
    test_pure_gpu_state_is_unchanged()
    test_pure_cpu_state_is_unchanged()

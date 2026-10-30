# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.cuda.implementation`` selects the GPU code generator, read at every ``generate_code`` call."""

import dace
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.targets.cuda import CUDACodeGen
from dace.codegen.targets.experimental_cuda import ExperimentalCUDACodeGen


def build_gpu_sdfg():
    sdfg = dace.SDFG("gpu_codegen_impl_selection")
    sdfg.add_array("A", (16,), dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("B", (16,), dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    rd = state.add_read("A")
    wr = state.add_write("B")
    me, mx = state.add_map("m", dict(i="0:16"), schedule=dace.ScheduleType.GPU_Device)
    tasklet = state.add_tasklet("double", {"inp": None}, {"out": None}, "out = inp * 2.0")
    state.add_memlet_path(rd, me, tasklet, dst_conn="inp", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(tasklet, mx, wr, src_conn="out", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def gpu_codegen_classes(sdfg):
    return {
        code_object.target
        for code_object in sdfg.generate_code()
        if code_object.target.target_name in ("cuda", "experimental_cuda")
    }


def test_both_gpu_codegens_are_registered():
    registered = {v["name"] for v in TargetCodeGenerator.extensions().values()}
    assert "cuda" in registered
    assert "experimental_cuda" in registered


def test_config_selects_active_gpu_codegen_at_runtime():
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="legacy"):
        used = gpu_codegen_classes(build_gpu_sdfg())
    assert used == {CUDACodeGen}

    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        used = gpu_codegen_classes(build_gpu_sdfg())
    assert used == {ExperimentalCUDACodeGen}

    with dace.config.set_temporary("compiler", "cuda", "implementation", value="legacy"):
        used = gpu_codegen_classes(build_gpu_sdfg())
    assert used == {CUDACodeGen}


if __name__ == "__main__":
    test_both_gpu_codegens_are_registered()
    test_config_selects_active_gpu_codegen_at_runtime()

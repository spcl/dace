# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``CPU_Pinned`` allocation passes ``hipMallocHost`` a ``void **``.

CUDA's runtime header templates ``cudaMallocHost`` over the pointer type, so ``&ptr`` of any type
binds; HIP declares only ``hipMallocHost(void **, size_t)``, and g++ refuses the implicit
``int **`` -> ``void **`` conversion (contour_integral's LAPACK ``info`` on the GPU canon column).
"""

import dace
from dace.codegen import codegen


def pinned_int_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("pinned_int_alloc")
    sdfg.add_array("out", [4], dace.int32, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient("info", [4], dace.int32, storage=dace.StorageType.CPU_Pinned)
    state = sdfg.add_state()
    me, mx = state.add_map("fill", dict(i="0:4"))
    t = state.add_tasklet("set", {}, {"o"}, "o = i")
    info = state.add_access("info")
    state.add_memlet_path(me, t, memlet=dace.Memlet())
    state.add_memlet_path(t, mx, info, src_conn="o", memlet=dace.Memlet("info[i]"))
    state.add_nedge(info, state.add_write("out"), dace.Memlet("info[0:4]"))
    return sdfg


def test_a_pinned_host_allocation_passes_a_void_pointer_pointer():
    sdfg = pinned_int_sdfg()
    with dace.config.set_temporary("compiler", "cuda", "backend", value="hip"):
        code = "\n".join(obj.clean_code for obj in codegen.generate_code(sdfg))
    calls = [line.strip() for line in code.splitlines() if "MallocHost(" in line]
    assert calls, "the pinned array must be allocated with MallocHost"
    assert all("(void**)&info" in call for call in calls), calls


if __name__ == "__main__":
    test_a_pinned_host_allocation_passes_a_void_pointer_pointer()

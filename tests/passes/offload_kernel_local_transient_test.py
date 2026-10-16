# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet


def test_kernel_local_transient_is_thread_private():
    # ``t`` is written and read only inside the kernel: one ``GPU_Global`` buffer would be shared by every thread
    sdfg = dace.SDFG("kernel_local_transient")
    sdfg.add_array("A", (64,), dace.float32)
    sdfg.add_array("B", (64,), dace.float32)
    sdfg.add_transient("t", (2,), dace.float32)
    state = sdfg.add_state()
    me, mx = state.add_map("m", dict(i="0:64"))
    t = state.add_access("t")
    fill = state.add_tasklet("fill", {"a"}, {"o0", "o1"}, "o0 = a; o1 = 2 * a")
    use = state.add_tasklet("use", {"x0", "x1"}, {"b"}, "b = x0 + x1")
    state.add_memlet_path(state.add_read("A"), me, fill, dst_conn="a", memlet=Memlet("A[i]"))
    state.add_edge(fill, "o0", t, None, Memlet("t[0]"))
    state.add_edge(fill, "o1", t, None, Memlet("t[1]"))
    state.add_edge(t, None, use, "x0", Memlet("t[0]"))
    state.add_edge(t, None, use, "x1", Memlet("t[1]"))
    state.add_memlet_path(use, mx, state.add_write("B"), src_conn="b", memlet=Memlet("B[i]"))
    sdfg.validate()

    sdfg.apply_gpu_transformations()
    assert sdfg.arrays["t"].storage == dace.StorageType.Register
    assert "t_gpu" not in sdfg.arrays
    sdfg.generate_code()


if __name__ == "__main__":
    test_kernel_local_transient_is_thread_private()

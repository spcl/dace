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


def test_a_constant_array_a_kernel_reads_gets_a_device_copy():
    """A closure array the caller stores as an SDFG constant is a host ``constexpr`` no caller passes in: a kernel
    reading it needs a device copy, not the host address (cp2k_density_matrix_trs4's ``np.arange(3)``)."""
    import numpy as np

    from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    n = dace.symbol("n", dtype=dace.int64, positive=True)

    @dace.program
    def offsets(idx: dace.int64[n], out: dace.int64[n, 3]):
        out[:] = idx[:, None] + np.arange(3, dtype=np.int64)[None, :]

    sdfg = offsets.to_sdfg(simplify=True)
    (name,) = [k for k in sdfg.arrays if k.startswith("__g_")]
    # As HB's CPF cache stores it: a transient whose value lives in ``constants_prop``
    sdfg.arrays[name].transient = True
    sdfg.add_constant(name, np.arange(3, dtype=np.int64), sdfg.arrays[name])
    canonicalize(sdfg, target="gpu", validate_all=False)
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target="gpu")

    constants = {k for k, (_, v) in sdfg.constants_prop.items() if isinstance(v, np.ndarray)}
    assert constants and not any(sdfg.arrays[k].storage == dace.StorageType.GPU_Global for k in constants)
    assert sdfg.arrays[name].storage == dace.StorageType.GPU_Global, "the device copy the kernel reads"
    code = "\n".join(c.clean_code for c in sdfg.generate_code())
    launches = [line for line in code.splitlines() if "__dace_runkernel_" in line and "(__state" in line]
    assert launches and not any(k in line for k in constants for line in launches), "a kernel takes a host constant"


if __name__ == "__main__":
    test_kernel_local_transient_is_thread_private()
    test_a_constant_array_a_kernel_reads_gets_a_device_copy()

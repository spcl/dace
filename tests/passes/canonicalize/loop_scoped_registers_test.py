# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_transient_storage

N, K = dace.symbol("N"), dace.symbol("K")


def per_iteration_scratch(name: str, extent: str, small: bool) -> dace.SDFG:
    """``for ip: c = d[ip]; tmp[0:extent] = w[ip] * j; out[ip] = sum(tmp)``, ``K`` declared SMALL when ``small``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("d", [N], dace.int64)
    sdfg.add_array("w", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_symbol("K", dace.int64)
    sdfg.add_symbol("c", dace.int64)
    sdfg.add_transient("tmp", [extent], dace.float64)
    if small:
        sdfg.set_symbol_scale("K", dace.symbolic.Scale.SMALL)
    loop = LoopRegion("particles", "ip < N", "ip", "ip = 0", "ip = ip + 1")
    sdfg.add_node(loop, is_start_block=True)
    read = loop.add_state("read", is_start_block=True)
    body = loop.add_state("body")
    loop.add_edge(read, body, dace.InterstateEdge(assignments={"c": "d[ip]"}))
    body.add_mapped_tasklet(
        "fill",
        {"j": f"0:{extent}"},
        {"wi": dace.Memlet("w[ip]")},
        "t = wi * j",
        {"t": dace.Memlet("tmp[j]")},
        external_edges=True,
    )
    tmp = next(n for n in body.data_nodes() if n.data == "tmp")
    body.add_mapped_tasklet(
        "sum",
        {"j": f"0:{extent}"},
        {"t": dace.Memlet("tmp[j]")},
        "o = t",
        {"o": dace.Memlet("out[ip]", wcr="lambda a, b: a + b")},
        external_edges=True,
        input_nodes={"tmp": tmp},
    )
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("small", [True, False])
def test_a_scratch_sized_by_the_iteration_goes_on_the_stack(small):
    """``K + 2 - c`` with ``c`` read per iteration: a VLA, not a ``new``/``delete`` per iteration, whatever ``K``."""
    sdfg = per_iteration_scratch(f"loop_vla_{small}", "K + 2 - c", small)
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    storage = sdfg.arrays["tmp"].storage
    assert storage == dtypes.StorageType.Register and storage.force
    n, k = 50, 3
    rng = np.random.default_rng(0)
    d, w = rng.integers(0, 2, n), rng.random(n)
    out = np.zeros(n)
    sdfg(d=d, w=w, out=out, N=n, K=k)
    ref = np.array([w[i] * sum(range(k + 2 - d[i])) for i in range(n)])
    assert np.allclose(out, ref)


def test_a_scratch_the_loop_does_not_resize_is_allocated_once_per_call():
    sdfg = per_iteration_scratch("loop_invariant_scratch", "N", small=True)
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    desc = sdfg.arrays["tmp"]
    assert desc.storage != dtypes.StorageType.Register
    assert desc.lifetime in (dtypes.AllocationLifetime.SDFG, dtypes.AllocationLifetime.Persistent)


def test_a_scratch_an_inner_loop_reads_goes_on_the_stack():
    """warpx's cumulative sum reads the scratch in a scan loop nested in the particle loop: the particle loop still
    allocates it once per iteration."""
    sdfg = per_iteration_scratch("loop_vla_inner_scan", "K + 2 - c", small=True)
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    body = next(st for st in loop.nodes() if st.label == "body")
    scan = LoopRegion("scan", "k < K + 2 - c", "k", "k = 0", "k = k + 1")
    loop.add_node(scan)
    loop.add_edge(body, scan, dace.InterstateEdge())
    step = scan.add_state("step", is_start_block=True)
    step.add_mapped_tasklet(
        "scale",
        {"z": "0:1"},
        {"t": dace.Memlet("tmp[k]")},
        "o = 2 * t",
        {"o": dace.Memlet("tmp[k]")},
        external_edges=True,
    )
    sdfg.validate()
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    storage = sdfg.arrays["tmp"].storage
    assert storage == dtypes.StorageType.Register and storage.force


def test_a_heap_intermediate_inside_a_map_inside_a_loop_goes_on_the_stack():
    """warpx_field_gather: a fusion intermediate of one element, inferred ``CPU_Heap``, in a sequential map in the
    particle loop -- a ``new``/``delete`` per inner iteration unless pinned to the stack."""
    sdfg = dace.SDFG("heap_intermediate_in_map_in_loop")
    sdfg.add_array("w", [N], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    sdfg.add_transient("weight", [1, 1], dace.float64, storage=dtypes.StorageType.CPU_Heap)
    loop = LoopRegion("particles", "ip < N", "ip", "ip = 0", "ip = ip + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    entry, exit_ = body.add_map("taps", {"j": "0:3"}, schedule=dtypes.ScheduleType.Sequential)
    scale = body.add_tasklet("scale", {"wi"}, {"o"}, "o = wi * j")
    weight = body.add_access("weight")
    acc = body.add_tasklet("acc", {"t"}, {"o"}, "o = t")
    body.add_memlet_path(body.add_read("w"), entry, scale, dst_conn="wi", memlet=dace.Memlet("w[ip]"))
    body.add_edge(scale, "o", weight, None, dace.Memlet("weight[0, 0]"))
    body.add_edge(weight, None, acc, "t", dace.Memlet("weight[0, 0]"))
    body.add_memlet_path(
        acc, exit_, body.add_write("out"), src_conn="o", memlet=dace.Memlet("out[ip]", wcr="lambda a, b: a + b")
    )
    sdfg.validate()
    finalize_transient_storage(sdfg, dtypes.DeviceType.CPU)
    storage = sdfg.arrays["weight"].storage
    assert storage == dtypes.StorageType.Register and storage.force
    n = 20
    w, out = np.random.default_rng(0).random(n), np.zeros(n)
    sdfg(w=w, out=out, N=n)
    assert np.allclose(out, 3 * w)


def test_only_host_register_arrays_of_a_device_program_are_pinned():
    """A host map's register scratch goes on the stack as on the CPU; a kernel's stays for MoveArrayOutOfKernel."""
    from dace.transformation.passes.canonicalize.finalize import force_registers

    sdfg = dace.SDFG("pin_host_registers")
    sdfg.add_array("a", ["N"], dace.float64)
    for name in ("host", "dev"):
        sdfg.add_transient(name, ["N"], dace.float64, storage=dtypes.StorageType.Register)
    state = sdfg.add_state()
    for name, schedule in (("host", dtypes.ScheduleType.CPU_Multicore), ("dev", dtypes.ScheduleType.GPU_Device)):
        entry, exit_ = state.add_map(f"m_{name}", {"i": "0:N"}, schedule=schedule)
        put = state.add_tasklet(f"put_{name}", {"v"}, {"o"}, "o = v")
        state.add_memlet_path(state.add_read("a"), entry, put, dst_conn="v", memlet=dace.Memlet("a[i]"))
        state.add_memlet_path(put, exit_, state.add_write(name), src_conn="o", memlet=dace.Memlet(f"{name}[i]"))
        inner = state.add_access(name)
        state.add_nedge(entry, inner, dace.Memlet())
        state.add_nedge(inner, exit_, dace.Memlet())
    assert force_registers(sdfg, host_only=True) == 1
    assert sdfg.arrays["host"].storage.force and not sdfg.arrays["dev"].storage.force


if __name__ == "__main__":
    for small in [True, False]:
        test_a_scratch_sized_by_the_iteration_goes_on_the_stack(small)
    test_a_scratch_the_loop_does_not_resize_is_allocated_once_per_call()
    test_a_scratch_an_inner_loop_reads_goes_on_the_stack()
    test_a_heap_intermediate_inside_a_map_inside_a_loop_goes_on_the_stack()
    test_only_host_register_arrays_of_a_device_program_are_pinned()

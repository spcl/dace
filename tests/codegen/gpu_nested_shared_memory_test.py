# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared memory declared in nested SDFGs of a kernel, at several depths and in sibling nested SDFGs, placed
statically or dynamically by ``gpu_shared_memory.PlanSharedMemory``."""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.memlet import Memlet

M = dace.symbol("M")


def block_loop(state: dace.SDFGState, src: str, dst: str, size, scale: int, src_offset=0, dst_offset=0) -> None:
    """The 32 threads of a block copy every 32nd element: ``dst[dst_offset + k] = src[src_offset + k] * scale``."""
    tme, tmx = state.add_map("t", {"t": "0:32"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    jme, jmx = state.add_map("j", {"j": f"0:{size} / 32"}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet("scale", {"x"}, {"y"}, f"y = x * {scale}")
    state.add_memlet_path(
        state.add_read(src), tme, jme, tasklet, dst_conn="x", memlet=Memlet(f"{src}[{src_offset} + j * 32 + t]")
    )
    state.add_memlet_path(
        tasklet, jmx, tmx, state.add_write(dst), src_conn="y", memlet=Memlet(f"{dst}[{dst_offset} + j * 32 + t]")
    )


def pass_through(name: str, size, depth: int, total, offset) -> dace.SDFG:
    """``depth`` nested SDFGs over whole ``total``-element arrays, as a nested connector takes its container's
    descriptor; the innermost stages the ``size`` elements from ``offset`` through its own shared array."""
    sdfg = dace.SDFG(name)
    for arr in ("inp", "out"):
        sdfg.add_array(arr, [total], dace.float32, storage=dtypes.StorageType.GPU_Global)
    if depth == 1:
        sdfg.add_array("shared", [size], dace.float32, storage=dtypes.StorageType.GPU_Shared, transient=True)
        load = sdfg.add_state("load")
        block_loop(load, "inp", "shared", size, 2, src_offset=offset)
        block_loop(sdfg.add_state_after(load, "store"), "shared", "out", size, 3, dst_offset=offset)
        return sdfg
    state = sdfg.add_state()
    node = state.add_nested_sdfg(pass_through(f"{name}_{depth - 1}", size, depth - 1, total, offset), {"inp"}, {"out"})
    state.add_edge(state.add_read("inp"), None, node, "inp", Memlet(f"inp[0:{total}]"))
    state.add_edge(node, "out", state.add_write("out"), None, Memlet(f"out[0:{total}]"))
    return sdfg


def kernel_of_siblings(name: str, size, depth: int, siblings: int) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    for arr in ("A", "B"):
        sdfg.add_array(arr, [size * siblings], dace.float32, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map("kernel", {"i": "0:1"}, schedule=dtypes.ScheduleType.GPU_Device)
    me.map.gpu_block_size = [32, 1, 1]
    a, b = state.add_read("A"), state.add_write("B")
    total = size * siblings
    for k in range(siblings):
        node = state.add_nested_sdfg(pass_through(f"{name}_{k}", size, depth, total, k * size), {"inp"}, {"out"})
        state.add_memlet_path(a, me, node, dst_conn="inp", memlet=Memlet(f"A[0:{total}]"))
        state.add_memlet_path(node, mx, b, src_conn="out", memlet=Memlet(f"B[0:{total}]"))
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, depth, siblings, symbols",
    [
        (64, 1, 1, {}),
        (64, 3, 1, {}),
        (64, 2, 2, {}),
        (M, 2, 2, {"M": 64}),
        (16384, 2, 1, {}),
    ],
)
def test_nested_shared_memory(size, depth, siblings, symbols):
    import cupy as cp

    sdfg = kernel_of_siblings(f"nested_shared_{depth}_{siblings}_{size}", size, depth, siblings)
    n = int(dace.symbolic.evaluate(size, symbols)) * siblings
    A = cp.arange(n, dtype=np.float32) + 1
    B = cp.zeros(n, dtype=np.float32)
    sdfg(A=A, B=B, **symbols)
    assert cp.array_equal(A * 6, B)


if __name__ == "__main__":
    for size, depth, siblings, symbols in [
        (64, 1, 1, {}),
        (64, 3, 1, {}),
        (64, 2, 2, {}),
        (M, 2, 2, {"M": 64}),
        (16384, 2, 1, {}),
    ]:
        test_nested_shared_memory(size, depth, siblings, symbols)

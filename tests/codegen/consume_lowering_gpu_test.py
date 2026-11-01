# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests consume scopes on the GPU: every thread a PE (kernel level), or every thread block a PE."""

import pytest

import dace
from dace.sdfg import nodes

GLOBAL = dace.StorageType.GPU_Global


def _fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _fibonacci_sdfg(name: str, num_pes: int, chunksize: int, block_pes: bool) -> dace.SDFG:
    """Recursive Fibonacci on the GPU: every task pushes two smaller ones into the consumed stream.

    With ``block_pes``, each PE is a thread block whose threads share a chunk of up to ``chunksize``
    elements (one per thread); otherwise each thread is a PE.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("iv", [1], dace.int32, storage=GLOBAL)
    sdfg.add_array("res", [1], dace.int64, storage=GLOBAL)
    sdfg.add_stream("S", dace.int32, transient=True, buffer_size=1 << 16, storage=GLOBAL)
    fill = sdfg.add_state("fill")
    fill.add_mapped_tasklet(
        "fill",
        dict(i="0:1"),
        {"inp": dace.Memlet("iv[i]")},
        "out = inp",
        {"out": dace.Memlet("S[0]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    state = sdfg.add_state_after(fill, "consume")
    entry, exit_node = state.add_consume(
        "cons", ("p", str(num_pes)), schedule=dace.ScheduleType.GPU_Device, chunksize=chunksize
    )
    state.add_edge(state.add_read("S"), None, entry, "IN_stream", dace.Memlet("S[0]"))
    chunk = dace.Memlet("S[0]") if chunksize == 1 else dace.Memlet(f"S[0:{chunksize}]", allow_oob=True)
    if block_pes:
        tb_entry, tb_exit = state.add_map("threads", dict(j=f"0:{chunksize}"), dace.ScheduleType.GPU_ThreadBlock)
        code = "if j < __dace_cons_numelems:\n    if s == 1:\n        val = 1\n    elif s > 1:\n        sout = s - 1\n        sout = s - 2"
        element = dace.Memlet("S[j]", allow_oob=True)
        tasklet = state.add_tasklet("fib", {"s"}, {"sout", "val"}, code)
        state.add_memlet_path(entry, tb_entry, tasklet, src_conn="OUT_stream", dst_conn="s", memlet=element)
        next(e for e in state.out_edges(entry) if e.src_conn == "OUT_stream").data = chunk
        exits = [tb_exit, exit_node]
    elif chunksize == 1:
        code = "if s == 1:\n    val = 1\nelif s > 1:\n    sout = s - 1\n    sout = s - 2"
        tasklet = state.add_tasklet("fib", {"s"}, {"sout", "val"}, code)
        state.add_memlet_path(entry, tasklet, src_conn="OUT_stream", dst_conn="s", memlet=chunk)
        exits = [exit_node]
    else:
        code = (
            "acc = 0\nfor i in range(__dace_cons_numelems):\n    if s[i] == 1:\n        acc += 1\n"
            "    elif s[i] > 1:\n        sout = s[i] - 1\n        sout = s[i] - 2\nval = acc"
        )
        tasklet = state.add_tasklet("fib", {"s"}, {"sout", "val"}, code)
        state.add_memlet_path(entry, tasklet, src_conn="OUT_stream", dst_conn="s", memlet=chunk)
        exits = [exit_node]
    state.add_memlet_path(
        tasklet, *exits, state.add_write("S"), src_conn="sout", memlet=dace.Memlet("S[0]", dynamic=True)
    )
    state.add_memlet_path(
        tasklet,
        *exits,
        state.add_write("res"),
        src_conn="val",
        memlet=dace.Memlet("res[0]", wcr="lambda a, b: a + b", dynamic=True),
    )
    return sdfg


def _block_local_sdfg(name: str, num_blocks: int, threads: int) -> dace.SDFG:
    """Every thread block consumes a shared-memory stream of its own with all its threads.

    Block ``b`` computes Fibonacci of ``iv[b]`` into ``res[b]``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("iv", [num_blocks], dace.int32, storage=GLOBAL)
    sdfg.add_array("res", [num_blocks], dace.int64, storage=GLOBAL)
    sdfg.add_stream("S", dace.int32, transient=True, buffer_size=2048, storage=dace.StorageType.GPU_Shared)
    state = sdfg.add_state()
    blocks_entry, blocks_exit = state.add_map("blocks", dict(b=f"0:{num_blocks}"), dace.ScheduleType.GPU_Device)
    seed_entry, seed_exit = state.add_map("seed", dict(j="0:1"), dace.ScheduleType.GPU_ThreadBlock)
    seed = state.add_tasklet("seed", {"inp"}, {"out"}, "out = inp")
    stream = state.add_access("S")
    state.add_memlet_path(
        state.add_read("iv"), blocks_entry, seed_entry, seed, dst_conn="inp", memlet=dace.Memlet("iv[b]")
    )
    state.add_memlet_path(seed, seed_exit, stream, src_conn="out", memlet=dace.Memlet("S[0]"))
    entry, exit_node = state.add_consume("cons", ("p", str(threads)), schedule=dace.ScheduleType.GPU_ThreadBlock)
    tasklet = state.add_tasklet(
        "fib", {"s"}, {"sout", "val"}, "if s == 1:\n    val = 1\nelif s > 1:\n    sout = s - 1\n    sout = s - 2"
    )
    state.add_edge(stream, None, entry, "IN_stream", dace.Memlet("S[0]"))
    state.add_edge(entry, "OUT_stream", tasklet, "s", dace.Memlet("S[0]"))
    pushed = state.add_access("S")
    state.add_memlet_path(tasklet, exit_node, pushed, src_conn="sout", memlet=dace.Memlet("S[0]", dynamic=True))
    state.add_nedge(pushed, blocks_exit, dace.Memlet())
    state.add_memlet_path(
        tasklet,
        exit_node,
        blocks_exit,
        state.add_write("res"),
        src_conn="val",
        memlet=dace.Memlet("res[b]", wcr="lambda a, b: a + b", dynamic=True),
    )
    return sdfg


def _cuda_code(sdfg: dace.SDFG) -> str:
    return next(obj.clean_code for obj in sdfg.generate_code() if obj.language == "cu")


def test_device_popped_stream_is_a_work_queue():
    """A stream popped in a kernel while tasks push is a ``GPUQueue`` everywhere it is passed."""
    code = _cuda_code(_fibonacci_sdfg("consume_queue_type", 64, 1, block_pes=False))
    assert "dace::GPUQueue<int>" in code
    assert "dace::GPUStream" not in code
    assert "S.pop_try(" in code


def test_block_pes_pop_collectively():
    """A body of thread-block maps makes each block one PE, which pops its chunk into shared memory."""
    sdfg = _fibonacci_sdfg("consume_block_code", 16, 64, block_pes=True)
    code = _cuda_code(sdfg)
    assert ".block_pop_try(" in code and ".block_task_done(" in code and ".block_pending()" in code
    assert "__shared__ __align__(16) int __dace_cons_elements[64];" in code


@pytest.mark.gpu
@pytest.mark.parametrize(
    "num_pes, chunksize, block_pes",
    [(256, 1, False), (4096, 1, False), (1024, 4, False), (32, 64, True), (8, 128, True)],
)
def test_recursive_pushes_on_gpu(num_pes: int, chunksize: int, block_pes: bool):
    import cupy as cp

    name = f"consume_gpu_fib_{num_pes}_{chunksize}_{int(block_pes)}"
    sdfg = _fibonacci_sdfg(name, num_pes, chunksize, block_pes)
    iv = cp.array([20], dtype=cp.int32)
    res = cp.zeros(1, dtype=cp.int64)
    sdfg(iv=iv, res=res)
    assert int(res[0]) == _fibonacci(20)


def test_block_local_queue_in_shared_memory():
    """A stream in shared memory is a queue of the block, set up by all its threads."""
    code = _cuda_code(_block_local_sdfg("consume_local_code", 4, 64))
    assert "dace::GPUQueue<int, true> S;" in code
    assert "S.init_block(" in code


@pytest.mark.gpu
@pytest.mark.parametrize("num_blocks, threads", [(4, 64), (100, 256)])
def test_block_local_consume_on_gpu(num_blocks: int, threads: int):
    import cupy as cp
    import numpy as np

    sdfg = _block_local_sdfg(f"consume_gpu_local_{num_blocks}_{threads}", num_blocks, threads)
    values = (np.arange(num_blocks) % 12 + 5).astype(np.int32)
    res = cp.zeros(num_blocks, dtype=cp.int64)
    sdfg(iv=cp.asarray(values), res=res)
    assert np.array_equal(res.get(), [_fibonacci(int(v)) for v in values])


@pytest.mark.gpu
def test_persistent_queue_is_reusable():
    """After a consume, the queue is empty and in a state the next call starts from."""
    import cupy as cp

    sdfg = _fibonacci_sdfg("consume_gpu_persistent", 512, 1, block_pes=False)
    # Above the most elements queued at once, below the pushes of one call: the ring wraps around
    sdfg.arrays["S"].buffer_size = 1 << 12
    sdfg.arrays["S"].lifetime = dace.AllocationLifetime.Persistent
    compiled = sdfg.compile()
    iv = cp.array([18], dtype=cp.int32)
    res = cp.zeros(1, dtype=cp.int64)
    for _ in range(5):
        res[:] = 0
        compiled(iv=iv, res=res)
        assert int(res[0]) == _fibonacci(18)


def test_lowered_gpu_consume_structure():
    """Thread PEs keep a register chunk; block PEs share theirs."""
    from dace.transformation.passes.lower_consume import LowerConsumeScopes

    for block_pes, storage in ((False, dace.StorageType.Register), (True, dace.StorageType.GPU_Shared)):
        sdfg = _fibonacci_sdfg(f"consume_gpu_structure_{int(block_pes)}", 16, 64 if block_pes else 1, block_pes)
        LowerConsumeScopes().apply_pass(sdfg, {})
        sdfg.validate()
        nsdfg = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))
        chunk = next(name for name in nsdfg.sdfg.arrays if name.startswith("__dace_cons_element"))
        assert nsdfg.sdfg.arrays[chunk].storage == storage


if __name__ == "__main__":
    test_device_popped_stream_is_a_work_queue()
    test_block_pes_pop_collectively()
    for args in [(256, 1, False), (4096, 1, False), (1024, 4, False), (32, 64, True), (8, 128, True)]:
        test_recursive_pushes_on_gpu(*args)
    test_block_local_queue_in_shared_memory()
    test_block_local_consume_on_gpu(4, 64)
    test_block_local_consume_on_gpu(100, 256)
    test_persistent_queue_is_reusable()
    test_lowered_gpu_consume_structure()

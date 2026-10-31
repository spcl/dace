# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A GPU kernel's scatter WCR into a small array is privatized per block in shared memory: threads fold with
shared-memory atomics and each block commits every element with one global atomic, instead of one global atomic
per thread. The emitted HIP/CUDA is what is under test; the GPU run checks the values."""

import numpy as np
import pytest

import dace
from dace.codegen.codegen import generate_code

N = dace.symbol("N")
M = dace.symbol("M")
K = 256


@dace.program
def histogram(idx: dace.int64[N], w: dace.float64[N], h: dace.float64[K]):
    for i in dace.map[0:N]:
        h[idx[i]] += w[i]


@dace.program
def column_sums(a: dace.float64[N, 64], s: dace.float64[64]):
    for i, j in dace.map[0:N, 0:64]:
        s[j] += a[i, j]


def _gpu_sdfg(prog):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    return sdfg


def _device_code(sdfg) -> str:
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        return "\n".join(
            c.clean_code for c in generate_code(sdfg) if c.target.target_name in ("cuda", "experimental_cuda")
        )


@pytest.mark.parametrize("prog", [histogram, column_sums], ids=["scatter", "column"])
def test_a_small_scatter_target_is_privatized_in_shared_memory(prog):
    code = _device_code(_gpu_sdfg(prog))
    assert "__shared__ double __bpart_" in code, "the block's copy of the target is not in shared memory"
    assert "&__bpart_" in code, "threads must fold into the shared copy"
    assert "__syncthreads();" in code


def nested_scatter_of_symbolic_size() -> dace.SDFG:
    """``h[idx[i]] += w[i]`` in a nested body that reads the slot into a symbol, on arrays already in device memory:
    nothing in the kernel uses the length ``M`` of ``h``."""
    body = dace.SDFG("scatter_body")
    body.add_symbol("i", dace.int64)
    body.add_array("idx", [N], dace.int64)
    body.add_array("w", [N], dace.float64)
    body.add_array("h", [M], dace.float64)
    load = body.add_state("load", is_start_block=True)
    add = body.add_state("add")
    body.add_edge(load, add, dace.InterstateEdge(assignments={"j": "idx[i]"}))
    t = add.add_tasklet("add", {"v"}, {"o"}, "o = v")
    add.add_edge(add.add_read("w"), None, t, "v", dace.Memlet("w[i]"))
    add.add_edge(t, "o", add.add_write("h"), None, dace.Memlet("h[j]", wcr="lambda a, b: a + b"))

    sdfg = dace.SDFG("nested_scatter_of_symbolic_size")
    gpu = dace.StorageType.GPU_Global
    sdfg.add_array("idx", [N], dace.int64, storage=gpu)
    sdfg.add_array("w", [N], dace.float64, storage=gpu)
    sdfg.add_array("h", [M], dace.float64, storage=gpu)
    state = sdfg.add_state()
    me, mx = state.add_map("scatter", {"i": "0:N"}, schedule=dace.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(body, {"idx", "w"}, {"h"}, symbol_mapping={"i": "i", "M": "M", "N": "N"})
    state.add_memlet_path(state.add_read("idx"), me, nsdfg, dst_conn="idx", memlet=dace.Memlet("idx[0:N]"))
    state.add_memlet_path(state.add_read("w"), me, nsdfg, dst_conn="w", memlet=dace.Memlet("w[0:N]"))
    state.add_memlet_path(
        nsdfg,
        mx,
        state.add_write("h"),
        src_conn="h",
        memlet=dace.Memlet("h[0:M]", wcr="lambda a, b: a + b", dynamic=True),
    )
    return sdfg


def tasklet_scatter() -> dace.SDFG:
    """``h[idx[i]] += w[i]`` with the tasklet subscripting its output pointer: ``o[ix] = v`` into ``h[0:M]``."""
    sdfg = dace.SDFG("tasklet_scatter")
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("w", [N], dace.float64)
    sdfg.add_array("h", [M], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("scatter", {"i": "0:N"})
    t = state.add_tasklet("add", {"ix", "v"}, {"o"}, "o[ix] = v")
    state.add_memlet_path(state.add_read("idx"), me, t, dst_conn="ix", memlet=dace.Memlet("idx[i]"))
    state.add_memlet_path(state.add_read("w"), me, t, dst_conn="v", memlet=dace.Memlet("w[i]"))
    state.add_memlet_path(
        t, mx, state.add_write("h"), src_conn="o", memlet=dace.Memlet("h[0:M]", wcr="lambda a, b: a + b", dynamic=True)
    )
    sdfg.apply_gpu_transformations()
    return sdfg


def test_a_subscripted_scatter_folds_into_the_slot_it_subscripts():
    """``o[ix] = v`` writes ``ix`` past the start of ``h``: the shared copy is updated there too, not at slot 0."""
    code = _device_code(tasklet_scatter())
    folds = [line for line in code.splitlines() if "reduce_atomic(&__bpart_" in line]
    assert folds and all("idx_gpu[" in line.split("reduce_atomic(&__bpart_")[1].split("]")[0] + "]" for line in folds)


def test_a_scatter_target_sized_by_a_symbol_the_kernel_does_not_receive_keeps_global_atomics():
    """``h`` is ``M`` long and nothing in the kernel reads ``M``: the kernel is not passed ``M``, so the run-time
    capacity check ``M <= ...`` of a shared copy could not compile there."""
    code = _device_code(nested_scatter_of_symbolic_size())
    kernel = next(line for line in code.splitlines() if "__global__" in line)
    assert "M" in kernel.replace("(", " ").replace(",", " ").split() or "__bpart_" not in code


@pytest.mark.gpu
@pytest.mark.parametrize("prog", [histogram, column_sums], ids=["scatter", "column"])
def test_a_privatized_scatter_matches_numpy(prog):
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg = _gpu_sdfg(prog)
        rng = np.random.default_rng(0)
        n = 100003
        if prog is histogram:
            idx, w, h = rng.integers(0, K, n), rng.random(n), np.zeros(K)
            sdfg(idx=idx, w=w, h=h, N=n)
            assert np.allclose(h, np.bincount(idx, w, K))
        else:
            a, s = rng.random((n, 64)), np.zeros(64)
            sdfg(a=a, s=s, N=n)
            assert np.allclose(s, a.sum(0))


@pytest.mark.gpu
def test_a_subscripted_scatter_matches_numpy():
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg = tasklet_scatter()
        rng = np.random.default_rng(0)
        n, m = 100003, 300
        idx, w, h = rng.integers(0, m, n), rng.random(n), np.zeros(m)
        sdfg(idx=idx, w=w, h=h, N=n, M=m)
        assert np.allclose(h, np.bincount(idx, w, m))


if __name__ == "__main__":
    for p in (histogram, column_sums):
        test_a_small_scatter_target_is_privatized_in_shared_memory(p)
    test_a_scatter_target_sized_by_a_symbol_the_kernel_does_not_receive_keeps_global_atomics()
    test_a_subscripted_scatter_folds_into_the_slot_it_subscripts()

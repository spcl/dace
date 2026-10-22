# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace import Memlet, dtypes
from dace import data as dt
from dace.config import Config
from dace.sdfg import infer_types
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.transient_reuse import TransientReuse

N_VALUE = 37
K_VALUE = 3


def buffers(sdfg: dace.SDFG) -> list[str]:
    return sorted(name for name, desc in sdfg.arrays.items() if desc.transient and type(desc) is dt.Array)


def elementwise(state, src, dst, expr, extent="N", src_index="i", dst_index="i", wcr=None):
    state.add_mapped_tasklet(
        f"{src.data}_to_{dst.data}",
        {"i": f"0:{extent}"},
        {"x": Memlet(f"{src.data}[{src_index}]")},
        f"y = {expr}",
        {"y": Memlet(f"{dst.data}[{dst_index}]", wcr=wcr)},
        input_nodes={src.data: src},
        output_nodes={dst.data: dst},
        external_edges=True,
    )


def new_sdfg(name, arrays, transients, sizes=None):
    sdfg = dace.SDFG(name)
    sizes = sizes or {}
    for array in arrays:
        sdfg.add_array(array, [sizes.get(array, "N")], dace.float64)
    for array in transients:
        sdfg.add_array(array, [sizes.get(array, "N")], dace.float64, transient=True)
    return sdfg


def chain(state, *steps):
    """``steps`` alternate data names and expressions; each expression maps the previous data to the next."""
    node = state.add_access(steps[0])
    for expr, name in zip(steps[1::2], steps[2::2]):
        nxt = state.add_access(name)
        elementwise(state, node, nxt, expr)
        node = nxt


def reuse(sdfg):
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    return TransientReuse().apply_pass(sdfg, {})


def run(sdfg, **arrays):
    sdfg.validate()
    sdfg(N=N_VALUE, **arrays)


def test_sequential_states_share_one_buffer():
    sdfg = new_sdfg("reuse_sequential_states", ["A", "B", "C"], ["T1", "T2"])
    chain(sdfg.add_state("s0", is_start_block=True), "A", "2 * x", "T1", "x + 1", "B")
    chain(sdfg.add_state_after(sdfg.start_block, "s1"), "B", "3 * x", "T2", "x - 1", "C")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"T2": "T1"}}
    assert buffers(sdfg) == ["T1"]
    A = np.random.rand(N_VALUE)
    B, C = np.zeros(N_VALUE), np.zeros(N_VALUE)
    run(sdfg, A=A, B=B, C=C)
    assert np.allclose(C, 3 * (2 * A + 1) - 1)


def test_same_state_reader_of_one_writing_the_other_keeps_both():
    sdfg = new_sdfg("reuse_same_state_reversal", ["A", "B", "C"], ["T1", "T2", "T3"])
    state = sdfg.add_state("s0", is_start_block=True)
    a, t1, t2, b, t3, c = (state.add_access(n) for n in ("A", "T1", "T2", "B", "T3", "C"))
    elementwise(state, a, t1, "2 * x")
    elementwise(state, t1, t2, "x", src_index="N - 1 - i")
    elementwise(state, t2, b, "x + 1")
    elementwise(state, b, t3, "3 * x")
    elementwise(state, t3, c, "x - 1")

    reuse(sdfg)

    assert buffers(sdfg) == ["T1", "T2"]
    A = np.random.rand(N_VALUE)
    B, C = np.zeros(N_VALUE), np.zeros(N_VALUE)
    run(sdfg, A=A, B=B, C=C)
    assert np.allclose(C, 3 * ((2 * A)[::-1] + 1) - 1)


def test_loop_carried_transient_is_not_shared():
    sdfg = new_sdfg("reuse_loop_carried", ["A", "C"], ["T1", "T2", "T3"])
    sdfg.add_symbol("K", dace.int64)
    init = sdfg.add_state("init", is_start_block=True)
    chain(init, "A", "x", "T1")
    loop = LoopRegion("loop", "k < K", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    first = loop.add_state("first", is_start_block=True)
    chain(first, "T1", "2 * x", "T2")
    chain(loop.add_state_after(first, "second"), "T2", "x + 1", "T1")
    after = sdfg.add_state_after(loop, "after")
    chain(after, "T1", "x", "C")
    chain(sdfg.add_state_after(after, "last"), "C", "x + 1", "T3", "2 * x", "C")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"T3": "T1"}}
    assert buffers(sdfg) == ["T1", "T2"]
    A = np.random.rand(N_VALUE)
    C = np.zeros(N_VALUE)
    run(sdfg, A=A, C=C, K=K_VALUE)
    expected = A.copy()
    for _ in range(K_VALUE):
        expected = 2 * expected + 1
    assert np.allclose(C, 2 * (expected + 1))


def test_larger_buffer_is_reused_only_when_provably_large_enough():
    sizes = {"Small": "N - 1", "D": "M", "E": "M", "Other": "M"}
    sdfg = new_sdfg("reuse_symbolic_sizes", ["A", "B", "C", "D", "E"], ["Small", "Big", "Other"], sizes)
    s0 = sdfg.add_state("s0", is_start_block=True)
    a, small, b = s0.add_access("A"), s0.add_access("Small"), s0.add_access("B")
    elementwise(s0, a, small, "2 * x", extent="N - 1")
    elementwise(s0, small, b, "x + 1", extent="N - 1")
    s1 = sdfg.add_state_after(s0, "s1")
    chain(s1, "A", "3 * x", "Big", "x - 1", "C")
    s2 = sdfg.add_state_after(s1, "s2")
    d, other, e = s2.add_access("D"), s2.add_access("Other"), s2.add_access("E")
    elementwise(s2, d, other, "4 * x", extent="M")
    elementwise(s2, other, e, "x + 2", extent="M")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"Small": "Big"}}
    assert buffers(sdfg) == ["Big", "Other"]
    assert sdfg.arrays["Big"].lifetime == dtypes.AllocationLifetime.SDFG
    A, D = np.random.rand(N_VALUE), np.random.rand(N_VALUE + 5)
    B, C, E = np.zeros(N_VALUE), np.zeros(N_VALUE), np.zeros(N_VALUE + 5)
    run(sdfg, A=A, B=B, C=C, D=D, E=E, M=N_VALUE + 5)
    assert np.allclose(B[:-1], 2 * A[:-1] + 1)
    assert np.allclose(C, 3 * A - 1)
    assert np.allclose(E, 4 * D + 2)


def nested_increment() -> dace.SDFG:
    inner = dace.SDFG("reuse_nested_increment")
    inner.add_array("inp", ["N"], dace.float64)
    inner.add_array("out", ["N"], dace.float64)
    chain(inner.add_state("body", is_start_block=True), "inp", "x + 1", "out")
    return inner


def test_transient_read_by_later_nested_sdfg_is_not_shared():
    sdfg = new_sdfg("reuse_nested_reader", ["A", "C", "D", "E"], ["T1", "T2", "T3"])
    s0 = sdfg.add_state("s0", is_start_block=True)
    chain(s0, "A", "2 * x", "T1")
    s1 = sdfg.add_state_after(s0, "s1")
    chain(s1, "A", "3 * x", "T2", "x", "C")
    s2 = sdfg.add_state_after(s1, "s2")
    call = s2.add_nested_sdfg(nested_increment(), {"inp": None}, {"out": None}, symbol_mapping={"N": "N"})
    s2.add_edge(s2.add_access("T1"), None, call, "inp", Memlet("T1[0:N]"))
    s2.add_edge(call, "out", s2.add_access("D"), None, Memlet("D[0:N]"))
    chain(sdfg.add_state_after(s2, "s3"), "C", "2 * x", "T3", "x", "E")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"T3": "T1"}}
    assert buffers(sdfg) == ["T1", "T2"]
    A = np.random.rand(N_VALUE)
    C, D, E = np.zeros(N_VALUE), np.zeros(N_VALUE), np.zeros(N_VALUE)
    run(sdfg, A=A, C=C, D=D, E=E)
    assert np.allclose(D, 2 * A + 1)
    assert np.allclose(E, 6 * A)


def test_viewed_transient_is_not_shared():
    sdfg = new_sdfg("reuse_viewed", ["A", "B", "C", "D"], ["T1", "T2", "T3"])
    sdfg.add_view("V", ["N"], dace.float64)
    s0 = sdfg.add_state("s0", is_start_block=True)
    a, t1, view, b = s0.add_access("A"), s0.add_access("T1"), s0.add_access("V"), s0.add_access("B")
    elementwise(s0, a, t1, "2 * x")
    s0.add_edge(t1, None, view, "views", Memlet("T1[0:N]"))
    elementwise(s0, view, b, "x + 1")
    s1 = sdfg.add_state_after(s0, "s1")
    chain(s1, "A", "3 * x", "T2", "x", "C")
    chain(sdfg.add_state_after(s1, "s2"), "A", "4 * x", "T3", "x", "D")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"T3": "T2"}}
    assert buffers(sdfg) == ["T1", "T2"]
    A = np.random.rand(N_VALUE)
    B, C, D = np.zeros(N_VALUE), np.zeros(N_VALUE), np.zeros(N_VALUE)
    run(sdfg, A=A, B=B, C=C, D=D)
    assert np.allclose(B, 2 * A + 1)
    assert np.allclose(C, 3 * A)
    assert np.allclose(D, 4 * A)


def test_wcr_accumulator_is_not_shared():
    sdfg = new_sdfg("reuse_wcr_accumulator", ["A", "B", "C", "D"], ["Acc", "T2", "T3"])
    s0 = sdfg.add_state("s0", is_start_block=True)
    a0, acc0 = s0.add_access("A"), s0.add_access("Acc")
    elementwise(s0, a0, acc0, "x")
    s1 = sdfg.add_state_after(s0, "s1")
    a1, acc1, b = s1.add_access("A"), s1.add_access("Acc"), s1.add_access("B")
    elementwise(s1, a1, acc1, "2 * x", wcr="lambda u, v: u + v")
    elementwise(s1, acc1, b, "x")
    s2 = sdfg.add_state_after(s1, "s2")
    chain(s2, "A", "3 * x", "T2", "x", "C")
    chain(sdfg.add_state_after(s2, "s3"), "A", "4 * x", "T3", "x", "D")

    renamed = reuse(sdfg)

    assert renamed == {sdfg.cfg_id: {"T3": "T2"}}
    assert buffers(sdfg) == ["Acc", "T2"]
    A = np.random.rand(N_VALUE)
    B, C, D = np.zeros(N_VALUE), np.zeros(N_VALUE), np.zeros(N_VALUE)
    run(sdfg, A=A, B=B, C=C, D=D)
    assert np.allclose(B, 3 * A)
    assert np.allclose(C, 3 * A)
    assert np.allclose(D, 4 * A)


def gpu_sequential_states(name: str) -> dace.SDFG:
    sdfg = new_sdfg(name, ["A", "B", "C"], ["T1", "T2"])
    chain(sdfg.add_state("s0", is_start_block=True), "A", "2 * x", "T1", "x + 1", "B")
    chain(sdfg.add_state_after(sdfg.start_block, "s1"), "B", "3 * x", "T2", "x - 1", "C")
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize("streams", [-1, 0])
def test_gpu_transients_shared_only_on_one_stream(streams):
    backend = Config.get("compiler", "cuda", "backend")
    sdfg = gpu_sequential_states(f"reuse_gpu_{backend}_streams_{abs(streams)}")
    gpu = [n for n, d in sdfg.arrays.items() if d.transient and d.storage == dtypes.StorageType.GPU_Global]
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=streams):
        reuse(sdfg)
        after = [n for n, d in sdfg.arrays.items() if d.transient and d.storage == dtypes.StorageType.GPU_Global]
        assert len(after) == (len(gpu) - 2 if streams == -1 else len(gpu))
        A = np.random.rand(N_VALUE)
        B, C = np.zeros(N_VALUE), np.zeros(N_VALUE)
        run(sdfg, A=A, B=B, C=C)
    assert np.allclose(C, 3 * (2 * A + 1) - 1)


if __name__ == "__main__":
    test_sequential_states_share_one_buffer()
    test_same_state_reader_of_one_writing_the_other_keeps_both()
    test_loop_carried_transient_is_not_shared()
    test_larger_buffer_is_reused_only_when_provably_large_enough()
    test_transient_read_by_later_nested_sdfg_is_not_shared()
    test_viewed_transient_is_not_shared()
    test_wcr_accumulator_is_not_shared()
    test_gpu_transients_shared_only_on_one_stream(-1)
    test_gpu_transients_shared_only_on_one_stream(0)

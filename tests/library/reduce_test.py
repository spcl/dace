# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import itertools

import numpy as np
import pytest

import dace
import dace.libraries.standard as std
from dace import SDFG, Memlet

C_in, C_out, H, K, N, W = (dace.symbol(s, dace.int64) for s in ("C_in", "C_out", "H", "K", "N", "W"))


def make_sdfg():
    g = SDFG("prog")
    g.add_array("A", (N, 1, 1, C_in, C_out), dace.float32, strides=(C_in * C_out, C_in * C_out, C_in * C_out, C_out, 1))
    g.add_array("C", (N, H, W, C_out), dace.float32, strides=(C_out * H * W, C_out * W, C_out, 1))

    st0 = g.add_state("st0", is_start_block=True)
    st = st0

    A = st.add_access("A")
    C = st.add_access("C")
    R = st.add_reduce("lambda x, y: x + y", [1, 2, 3], 0)
    st.add_nedge(A, R, Memlet(expr="A[0:N, 0, 0, 0:C_in, 0:C_out]"))
    st.add_nedge(R, C, Memlet(expr="C[0:N, 5, 5, 0:C_out]"))

    return g, R


def test_library_node_expand_reduce_pure():
    n, cin, cout = 7, 7, 7
    h, k, w = 25, 35, 45
    A = np.ones((n, 1, 1, cin, cout), np.float32)

    g, R = make_sdfg()
    R.implementation = "pure-seq"
    g.validate()
    g.compile()

    wantC = np.ones((n, h, w, cout), np.float32) * 42
    g(A=A, C=wantC, N=n, C_in=cin, C_out=cout, H=h, K=k, W=w)

    g, R = make_sdfg()
    R.implementation = "pure"
    g.validate()
    g.compile()

    gotC = np.ones((n, h, w, cout), np.float32) * 42
    g(A=A, C=gotC, N=n, C_in=cin, C_out=cout, H=h, K=k, W=w)
    assert np.allclose(wantC, gotC)


def test_pure_seq_row_sums_in_a_map():
    """A sequential reduction per map iteration: its output is one element of the destination, so the
    nested SDFG connects the outer containers and indexes them absolutely."""

    @dace.program
    def row_sums(A: dace.float64[10, 4], B: dace.float64[10]):
        for i in dace.map[0:10]:
            B[i] = np.sum(A[i, :])

    sdfg = row_sums.to_sdfg()
    rednode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
    rednode.implementation = "pure-seq"

    A = np.random.rand(10, 4)
    B = np.zeros(10)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.sum(axis=1))


_impls = ["pure", "CUDA (device)", "pure-seq", "GPUAuto"]
_case_params = [
    ([1, 64, 60, 60], (0, 2, 3), [64], np.float32),
    ([8, 512, 4096], (0, 1), [4096], np.float32),
    ([8, 512, 4096], (0, 1), [4096], np.float64),
    ([1024, 8], (0), [8], np.float32),
    ([111, 111, 111], (0, 1), [111], np.float64),
    ([111, 111, 111], (1, 2), [111], np.float64),
    ([1000000], (0), [1], np.float64),
    ([1111111], (0), [1], np.float64),
    ([123, 21, 26, 8], (1, 2), [123, 8], np.float32),
    ([2, 512, 2], (0, 2), [512], np.float32),
    ([512, 555, 257], (0, 2), [555], np.float64),
]


@pytest.mark.gpu
@pytest.mark.parametrize("impl,test_case", itertools.product(_impls, _case_params))
def test_multidim_gpu(impl, test_case):
    in_shape, ax, out_shape, dtype = test_case
    print(in_shape, ax, out_shape, dtype)
    axes = ax

    @dace.program
    def multidimred(a, b):
        b[:] = np.sum(a, axis=axes)

    a = np.random.rand(*in_shape).astype(dtype)
    b = np.random.rand(*out_shape).astype(dtype)
    sdfg = multidimred.to_sdfg(a, b)
    # One build folder per case: parallel workers compiling the same name overwrite each other's library.
    sdfg.name = f"multidimred_{_impls.index(impl)}_{_case_params.index(test_case)}"
    sdfg.apply_gpu_transformations()
    rednode = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, std.Reduce))
    rednode.implementation = impl

    sdfg(a, b)

    assert np.allclose(b, np.sum(a, axis=axes))


def device_reduce_through_connectors() -> dace.SDFG:
    """A top-level ``CUDA (device)`` sum wired through the node's own ``_in``/``_out`` connectors."""
    sdfg = dace.SDFG("device_reduce_through_connectors")
    sdfg.add_array("A", [64], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("out", [1], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    reduce = state.add_reduce("lambda a, b: a + b", None, 0)
    reduce.implementation = "CUDA (device)"
    reduce.schedule = dace.ScheduleType.GPU_Device
    reduce.add_in_connector("_in")
    reduce.add_out_connector("_out")
    state.add_edge(state.add_read("A"), None, reduce, "_in", dace.Memlet("A[0:64]"))
    state.add_edge(reduce, "_out", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def test_device_reduce_expands_with_typed_connectors():
    """The expansion read the output type with ``next`` on a dict view, a TypeError once the connector exists."""
    sdfg = device_reduce_through_connectors()
    sdfg.expand_library_nodes()
    assert not any(isinstance(n, std.Reduce) for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.gpu
def test_device_reduce_through_connectors_computes_the_sum():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    sdfg = device_reduce_through_connectors()
    A = cupy.asarray(np.random.default_rng(0).random(64))
    out = cupy.zeros(1)
    sdfg(A=A, out=out)
    assert np.allclose(out.get(), A.get().sum())


if __name__ == "__main__":
    for params in itertools.product(_impls, _case_params):
        test_multidim_gpu(params[0], params[1])
    test_library_node_expand_reduce_pure()
    test_pure_seq_row_sums_in_a_map()

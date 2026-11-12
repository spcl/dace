# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The pure-only standard library nodes run inside a GPU kernel as one thread block.

One kernel iteration per row: a whole-row reduction (``Norm2``, ``ArgMax`` / ``ArgMin``, ``ALL`` / ``ANY`` /
``COUNT``) becomes a ``GPU_ThreadBlock`` lane map around one ``gpucub::BlockReduce``; an element-wise node
(``Broadcast``, ``CShift``, ``Merge``) becomes its ``pure`` map strided over the same lanes. A reduction along one
axis and a node writing a thread's own copy keep their per-thread ``pure`` lowering.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.libraries.standard.block_reduce import gpu_block_implementation, lane_map_nodes
from dace.libraries.standard.nodes.argminmax import ArgMax, ArgMin
from dace.libraries.standard.nodes.broadcast import Broadcast
from dace.libraries.standard.nodes.cshift import CShift, ShiftDirection
from dace.libraries.standard.nodes.mask_reduction import AllNode, AnyNode, CountLibraryNode
from dace.libraries.standard.nodes.merge_node import MergeLibraryNode
from dace.libraries.standard.nodes.norm2 import Norm2
from dace.sdfg import nodes

R = dace.symbol("R", dtype=dace.int64)
N = dace.symbol("N", dtype=dace.int64)
P = dace.symbol("P", dtype=dace.int64)
ROWS, COLS, PANEL = 5, 1000, 3


def row_kernel(name: str, node: nodes.LibraryNode, arrays: dict, inputs: dict, outputs: dict):
    """A ``GPU_Device`` map over rows ``i`` holding ``node``, wired to ``GPU_Global`` ``arrays`` (name -> (shape,
    dtype)) by the given connector memlets."""
    sdfg = dace.SDFG(name)
    for array, (shape, dtype) in arrays.items():
        sdfg.add_array(array, shape, dtype, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("rows", {"i": "0:R"}, schedule=dtypes.ScheduleType.GPU_Device)
    state.add_node(node)
    for conn, subset in inputs.items():
        read = state.add_read(subset.split("[")[0])
        state.add_memlet_path(read, entry, node, dst_conn=conn, memlet=dace.Memlet(subset))
    for conn, subset in outputs.items():
        write = state.add_write(subset.split("[")[0])
        state.add_memlet_path(node, exit_node, write, src_conn=conn, memlet=dace.Memlet(subset))
    return sdfg, state


def norm2_kernel():
    arrays = {"A": ([R, N], dace.float64), "out": ([R], dace.float64)}
    return row_kernel("norm2_rows", Norm2("norm2"), arrays, {"_x": "A[i, 0:N]"}, {"_out": "out[i]"})


def extreme_index_kernel(node_class=ArgMax, back=False, mask=False):
    arrays = {"A": ([R, N], dace.float64), "M": ([R, N], dace.int32), "idx": ([R, 2], dace.int64)}
    inputs = {"_x": "A[i, 0:N]", **({"_mask": "M[i, 0:N]"} if mask else {})}
    node = node_class("extreme", back=back, mask=mask)
    return row_kernel(f"{node_class.__name__}_rows", node, arrays, inputs, {"_idx": "idx[i, 0:2]"})


def mask_kernel(node_class=AllNode):
    out_type = dace.int32 if node_class is CountLibraryNode else dace.bool_
    node = node_class("fold")
    arrays = {"M": ([R, N], dace.int32), "out": ([R], out_type)}
    inputs = {node.INPUT_CONNECTOR_NAME: "M[i, 0:N]"}
    return row_kernel(f"{node_class.__name__}_rows", node, arrays, inputs, {node.OUTPUT_CONNECTOR_NAME: "out[i]"})


def broadcast_kernel():
    arrays = {"x": ([R], dace.float64), "B": ([R, N], dace.float64)}
    node = Broadcast("spread", dim=None)
    return row_kernel("broadcast_rows", node, arrays, {"_src": "x[i]"}, {"_dst": "B[i, 0:N]"})


def cshift_kernel():
    arrays = {"A": ([R, N], dace.float64), "B": ([R, N], dace.float64)}
    node = CShift("roll", dim=2, shift=3, direction=ShiftDirection.NUMPY)
    return row_kernel("cshift_rows", node, arrays, {"_x": "A[i, 0:N]"}, {"_out": "B[i, 0:N]"})


def cshift_cube_kernel():
    """Each row a 3 x 5 tile: no axis alone comes near the block's width."""
    arrays = {"A": ([R, 3, 5], dace.float64), "B": ([R, 3, 5], dace.float64)}
    node = CShift("roll", dim=3, shift=2, direction=ShiftDirection.NUMPY)
    return row_kernel("cshift_cube_rows", node, arrays, {"_x": "A[i, 0:3, 0:5]"}, {"_out": "B[i, 0:3, 0:5]"})


def merge_kernel():
    arrays = {"A": ([R, N], dace.float64), "C": ([R, N], dace.float64), "M": ([R, N], dace.int32)}
    arrays["B"] = ([R, N], dace.float64)
    inputs = {"_mrg_t": "A[i, 0:N]", "_mrg_f": "C[i, 0:N]", "_mrg_mask": "M[i, 0:N]"}
    return row_kernel("merge_rows", MergeLibraryNode("where"), arrays, inputs, {"_mrg_out": "B[i, 0:N]"})


def tensordot_kernel():
    arrays = {"L": ([R, N], dace.float64), "W": ([N, P], dace.float64), "O": ([R, P], dace.float64)}
    node = TensorDot("contract", left_axes=[1], right_axes=[0])
    inputs = {"_left_tensor": "L[i, 0:N]", "_right_tensor": "W[0:N, 0:P]"}
    return row_kernel("tensordot_rows", node, arrays, inputs, {"_out_tensor": "O[i, 0:P]"})


REDUCTIONS = [
    norm2_kernel,
    extreme_index_kernel,
    lambda: extreme_index_kernel(ArgMin),
    lambda: mask_kernel(AllNode),
    lambda: mask_kernel(AnyNode),
    lambda: mask_kernel(CountLibraryNode),
]
ELEMENTWISE = [broadcast_kernel, cshift_kernel, merge_kernel]


def library_node(state: dace.SDFGState) -> nodes.LibraryNode:
    return next(n for n in state.nodes() if isinstance(n, nodes.LibraryNode))


def lane_maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]


@pytest.mark.parametrize("build", REDUCTIONS)
def test_an_in_kernel_reduction_lowers_to_one_block_collective(build):
    sdfg, state = build()
    node = library_node(state)
    sdfg.expand_library_nodes(recursive=False)
    assert node.implementation == "CUDA (block)"
    assert [lane.map.range.size()[0] for lane in lane_maps(sdfg)] == [256]
    (tasklet,) = [n for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    assert "gpucub::BlockReduce<" in tasklet.code.as_string and ", 256>" in tasklet.code.as_string


@pytest.mark.parametrize("build", ELEMENTWISE)
def test_an_in_kernel_elementwise_node_strides_its_map_over_the_lanes(build):
    sdfg, state = build()
    node = library_node(state)
    sdfg.expand_library_nodes(recursive=False)
    assert node.implementation == "CUDA (block)"
    (lanes,) = lane_maps(sdfg)
    assert lanes.map.range.size()[0] == 256 and lanes.map.gpu_force_syncthreads
    inner = [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.Sequential
    ]
    (elements,) = inner
    begin, end, step = elements.map.range[-1]
    assert str(begin) == "__lane" and step == 256


def test_an_in_kernel_map_of_narrow_axes_strides_its_whole_iteration_space_over_the_lanes():
    sdfg, state = cshift_cube_kernel()
    sdfg.expand_library_nodes(recursive=False)
    (elements,) = [
        n
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.Sequential
    ]
    ((begin, end, step),) = elements.map.range
    assert len(elements.map.params) == 1 and str(begin) == "__lane" and end == 14 and step == 256
    sdfg.validate()


def test_an_in_kernel_contraction_strides_both_of_its_maps_over_the_lanes():
    """The zero fill and the contraction each get a lane map; the barrier between them orders the two."""
    sdfg, state = tensordot_kernel()
    node = library_node(state)
    node.implementation = "CUDA (block)"
    sdfg.expand_library_nodes(recursive=False)
    lanes = lane_maps(sdfg)
    assert len(lanes) == 2 and all(lane.map.gpu_force_syncthreads for lane in lanes)


def test_the_lowered_nodes_opt_into_kernels():
    """A block lowering is what lets the offloader keep the map around the node a kernel."""
    for build in REDUCTIONS + ELEMENTWISE + [tensordot_kernel]:
        sdfg, state = build()
        assert gpu_block_implementation(library_node(state), state, sdfg) == "CUDA (block)"
    assert {Broadcast, CShift, MergeLibraryNode, TensorDot} <= set(lane_map_nodes())


def test_a_reduction_along_one_axis_keeps_its_per_thread_lowering():
    """The block lowering folds the whole array; ``dim`` keeps a result per remaining coordinate."""
    sdfg, state = norm2_kernel()
    node = library_node(state)
    node.dim = 2
    assert gpu_block_implementation(node, state, sdfg) is None
    sdfg.expand_library_nodes(recursive=False)
    assert node.implementation != "CUDA (block)" and not lane_maps(sdfg)


def test_an_elementwise_node_writing_a_threads_own_copy_keeps_its_per_thread_lowering():
    """Lanes splitting a per-thread transient would each fill only their share of it."""
    sdfg, state = cshift_kernel()
    sdfg.arrays["B"].transient = True
    sdfg.arrays["B"].storage = dtypes.StorageType.Register
    node = library_node(state)
    sdfg.expand_library_nodes(recursive=False)
    assert node.implementation != "CUDA (block)" and not lane_maps(sdfg)


def test_a_host_level_node_keeps_its_pure_lowering():
    sdfg = dace.SDFG("norm2_host")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state()
    node = Norm2("norm2")
    state.add_edge(state.add_read("A"), None, node, "_x", dace.Memlet("A[0:N]"))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.expand_library_nodes(recursive=False)
    assert not lane_maps(sdfg)
    a = np.random.default_rng(0).random(100)
    out = np.zeros(1)
    sdfg(A=a, out=out, N=100)
    assert np.allclose(out[0], np.linalg.norm(a))


def test_the_offloader_keeps_a_row_loop_around_a_norm_a_kernel():
    """Without a block lowering the map holding the node would be pinned to the host."""
    from dace.transformation.passes.canonicalize.finalize import offload_to_gpu
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    @dace.program
    def row_norms(a: dace.float64[R, N], out: dace.float64[R]):
        for i in dace.map[0:R]:
            dace.libraries.standard.norm2(a[i], out[i : i + 1])

    sdfg = row_norms.to_sdfg(simplify=True)
    canonicalize(sdfg, target="gpu", validate_all=False)
    offload_to_gpu(sdfg)
    ((node, state),) = [(n, st) for n, st in sdfg.all_nodes_recursive() if isinstance(n, Norm2)]
    scopes = [
        s.map.schedule
        for s in dace.transformation.helpers.get_parent_map_and_loop_scopes(state.sdfg, node, state)
        if isinstance(s, nodes.MapEntry)
    ]
    assert dtypes.ScheduleType.GPU_Device in scopes


def run_on_gpu(sdfg: dace.SDFG, **arrays) -> dict:
    import cupy

    device = {name: cupy.asarray(value) for name, value in arrays.items()}
    sdfg(**device, R=ROWS, N=COLS, P=PANEL)
    return {name: value.get() for name, value in device.items()}


def mask_rows(rng: np.random.Generator) -> np.ndarray:
    m = (rng.random((ROWS, COLS)) < 0.5).astype(np.int32) * 3  # any non-zero value is true
    m[1, :] = 1
    m[2, :] = 0
    m[3, :] = 0
    m[3, 777] = -2
    return m


@pytest.mark.gpu
def test_row_norm2_in_a_kernel_matches_numpy():
    a = np.random.default_rng(1).random((ROWS, COLS))
    out = run_on_gpu(norm2_kernel()[0], A=a, out=np.zeros(ROWS))["out"]
    assert np.allclose(out, np.linalg.norm(a, axis=1))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "node_class, back, mask", [(ArgMax, False, False), (ArgMin, True, False), (ArgMax, False, True)]
)
def test_row_extreme_index_in_a_kernel_matches_numpy(node_class, back, mask):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 9, (ROWS, COLS)).astype(np.float64)  # many ties: back picks the last, else the first
    m = mask_rows(rng)
    m[2, 5] = 1  # every row selects something
    sdfg, state = extreme_index_kernel(node_class, back, mask)
    idx = run_on_gpu(sdfg, A=a, M=m, idx=np.zeros((ROWS, 2), dtype=np.int64))["idx"]
    pick = np.argmax if node_class is ArgMax else np.argmin
    filler = -np.inf if node_class is ArgMax else np.inf
    values = np.where(m != 0, a, filler) if mask else a
    expected = [
        COLS - 1 - pick(row[::-1]) if back else pick(row) for row in values
    ]  # reversed search finds the last extreme
    assert idx[:, 0].tolist() == [1] * ROWS
    assert (idx[:, 1] - 1).tolist() == expected


@pytest.mark.gpu
@pytest.mark.parametrize("node_class", [AllNode, AnyNode, CountLibraryNode])
def test_row_mask_reduction_in_a_kernel_matches_numpy(node_class):
    m = mask_rows(np.random.default_rng(3))
    sdfg, state = mask_kernel(node_class)
    out_type = np.int32 if node_class is CountLibraryNode else np.bool_
    out = run_on_gpu(sdfg, M=m, out=np.zeros(ROWS, dtype=out_type))["out"]
    fold = {AllNode: np.all, AnyNode: np.any, CountLibraryNode: np.count_nonzero}[node_class]
    assert out.tolist() == [fold(row != 0) for row in m]


@pytest.mark.gpu
def test_row_broadcast_in_a_kernel_matches_numpy():
    x = np.arange(ROWS, dtype=np.float64) + 0.5
    b = run_on_gpu(broadcast_kernel()[0], x=x, B=np.zeros((ROWS, COLS)))["B"]
    assert np.array_equal(b, np.broadcast_to(x[:, None], (ROWS, COLS)))


@pytest.mark.gpu
def test_row_cshift_in_a_kernel_matches_numpy():
    a = np.random.default_rng(4).random((ROWS, COLS))
    b = run_on_gpu(cshift_kernel()[0], A=a, B=np.zeros((ROWS, COLS)))["B"]
    assert np.array_equal(b, np.roll(a, 3, axis=1))


@pytest.mark.gpu
def test_row_cshift_of_a_cube_in_a_kernel_matches_numpy():
    a = np.random.default_rng(7).random((ROWS, 3, 5))
    b = run_on_gpu(cshift_cube_kernel()[0], A=a, B=np.zeros((ROWS, 3, 5)))["B"]
    assert np.array_equal(b, np.roll(a, 2, axis=2))


@pytest.mark.gpu
def test_row_merge_in_a_kernel_matches_numpy():
    rng = np.random.default_rng(5)
    a, c, m = rng.random((ROWS, COLS)), rng.random((ROWS, COLS)), mask_rows(rng)
    b = run_on_gpu(merge_kernel()[0], A=a, C=c, M=m, B=np.zeros((ROWS, COLS)))["B"]
    assert np.array_equal(b, np.where(m != 0, a, c))


@pytest.mark.gpu
def test_row_tensordot_in_a_kernel_matches_numpy():
    rng = np.random.default_rng(6)
    left, weights = rng.random((ROWS, COLS)), rng.random((COLS, PANEL))
    sdfg, state = tensordot_kernel()
    library_node(state).implementation = "CUDA (block)"
    out = run_on_gpu(sdfg, L=left, W=weights, O=np.full((ROWS, PANEL), 7.0))["O"]
    assert np.allclose(out, left @ weights)


if __name__ == "__main__":
    for build in REDUCTIONS:
        test_an_in_kernel_reduction_lowers_to_one_block_collective(build)
    for build in ELEMENTWISE:
        test_an_in_kernel_elementwise_node_strides_its_map_over_the_lanes(build)
    test_an_in_kernel_map_of_narrow_axes_strides_its_whole_iteration_space_over_the_lanes()
    test_an_in_kernel_contraction_strides_both_of_its_maps_over_the_lanes()
    test_the_lowered_nodes_opt_into_kernels()
    test_a_reduction_along_one_axis_keeps_its_per_thread_lowering()
    test_an_elementwise_node_writing_a_threads_own_copy_keeps_its_per_thread_lowering()
    test_a_host_level_node_keeps_its_pure_lowering()
    test_the_offloader_keeps_a_row_loop_around_a_norm_a_kernel()
    test_row_norm2_in_a_kernel_matches_numpy()
    for args in [(ArgMax, False, False), (ArgMin, True, False), (ArgMax, False, True)]:
        test_row_extreme_index_in_a_kernel_matches_numpy(*args)
    for node_class in (AllNode, AnyNode, CountLibraryNode):
        test_row_mask_reduction_in_a_kernel_matches_numpy(node_class)
    test_row_broadcast_in_a_kernel_matches_numpy()
    test_row_cshift_in_a_kernel_matches_numpy()
    test_row_cshift_of_a_cube_in_a_kernel_matches_numpy()
    test_row_merge_in_a_kernel_matches_numpy()
    test_row_tensordot_in_a_kernel_matches_numpy()

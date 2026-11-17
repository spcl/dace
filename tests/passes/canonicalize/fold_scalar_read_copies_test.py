# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scalar read copy folds into its tasklet exactly when no write can land between the copy and the read."""

import numpy as np

import dace
from dace import data
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.canonicalize.fold_scalar_read_copies import FoldScalarReadCopies

N = dace.symbol("N")


def scalar_copies(sdfg: dace.SDFG) -> int:
    return sum(
        1
        for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.AccessNode)
        and isinstance(state.sdfg.arrays[node.data], data.Scalar)
        and state.sdfg.arrays[node.data].transient
        and state.in_degree(node) == 1
        and state.in_edges(node)[0].data.data != node.data
    )


def folded(program, **arrays):
    sdfg = program.to_sdfg(simplify=True)
    before = scalar_copies(sdfg)
    FoldScalarReadCopies().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg(**arrays)
    return before, scalar_copies(sdfg)


@dace.program
def doubled(A: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = A[i] * 2.0


@dace.program
def rowsum(B: dace.float64[N, N], out: dace.float64[N]):
    for k in range(N):
        for i in range(N):
            out[i] = out[i] + B[k, i]


@dace.program
def recurrence(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i + 1] = a[i] + b[i]


@dace.program
def written_then_read(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
        c[i] = b[i] + 1.0


def test_a_read_only_copy_folds():
    A, out = np.random.rand(8), np.zeros(8)
    before, after = folded(doubled, A=A, out=out, N=8)
    assert before > after
    assert np.allclose(out, 2 * A)


def test_a_same_element_update_folds():
    B, out = np.random.rand(6, 6), np.zeros(6)
    before, after = folded(rowsum, B=B, out=out, N=6)
    assert before > after
    assert np.allclose(out, B.sum(axis=0))


def test_a_recurrence_whose_write_follows_the_read_folds():
    a, b = np.random.rand(8), np.random.rand(8)
    expected = a.copy()
    for i in range(7):
        expected[i + 1] = expected[i] + b[i]
    before, after = folded(recurrence, a=a, b=b, N=8)
    assert before > after
    assert np.allclose(a, expected)


def test_a_copy_out_of_the_node_its_writer_fills_folds():
    a, b, c = np.random.rand(8), np.zeros(8), np.zeros(8)
    before, after = folded(written_then_read, a=a, b=b, c=c, N=8)
    assert before > 0 and after == 0
    assert np.allclose(b, 2 * a) and np.allclose(c, 2 * a + 1)


def copy_beside_a_writer(ordered: bool) -> dace.SDFG:
    """``A[0] -> s -> t1 -> B[0]`` beside ``t2 -> A[0]``; with ``ordered`` the write follows ``t1``."""
    sdfg = dace.SDFG("copy_beside_a_writer" + ("_ordered" if ordered else ""))
    sdfg.add_array("A", [2], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state()
    s = state.add_access("s")
    state.add_edge(state.add_read("A"), None, s, None, dace.Memlet("A[0]"))
    t1 = state.add_tasklet("use", {"x"}, {"y"}, "y = x")
    state.add_edge(s, None, t1, "x", dace.Memlet("s[0]"))
    b_node = state.add_write("B")
    state.add_edge(t1, "y", b_node, None, dace.Memlet("B[0]"))
    t2 = state.add_tasklet("clobber", {}, {"z"}, "z = 7.0")
    state.add_edge(t2, "z", state.add_write("A"), None, dace.Memlet("A[0]"))
    if ordered:
        state.add_nedge(b_node, t2, dace.Memlet())
    return sdfg


def test_a_copy_beside_an_unordered_writer_stays():
    sdfg = copy_beside_a_writer(ordered=False)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_copy_ordering_the_overwrite_through_the_snapshot_stays():
    """CloudSC's ``zqx`` shape: only the snapshot's empty edge orders the clobber after the read, so dropping the
    snapshot would leave ``use`` and ``clobber`` unordered."""
    sdfg = copy_beside_a_writer(ordered=False)
    state = sdfg.start_state
    snapshot = next(node for node in state.data_nodes() if node.data == "s")
    clobber = next(node for node in state.nodes() if isinstance(node, nodes.Tasklet) and node.label == "clobber")
    state.add_nedge(snapshot, clobber, dace.Memlet())
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_copy_whose_writer_follows_the_read_folds():
    sdfg = copy_beside_a_writer(ordered=True)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()


def test_a_scalar_read_again_in_another_state_stays():
    sdfg = copy_beside_a_writer(ordered=True)
    later = sdfg.add_state_after(sdfg.start_state)
    t = later.add_tasklet("again", {"x"}, {"y"}, "y = x")
    later.add_edge(later.add_read("s"), None, t, "x", dace.Memlet("s[0]"))
    later.add_edge(t, "y", later.add_write("B"), None, dace.Memlet("B[0]"))
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_scalar_a_branch_condition_reads_stays():
    """cp2k_density_matrix_trs4: ``if s < ...`` read the scalar the fold removed, and C++ saw an undeclared name."""
    sdfg = copy_beside_a_writer(ordered=True)
    start = sdfg.start_state
    branch = ConditionalBlock("branch")
    sdfg.add_node(branch)
    sdfg.add_edge(start, branch, dace.InterstateEdge())
    body = ControlFlowRegion("body", sdfg=sdfg)
    body.add_state("taken", is_start_block=True)
    branch.add_branch(CodeBlock("s > 0.5"), body)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_scalar_an_interstate_edge_reads_stays():
    sdfg = copy_beside_a_writer(ordered=True)
    sdfg.add_state_after(sdfg.start_state, assignments={"k": "s"})
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def result_copy_beside_a_reader(ordered: bool) -> dace.SDFG:
    """``t -> r -> A[0]`` beside ``A[0] -> peek -> B[0]``; with ``ordered`` the reader runs first."""
    sdfg = dace.SDFG("result_copy_beside_a_reader" + ("_ordered" if ordered else ""))
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_scalar("r", dace.float64, transient=True)
    state = sdfg.add_state()
    produce = state.add_tasklet("produce", {}, {"o"}, "o = 3.0")
    r = state.add_access("r")
    state.add_edge(produce, "o", r, None, dace.Memlet("r[0]"))
    state.add_edge(r, None, state.add_write("A"), None, dace.Memlet("r[0] -> [0]"))
    peek = state.add_tasklet("peek", {"x"}, {"y"}, "y = x")
    state.add_edge(state.add_read("A"), None, peek, "x", dace.Memlet("A[0]"))
    b_node = state.add_write("B")
    state.add_edge(peek, "y", b_node, None, dace.Memlet("B[0]"))
    if ordered:
        state.add_nedge(b_node, produce, dace.Memlet())
    return sdfg


def test_a_result_copy_beside_an_unordered_reader_stays():
    sdfg = result_copy_beside_a_reader(ordered=False)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_result_copy_whose_reader_runs_first_folds():
    sdfg = result_copy_beside_a_reader(ordered=True)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert "r" not in {node.data for node in sdfg.start_state.data_nodes()}


def test_a_copy_out_of_a_view_bound_to_the_scalar_alone_stays():
    """velocity_tendencies' ``max_value`` shape: ``produce -> v -views-> s -> use``; ``v`` views ``s`` and
    nothing else, so dropping ``s`` would leave the view without data."""
    sdfg = dace.SDFG("copy_out_of_a_view_bound_to_the_scalar")
    sdfg.add_view("v", [1], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    produce = state.add_tasklet("produce", {}, {"o"}, "o = 3.0")
    v = state.add_access("v")
    state.add_edge(produce, "o", v, None, dace.Memlet("v[0]"))
    s = state.add_access("s")
    state.add_edge(v, "views", s, None, dace.Memlet("s[0]"))
    use = state.add_tasklet("use", {"x"}, {"y"}, "y = x")
    state.add_edge(s, None, use, "x", dace.Memlet("s[0]"))
    state.add_edge(use, "y", state.add_write("B"), None, dace.Memlet("B[0]"))
    sdfg.validate()
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None
    sdfg.validate()


def nested_store(name: str) -> dace.SDFG:
    """``y[i] = x`` with the scalar ``x`` a nested SDFG's own descriptor."""
    inner = dace.SDFG(name)
    inner.add_scalar("x", dace.float64)
    inner.add_array("y", [N], dace.float64)
    inner.add_symbol("i", dace.int64)
    state = inner.add_state()
    store = state.add_tasklet("store", {"a"}, {"b"}, "b = a")
    state.add_edge(state.add_read("x"), None, store, "a", dace.Memlet("x[0]"))
    state.add_edge(store, "b", state.add_write("y"), None, dace.Memlet("y[i]"))
    return inner


def broadcast_by_a_map(
    name: str,
    writes_source: bool = False,
    nested: bool = False,
    storage: dace.StorageType = dace.StorageType.Default,
    schedule: dace.ScheduleType = dace.ScheduleType.Default,
) -> dace.SDFG:
    """``A[1] -> s -> map i: B[i] = s`` (fdtd_2d's ``ey[0, :] = fict[t]``); with ``writes_source`` the map also stores
    ``A[1]``, with ``nested`` a nested SDFG reads ``s`` as its scalar ``x``. ``A`` lives in ``storage`` and the map
    runs on ``schedule``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [2], dace.float64, storage=storage)
    sdfg.add_array("B", [N], dace.float64, storage=storage)
    sdfg.add_scalar("s", dace.float64, transient=True)
    state = sdfg.add_state()
    s = state.add_access("s")
    state.add_edge(state.add_read("A"), None, s, None, dace.Memlet("A[1]"))
    entry, exit_ = state.add_map("broadcast", dict(i="0:N"), schedule=schedule)
    if nested:
        use = state.add_nested_sdfg(nested_store(f"{name}_store"), {"x"}, {"y"}, {"N": N, "i": "i"})
        state.add_memlet_path(s, entry, use, dst_conn="x", memlet=dace.Memlet("s[0]"))
        state.add_memlet_path(use, exit_, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    else:
        use = state.add_tasklet("use", {"x"}, {"y"}, "y = x")
        state.add_memlet_path(s, entry, use, dst_conn="x", memlet=dace.Memlet("s[0]"))
        state.add_memlet_path(use, exit_, state.add_write("B"), src_conn="y", memlet=dace.Memlet("B[i]"))
    if writes_source:
        clobber = state.add_tasklet("clobber", {}, {"z"}, "z = 7.0")
        state.add_nedge(entry, clobber, dace.Memlet())
        state.add_memlet_path(clobber, exit_, state.add_write("A"), src_conn="z", memlet=dace.Memlet("A[1]"))
    sdfg.validate()
    return sdfg


def test_a_copy_broadcast_by_a_map_folds_into_its_memlet_tree():
    sdfg = broadcast_by_a_map("broadcast_fold")
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert scalar_copies(sdfg) == 0 and "s" not in {n.data for n in sdfg.start_state.data_nodes()}
    A, B = np.random.rand(2), np.zeros(5)
    sdfg(A=A, B=B, N=5)
    assert np.allclose(B, A[1])


def test_a_copy_broadcast_by_a_map_writing_its_source_stays():
    sdfg = broadcast_by_a_map("broadcast_clobbered", writes_source=True)
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) is None


def test_a_copy_broadcast_into_a_nested_sdfg_stays():
    """A nested SDFG's connector descriptor must match the container it reads: the scalar stays its input."""
    sdfg = broadcast_by_a_map("broadcast_nested", nested=True)
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) is None
    sdfg.validate()


def test_a_host_element_broadcast_into_a_kernel_stays():
    """The copy into the scalar is the element's only way onto the device: a kernel cannot read host ``A``."""
    sdfg = broadcast_by_a_map(
        "broadcast_host_into_kernel", storage=dace.StorageType.CPU_Heap, schedule=dace.ScheduleType.GPU_Device
    )
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) is None


def test_a_device_element_broadcast_into_a_kernel_folds():
    sdfg = broadcast_by_a_map(
        "broadcast_device_into_kernel", storage=dace.StorageType.GPU_Global, schedule=dace.ScheduleType.GPU_Device
    )
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) == 1
    assert scalar_copies(sdfg) == 0


def test_a_device_element_broadcast_by_a_host_map_stays():
    sdfg = broadcast_by_a_map("broadcast_device_on_host", storage=dace.StorageType.GPU_Global)
    assert FoldScalarReadCopies(scopes=True).apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_a_read_only_copy_folds()
    test_a_same_element_update_folds()
    test_a_recurrence_whose_write_follows_the_read_folds()
    test_a_copy_out_of_the_node_its_writer_fills_folds()
    test_a_copy_beside_an_unordered_writer_stays()
    test_a_copy_ordering_the_overwrite_through_the_snapshot_stays()
    test_a_copy_whose_writer_follows_the_read_folds()
    test_a_scalar_read_again_in_another_state_stays()
    test_a_scalar_a_branch_condition_reads_stays()
    test_a_scalar_an_interstate_edge_reads_stays()
    test_a_result_copy_beside_an_unordered_reader_stays()
    test_a_result_copy_whose_reader_runs_first_folds()
    test_a_copy_out_of_a_view_bound_to_the_scalar_alone_stays()
    test_a_copy_broadcast_by_a_map_folds_into_its_memlet_tree()
    test_a_copy_broadcast_by_a_map_writing_its_source_stays()
    test_a_copy_broadcast_into_a_nested_sdfg_stays()
    test_a_host_element_broadcast_into_a_kernel_stays()
    test_a_device_element_broadcast_into_a_kernel_folds()
    test_a_device_element_broadcast_by_a_host_map_stays()

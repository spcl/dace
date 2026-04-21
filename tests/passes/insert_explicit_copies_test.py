# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``InsertExplicitCopies`` pass (Layer 2).

Part 1: Artificial SDFG-builder-API tests with direct AccessNode->AccessNode
         edges and map boundary staging patterns.
Part 2: Polybench-derived functional tests that verify the pass preserves
         numerical correctness on real programs.
"""
import copy as _copy
import math

import dace
import numpy as np
import pytest
from dace import nodes
from dace.memlet import Memlet
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies


def _count_copy_nodes(sdfg):
    """Count CopyLibraryNode instances across all states (recursive)."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CopyLibraryNode))


def _count_direct_copy_edges(sdfg):
    """Count AccessNode -> AccessNode non-empty edges (recursive)."""
    count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for e in state.edges():
                if (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)
                        and not e.data.is_empty()):
                    count += 1
    return count


def _assert_no_other_subset(sdfg: dace.SDFG) -> None:
    """Postcondition: after copy-node insertion, no memlet in any state/nsdfg
    should still carry an ``other_subset`` copies are represented by ``CopyLibraryNode``."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for edge in state.edges():
                memlet = edge.data
                if memlet.is_empty():
                    continue
                assert memlet.other_subset is None, (
                    f"Memlet on edge {edge.src}->{edge.dst} in SDFG '{nsdfg.name}' still "
                    f"has other_subset={memlet.other_subset}; expected None after copy insertion.")


def _build_copy_sdfg(name, arrays, edge_memlet):
    """Build an SDFG with two AccessNodes wired by a single edge."""
    sdfg = dace.SDFG(name)
    for arr_name, shape, storage in arrays:
        sdfg.add_array(arr_name, shape, dace.float64, storage)
    st = sdfg.add_state("s")
    src = st.add_access(arrays[0][0])
    dst = st.add_access(arrays[1][0])
    st.add_edge(src, None, dst, None, edge_memlet)
    return sdfg, st, src, dst


def _assert_copy_storages(sdfg, src_storage, dst_storage):
    """Assert that every CopyLibraryNode in ``sdfg`` has the given storages."""
    found = False
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, CopyLibraryNode):
            assert n.src_storage(parent, parent.sdfg) == src_storage
            assert n.dst_storage(parent, parent.sdfg) == dst_storage
            found = True
    assert found, "No CopyLibraryNode found in SDFG"


def _compile_and_run(sdfg, inputs):
    """Expand library nodes, compile, and run with ``inputs`` as kwargs."""
    sdfg.expand_library_nodes()
    exe = sdfg.compile()
    exe(**inputs)


def test_insert_cpu_to_cpu_1d():
    """CPU_Heap -> CPU_Heap 1D copy."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("insert_cpu_cpu_1d", [("A", [100], cpu), ("B", [100], cpu)],
                                     Memlet("A[10:60]", other_subset="20:70"))

    assert _count_direct_copy_edges(sdfg) == 1
    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_direct_copy_edges(sdfg) == 0
    assert _count_copy_nodes(sdfg) == 1
    _assert_copy_storages(sdfg, cpu, cpu)

    A = np.arange(100, dtype=np.float64)
    B = np.zeros(100, dtype=np.float64)
    _compile_and_run(sdfg, dict(A=A, B=B))
    np.testing.assert_array_equal(B[20:70], A[10:60])
    assert np.all(B[:20] == 0) and np.all(B[70:] == 0)


def test_insert_cpu_to_cpu_2d_slice():
    """CPU 2D slice copy with explicit other_subset."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("insert_cpu_2d", [("A", [10, 20], cpu), ("B", [10, 20], cpu)],
                                     Memlet(data="A", subset="2:8, 5:15", other_subset="0:6, 0:10"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_direct_copy_edges(sdfg) == 0
    assert _count_copy_nodes(sdfg) == 1

    A = np.arange(200, dtype=np.float64).reshape(10, 20).copy()
    B = np.zeros((10, 20), dtype=np.float64)
    _compile_and_run(sdfg, dict(A=A, B=B))
    np.testing.assert_array_equal(B[0:6, 0:10], A[2:8, 5:15])


@pytest.mark.parametrize("sdfg_name,memlet", [
    ("insert_other_dst", Memlet(data="B", subset="0:8", other_subset="2:10")),
    ("insert_other_src", Memlet(data="A", subset="2:10", other_subset="0:8")),
],
                         ids=["data_is_dst", "data_is_src"])
def test_insert_other_subset_data_convention(sdfg_name, memlet):
    """Both memlet conventions (data=src or data=dst) must produce the same
    copy: _in=A[2:10], _out=B[0:8], no other_subset."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, st, _, _ = _build_copy_sdfg(sdfg_name, [("A", [20], cpu), ("B", [20], cpu)], memlet)

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_copy_nodes(sdfg) == 1

    for n in st.nodes():
        if isinstance(n, CopyLibraryNode):
            in_m = list(st.in_edges(n))[0].data
            out_m = list(st.out_edges(n))[0].data
            assert in_m.data == "A" and str(in_m.subset) == "2:10"
            assert in_m.other_subset is None
            assert out_m.data == "B" and str(out_m.subset) == "0:8"
            assert out_m.other_subset is None
            break

    A = np.arange(20, dtype=np.float64)
    B = np.full(20, -1.0, dtype=np.float64)
    _compile_and_run(sdfg, dict(A=A, B=B))
    np.testing.assert_array_equal(B[0:8], A[2:10])
    assert np.all(B[8:] == -1.0)


def test_insert_cpu_to_cpu_full_array():
    """Full array copy."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("insert_full", [("A", [64], cpu), ("B", [64], cpu)], Memlet("A[0:64]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    A = np.arange(64, dtype=np.float64)
    B = np.zeros(64, dtype=np.float64)
    _compile_and_run(sdfg, dict(A=A, B=B))
    np.testing.assert_array_equal(B, A)


def test_insert_multiple_copies_same_state():
    """Two copies in the same state: A->B and A->C."""
    sdfg = dace.SDFG("insert_multi")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, [32], dace.float64, dace.StorageType.CPU_Heap)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    c = st.add_access("C")
    st.add_edge(a, None, b, None, Memlet("A[0:32]"))
    st.add_edge(a, None, c, None, Memlet("A[0:32]"))

    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result == 2
    assert _count_copy_nodes(sdfg) == 2

    A = np.arange(32, dtype=np.float64)
    B = np.zeros(32, dtype=np.float64)
    C = np.zeros(32, dtype=np.float64)
    _compile_and_run(sdfg, dict(A=A, B=B, C=C))
    np.testing.assert_array_equal(B, A)
    np.testing.assert_array_equal(C, A)


def test_insert_empty_memlet_skipped():
    """Empty memlets (control edges) are not replaced."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("insert_empty", [("A", [10], cpu), ("B", [10], cpu)], Memlet())

    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result is None
    assert _count_copy_nodes(sdfg) == 0


def test_insert_no_copies_returns_none():
    """If there are no copy edges, return None."""
    sdfg = dace.SDFG("no_copies")
    sdfg.add_array("A", [10], dace.float64, dace.StorageType.CPU_Heap)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    t = st.add_tasklet("noop", {"_in"}, {"_out"}, "_out = _in + 1")
    a2 = st.add_access("A")
    st.add_edge(a, None, t, "_in", Memlet("A[0]"))
    st.add_edge(t, "_out", a2, None, Memlet("A[0]"))

    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result is None


def test_insert_nested_sdfg():
    """Copy inside a nested SDFG is also replaced."""
    inner = dace.SDFG("inner")
    inner.add_array("X", [20], dace.float64, dace.StorageType.CPU_Heap)
    inner.add_array("Y", [20], dace.float64, dace.StorageType.CPU_Heap)
    ist = inner.add_state("is")
    x = ist.add_access("X")
    y = ist.add_access("Y")
    ist.add_edge(x, None, y, None, Memlet("X[0:20]"))

    outer = dace.SDFG("outer")
    outer.add_array("A", [20], dace.float64, dace.StorageType.CPU_Heap)
    outer.add_array("B", [20], dace.float64, dace.StorageType.CPU_Heap)
    ost = outer.add_state("os")
    nsdfg = ost.add_nested_sdfg(inner, {"X"}, {"Y"})
    a = ost.add_access("A")
    b = ost.add_access("B")
    ost.add_edge(a, None, nsdfg, "X", Memlet("A[0:20]"))
    ost.add_edge(nsdfg, "Y", b, None, Memlet("B[0:20]"))

    result = InsertExplicitCopies().apply_pass(outer, {})
    _assert_no_other_subset(outer)
    assert result == 1
    assert _count_copy_nodes(outer) == 1


def test_insert_validates_after_pass():
    """SDFG passes validation after InsertExplicitCopies."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("validate_after", [("A", [100], cpu), ("B", [100], cpu)], Memlet("A[0:100]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    sdfg.validate()


def test_insert_map_staging_copies():
    """Map with transient staging buffers:
    AccessNode(A) -> MapEntry -> AccessNode(local_in) -> ... -> AccessNode(local_out) -> MapExit -> AccessNode(B)
    should insert CopyNodes at the map boundaries:
    AccessNode(A) -> MapEntry -> CopyNode -> AccessNode(local_in) -> ... -> AccessNode(local_out) -> CopyNode -> MapExit -> AccessNode(B)

    The compute Tasklet must NOT be replaced."""
    N = 64
    TILE = 8
    sdfg = dace.SDFG("map_staging")
    sdfg.add_array("A", [N], dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_array("B", [N], dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_transient("local_in", [TILE], dace.float64)
    sdfg.add_transient("local_out", [TILE], dace.float64)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    li = st.add_access("local_in")
    lo = st.add_access("local_out")

    me, mx = st.add_map("block", {"bi": f"0:{N}:{TILE}"})
    ime, imx = st.add_map("elem", {"ti": f"0:{TILE}"})
    t = st.add_tasklet("compute", {"_in"}, {"_out"}, "_out = _in * 2.0")

    st.add_memlet_path(a, me, li, memlet=dace.Memlet(f"A[bi:bi+{TILE}]"))
    st.add_memlet_path(li, ime, t, dst_conn="_in", memlet=dace.Memlet("local_in[ti]"))
    st.add_memlet_path(t, imx, lo, src_conn="_out", memlet=dace.Memlet("local_out[ti]"))
    st.add_memlet_path(lo, mx, b, memlet=dace.Memlet(f"B[bi:bi+{TILE}]"))

    sdfg.validate()
    assert _count_copy_nodes(sdfg) == 0

    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result == 2, f"Expected 2 staging copies, got {result}"
    assert _count_copy_nodes(sdfg) == 2

    # Compute Tasklet must survive
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet) and "compute" in n.label]
    assert len(tasklets) == 1

    # Each CopyNode connects MapEntry/AccessNode on one side,
    # AccessNode/MapExit on the other
    for cn in (n for n in st.nodes() if isinstance(n, CopyLibraryNode)):
        in_types = {type(e.src).__name__ for e in st.in_edges(cn)}
        out_types = {type(e.dst).__name__ for e in st.out_edges(cn)}
        assert ("MapEntry" in in_types or "AccessNode" in in_types)
        assert ("MapExit" in out_types or "AccessNode" in out_types)

    sdfg.validate()
    sdfg.expand_library_nodes()
    exe = sdfg.compile()
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B, A * 2.0)


def test_insert_nested_map_staging_copies():
    """Nested maps: staging path crosses two map scopes on each side.

    ``A -> MapEntry_outer -> MapEntry_inner -> local_in -> compute ->
    local_out -> MapExit_inner -> MapExit_outer -> B``

    Both stage-in and stage-out should be lifted into ``CopyLibraryNode``
    instances at the innermost-scope boundary (innermost ``MapEntry`` /
    ``MapExit``), even though the outer ``AccessNode`` is two scope levels
    away.
    """
    N = 64
    TILE = 8
    sdfg = dace.SDFG("nested_map_staging")
    sdfg.add_array("A", [N], dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_array("B", [N], dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_transient("local_in", [TILE], dace.float64)
    sdfg.add_transient("local_out", [TILE], dace.float64)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    li = st.add_access("local_in")
    lo = st.add_access("local_out")

    me_o, mx_o = st.add_map("outer", {"bi": f"0:{N}:{TILE}"})
    me_i, mx_i = st.add_map("inner", {"_": "0:1"})

    # Stage-in crosses both outer and inner map entries to reach ``local_in``.
    st.add_memlet_path(a, me_o, me_i, li, memlet=dace.Memlet(f"A[bi:bi+{TILE}]"))

    # Per-element compute inside the inner scope via an element-wise map.
    ime, imx = st.add_map("elem", {"ti": f"0:{TILE}"})
    t = st.add_tasklet("compute", {"_in"}, {"_out"}, "_out = _in * 2.0")
    st.add_memlet_path(li, ime, t, dst_conn="_in", memlet=dace.Memlet("local_in[ti]"))
    st.add_memlet_path(t, imx, lo, src_conn="_out", memlet=dace.Memlet("local_out[ti]"))

    # Stage-out mirrors back through both map exits to ``B``.
    st.add_memlet_path(lo, mx_i, mx_o, b, memlet=dace.Memlet(f"B[bi:bi+{TILE}]"))

    sdfg.validate()
    assert _count_copy_nodes(sdfg) == 0

    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result == 2, f"Expected 2 staging copies across nested maps, got {result}"
    assert _count_copy_nodes(sdfg) == 2

    # Compute tasklet must survive.
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet) and "compute" in n.label]
    assert len(tasklets) == 1

    # Each inserted copy must sit at the innermost-scope boundary.
    for cn in (n for n in st.nodes() if isinstance(n, CopyLibraryNode)):
        in_types = {type(e.src).__name__ for e in st.in_edges(cn)}
        out_types = {type(e.dst).__name__ for e in st.out_edges(cn)}
        assert "MapEntry" in in_types or "AccessNode" in in_types
        assert "MapExit" in out_types or "AccessNode" in out_types

    sdfg.validate()
    sdfg.expand_library_nodes()
    exe = sdfg.compile()
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    exe(A=A, B=B)
    np.testing.assert_array_equal(B, A * 2.0)


def test_insert_staging_requires_outer_access_node():
    """Stage-in pattern must have an AccessNode feeding MapEntry on the
    matching connector; a transient inside the map with no outer
    AccessNode should NOT trigger the staging rewrite."""
    N = 32
    sdfg = dace.SDFG("no_staging_no_outer_an")
    sdfg.add_array("B", [N], dace.float64, dace.StorageType.CPU_Heap)
    sdfg.add_transient("local", [1], dace.float64)

    st = sdfg.add_state("s")
    b = st.add_access("B")
    lo = st.add_access("local")
    me, mx = st.add_map("m", {"i": f"0:{N}"})
    # Tasklet writes a constant to a transient; no AccessNode feeds MapEntry.
    t = st.add_tasklet("produce", set(), {"_out"}, "_out = 3.14")
    # Wire: me (entry, no incoming data) -> tasklet -> local -> mx -> B
    st.add_edge(me, None, t, None, dace.Memlet())
    st.add_edge(t, "_out", lo, None, dace.Memlet("local[0]"))
    st.add_memlet_path(lo, mx, b, memlet=dace.Memlet("B[i]"))

    # Stage-out is a valid pattern (lo -> MapExit -> B), but stage-in must NOT
    # fire because there is no AccessNode feeding MapEntry for "local".
    result = InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)

    # Only the stage-out side should have been rewritten.
    assert result == 1, f"Expected 1 staging copy (stage-out only), got {result}"
    assert _count_copy_nodes(sdfg) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("sdfg_name,src_name,src_storage,dst_name,dst_storage,size", [
    ("insert_cpu_gpu", "H", dace.StorageType.CPU_Heap, "G", dace.StorageType.GPU_Global, 64),
    ("insert_gpu_cpu", "G", dace.StorageType.GPU_Global, "H", dace.StorageType.CPU_Heap, 64),
    ("insert_gpu_gpu", "A", dace.StorageType.GPU_Global, "B", dace.StorageType.GPU_Global, 128),
],
                         ids=["cpu_to_gpu", "gpu_to_cpu", "gpu_to_gpu"])
def test_insert_cross_storage_transfer(sdfg_name, src_name, src_storage, dst_name, dst_storage, size):
    """Structural check for cross-storage (CPU<->GPU, GPU<->GPU) transfers."""
    sdfg, _, _, _ = _build_copy_sdfg(sdfg_name, [(src_name, [size], src_storage), (dst_name, [size], dst_storage)],
                                     Memlet(f"{src_name}[0:{size}]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_copy_nodes(sdfg) == 1
    assert _count_direct_copy_edges(sdfg) == 0
    _assert_copy_storages(sdfg, src_storage, dst_storage)


# Part 2: Polybench-derived numerical correctness tests. Pattern 1 (direct
# AccessNode->AccessNode edges) is not present in polybench; Pattern 2
# (map-boundary staging) may or may not fire. Either way, output must match
# the reference.

_ = None  # needed for dace.map range syntax
datatype = dace.float64


def _run_and_compare(program, init_fn, check_arrays, sizes, name):
    """Run a DaCe program before and after InsertExplicitCopies,
    assert numerical correctness."""
    sdfg_ref = program.to_sdfg(simplify=True)
    ref_exe = sdfg_ref.compile()
    ref_arrays = init_fn(**sizes)
    ref_exe(**{k: v for k, v in ref_arrays.items()}, **sizes)
    ref_values = {k: ref_arrays[k].copy() for k in check_arrays}

    sdfg_pass = _copy.deepcopy(sdfg_ref)
    InsertExplicitCopies().apply_pass(sdfg_pass, {})
    _assert_no_other_subset(sdfg_pass)
    sdfg_pass.expand_library_nodes()
    pass_exe = sdfg_pass.compile()
    pass_arrays = init_fn(**sizes)
    pass_exe(**{k: v for k, v in pass_arrays.items()}, **sizes)

    for arr_name in check_arrays:
        np.testing.assert_allclose(pass_arrays[arr_name],
                                   ref_values[arr_name],
                                   rtol=1e-10,
                                   atol=1e-12,
                                   err_msg=f"{name}: array '{arr_name}' mismatch after pass")


NX = dace.symbol('NX')
NY = dace.symbol('NY')
TMAX = dace.symbol('TMAX')


@dace.program(datatype[NX, NY], datatype[NX, NY], datatype[NX, NY], datatype[TMAX])
def fdtd2d_v(ex, ey, hz, _fict_):
    for t in range(TMAX):

        @dace.map
        def col0(j: _[0:NY]):
            fict << _fict_[t]
            out >> ey[0, j]
            out = fict

        @dace.map
        def update_ey(i: _[1:NX], j: _[0:NY]):
            eyin << ey[i, j]
            hz1 << hz[i, j]
            hz2 << hz[i - 1, j]
            eyout >> ey[i, j]
            eyout = eyin - datatype(0.5) * (hz1 - hz2)

        @dace.map
        def update_ex(i: _[0:NX], j: _[1:NY]):
            exin << ex[i, j]
            hz1 << hz[i, j]
            hz2 << hz[i, j - 1]
            exout >> ex[i, j]
            exout = exin - datatype(0.5) * (hz1 - hz2)

        @dace.map
        def update_hz(i: _[0:NX - 1], j: _[0:NY - 1]):
            hzin << hz[i, j]
            ex1 << ex[i, j + 1]
            ex2 << ex[i, j]
            ey1 << ey[i + 1, j]
            ey2 << ey[i, j]
            hzout >> hz[i, j]
            hzout = hzin - datatype(0.7) * (ex1 - ex2 + ey1 - ey2)


def _init_fdtd2d(NX, NY, TMAX):
    nx, ny, tmax = NX, NY, TMAX
    _fict_ = np.array([np.float64(i) for i in range(tmax)])
    ex = np.zeros((nx, ny), dtype=np.float64)
    ey = np.zeros((nx, ny), dtype=np.float64)
    hz = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = np.float64(i * (j + 1)) / nx
            ey[i, j] = np.float64(i * (j + 2)) / ny
            hz[i, j] = np.float64(i * (j + 3)) / nx
    return {"ex": ex, "ey": ey, "hz": hz, "_fict_": _fict_}


def test_polybench_fdtd2d():
    _run_and_compare(fdtd2d_v, _init_fdtd2d, ["ex", "ey", "hz"], {"NX": 20, "NY": 30, "TMAX": 10}, "fdtd2d")


M_corr = dace.symbol('M_corr')
N_corr = dace.symbol('N_corr')


@dace.program(datatype[N_corr, M_corr], datatype[M_corr, M_corr], datatype[M_corr], datatype[M_corr])
def correlation_v(data, corr, mean, stddev):

    @dace.map
    def comp_mean(j: _[0:M_corr], i: _[0:N_corr]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y, 0)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M_corr]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N_corr

    @dace.map
    def comp_stddev(j: _[0:M_corr], i: _[0:N_corr]):
        inp << data[i, j]
        inmean << mean[j]
        out >> stddev(1, lambda x, y: x + y, 0)[j]
        out = (inp - inmean) * (inp - inmean)

    @dace.map
    def comp_stddev2(j: _[0:M_corr]):
        inp << stddev[j]
        out >> stddev[j]
        out = math.sqrt(inp / N_corr)
        if out <= 0.1:
            out = 1.0

    @dace.map
    def center_data(i: _[0:N_corr], j: _[0:M_corr]):
        ind << data[i, j]
        m << mean[j]
        sd << stddev[j]
        oud >> data[i, j]
        oud = (ind - m) / (math.sqrt(datatype(N_corr)) * sd)

    @dace.map
    def comp_corr_diag(i: _[0:M_corr]):
        corrout >> corr[i, i]
        corrout = 1.0

    @dace.mapscope
    def comp_corr_row(i: _[0:M_corr - 1]):

        @dace.mapscope
        def comp_corr_col(j: _[i + 1:M_corr]):

            @dace.map
            def comp_cov_k(k: _[0:N_corr]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> corr(1, lambda x, y: x + y, 0)[i, j]
                cov_ij = (indi * indj)

    @dace.mapscope
    def symmetrize(i: _[0:M_corr - 1]):

        @dace.map
        def symmetrize_col(j: _[i + 1:M_corr]):
            corrin << corr[i, j]
            corrout >> corr[j, i]
            corrout = corrin


def _init_correlation(N_corr, M_corr):
    n, m = N_corr, M_corr
    data = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            data[i, j] = np.float64(i * j) / m + i
    corr = np.zeros((m, m), dtype=np.float64)
    mean = np.zeros(m, dtype=np.float64)
    stddev = np.zeros(m, dtype=np.float64)
    return {"data": data, "corr": corr, "mean": mean, "stddev": stddev}


def test_polybench_correlation():
    _run_and_compare(correlation_v, _init_correlation, ["corr"], {"N_corr": 32, "M_corr": 28}, "correlation")


M_cov = dace.symbol('M_cov')
N_cov = dace.symbol('N_cov')


@dace.program(datatype[N_cov, M_cov], datatype[M_cov, M_cov], datatype[M_cov])
def covariance_v(data, cov, mean):
    mean[:] = 0.0

    @dace.map
    def comp_mean(j: _[0:M_cov], i: _[0:N_cov]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M_cov]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N_cov

    @dace.map
    def sub_mean(i: _[0:N_cov], j: _[0:M_cov]):
        ind << data[i, j]
        m << mean[j]
        oud >> data[i, j]
        oud = ind - m

    @dace.mapscope
    def comp_cov_row(i: _[0:M_cov]):

        @dace.mapscope
        def comp_cov_col(j: _[i:M_cov]):
            with dace.tasklet:
                cov_ij >> cov[i, j]
                cov_ij = 0.0

            @dace.map
            def comp_cov_k(k: _[0:N_cov]):
                indi << data[k, i]
                indj << data[k, j]
                cov_ij >> cov(1, lambda x, y: x + y)[i, j]
                cov_ij = (indi * indj)

            with dace.tasklet:
                cov_ij_in << cov[i, j]
                cov_ij_out >> cov[i, j]
                cov_ji_out >> cov[j, i]
                cov_ij_out = cov_ij_in / (N_cov - 1)
                cov_ji_out = cov_ij_out


def _init_covariance(N_cov, M_cov):
    n, m = N_cov, M_cov
    data = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            data[i, j] = np.float64(i * j) / m
    cov = np.zeros((m, m), dtype=np.float64)
    mean = np.zeros(m, dtype=np.float64)
    return {"data": data, "cov": cov, "mean": mean}


def test_polybench_covariance():
    _run_and_compare(covariance_v, _init_covariance, ["cov"], {"N_cov": 32, "M_cov": 28}, "covariance")


if __name__ == "__main__":
    test_insert_cpu_to_cpu_1d()
    test_insert_cpu_to_cpu_2d_slice()
    test_insert_other_subset_data_convention("insert_other_dst", Memlet(data="B", subset="0:8", other_subset="2:10"))
    test_insert_other_subset_data_convention("insert_other_src", Memlet(data="A", subset="2:10", other_subset="0:8"))
    test_insert_cpu_to_cpu_full_array()
    test_insert_multiple_copies_same_state()
    test_insert_empty_memlet_skipped()
    test_insert_no_copies_returns_none()
    test_insert_nested_sdfg()
    test_insert_validates_after_pass()

    # Pattern 2: map staging
    test_insert_map_staging_copies()
    test_insert_nested_map_staging_copies()
    test_insert_staging_requires_outer_access_node()

    # Polybench correctness
    test_polybench_fdtd2d()
    test_polybench_correlation()
    test_polybench_covariance()

    for params in [
        ("insert_cpu_gpu", "H", dace.StorageType.CPU_Heap, "G", dace.StorageType.GPU_Global, 64),
        ("insert_gpu_cpu", "G", dace.StorageType.GPU_Global, "H", dace.StorageType.CPU_Heap, 64),
        ("insert_gpu_gpu", "A", dace.StorageType.GPU_Global, "B", dace.StorageType.GPU_Global, 128),
    ]:
        test_insert_cross_storage_transfer(*params)

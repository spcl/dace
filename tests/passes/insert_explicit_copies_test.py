# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``InsertExplicitCopies`` pass: builder-API structural cases plus Polybench numerical checks."""
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
    """Assert no memlet in any state or nested SDFG still carries an ``other_subset`` after copy-node insertion."""
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for edge in state.edges():
                memlet = edge.data
                if memlet.is_empty():
                    continue
                assert memlet.other_subset is None, (
                    f"Memlet on edge {edge.src}->{edge.dst} in SDFG '{nsdfg.name}' still "
                    f"has other_subset={memlet.other_subset}; expected None after copy insertion.")


def _assert_no_copynd(sdfg: dace.SDFG) -> None:
    """Assert ``generate_code`` emits no ``dace::CopyND`` template instantiations.

    The libnodes are designed to displace the runtime CopyND fallback entirely.
    Generated code for any SDFG run through ``InsertExplicitCopies`` + the
    libnode expansions should reach zero ``CopyND<`` references.
    """
    sdfg.expand_library_nodes()
    for obj in sdfg.generate_code():
        code = obj.code if isinstance(obj.code, str) else getattr(obj.code, 'code', str(obj.code))
        assert 'CopyND<' not in code, f"unexpected CopyND in code object {obj.title}"


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
    """Either memlet convention (``data=src`` or ``data=dst``) yields the same copy ``_in=A[2:10]``,
    ``_out=B[0:8]`` with no ``other_subset``."""
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


def _count_nested_sdfgs(sdfg):
    """Count NestedSDFGs in ``sdfg`` (top level only -- not recursive into them)."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def test_single_element_copies_expand_to_tasklets_no_nested_sdfg():
    """Single-element copies expand to direct ``_cpy_out = _cpy_in`` Tasklets, never a NestedSDFG.

    The ``MappedTasklet`` path would build a 0-D map for these and crash
    propagation, so routing must short-circuit to the ``Tasklet`` impl.
    """
    cpu = dace.StorageType.CPU_Heap
    pinned = dace.StorageType.CPU_Pinned
    register = dace.StorageType.Register
    gpu = dace.StorageType.GPU_Global

    sdfg = dace.SDFG("scalar_copies")
    # Cross-CPU storage scalars (CPU_Heap -> CPU_Pinned, single element).
    sdfg.add_array("c_in", [1], dace.float64, cpu)
    sdfg.add_array("c_out", [1], dace.float64, pinned)
    # Same-side GPU register scalars.
    sdfg.add_array("r_in", [1], dace.float64, register, transient=True)
    sdfg.add_array("r_out", [1], dace.float64, register, transient=True)

    st = sdfg.add_state("s")
    c_in = st.add_access("c_in")
    c_out = st.add_access("c_out")
    r_in = st.add_access("r_in")
    r_out = st.add_access("r_out")
    st.add_edge(c_in, None, c_out, None, Memlet("c_in[0]"))
    st.add_edge(r_in, None, r_out, None, Memlet("r_in[0]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    assert _count_copy_nodes(sdfg) == 2

    sdfg.expand_library_nodes()

    assert _count_nested_sdfgs(sdfg) == 0, (
        "Single-element copies should expand to a direct Tasklet, not a NestedSDFG. "
        f"Found {_count_nested_sdfgs(sdfg)} NestedSDFG(s) after expansion.")

    # Sanity: the expansions left tasklets behind that do the copy assignment.
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    assert any(
        "_cpy_out = _cpy_in" in t.code.as_string
        for t in tasklets), (f"Expected at least one ``_cpy_out = _cpy_in`` Tasklet from CopyLibraryNode expansion; "
                             f"got tasklets with code: {[t.code.as_string for t in tasklets]}")


def test_insert_validates_after_pass():
    """SDFG passes validation after InsertExplicitCopies."""
    cpu = dace.StorageType.CPU_Heap
    sdfg, _, _, _ = _build_copy_sdfg("validate_after", [("A", [100], cpu), ("B", [100], cpu)], Memlet("A[0:100]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    sdfg.validate()


def test_insert_view_dst_round_trip_lifts_source_edge():
    """An AN -> View -> AN round-trip collapses so the source array's single out-edge targets the
    ``CopyLibraryNode`` and the View is removed."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG("view_dst_intermediate")
    sdfg.add_array("A", [4, 5, 6], dace.float64, storage=cpu)
    sdfg.add_view("A_view", [5, 6], dace.float64, storage=cpu)
    sdfg.add_array("sink", [5, 6], dace.float64, storage=cpu)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    v = st.add_access("A_view")
    out = st.add_access("sink")
    st.add_edge(a, None, v, None, Memlet("A[1, 0:5, 0:6]"))
    st.add_edge(v, None, out, None, Memlet("A_view[0:5, 0:6]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()

    out_edges = list(st.out_edges(a))
    assert len(out_edges) == 1 and isinstance(out_edges[0].dst, CopyLibraryNode)
    assert v not in st.nodes()


def test_insert_view_src_round_trip_lifts_destination_edge():
    """An AN -> View -> AN round-trip collapses so the sink's single in-edge comes from the
    ``CopyLibraryNode`` and the View is removed."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG("view_src_intermediate")
    sdfg.add_array("A", [4, 5, 6], dace.float64, storage=cpu)
    sdfg.add_view("A_view", [5, 6], dace.float64, storage=cpu)
    sdfg.add_array("sink", [5, 6], dace.float64, storage=cpu)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    v = st.add_access("A_view")
    out = st.add_access("sink")
    st.add_edge(a, None, v, None, Memlet("A[1, 0:5, 0:6]"))
    st.add_edge(v, None, out, None, Memlet("A_view[0:5, 0:6]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()

    in_edges = list(st.in_edges(out))
    assert len(in_edges) == 1 and isinstance(in_edges[0].src, CopyLibraryNode)
    assert v not in st.nodes()


def test_insert_view_round_trip_collapses_to_single_copy():
    """An AN -> View -> AN round-trip yields exactly one ``CopyLibraryNode`` and the now-unused View is removed."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG("view_round_trip")
    sdfg.add_array("A", [4, 5, 6], dace.float64, storage=cpu)
    sdfg.add_view("A_view", [5, 6], dace.float64, storage=cpu)
    sdfg.add_array("sink", [5, 6], dace.float64, storage=cpu)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    v = st.add_access("A_view")
    out = st.add_access("sink")
    st.add_edge(a, None, v, None, Memlet("A[1, 0:5, 0:6]"))
    st.add_edge(v, None, out, None, Memlet("A_view[0:5, 0:6]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()

    assert _count_copy_nodes(sdfg) == 1
    assert v not in st.nodes()


def test_insert_self_copy_subset_is_dst_side():
    """On a self-copy ``p -> p`` the ``subset`` side maps to the ``_out`` (dst) edge and ``other_subset`` to
    ``_in`` (src); reversing them would silently produce a backwards copy."""
    sdfg = dace.SDFG("self_copy_subset_dst")
    sdfg.add_array("p", [4, 5], dace.float64)

    st = sdfg.add_state("s")
    a = st.add_access("p")
    b = st.add_access("p")
    st.add_edge(a, None, b, None, Memlet(data="p", subset="0:4, 4", other_subset="0:4, 3"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()

    copies = [n for n in st.nodes() if isinstance(n, CopyLibraryNode)]
    assert len(copies) == 1
    cn = copies[0]
    in_e = [e for e in st.in_edges(cn) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME][0]
    out_e = [e for e in st.out_edges(cn) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME][0]

    assert str(in_e.data.subset) == "0:4, 3", (f"src side should read column 3 (other_subset); got {in_e.data.subset}")
    assert str(out_e.data.subset) == "0:4, 4", (f"dst side should write column 4 (subset); got {out_e.data.subset}")


def _check_reshape_copy(sdfg, dst_name, dst_shape):
    """Assert the SDFG validates and the single lifted ``CopyLibraryNode``'s output memlet spans the full
    ``dst_shape``."""
    sdfg.validate()
    copies = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CopyLibraryNode)]
    assert len(copies) == 1, f"expected exactly one CopyLibraryNode, got {len(copies)}"
    cn = copies[0]
    parent = next(p for n, p in sdfg.all_nodes_recursive() if n is cn)
    out_e = [e for e in parent.out_edges(cn) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME][0]
    assert out_e.data.data == dst_name
    assert str(out_e.data.subset) == ', '.join(
        f"0:{s}" for s in dst_shape), (f"dst memlet subset should span full {dst_shape}, got {out_e.data.subset}")


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([8, 12, 5, 3], [96, 5, 3]),  # collapse leading two: einsum_blas test_4x4 pattern
        ([8, 10, 12], [80, 12]),  # collapse leading two: einsum_blas test_3x2 pattern
        ([8, 12, 5, 3], [8, 60, 3]),  # collapse middle two
        ([2, 3, 4, 5], [6, 20]),  # double collapse: dims 0-1 and dims 2-3
        ([8, 12, 5, 3], [1440]),  # full flatten
    ])
def test_insert_consecutive_collapse_reshape(src_shape, dst_shape):
    """When the destination shape collapses contiguous source dims, the pass derives a full-destination subset
    rather than reusing the rank-mismatched ``src_subset``."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG(f"reshape_collapse_{len(src_shape)}_to_{len(dst_shape)}")
    sdfg.add_array("A", src_shape, dace.float64, storage=cpu)
    sdfg.add_array("B", dst_shape, dace.float64, storage=cpu)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    # No other_subset, so the pass must derive the destination range itself.
    st.add_edge(a, None, b, None, Memlet(data="A", subset=', '.join(f"0:{s}" for s in src_shape)))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _check_reshape_copy(sdfg, "B", dst_shape)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([80, 12], [8, 10, 12]),  # split leading dim
        ([96, 5, 3], [8, 12, 5, 3]),  # split leading dim
        ([1440], [8, 12, 5, 3]),  # full unflatten
        ([6, 20], [2, 3, 4, 5]),  # double split
    ])
def test_insert_consecutive_split_reshape(src_shape, dst_shape):
    """The inverse split case: a higher-rank destination reached by splitting source dims is handled by the same
    symmetric code path."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG(f"reshape_split_{len(src_shape)}_to_{len(dst_shape)}")
    sdfg.add_array("A", src_shape, dace.float64, storage=cpu)
    sdfg.add_array("B", dst_shape, dace.float64, storage=cpu)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    st.add_edge(a, None, b, None, Memlet(data="A", subset=', '.join(f"0:{s}" for s in src_shape)))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _check_reshape_copy(sdfg, "B", dst_shape)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([8, 1, 12], [8, 12]),  # squeeze a length-1 dim
        ([8, 12, 1, 5], [96, 5]),  # squeeze + collapse
        ([1, 96, 5, 3], [8, 12, 5, 3]),  # leading 1 + split
    ])
def test_insert_reshape_with_squeezed_ones(src_shape, dst_shape):
    """Unit-length dimensions on either side are ignored when matching a consecutive collapse or split."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG(f"reshape_squeeze_{len(src_shape)}_to_{len(dst_shape)}")
    sdfg.add_array("A", src_shape, dace.float64, storage=cpu)
    sdfg.add_array("B", dst_shape, dace.float64, storage=cpu)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    st.add_edge(a, None, b, None, Memlet(data="A", subset=', '.join(f"0:{s}" for s in src_shape)))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _check_reshape_copy(sdfg, "B", dst_shape)


def test_insert_view_rewrite_is_idempotent_under_repeated_apply():
    """Repeated ``apply_pass`` calls do not accumulate extra ``CopyLibraryNode``s; runs after the first are
    no-ops since the round-trip pattern is gone."""
    cpu = dace.StorageType.CPU_Heap
    sdfg = dace.SDFG("view_rewrite_idempotent")
    sdfg.add_array("A", [4, 5, 6], dace.float64, storage=cpu)
    sdfg.add_view("A_view", [5, 6], dace.float64, storage=cpu)
    sdfg.add_array("sink", [5, 6], dace.float64, storage=cpu)

    st = sdfg.add_state("s")
    a = st.add_access("A")
    v = st.add_access("A_view")
    out = st.add_access("sink")
    st.add_edge(a, None, v, None, Memlet("A[1, 0:5, 0:6]"))
    st.add_edge(v, None, out, None, Memlet("A_view[0:5, 0:6]"))

    p = InsertExplicitCopies()
    p.apply_pass(sdfg, {})
    n_after_first = _count_copy_nodes(sdfg)
    assert n_after_first == 1

    for _ in range(5):
        p.apply_pass(sdfg, {})

    assert _count_copy_nodes(sdfg) == n_after_first
    sdfg.validate()


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


_N = dace.symbol('_N')


def test_iec_skips_array_to_view_edge():
    """An AccessNode -> View edge is left direct (no ``CopyLibraryNode`` inserted)."""
    sdfg = dace.SDFG('skip_array_to_view')
    sdfg.add_array('A', [4, 5, 6], dace.float64)
    sdfg.add_view('Av', [5, 6], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    v = state.add_access('Av')
    state.add_edge(a, None, v, None, Memlet('A[1, 0:5, 0:6]'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    assert _count_copy_nodes(sdfg) == 0
    in_e = list(state.in_edges(v))
    assert len(in_e) == 1 and in_e[0].src is a


def test_iec_collapses_round_trip_view():
    """An A -> View -> sink round-trip collapses to one ``CopyLibraryNode``, removes the View, and stays
    numerically correct."""
    sdfg = dace.SDFG('collapse_round_trip_view')
    sdfg.add_array('A', [4, 5, 6], dace.float64)
    sdfg.add_view('Av', [5, 6], dace.float64)
    sdfg.add_array('sink', [5, 6], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    v = state.add_access('Av')
    out = state.add_access('sink')
    state.add_edge(a, None, v, None, Memlet('A[1, 0:5, 0:6]'))
    state.add_edge(v, None, out, None, Memlet('Av[0:5, 0:6]'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    assert _count_copy_nodes(sdfg) == 1
    assert v not in state.nodes()
    sdfg.validate()
    A = np.copy(np.arange(120, dtype=np.float64).reshape(4, 5, 6))
    sink = np.zeros((5, 6), dtype=np.float64)
    sdfg(A=A, sink=sink)
    assert np.array_equal(sink, A[1])


def test_iec_collapse_keeps_view_with_other_consumers():
    """A round-trip collapse keeps the View when it still has another consumer edge."""
    sdfg = dace.SDFG('collapse_keeps_view_with_other_consumers')
    sdfg.add_array('A', [4, 5, 6], dace.float64)
    sdfg.add_view('Av', [5, 6], dace.float64)
    sdfg.add_array('sink', [5, 6], dace.float64)
    sdfg.add_array('also_reads', [5, 6], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    v = state.add_access('Av')
    out = state.add_access('sink')
    other = state.add_access('also_reads')
    state.add_edge(a, None, v, None, Memlet('A[1, 0:5, 0:6]'))
    state.add_edge(v, None, out, None, Memlet('Av[0:5, 0:6]'))
    state.add_edge(v, None, other, None, Memlet('Av[0:5, 0:6]'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    # v survives because v->also_reads still consumes it after the collapse.
    assert v in state.nodes()
    sdfg.validate()


def test_iec_skips_reshape_view_edge():
    """A reshape (rank-changing) AccessNode -> View edge is left direct with no ``CopyLibraryNode``."""
    sdfg = dace.SDFG('skip_reshape_view')
    sdfg.add_array('A', [2, 3, 4], dace.float64)
    sdfg.add_view('Av', [8, 3], dace.float64)
    state = sdfg.add_state()
    a = state.add_access('A')
    v = state.add_access('Av')
    state.add_edge(a, None, v, None, Memlet(data='A', subset='0:2, 0:3, 0:4', other_subset='0:8, 0:3'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    assert _count_copy_nodes(sdfg) == 0


def test_iec_array_to_array_constant_first_dim():
    """A rank-mismatched copy whose extra leading dim is a constant index collapses to matching rank and copies
    correctly."""
    sdfg = dace.SDFG('const_first_dim_copy')
    sdfg.add_array('src', [5, 4, 3], dace.float64)
    sdfg.add_array('dst', [4, 3], dace.float64)
    state = sdfg.add_state()
    s = state.add_access('src')
    d = state.add_access('dst')
    state.add_edge(s, None, d, None, Memlet(data='src', subset='2, 0:4, 0:3', other_subset='0:4, 0:3'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    src = np.copy(np.arange(60, dtype=np.float64).reshape(5, 4, 3))
    dst = np.zeros((4, 3), dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src[2])


def test_iec_array_to_array_constant_middle_dim():
    """A rank-mismatched copy whose extra middle dim is a constant index collapses to matching rank and copies
    correctly."""
    sdfg = dace.SDFG('const_middle_dim_copy')
    sdfg.add_array('src', [4, 5, 3], dace.float64)
    sdfg.add_array('dst', [4, 3], dace.float64)
    state = sdfg.add_state()
    s = state.add_access('src')
    d = state.add_access('dst')
    state.add_edge(s, None, d, None, Memlet(data='src', subset='0:4, 2, 0:3', other_subset='0:4, 0:3'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    src = np.copy(np.arange(60, dtype=np.float64).reshape(4, 5, 3))
    dst = np.zeros((4, 3), dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src[:, 2, :])


def test_iec_array_to_array_rank_change_uses_mapped_tasklet():
    """A volume-equal rank-changing copy with no constant dim takes the ``MappedTasklet`` rank-mismatch path and
    copies correctly."""
    sdfg = dace.SDFG('rank_change_copynd')
    sdfg.add_array('src', [2, 3, 4], dace.float64)
    sdfg.add_array('dst', [8, 3], dace.float64)
    state = sdfg.add_state()
    s = state.add_access('src')
    d = state.add_access('dst')
    state.add_edge(s, None, d, None, Memlet(data='src', subset='0:2, 0:3, 0:4', other_subset='0:8, 0:3'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    src = np.copy(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    dst = np.zeros((8, 3), dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.reshape(8, 3))


def test_iec_array_to_array_flatten_uses_mapped_tasklet():
    """A 2D -> 1D flatten copy takes the ``MappedTasklet`` rank-mismatch path and copies correctly."""
    sdfg = dace.SDFG('flatten_copynd')
    sdfg.add_array('src', [4, 3], dace.float64)
    sdfg.add_array('dst', [12], dace.float64)
    state = sdfg.add_state()
    s = state.add_access('src')
    d = state.add_access('dst')
    state.add_edge(s, None, d, None, Memlet(data='src', subset='0:4, 0:3', other_subset='0:12'))
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    src = np.copy(np.arange(12, dtype=np.float64).reshape(4, 3))
    dst = np.zeros(12, dtype=np.float64)
    sdfg(src=src, dst=dst)
    assert np.array_equal(dst, src.reshape(12))


@dace.program
def _iec_pin_reshape_rank_change(A: dace.float64[2, 3, 4], B: dace.float64[8, 3]):
    C = np.reshape(A, [8, 3])
    B[:] += C


def test_iec_reshape_does_not_lift_view():
    """The pass does not lift a reshape view in a real program; output stays numerically correct."""
    sdfg = _iec_pin_reshape_rank_change.to_sdfg(simplify=True)
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    A = np.random.rand(2, 3, 4)
    B = np.random.rand(8, 3)
    expected = np.reshape(A, [8, 3]) + B
    sdfg(A=A, B=B)
    assert np.allclose(B, expected)


@dace.program
def _iec_pin_reinterpret_dtype(A: dace.int32[_N]):
    C = A.view(dace.int16)
    C[:] += 1


def test_iec_reinterpret_does_not_lift_view():
    """The pass does not lift a dtype-reinterpret view; output stays numerically correct."""
    sdfg = _iec_pin_reinterpret_dtype.to_sdfg(simplify=True)
    InsertExplicitCopies().apply_pass(sdfg, {})
    sdfg.validate()
    A = np.random.randint(0, 262144, size=[10], dtype=np.int32)
    expected = np.copy(A)
    expected.view(np.int16)[:] += 1
    sdfg(A=A, _N=10)
    assert np.array_equal(A, expected)


# Map-staging lift: AN -> MapEntry -> AN and AN -> MapExit -> AN copies are
# rewritten to put a CopyLibraryNode INSIDE the map scope, wired directly to
# the map node's connector. Views on the outer side stay in place. Chained
# MapEntries / MapExits are followed via memlet_path. Generated code emits
# no CopyND template instantiations.


_CPU = dace.dtypes.StorageType.CPU_Heap
_N_STAGE = 128
_TILE = 32


def _build_stage_in_sdfg(name: str, with_view: bool = False) -> dace.SDFG:
    """Build ``A -> MapEntry -> local -> inner work -> B``, optionally with a View aliasing ``A``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("B", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("local", [_TILE], dace.float64, storage=_CPU, transient=True)
    if with_view:
        sdfg.add_view("Av", [_N_STAGE], dace.float64, storage=_CPU)

    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    local = state.add_access("local")
    me, mx = state.add_map("tile", {"bi": f"0:{_N_STAGE}:{_TILE}"})

    if with_view:
        av = state.add_access("Av")
        state.add_edge(a, None, av, None, Memlet(f"A[0:{_N_STAGE}]"))
        state.add_memlet_path(av, me, local, memlet=Memlet(f"Av[bi:bi+{_TILE}]"))
    else:
        state.add_memlet_path(a, me, local, memlet=Memlet(f"A[bi:bi+{_TILE}]"))

    ime, imx = state.add_map("inner", {"ti": f"0:{_TILE}"})
    t = state.add_tasklet("incr", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_memlet_path(local, ime, t, dst_conn="_in", memlet=Memlet("local[ti]"))
    state.add_memlet_path(t, imx, mx, b, src_conn="_out", memlet=Memlet("B[bi+ti]"))
    return sdfg


def _build_stage_out_sdfg(name: str, with_view: bool = False) -> dace.SDFG:
    """Build ``A -> inner work -> local -> MapExit -> B``, optionally with a View aliasing ``B``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("B", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("local", [_TILE], dace.float64, storage=_CPU, transient=True)
    if with_view:
        sdfg.add_view("Bv", [_N_STAGE], dace.float64, storage=_CPU)

    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    local = state.add_access("local")
    me, mx = state.add_map("tile", {"bi": f"0:{_N_STAGE}:{_TILE}"})

    ime, imx = state.add_map("inner", {"ti": f"0:{_TILE}"})
    t = state.add_tasklet("incr", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_memlet_path(a, me, ime, t, dst_conn="_in", memlet=Memlet("A[bi+ti]"))
    state.add_memlet_path(t, imx, local, src_conn="_out", memlet=Memlet("local[ti]"))

    if with_view:
        bv = state.add_access("Bv")
        state.add_memlet_path(local, mx, bv, memlet=Memlet(f"Bv[bi:bi+{_TILE}]"))
        state.add_edge(bv, None, b, None, Memlet(f"B[0:{_N_STAGE}]"))
    else:
        state.add_memlet_path(local, mx, b, memlet=Memlet(f"B[bi:bi+{_TILE}]"))
    return sdfg


def _find_libnode_and_scope(state):
    libnodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
    assert len(libnodes) == 1, f"expected exactly one CopyLibraryNode, got {len(libnodes)}"
    cn = libnodes[0]
    return cn, state.entry_node(cn)


def _assert_lifted_libnode(state, side: str, expected_scope=None):
    """Assert exactly one libnode in ``state`` is inside a map scope and wired directly to it.

    :param side: ``'in'`` for stage-in (libnode input edge from MapEntry) or
        ``'out'`` for stage-out (libnode output edge to MapExit).
    :param expected_scope: optional MapEntry node identity to require for the
        libnode's enclosing scope; when ``None``, any MapEntry passes.
    :returns: ``(libnode, enclosing_map_entry)``.
    """
    cn, parent = _find_libnode_and_scope(state)
    assert isinstance(parent, nodes.MapEntry), f"libnode parent scope is {type(parent).__name__}, expected MapEntry"
    if expected_scope is not None:
        assert parent is expected_scope, "libnode must sit in the expected (innermost) map scope"
    if side == "in":
        in_edges = [e for e in state.in_edges(cn) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME]
        assert len(in_edges) == 1 and in_edges[0].src is parent, \
            "libnode's input must wire directly to the MapEntry connector"
    else:
        out_edges = [e for e in state.out_edges(cn) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME]
        assert len(out_edges) == 1 and isinstance(out_edges[0].dst, nodes.MapExit), \
            "libnode's output must wire directly to the MapExit connector"
    return cn, parent


def _run_and_check(sdfg: dace.SDFG, expected_b):
    A = np.arange(_N_STAGE, dtype=np.float64)
    B = np.zeros(_N_STAGE, dtype=np.float64)
    sdfg(A=A, B=B)
    np.testing.assert_array_equal(B, expected_b(A))


def test_lift_stage_in_copy():
    """``A -> MapEntry -> local`` lifts to a libnode INSIDE the map scope, wired directly to MapEntry."""
    sdfg = _build_stage_in_sdfg("stage_in")
    InsertExplicitCopies().apply_pass(sdfg, {})

    _assert_lifted_libnode(sdfg.start_state, side="in")
    _assert_no_copynd(sdfg)
    _run_and_check(sdfg, lambda A: A + 1.0)


def test_lift_stage_out_copy():
    """``local -> MapExit -> B`` lifts to a libnode INSIDE the map scope, wired directly to MapExit."""
    sdfg = _build_stage_out_sdfg("stage_out")
    InsertExplicitCopies().apply_pass(sdfg, {})

    _assert_lifted_libnode(sdfg.start_state, side="out")
    _assert_no_copynd(sdfg)
    _run_and_check(sdfg, lambda A: A + 1.0)


def _view_an_names(sdfg, state):
    return [n.data for n in state.nodes()
            if isinstance(n, nodes.AccessNode) and isinstance(sdfg.arrays[n.data], dace.data.View)]


def test_lift_stage_in_copy_through_view():
    """``A -> A_view -> MapEntry -> local``: View stays in place; libnode placed between MapEntry and inner AN."""
    sdfg = _build_stage_in_sdfg("stage_in_view", with_view=True)
    InsertExplicitCopies().apply_pass(sdfg, {})

    _assert_lifted_libnode(sdfg.start_state, side="in")
    assert _view_an_names(sdfg, sdfg.start_state) == ["Av"]
    _assert_no_copynd(sdfg)
    _run_and_check(sdfg, lambda A: A + 1.0)


def test_lift_stage_out_copy_through_view():
    """``local -> MapExit -> B_view -> B``: View stays in place; libnode placed between local and MapExit."""
    sdfg = _build_stage_out_sdfg("stage_out_view", with_view=True)
    InsertExplicitCopies().apply_pass(sdfg, {})

    _assert_lifted_libnode(sdfg.start_state, side="out")
    assert _view_an_names(sdfg, sdfg.start_state) == ["Bv"]
    _assert_no_copynd(sdfg)
    _run_and_check(sdfg, lambda A: A + 1.0)


def test_lift_stage_in_copy_chained_map_entries():
    """``A -> ME1 -> ME2 -> local``: lift through nested MapEntries; libnode at innermost scope."""
    N, TILE, INNER = 64, 16, 4
    sdfg = dace.SDFG("stage_in_nested")
    sdfg.add_array("A", [N], dace.float64, storage=_CPU)
    sdfg.add_array("B", [N], dace.float64, storage=_CPU)
    sdfg.add_array("local", [INNER], dace.float64, storage=_CPU, transient=True)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    local = state.add_access("local")
    me1, mx1 = state.add_map("outer", {"bi": f"0:{N}:{TILE}"})
    me2, mx2 = state.add_map("inner_block", {"si": f"0:{TILE}:{INNER}"})
    # Stage-in flows through both MapEntries down to `local`.
    state.add_memlet_path(a, me1, me2, local, memlet=Memlet(f"A[bi+si:bi+si+{INNER}]"))
    ime, imx = state.add_map("inner", {"ti": f"0:{INNER}"})
    t = state.add_tasklet("incr", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_memlet_path(local, ime, t, dst_conn="_in", memlet=Memlet("local[ti]"))
    state.add_memlet_path(t, imx, mx2, mx1, b, src_conn="_out", memlet=Memlet("B[bi+si+ti]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_lifted_libnode(state, side="in", expected_scope=me2)
    _assert_no_copynd(sdfg)
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    sdfg(A=A, B=B)
    np.testing.assert_array_equal(B, A + 1.0)


def test_lift_stage_out_copy_chained_map_exits():
    """Symmetric: ``local -> MX2 -> MX1 -> B`` -- libnode at innermost scope, wired directly to MX2."""
    N, TILE, INNER = 64, 16, 4
    sdfg = dace.SDFG("stage_out_nested")
    sdfg.add_array("A", [N], dace.float64, storage=_CPU)
    sdfg.add_array("B", [N], dace.float64, storage=_CPU)
    sdfg.add_array("local", [INNER], dace.float64, storage=_CPU, transient=True)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    local = state.add_access("local")
    me1, mx1 = state.add_map("outer", {"bi": f"0:{N}:{TILE}"})
    me2, mx2 = state.add_map("inner_block", {"si": f"0:{TILE}:{INNER}"})
    ime, imx = state.add_map("inner", {"ti": f"0:{INNER}"})
    t = state.add_tasklet("incr", {"_in"}, {"_out"}, "_out = _in + 1.0")
    state.add_memlet_path(a, me1, me2, ime, t, dst_conn="_in", memlet=Memlet("A[bi+si+ti]"))
    state.add_memlet_path(t, imx, local, src_conn="_out", memlet=Memlet("local[ti]"))
    state.add_memlet_path(local, mx2, mx1, b, memlet=Memlet(f"B[bi+si:bi+si+{INNER}]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    _assert_lifted_libnode(state, side="out", expected_scope=me2)
    _assert_no_copynd(sdfg)
    A = np.arange(N, dtype=np.float64)
    B = np.zeros(N, dtype=np.float64)
    sdfg(A=A, B=B)
    np.testing.assert_array_equal(B, A + 1.0)


def _make_inner_nested_sdfg(body_name: str, inout_name: str, size: int, op: str) -> dace.SDFG:
    """Tiny NestedSDFG: ``inout[i] = op(inout[i])`` over ``i = 0:size``."""
    nsdfg = dace.SDFG(body_name)
    nsdfg.add_array(inout_name, [size], dace.float64)
    st = nsdfg.add_state("body")
    a = st.add_access(inout_name)
    b = st.add_access(inout_name)
    me, mx = st.add_map("inner", {"ti": f"0:{size}"})
    t = st.add_tasklet("op", {"_in"}, {"_out"}, f"_out = {op}")
    st.add_memlet_path(a, me, t, dst_conn="_in", memlet=Memlet(f"{inout_name}[ti]"))
    st.add_memlet_path(t, mx, b, src_conn="_out", memlet=Memlet(f"{inout_name}[ti]"))
    return nsdfg


def test_lift_stage_in_copy_with_nested_sdfg_consumer():
    """``A -> MapEntry -> local`` where ``local`` feeds a NestedSDFG inside the map: lift unaffected."""
    sdfg = dace.SDFG("stage_in_nsdfg")
    sdfg.add_array("A", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("B", [_N_STAGE], dace.float64, storage=_CPU)
    sdfg.add_array("local", [_TILE], dace.float64, storage=_CPU, transient=True)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    local = state.add_access("local")
    me, mx = state.add_map("tile", {"bi": f"0:{_N_STAGE}:{_TILE}"})
    state.add_memlet_path(a, me, local, memlet=Memlet(f"A[bi:bi+{_TILE}]"))

    nsdfg = _make_inner_nested_sdfg("inner_body", "buf", _TILE, "_in + 1.0")
    nnode = state.add_nested_sdfg(nsdfg, {"buf"}, {"buf"})
    state.add_edge(local, None, nnode, "buf", Memlet(f"local[0:{_TILE}]"))
    out_local = state.add_access("local")
    state.add_edge(nnode, "buf", out_local, None, Memlet(f"local[0:{_TILE}]"))
    state.add_memlet_path(out_local, mx, b, memlet=Memlet(f"B[bi:bi+{_TILE}]"))

    InsertExplicitCopies().apply_pass(sdfg, {})
    state = sdfg.start_state
    # Both the stage-in and stage-out edges lift.
    libnodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
    assert len(libnodes) == 2
    for cn in libnodes:
        assert isinstance(state.entry_node(cn), nodes.MapEntry)

    _assert_no_copynd(sdfg)
    A = np.arange(_N_STAGE, dtype=np.float64)
    B = np.zeros(_N_STAGE, dtype=np.float64)
    sdfg(A=A, B=B)
    np.testing.assert_array_equal(B, A + 1.0)


# Polybench-derived tests: the pass must preserve numerical output on real programs.

_ = None  # needed for dace.map range syntax
_DATATYPE = dace.float64


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


@dace.program(_DATATYPE[NX, NY], _DATATYPE[NX, NY], _DATATYPE[NX, NY], _DATATYPE[TMAX])
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
            eyout = eyin - _DATATYPE(0.5) * (hz1 - hz2)

        @dace.map
        def update_ex(i: _[0:NX], j: _[1:NY]):
            exin << ex[i, j]
            hz1 << hz[i, j]
            hz2 << hz[i, j - 1]
            exout >> ex[i, j]
            exout = exin - _DATATYPE(0.5) * (hz1 - hz2)

        @dace.map
        def update_hz(i: _[0:NX - 1], j: _[0:NY - 1]):
            hzin << hz[i, j]
            ex1 << ex[i, j + 1]
            ex2 << ex[i, j]
            ey1 << ey[i + 1, j]
            ey2 << ey[i, j]
            hzout >> hz[i, j]
            hzout = hzin - _DATATYPE(0.7) * (ex1 - ex2 + ey1 - ey2)


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
    """``InsertExplicitCopies`` preserves fdtd2d output versus the untransformed reference."""
    _run_and_compare(fdtd2d_v, _init_fdtd2d, ["ex", "ey", "hz"], {"NX": 20, "NY": 30, "TMAX": 10}, "fdtd2d")


M_corr = dace.symbol('M_corr')
N_corr = dace.symbol('N_corr')


@dace.program(_DATATYPE[N_corr, M_corr], _DATATYPE[M_corr, M_corr], _DATATYPE[M_corr], _DATATYPE[M_corr])
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
        oud = (ind - m) / (math.sqrt(_DATATYPE(N_corr)) * sd)

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
    """``InsertExplicitCopies`` preserves correlation output versus the untransformed reference."""
    _run_and_compare(correlation_v, _init_correlation, ["corr"], {"N_corr": 32, "M_corr": 28}, "correlation")


M_cov = dace.symbol('M_cov')
N_cov = dace.symbol('N_cov')


@dace.program(_DATATYPE[N_cov, M_cov], _DATATYPE[M_cov, M_cov], _DATATYPE[M_cov])
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
    """``InsertExplicitCopies`` preserves covariance output versus the untransformed reference."""
    _run_and_compare(covariance_v, _init_covariance, ["cov"], {"N_cov": 32, "M_cov": 28}, "covariance")


if __name__ == "__main__":
    pytest.main([__file__])

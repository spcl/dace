# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``InsertMemoryEngineBulkCopies`` pass."""
import copy as _copy
import os
import sys

import dace
import pytest
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import utils as sdutils
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import AddThreadBlockMaps
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUCodegenPreprocessPipeline
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.insert_memory_engine_bulk_copies import (InsertMemoryEngineBulkCopies,
                                                                         InsertMemoryEngineBulkCopiesPipeline)

import tests.polybench

# The polybench programs import their ``polybench`` harness as a top-level module, which only
# resolves when run as scripts (own directory on sys.path); importing them as a package needs it too.
sys.path.append(os.path.dirname(tests.polybench.__file__))

from tests.polybench.gemm import gemm, init_array as _gemm_init_array
from tests.library.copy_node_test import _TMA_KERNELS, _TMA_M, _TMA_N, _require_memory_engine

_GLOBAL = dace.StorageType.GPU_Global
_SHARED = dace.StorageType.GPU_Shared
_DIRECTIONS = pytest.mark.parametrize("load", [True, False], ids=["global_to_shared", "shared_to_global"])


def _count_copy_nodes(sdfg):
    """Count CopyLibraryNode instances across all states (recursive)."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CopyLibraryNode))


def _assert_no_other_subset(sdfg: dace.SDFG) -> None:
    """Assert no data-movement memlet still carries an ``other_subset`` after copy-node insertion.

    View-defining (alias) edges are excluded: they reference the underlying buffer rather than
    moving data, so ``InsertExplicitCopies`` correctly leaves them direct with ``other_subset``
    intact (mirrors the pass's own view-edge skip); a copy would change the SDFG's semantics.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for edge in state.edges():
                memlet = edge.data
                if memlet.is_empty():
                    continue
                if any(
                        isinstance(an, nodes.AccessNode) and isinstance(nsdfg.arrays[an.data], dace.data.View)
                        and sdutils.get_view_edge(state, an) is edge for an in (edge.src, edge.dst)):
                    continue
                assert memlet.other_subset is None, (
                    f"Memlet on edge {edge.src}->{edge.dst} in SDFG '{nsdfg.name}' still "
                    f"has other_subset={memlet.other_subset}; expected None after copy insertion.")


def _assert_libnode_implementation(sdfg, none=False):
    """Assert that every (if none == False) or no CopyLibraryNode in ``sdfg`` has `"MemoryEngineBulk"` as implementation."""
    found = False
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, CopyLibraryNode):
            if none:
                assert n.implementation != "MemoryEngineBulk"
            else:
                assert n.implementation == "MemoryEngineBulk"
            found = True
    assert found, "No CopyLibraryNode found in SDFG"


def _insert(sdfg):
    """Run ``InsertMemoryEngineBulkCopies`` (and the passes it depends on); return its result."""
    # A pipeline that changed nothing returns ``None`` instead of its results
    results = ppl.Pipeline([InsertMemoryEngineBulkCopies()]).apply_pass(sdfg, {})
    return (results or {}).get("InsertMemoryEngineBulkCopies")


def _add_kernel_copy(sdfg, state, load, global_name, shared_name, global_subset, shared_subset):
    """Add a kernel map to ``state`` whose body is a single direct copy edge between ``global_name``
    (outside the kernel) and ``shared_name`` (inside), in the direction given by ``load``."""
    me, mx = state.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    shared = state.add_access(shared_name)
    memlet = Memlet(data=global_name, subset=global_subset, other_subset=shared_subset)
    if load:
        state.add_memlet_path(state.add_access(global_name), me, shared, memlet=memlet)
        state.add_nedge(shared, mx, Memlet())
    else:
        state.add_nedge(me, shared, Memlet())
        state.add_memlet_path(shared, mx, state.add_access(global_name), memlet=memlet)
    return me


def _build_kernel_copy(name, load, global_shape, shared_shape, global_subset, shared_subset, dtype=dace.float64):
    """One-state SDFG with a kernel copying ``A[global_subset]`` <-> ``S[shared_subset]``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", global_shape, dtype, _GLOBAL)
    sdfg.add_array("S", shared_shape, dtype, _SHARED, transient=True)
    _add_kernel_copy(sdfg, sdfg.add_state("s"), load, "A", "S", global_subset, shared_subset)
    return sdfg


@_DIRECTIONS
def test_insert_kernel_copy(load):
    """A 2D tile copy inside a kernel becomes a ``MemoryEngineBulk`` copy that expands."""
    sdfg = _build_kernel_copy("insert_kernel_copy", load, [64, 64], [32, 32], "32:64, 0:32", "0:32, 0:32")
    sdfg.validate()

    assert _insert(sdfg) == 1
    _assert_no_other_subset(sdfg)
    assert _count_copy_nodes(sdfg) == 1
    _assert_libnode_implementation(sdfg)
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()


@_DIRECTIONS
def test_insert_gpu_1d(load):
    """GPU_Global <-> GPU_Shared 1D copy inside a kernel."""
    sdfg = _build_kernel_copy("insert_gpu_1d", load, [100], [64], "16:48", "32:64")
    assert _insert(sdfg) == 1
    _assert_libnode_implementation(sdfg)


@pytest.mark.parametrize("shared_shape,shared_subset,accepted", [
    ([64, 32], "32:64, 0:32", True),
    ([32, 64], "0:32, 32:64", False),
],
                         ids=["packed_rows", "strided_columns"])
def test_insert_requires_packed_shared_region(shared_shape, shared_subset, accepted):
    """The memory engine writes the box densely, so the shared region must be packed."""
    sdfg = _build_kernel_copy("insert_packed", True, [64, 64], shared_shape, "0:32, 0:32", shared_subset)
    assert _insert(sdfg) == (1 if accepted else None)
    _assert_libnode_implementation(sdfg, none=not accepted)


def test_insert_rejects_misaligned_pitch():
    """A global row pitch that is not a multiple of 16 bytes cannot be described by a tensor map."""
    sdfg = _build_kernel_copy("insert_pitch", True, [64, 66], [32, 32], "0:32, 0:32", "0:32, 0:32", dace.float32)
    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


_STORAGES = [dace.StorageType.CPU_Heap, _GLOBAL, _SHARED, dace.StorageType.Register]


@pytest.mark.parametrize("src_storage,dst_storage",
                         [(s, d) for s in _STORAGES
                          for d in _STORAGES if (s, d) not in [(_GLOBAL, _SHARED), (_SHARED, _GLOBAL)]])
def test_insert_rejects_other_storages(src_storage, dst_storage):
    """Only GPU_Global <-> GPU_Shared copies use the memory engine."""
    sdfg = dace.SDFG("insert_other_storages")
    sdfg.add_array("A", [64], dace.float64, src_storage, transient=src_storage == _SHARED)
    sdfg.add_array("B", [64], dace.float64, dst_storage, transient=dst_storage == _SHARED)
    st = sdfg.add_state("s")
    me, mx = st.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    a, b = st.add_access("A"), st.add_access("B")
    st.add_nedge(me, a, Memlet())
    st.add_edge(a, None, b, None, Memlet("A[0:32]", other_subset="32:64"))
    st.add_nedge(b, mx, Memlet())

    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


@_DIRECTIONS
def test_insert_rejects_outside_kernel(load):
    """A GPU_Global <-> GPU_Shared copy outside any kernel cannot use the memory engine."""
    sdfg = dace.SDFG("insert_outside_kernel")
    sdfg.add_array("A", [64], dace.float64, _GLOBAL)
    sdfg.add_array("S", [32], dace.float64, _SHARED, transient=True)
    st = sdfg.add_state("s")
    src, dst = ("A", "S") if load else ("S", "A")
    st.add_edge(st.add_access(src), None, st.add_access(dst), None, Memlet(data="A", subset="0:32",
                                                                           other_subset="0:32"))

    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


def test_insert_rejects_inside_threadblock_map():
    """A copy inside a ``GPU_ThreadBlock`` map is per-thread, not block-collective."""
    sdfg = dace.SDFG("insert_in_tblock")
    sdfg.add_array("A", [64], dace.float64, _GLOBAL)
    sdfg.add_array("S", [32], dace.float64, _SHARED, transient=True)
    st = sdfg.add_state("s")
    me, mx = st.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    te, tx = st.add_map("block", {"t": "0:1"}, schedule=dace.ScheduleType.GPU_ThreadBlock)
    s = st.add_access("S")
    st.add_memlet_path(st.add_access("A"), me, te, s, memlet=Memlet(data="A", subset="0:32", other_subset="0:32"))
    st.add_memlet_path(s, tx, mx, memlet=Memlet())

    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


@_DIRECTIONS
def test_insert_rejects_dynamic_shared(load):
    """A copy into explicitly dynamic shared memory, which is not 128 byte aligned, keeps another implementation."""
    sdfg = _build_kernel_copy("insert_dynamic_shared", load, [64, 64], [32, 32], "32:64, 0:32", "0:32, 0:32")
    sdfg.arrays["S"].storage = dace.StorageType.GPU_Shared(dynamic=True)
    _insert(sdfg)
    _assert_libnode_implementation(sdfg, none=True)


def test_insert_keeps_explicit_implementation():
    """A copy node with an explicitly chosen implementation is left alone."""
    sdfg = _build_kernel_copy("insert_explicit", True, [64], [32], "0:32", "0:32")
    InsertExplicitCopies().apply_pass(sdfg, {})
    copy_node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CopyLibraryNode))
    copy_node.implementation = "MappedTasklet"

    assert _insert(sdfg) is None
    assert copy_node.implementation == "MappedTasklet"


def test_insert_multiple_copies_same_kernel():
    """Two tile loads in the same kernel: A->S and A->T."""
    sdfg = dace.SDFG("insert_multi")
    sdfg.add_array("A", [64], dace.float64, _GLOBAL)
    for name in ("S", "T"):
        sdfg.add_array(name, [32], dace.float64, _SHARED, transient=True)
    st = sdfg.add_state("s")
    me, mx = st.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    a = st.add_access("A")
    for name, subset in (("S", "0:32"), ("T", "32:64")):
        shared = st.add_access(name)
        st.add_memlet_path(a, me, shared, memlet=Memlet(data="A", subset=subset, other_subset="0:32"))
        st.add_nedge(shared, mx, Memlet())

    assert _insert(sdfg) == 2
    _assert_no_other_subset(sdfg)
    _assert_libnode_implementation(sdfg)
    sdfg.expand_library_nodes()
    sdfg.validate()


def test_insert_inside_loop_region():
    """A kernel in the body of a loop region is reached (states nested in control flow regions)."""
    sdfg = dace.SDFG("insert_loop_region")
    sdfg.add_array("A", [64, 64], dace.float64, _GLOBAL)
    sdfg.add_array("S", [32, 32], dace.float64, _SHARED, transient=True)
    loop = dace.sdfg.state.LoopRegion("loop", "k < 4", "k", "k = 0", "k = k + 1")
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state("body", is_start_block=True)
    _add_kernel_copy(sdfg, body, True, "A", "S", "0:32, 0:32", "0:32, 0:32")

    assert _insert(sdfg) == 1
    _assert_libnode_implementation(sdfg)


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

    assert _insert(sdfg) is None


def test_insert_rejects_view():
    """A view endpoint is not lifted onto the memory engine (its descriptor is not the viewed array's)."""
    sdfg = dace.SDFG("insert_view")
    sdfg.add_array("A", [4, 32], dace.float64, _GLOBAL)
    sdfg.add_view("A_view", [32], dace.float64, storage=_GLOBAL)
    sdfg.add_array("S", [32], dace.float64, _SHARED, transient=True)
    st = sdfg.add_state("s")
    me, mx = st.add_map("kernel", {"b": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    v, s = st.add_access("A_view"), st.add_access("S")
    st.add_memlet_path(st.add_access("A"), me, v, memlet=Memlet("A[1, 0:32]"))
    st.add_edge(v, None, s, None, Memlet("A_view[0:32]"))
    st.add_nedge(s, mx, Memlet())

    assert _insert(sdfg) is None
    assert v in st.nodes(), "the view must be preserved as a copy endpoint"


@pytest.mark.parametrize(
    "global_shape,shared_shape",
    [
        ([8, 12, 16], [96, 16]),  # collapse leading two
        ([8, 16], [128]),  # full flatten
        ([128], [8, 16]),  # split
        ([16, 8], [8, 16]),  # same size, different shape
    ])
def test_insert_rejects_reshape(global_shape, shared_shape):
    """The memory engine cannot reshape: the shared region must be the (squeezed) box."""
    sdfg = _build_kernel_copy(f"reshape_{len(global_shape)}_to_{len(shared_shape)}", True, global_shape, shared_shape,
                              ', '.join(f"0:{s}" for s in global_shape), ', '.join(f"0:{s}" for s in shared_shape))
    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


@pytest.mark.parametrize("global_shape,global_subset,shared_shape", [
    ([8, 1, 16], "0:8, 0:1, 0:16", [8, 16]),
    ([4, 32], "2, 0:32", [32]),
])
def test_insert_accepts_squeezed_unit_dims(global_shape, global_subset, shared_shape):
    """Unit-length box dimensions need no shared counterpart."""
    sdfg = _build_kernel_copy("squeeze", True, global_shape, shared_shape, global_subset,
                              ', '.join(f"0:{s}" for s in shared_shape))
    assert _insert(sdfg) == 1
    _assert_libnode_implementation(sdfg)


@_DIRECTIONS
def test_insert_accepts_bound_clamped_shared_region(load):
    """A partial tile clamps both regions to the array bound, which the memory engine handles itself."""
    NI = dace.symbol("NI")
    g = "b:Min(NI - 1, b + 31) + 1, 0:32"
    s = "0:Min(NI - 1, b + 31) - b + 1, 0:32"
    sdfg = _build_kernel_copy("insert_bound_clamp", load, [NI, 64], [32, 32], g, s)
    assert _insert(sdfg) == 1
    _assert_libnode_implementation(sdfg)


def test_insert_rejects_non_bound_clamp():
    """A clamp that is not the array bound changes the region the copy moves, so it is still a mismatch."""
    NI = dace.symbol("NI")
    sdfg = _build_kernel_copy("insert_other_clamp", True, [NI, 64], [32, 32], "b:b + 32, 0:32",
                              "0:Min(20, NI - b), 0:32")
    assert _insert(sdfg) is None
    _assert_libnode_implementation(sdfg, none=True)


def _tma_kernel_without_implementation(kernel, dtype=dace.float64):
    """A memory engine kernel of ``copy_node_test`` whose copy leaves the implementation to be selected."""
    builder, expected, is_load = _TMA_KERNELS[kernel]
    sdfg = builder(dtype)
    # Same kernel as in ``copy_node_test`` but another codegen: a build folder of its own
    sdfg.name += "_auto"
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, CopyLibraryNode):
            node.implementation = None
    return sdfg, expected, is_load


def _tasklet_labels(sdfg):
    return {n.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)}


@pytest.mark.parametrize("kernel", ["transpose", "store"])
def test_preprocess_pipeline_inserts_memory_engine_copies(kernel):
    """With ``memory_engine_copies``, the experimental GPU preprocessing selects and expands the memory engine."""
    sdfg, _, is_load = _tma_kernel_without_implementation(kernel)
    with dace.config.set_temporary('compiler', 'cuda', 'cuda_arch', value='90'):
        GPUCodegenPreprocessPipeline(memory_engine_copies=True).apply_pass(sdfg, {})
    assert _count_copy_nodes(sdfg) == 0
    labels = _tasklet_labels(sdfg)
    assert "tma_copy_tma_desc_init" in labels
    assert ("tma_copy_tma_barrier_init" in labels) == is_load
    sdfg.validate()


def test_preprocess_pipeline_memory_engine_copies_disabled():
    """By default (``compiler.cuda.memory_engine_copies`` is off) the memory engine is not selected."""
    sdfg, _, _ = _tma_kernel_without_implementation("transpose")
    with dace.config.set_temporary('compiler', 'cuda', 'cuda_arch', value='90'):
        GPUCodegenPreprocessPipeline().apply_pass(sdfg, {})
    assert not any(label.startswith("tma_copy_tma") for label in _tasklet_labels(sdfg))


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["transpose", "store"])
def test_experimental_codegen_memory_engine_copies(kernel):
    """End to end: the experimental codegen with ``compiler.cuda.memory_engine_copies`` computes correctly."""
    cp = _require_memory_engine()
    sdfg, expected, _ = _tma_kernel_without_implementation(kernel)
    A = (cp.arange(_TMA_M * _TMA_N) % 50).astype(cp.float64).reshape((_TMA_M, _TMA_N))
    B = cp.zeros_like(expected(A))
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'), \
            dace.config.set_temporary('compiler', 'cuda', 'memory_engine_copies', value=True):
        sdfg(A=A, B=B)
    cp.testing.assert_array_equal(expected(A), B)


# Polybench-derived tests: the pass must preserve numerical output on real programs.
# Init wrappers allocate the arrays and delegate to the canonical programs' ``init_array``.


def _to_gpu_sdfg(program, arrays):
    """``program`` as an SDFG with GPU kernels (32x32 blocks) and ``arrays`` in ``GPU_Global``."""
    sdfg = program.to_sdfg(simplify=True)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device
                node.gpu_block_size = (32, 32)
            elif isinstance(node, dace.nodes.AccessNode) and node.data in arrays:
                sdfg.arrays[node.data].storage = _GLOBAL
    AddThreadBlockMaps().apply_pass(sdfg, {})
    return sdfg


def _run_and_compare(program, init_fn, check_arrays, sizes, name, backend):
    """Run a program before and after ``InsertMemoryEngineBulkCopiesPipeline``, asserting numerical correctness."""
    ref_arrays = init_fn(**sizes)

    sdfg_ref = _to_gpu_sdfg(program, ref_arrays.keys())
    ref_exe = sdfg_ref.compile()

    sdfg_pass = _copy.deepcopy(sdfg_ref)
    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg_pass, {})
    # Scalar copies (e.g. ``alpha``) stay on another implementation: their box is below 16 bytes
    assert any(n.implementation == "MemoryEngineBulk" for n, _ in sdfg_pass.all_nodes_recursive()
               if isinstance(n, CopyLibraryNode))
    _assert_no_other_subset(sdfg_pass)
    sdfg_pass.expand_library_nodes()
    pass_exe = sdfg_pass.compile()

    ref_exe(**{k: v for k, v in ref_arrays.items()}, **sizes)
    ref_values = {k: ref_arrays[k].copy() for k in check_arrays}
    pass_arrays = init_fn(**sizes)
    pass_exe(**{k: v for k, v in pass_arrays.items()}, **sizes)

    for arr_name in check_arrays:
        backend.testing.assert_allclose(pass_arrays[arr_name],
                                        ref_values[arr_name],
                                        rtol=1e-10,
                                        atol=1e-12,
                                        err_msg=f"{name}: array '{arr_name}' mismatch after pass")


def _init_gemm(NI, NJ, NK, backend):
    A = backend.empty((NI, NK), dtype=backend.float64)
    B = backend.empty((NK, NJ), dtype=backend.float64)
    C = backend.empty((NI, NJ), dtype=backend.float64)
    alpha = backend.empty(1, dtype=backend.float64)
    beta = backend.empty(1, dtype=backend.float64)
    _gemm_init_array(C, A, B, alpha, beta, NI, NJ, NK)
    return {"A": A, "B": B, "C": C, "alpha": alpha, "beta": beta}


@pytest.mark.gpu
def test_polybench_gemm():
    """``InsertMemoryEngineBulkCopiesPipeline`` preserves gemm output versus the untransformed reference."""
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.Device().compute_capability) < 90:
        pytest.skip("The memory engine (TMA) requires compute capability 9.0 or newer")
    _run_and_compare(gemm, lambda **kwargs: _init_gemm(**kwargs, backend=cp), ["C"], {
        "NI": 32,
        "NJ": 64,
        "NK": 96
    }, "gemm", cp)


if __name__ == "__main__":
    pytest.main([__file__])

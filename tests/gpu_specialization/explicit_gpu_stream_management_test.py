# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for explicit GPU stream assignment and sync-tasklet insertion."""
import pytest

import dace
from dace.codegen import common
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.transformation.interstate import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_specialization_pipeline import GPUStreamPipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import InsertExplicitGPUGlobalMemoryCopies
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (STREAM_CONNECTOR,
                                                                               get_gpu_stream_array_name)

gpu_stream_pipeline = GPUStreamPipeline()

backend = common.get_gpu_backend()

_STREAM_ARRAY = get_gpu_stream_array_name()
_STREAM_VAR_PREFIX = STREAM_CONNECTOR


def _sync_tasklets(state):
    return [
        n for n in state.nodes()
        if isinstance(n, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in n.code.as_string
    ]


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_basic():
    """
    Single-component GPU program: exactly one stream, one end-of-state sync tasklet.
    Under the new path-based chain, the sync tasklet is itself a sink (no output
    edges); we verify that and its input wiring.
    """

    @dace.program
    def simple_copy(A: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:128:1] @ dace.dtypes.ScheduleType.GPU_Device:
            B[i] = A[i]

    sdfg = simple_copy.to_sdfg()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]

    syncs = _sync_tasklets(state)
    assert len(syncs) == 1, f"Expected exactly one end-of-state sync tasklet; got {len(syncs)}"
    sync = syncs[0]

    assert sync.label == "gpu_streams_synchronization", sync.label
    assert sync.side_effects is True
    assert state.out_degree(sync) == 0, "Sync tasklet must be a sink (no outgoing edges)"

    stream_conns = [c for c in sync.in_connectors if c.startswith(_STREAM_VAR_PREFIX)]
    assert len(stream_conns) == 1, f"Single-component program must sync exactly one stream; got {stream_conns}"

    # The sync's stream in-edge must come from a gpu_streams AccessNode.
    stream_in_edges = [e for e in state.in_edges(sync) if e.dst_conn in stream_conns]
    assert len(stream_in_edges) == 1
    src = stream_in_edges[0].src
    assert isinstance(src, dace.nodes.AccessNode) and src.data == _STREAM_ARRAY

    sdfg.compile()


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_extended():
    """
    Two independent connected components -> two streams -> one sync tasklet
    with two stream in-connectors.  Memcpy tasklets must also be wired to
    a stream.
    """

    @dace.program
    def independent_copies(A: dace.uint32[128], B: dace.uint32[128], C: dace.uint32[128], D: dace.uint32[128]):
        for i in dace.map[0:128:1]:
            B[i] = A[i]
        for i in dace.map[0:128:1]:
            D[i] = C[i]

    sdfg = independent_copies.to_sdfg()
    sdfg.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]

    syncs = _sync_tasklets(state)
    # Per-state syncs are fused into a single tasklet that synchronizes
    # every stream the state needs to wait on, with one
    # ``__stream_<id>`` ``gpuStream_t`` connector per stream id (the
    # offset into the ``gpu_streams`` array).
    assert len(syncs) == 1, f"Expected one fused sync tasklet (two streams); got {len(syncs)}"
    sync = syncs[0]
    assert sync.side_effects is True
    assert state.out_degree(sync) == 0
    assert len(sync.in_connectors) == 2
    for conn_name, conn_type in sync.in_connectors.items():
        assert conn_name.startswith(f"{STREAM_CONNECTOR}_"), conn_name
        assert conn_type == dace.dtypes.gpuStream_t

    # Memcpy tasklets emitted by the non-library GPU transformation still
    # need a stream connector (the library-node expansion handles its own
    # during codegen).
    memcopy_tasklets = [
        n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and f"{backend}MemcpyAsync(" in n.code.as_string
    ]
    for tasklet in memcopy_tasklets:
        assert len(tasklet.in_connectors) == 2, ("Memcpy tasklets must have one connector for the GPU stream"
                                                 " and one for the copy source/destination.")

    sdfg.compile()


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_numerical_correctness():
    """
    Simple element-wise computation, CPU vs. GPU parity.
    """
    import numpy as np

    @dace.program
    def compute(A: dace.float32[128], B: dace.float32[128], C: dace.float32[128]):
        for i in dace.map[0:128:1]:
            C[i] = A[i] * 2.0 + B[i]

    rng = np.random.default_rng(42)
    A = rng.random(128, dtype=np.float32)
    B = rng.random(128, dtype=np.float32)
    C_cpu = np.zeros(128, dtype=np.float32)
    C_gpu = np.zeros(128, dtype=np.float32)

    sdfg_cpu = compute.to_sdfg()
    sdfg_cpu(A=A.copy(), B=B.copy(), C=C_cpu)

    sdfg_gpu = compute.to_sdfg()
    sdfg_gpu.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg_gpu, {})
    sdfg_gpu(A=A.copy(), B=B.copy(), C=C_gpu)

    assert np.allclose(C_cpu, C_gpu, rtol=1e-5, atol=1e-7)
    expected = A * 2.0 + B
    assert np.allclose(C_cpu, expected, rtol=1e-5, atol=1e-7)
    assert np.allclose(C_gpu, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.new_gpu_codegen_only
def test_numerical_correctness_complex():
    """
    Two dependent maps, CPU vs. GPU parity including the intermediate array.
    """
    import numpy as np

    @dace.program
    def complex_compute(A: dace.float64[128], B: dace.float64[128], C: dace.float64[128], D: dace.float64[128]):
        for i in dace.map[0:128:1]:
            C[i] = A[i] * B[i]
        for i in dace.map[0:128:1]:
            D[i] = C[i] + A[i]

    rng = np.random.default_rng(123)
    A = rng.random(128, dtype=np.float64)
    B = rng.random(128, dtype=np.float64)
    C_cpu = np.zeros(128, dtype=np.float64)
    D_cpu = np.zeros(128, dtype=np.float64)
    C_gpu = np.zeros(128, dtype=np.float64)
    D_gpu = np.zeros(128, dtype=np.float64)

    sdfg_cpu = complex_compute.to_sdfg()
    sdfg_cpu(A=A.copy(), B=B.copy(), C=C_cpu, D=D_cpu)

    sdfg_gpu = complex_compute.to_sdfg()
    sdfg_gpu.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg_gpu, {})
    sdfg_gpu(A=A.copy(), B=B.copy(), C=C_gpu, D=D_gpu)

    assert np.allclose(C_cpu, C_gpu, rtol=1e-12, atol=1e-14)
    assert np.allclose(D_cpu, D_gpu, rtol=1e-12, atol=1e-14)
    expected_C = A * B
    expected_D = expected_C + A
    assert np.allclose(D_cpu, expected_D, rtol=1e-12, atol=1e-14)
    assert np.allclose(D_gpu, expected_D, rtol=1e-12, atol=1e-14)


def test_three_kernels_dependent_and_independent():
    """
    K1:  B = A * 2        -- produces B
    K2:  C = B + 1        -- depends on K1 through B
    K3:  E = D * 3        -- independent of K1 and K2

    K1 and K2 share one GPU stream (same weakly connected component via B);
    K3 gets its own stream; the state-end synchronization tasklet references
    both streams.
    """
    N = dace.symbol('N')

    @dace.program
    def three_kernels(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N],
                      E: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = A[i] * 2.0
        for i in dace.map[0:N]:
            C[i] = B[i] + 1.0
        for i in dace.map[0:N]:
            E[i] = D[i] * 3.0

    with dace.config.set_temporary('compiler', 'cuda', 'max_concurrent_streams', value=0):
        sdfg = three_kernels.to_sdfg(simplify=True)
        sdfg.apply_transformations_repeated(StateFusionExtended)
        sdfg.apply_gpu_transformations()
        sdfg.apply_transformations_repeated(StateFusionExtended)
        # Step 1: materialize explicit GPU memory copies so we can inspect the SDFG at that point.
        Pipeline([InsertExplicitGPUGlobalMemoryCopies()]).apply_pass(sdfg, {})

        # Step 2: run the remaining stream-specialization passes.
        Pipeline([
            NaiveGPUStreamScheduler(),
        ]).apply_pass(sdfg, {})

        kernel_states = []
        for state in sdfg.states():
            maps = [
                n for n in state.nodes()
                if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
            ]
            if maps:
                kernel_states.append((state, maps))
        assert len(kernel_states) == 1
        kernel_state, kernels = kernel_states[0]
        assert len(kernels) == 3

        def stream_id_of(map_entry):
            """Read the stream id from the wired ``gpu_streams[<i>]`` memlet
            on the kernel's stream connector. The connector name is
            uniformly ``__stream``; the id rides on the memlet subset."""
            stream_inputs = [e for e in kernel_state.in_edges(map_entry) if e.dst_conn == STREAM_CONNECTOR]
            assert len(stream_inputs) == 1
            return int(stream_inputs[0].data.subset[0][0])

        by_stream = {}
        for ker in kernels:
            by_stream.setdefault(stream_id_of(ker), []).append(ker)
        assert len(by_stream) == 2
        assert sorted(len(g) for g in by_stream.values()) == [1, 2]

        syncs = _sync_tasklets(kernel_state)
        # Per-state syncs are fused into one tasklet with N
        # ``__stream_<id>`` connectors (one per synced stream).
        assert len(syncs) == 1
        sync = syncs[0]
        assert sync.label == "gpu_streams_synchronization"
        assert sync.side_effects is True
        assert kernel_state.out_degree(sync) == 0, "Sync tasklet must be a sink under the path-based chain"
        sync_ids = set()
        for conn_name, conn_type in sync.in_connectors.items():
            assert conn_name.startswith(f"{STREAM_CONNECTOR}_"), conn_name
            assert conn_type == dace.dtypes.gpuStream_t
            inc = [e for e in kernel_state.in_edges(sync) if e.dst_conn == conn_name]
            assert len(inc) == 1
            sync_ids.add(int(inc[0].data.subset[0][0]))
        assert set(by_stream.keys()) == sync_ids
        # Body chains one ``cudaStreamSynchronize`` per ``__stream_<id>`` connector.
        for sid in sync_ids:
            assert f"{STREAM_CONNECTOR}_{sid}" in sync.code.as_string

        gpu = dace.dtypes.StorageType.GPU_Global
        cpu_like = {
            dace.dtypes.StorageType.Default,
            dace.dtypes.StorageType.CPU_Heap,
            dace.dtypes.StorageType.CPU_Pinned,
            dace.dtypes.StorageType.CPU_ThreadLocal,
        }
        copy_nodes = [n for n in kernel_state.nodes() if isinstance(n, CopyLibraryNode)]
        assert copy_nodes
        for c in copy_nodes:
            src = c.src_storage(kernel_state, kernel_state.sdfg)
            dst = c.dst_storage(kernel_state, kernel_state.sdfg)
            crosses = (src == gpu and dst in cpu_like) or (src in cpu_like and dst == gpu)
            assert crosses


# ---------------------------------------------------------------------------
# Structural sanity tests (no compile / run).
# ---------------------------------------------------------------------------


def test_empty_state():
    """An SDFG with a single empty state must pass through the pipeline without crashing."""
    sdfg = dace.SDFG("empty_sdfg")
    sdfg.add_state("empty_state")

    gpu_stream_pipeline.apply_pass(sdfg, {})

    # No stream users: no sync tasklets and no nodes in the state.
    assert len(sdfg.states()) == 1
    state = sdfg.states()[0]
    assert state.number_of_nodes() == 0
    assert _sync_tasklets(state) == []


def test_single_copy_library_node():
    """Single CopyLibraryNode (CPU->GPU) in one state: wired stream chain + sync tasklet."""
    sdfg = dace.SDFG("single_copy_node")
    sdfg.add_array("A", [128], dace.uint32, storage=dace.dtypes.StorageType.CPU_Heap)
    sdfg.add_array("B", [128], dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("copy_state")

    a = state.add_access("A")
    b = state.add_access("B")
    cp = CopyLibraryNode(name="copy_A_to_B")
    state.add_node(cp)
    state.add_edge(a, None, cp, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet("A[0:128]"))
    state.add_edge(cp, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.Memlet("B[0:128]"))

    Pipeline([
        NaiveGPUStreamScheduler(),
    ]).apply_pass(sdfg, {})

    assert _STREAM_ARRAY in sdfg.arrays
    assert STREAM_CONNECTOR in cp.in_connectors, "CopyLibraryNode must have its STREAM_CONNECTOR in-connector wired"

    stream_inputs = [e for e in state.in_edges(cp) if e.dst_conn == STREAM_CONNECTOR]
    assert len(stream_inputs) == 1
    assert isinstance(stream_inputs[0].src, dace.nodes.AccessNode)
    assert stream_inputs[0].src.data == _STREAM_ARRAY

    # One sync tasklet, and it must be a sink.
    syncs = _sync_tasklets(state)
    assert len(syncs) == 1
    assert syncs[0].side_effects is True
    assert state.out_degree(syncs[0]) == 0


def test_single_memset_library_node():
    """Single MemsetLibraryNode over a GPU buffer in one state."""
    sdfg = dace.SDFG("single_memset_node")
    sdfg.add_array("B", [128], dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("memset_state")

    b = state.add_access("B")
    ms = MemsetLibraryNode(name="memset_B")
    state.add_node(ms)
    state.add_edge(ms, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, b, None, dace.Memlet("B[0:128]"))

    Pipeline([
        NaiveGPUStreamScheduler(),
    ]).apply_pass(sdfg, {})

    assert _STREAM_ARRAY in sdfg.arrays
    assert STREAM_CONNECTOR in ms.in_connectors, "MemsetLibraryNode must have its STREAM_CONNECTOR in-connector wired"

    stream_inputs = [e for e in state.in_edges(ms) if e.dst_conn == STREAM_CONNECTOR]
    assert len(stream_inputs) == 1
    assert isinstance(stream_inputs[0].src, dace.nodes.AccessNode)
    assert stream_inputs[0].src.data == _STREAM_ARRAY

    syncs = _sync_tasklets(state)
    assert len(syncs) == 1
    assert syncs[0].side_effects is True
    assert state.out_degree(syncs[0]) == 0


def test_conditional_gpu_kernel_in_sequential_map():
    """
    Outer sequential map with a conditional guarding a GPU kernel.  After GPU
    transformations the inner kernel lives inside a nested SDFG reached via
    the outer Sequential map.  The stream pipeline must:
      * propagate the ``gpu_streams`` array down to the nested SDFG,
      * assign the inner GPU map to a stream,
      * add a sync tasklet where required.
    """

    @dace.program
    def conditional_gpu(A: dace.float64[10], B: dace.float64[128]):
        for i in dace.map[0:10] @ dace.dtypes.ScheduleType.Sequential:
            if A[i] > 0.0:
                for j in dace.map[0:128]:
                    B[j] = B[j] + 1.0

    sdfg = conditional_gpu.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    # Stream array must be present at the top level.
    assert _STREAM_ARRAY in sdfg.arrays

    # Locate the GPU kernel MapEntry wherever it ended up (top level or nested).
    gpu_maps = []
    for sub_sdfg in sdfg.all_sdfgs_recursive():
        for state in sub_sdfg.states():
            for node in state.nodes():
                if (isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device):
                    gpu_maps.append((sub_sdfg, state, node))
    assert gpu_maps, "Expected at least one GPU_Device MapEntry after apply_gpu_transformations"

    # Any SDFG that contains a GPU kernel must have the stream array declared.
    for sub_sdfg, _state, me in gpu_maps:
        assert _STREAM_ARRAY in sub_sdfg.arrays, (
            f"Nested SDFG containing a GPU kernel must have '{_STREAM_ARRAY}' declared")
        stream_conns = [c for c in me.in_connectors if c.startswith(_STREAM_VAR_PREFIX)]
        assert len(stream_conns) == 1, (f"GPU MapEntry must have exactly one stream connector, got {stream_conns}")

    # At least one sync tasklet was inserted somewhere in the hierarchy.
    any_sync = False
    for sub_sdfg in sdfg.all_sdfgs_recursive():
        for state in sub_sdfg.states():
            if _sync_tasklets(state):
                any_sync = True
                for sync in _sync_tasklets(state):
                    assert sync.side_effects is True
                    assert state.out_degree(sync) == 0
    assert any_sync, "Expected at least one stream-sync tasklet across the SDFG hierarchy"


def test_libnode_expansion_propagates_stream_to_child_libnode():
    """
    A library node whose expansion produces another library node (e.g.
    ``MatMul`` -> ``Gemm`` via ``SpecializeMatMul``) must have its stream
    binding propagated to the child. After the GPU stream pipeline +
    one level of expansion, the resulting child library node must have
    the same ``stream`` in-connector wiring as the original outer node.

    Pre-fix this fails because:
    - ``NaiveGPUStreamScheduler`` only assigns streams to ``CopyLibraryNode``
      / ``MemsetLibraryNode`` / GPU ``MapEntry`` (see ``_is_gpu_copy_or_memset``).
      ``MatMul`` is a generic GPU library node and is ignored.
    - Even if the parent were wired, ``ExpandTransformation.apply`` would
      have nothing to copy onto the child because the replacement
      (``Gemm``) declares no ``stream`` in-connector.

    The fix is twofold: extend the stream pass to cover all GPU library
    nodes, and add a follow-up pass that, after each round of library
    expansion, wires newly-introduced library nodes onto the parent's
    stream.
    """
    from dace.libraries.blas.nodes.matmul import MatMul

    M, K, N = 8, 8, 8
    sdfg = dace.SDFG("matmul_to_gemm_stream_propagation")
    sdfg.add_array("A", [M, K], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [K, N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("C", [M, N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("matmul_state")
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    matmul = MatMul("matmul")
    state.add_node(matmul)
    state.add_edge(a, None, matmul, "_a", dace.Memlet(f"A[0:{M}, 0:{K}]"))
    state.add_edge(b, None, matmul, "_b", dace.Memlet(f"B[0:{K}, 0:{N}]"))
    state.add_edge(matmul, "_c", c, None, dace.Memlet(f"C[0:{M}, 0:{N}]"))

    # Run the GPU stream pipeline on the un-expanded SDFG.
    Pipeline([
        NaiveGPUStreamScheduler(),
    ]).apply_pass(sdfg, {})

    assert _STREAM_ARRAY in sdfg.arrays, ("Stream array must be present after the pipeline runs")
    # The MatMul itself must have been wired with a `stream` in-connector
    # from a `gpu_streams` AccessNode (currently fails: scheduler ignores
    # generic GPU library nodes).
    assert STREAM_CONNECTOR in matmul.in_connectors, (
        "MatMul (a GPU library node) should be wired with a `stream` connector "
        "by the stream pipeline before it is expanded")
    matmul_stream_in = [e for e in state.in_edges(matmul) if e.dst_conn == STREAM_CONNECTOR]
    assert len(matmul_stream_in) == 1
    assert isinstance(matmul_stream_in[0].src, dace.nodes.AccessNode)
    assert matmul_stream_in[0].src.data == _STREAM_ARRAY

    # Expand exactly one level so MatMul -> Gemm (via SpecializeMatMul).
    matmul.expand(state)

    # Find the child library node that replaced MatMul.
    children = [n for n in state.nodes() if isinstance(n, dace.nodes.LibraryNode)]
    assert len(children) == 1, (f"Expected exactly one child library node after MatMul.specialize, got {len(children)}")
    child = children[0]
    assert type(child).__name__.endswith("Gemm"), (f"Expected Gemm-family child, got {type(child).__name__}")

    # The child must have inherited the parent's stream wiring.
    assert STREAM_CONNECTOR in child.in_connectors, (
        f"Child library node {type(child).__name__} (produced by expanding MatMul) "
        f"must have a `stream` in-connector inherited from the parent")
    child_stream_in = [e for e in state.in_edges(child) if e.dst_conn == STREAM_CONNECTOR]
    assert len(child_stream_in) == 1
    assert isinstance(child_stream_in[0].src, dace.nodes.AccessNode)
    assert child_stream_in[0].src.data == _STREAM_ARRAY


def test_libnode_expansion_to_nested_sdfg_wires_inner_libnodes():
    """
    A library node whose expansion produces a *nested SDFG* containing more
    library nodes (e.g. ``Cholesky`` (cuSolverDn) -> NestedSDFG{Potrf,
    Transpose, Transpose}) must have stream wiring propagated to every
    nested runtime call.

    Under the unified post-expansion pipeline, ``expand_library_nodes(
    recursive=True)`` flattens the chain in one pass; the stream
    scheduler then walks every kernel ``MapEntry`` and runtime Tasklet
    directly and wires each one's ``__stream`` connector. No follow-up
    pass / re-run is needed.
    """
    from dace.libraries.linalg.nodes.cholesky import Cholesky

    N = 8
    sdfg = dace.SDFG("cholesky_stream_propagation")
    sdfg.add_array("A", [N, N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [N, N], dace.float64, storage=dace.dtypes.StorageType.GPU_Global)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    b = state.add_access("B")
    chol = Cholesky("chol", lower=True)
    chol.implementation = "cuSolverDn"
    state.add_node(chol)
    state.add_edge(a, None, chol, "_a", dace.Memlet(f"A[0:{N}, 0:{N}]"))
    state.add_edge(chol, "_b", b, None, dace.Memlet(f"B[0:{N}, 0:{N}]"))

    # Recursive expand first (the unified pipeline does this), then run the
    # scheduler on the post-expansion shape.
    sdfg.expand_library_nodes(recursive=True)
    Pipeline([
        NaiveGPUStreamScheduler(),
    ]).apply_pass(sdfg, {})

    # Every runtime Tasklet (post-expansion) that takes a stream must have
    # its ``__stream`` connector wired to ``gpu_streams[<i>]``.
    from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (is_already_lowered_gpu_runtime_call)
    runtime_tasklets = [
        n for nsdfg in sdfg.all_sdfgs_recursive() for st in nsdfg.states() for n in st.nodes()
        if is_already_lowered_gpu_runtime_call(n)
    ]
    assert runtime_tasklets, "Cholesky cuSolverDn expansion should leave at least one runtime call Tasklet."
    for t in runtime_tasklets:
        assert STREAM_CONNECTOR in t.in_connectors, (
            f"Runtime tasklet {t.label} must have its `__stream` in-connector wired by the unified pipeline")

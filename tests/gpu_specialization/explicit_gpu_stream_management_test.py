# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.codegen import common
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.transformation.interstate import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import InsertExplicitGPUGlobalMemoryCopies
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_nodes import ConnectGPUStreamsToNodes
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
from dace.transformation.passes.gpu_specialization.helpers.gpu_helpers import (get_gpu_stream_array_name,
                                                                               get_gpu_stream_connector_name)

gpu_stream_pipeline = Pipeline([
    InsertExplicitGPUGlobalMemoryCopies(),
    NaiveGPUStreamScheduler(),
    InsertGPUStreams(),
    ConnectGPUStreamsToNodes(),
    InsertGPUStreamSyncTasklets(),
])

backend = common.get_gpu_backend()

_STREAM_ARRAY = get_gpu_stream_array_name()
_STREAM_VAR_PREFIX = get_gpu_stream_connector_name()


def _sync_tasklets(state):
    return [
        n for n in state.nodes()
        if isinstance(n, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in n.code.as_string
    ]


@pytest.mark.gpu
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
    assert len(syncs) == 1, f"Expected exactly one end-of-state sync tasklet, got {len(syncs)}"
    sync = syncs[0]
    assert sync.side_effects is True
    assert state.out_degree(sync) == 0

    stream_conns = sorted(c for c in sync.in_connectors if c.startswith(_STREAM_VAR_PREFIX))
    assert len(stream_conns) == 2, (f"Sync tasklet must carry one in-connector per synchronized stream "
                                    f"(two components -> two streams); got {stream_conns}")

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
        sdfg.save("three_kernels_gpu_copies.sdfg")

        # Step 2: run the remaining stream-specialization passes.
        Pipeline([
            NaiveGPUStreamScheduler(),
            InsertGPUStreams(),
            ConnectGPUStreamsToNodes(),
            InsertGPUStreamSyncTasklets(),
        ]).apply_pass(sdfg, {})
        sdfg.save("three_kernels_gpu_streams.sdfg")

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

        def stream_conn_of(map_entry):
            conns = [c for c in map_entry.in_connectors if c.startswith(_STREAM_VAR_PREFIX)]
            assert len(conns) == 1, f"Kernel {map_entry} must have exactly one stream connector, got {conns}"
            return conns[0]

        by_stream = {}
        for ker in kernels:
            by_stream.setdefault(stream_conn_of(ker), []).append(ker)
        assert len(by_stream) == 2
        assert sorted(len(g) for g in by_stream.values()) == [1, 2]

        syncs = _sync_tasklets(kernel_state)
        assert len(syncs) == 1
        sync = syncs[0]
        assert sync.label == "gpu_streams_synchronization"
        assert sync.side_effects is True
        assert kernel_state.out_degree(sync) == 0, "Sync tasklet must be a sink under the path-based chain"

        sync_in_conns = set(sync.in_connectors)
        assert set(by_stream.keys()).issubset(sync_in_conns)

        for stream_name in by_stream:
            assert stream_name in sync.code.as_string

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
            crosses = ((c.src_storage == gpu and c.dst_storage in cpu_like)
                       or (c.src_storage in cpu_like and c.dst_storage == gpu))
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
    cp = CopyLibraryNode(name="copy_A_to_B",
                         src_storage=dace.dtypes.StorageType.CPU_Heap,
                         dst_storage=dace.dtypes.StorageType.GPU_Global)
    state.add_node(cp)
    state.add_edge(a, None, cp, "_in", dace.Memlet("A[0:128]"))
    state.add_edge(cp, "_out", b, None, dace.Memlet("B[0:128]"))

    Pipeline([
        NaiveGPUStreamScheduler(),
        InsertGPUStreams(),
        ConnectGPUStreamsToNodes(),
        InsertGPUStreamSyncTasklets(),
    ]).apply_pass(sdfg, {})

    assert _STREAM_ARRAY in sdfg.arrays
    assert "stream" in cp.in_connectors, "CopyLibraryNode must have its 'stream' in-connector wired"

    stream_inputs = [e for e in state.in_edges(cp) if e.dst_conn == "stream"]
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
    ms = MemsetLibraryNode(name="memset_B", inputs=set(), outputs={"_out"})
    state.add_node(ms)
    state.add_edge(ms, "_out", b, None, dace.Memlet("B[0:128]"))

    Pipeline([
        NaiveGPUStreamScheduler(),
        InsertGPUStreams(),
        ConnectGPUStreamsToNodes(),
        InsertGPUStreamSyncTasklets(),
    ]).apply_pass(sdfg, {})

    assert _STREAM_ARRAY in sdfg.arrays
    assert "stream" in ms.in_connectors, "MemsetLibraryNode must have its 'stream' in-connector wired"

    stream_inputs = [e for e in state.in_edges(ms) if e.dst_conn == "stream"]
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

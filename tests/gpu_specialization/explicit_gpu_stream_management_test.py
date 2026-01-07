# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.codegen import common
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import NaiveGPUStreamScheduler
from dace.transformation.passes.gpu_specialization.insert_explicit_gpu_global_memory_copies import InsertExplicitGPUGlobalMemoryCopies
from dace.transformation.passes.gpu_specialization.insert_gpu_streams import InsertGPUStreams
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_kernels import ConnectGPUStreamsToKernels
from dace.transformation.passes.gpu_specialization.connect_gpu_streams_to_tasklets import ConnectGPUStreamsToTasklets
from dace.transformation.passes.gpu_specialization.insert_gpu_stream_sync_tasklets import InsertGPUStreamSyncTasklets
from dace.transformation.passes.gpu_specialization.gpu_stream_topology_simplification import GPUStreamTopologySimplification

gpu_stream_pipeline = Pipeline([
    InsertExplicitGPUGlobalMemoryCopies(),
    NaiveGPUStreamScheduler(),
    InsertGPUStreams(),
    ConnectGPUStreamsToKernels(),
    ConnectGPUStreamsToTasklets(),
    InsertGPUStreamSyncTasklets(),
    GPUStreamTopologySimplification(),
])

backend = common.get_gpu_backend()


@pytest.mark.gpu
def test_basic():
    """
    A simple memory copy program.
    Since the SDFG has a single connected component, exactly one GPU stream is used
    and must be synchronized at the end of the state. For each synchronized stream,
    the pipeline introduces a memlet from the synchronization tasklet to a GPU stream
    AccessNode. Therefore, it is sufficient to verify there is only one sink node with one ingoing
    edge, verify its dtype, and check for the presence of a preceeding synchronization tasklet.
    """

    @dace.program
    def simple_copy(A: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global,
                    B: dace.uint32[128] @ dace.dtypes.StorageType.GPU_Global):
        for i in dace.map[0:128:1] @ dace.dtypes.ScheduleType.GPU_Device:
            B[i] = A[i]

    sdfg = simple_copy.to_sdfg()
    gpu_stream_pipeline.apply_pass(sdfg, {})

    state = sdfg.states()[0]
    sink_nodes = state.sink_nodes()
    node = sink_nodes[0]
    assert (
        len(sink_nodes) == 1 and len(state.in_edges(node)) == 1 and isinstance(node, dace.nodes.AccessNode)
        and node.desc(state).dtype == dace.dtypes.gpuStream_t
    ), ("Only one sink node with should exist, which is a GPU stream AccessNode and it should have one ingoing edge.")

    assert (isinstance(pre, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in pre.code.as_string
            for pre in state.predecessors(node)), ("At then end of each state any used stream must be synchronized.")

    sdfg.compile()


@pytest.mark.gpu
def test_extended():
    """
    A program that performs two independent memory copies.
    The input arrays reside in host memory, and `gpu_transformations()` is applied to
    the program. As a result, the data is first copied to GPU global memory, after
    which the two copies are executed on the GPU. Since these copies form two
    independent connected components in the resulting SDFG, the naive GPU stream
    scheduler assigns them to different GPU streams.
    This test verifies that exactly two GPU streams are used, that both streams are
    synchronized at the end of the state, and that the corresponding asynchronous
    memory copy tasklets are correctly associated with their assigned streams.
    """

    @dace.program
    def independent_copies(A: dace.uint32[128], B: dace.uint32[128], C: dace.uint32[128], D: dace.uint32[128]):
        for i in dace.map[0:128:1]:
            B[i] = A[i]
        for i in dace.map[0:128:1]:
            D[i] = C[i]

    sdfg = independent_copies.to_sdfg()

    # Transform such that program can run on GPU and apply GPU stream pipeline
    sdfg.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg, {})


    # Test 1: Two GPU streams were used since we use the Naive Stream scheduler
    state = sdfg.states()[0]
    sink_nodes = state.sink_nodes()
    node = sink_nodes[0]
    assert (len(sink_nodes) == 1 and len(state.in_edges(node)) == 2 and isinstance(node, dace.nodes.AccessNode)
            and node.desc(state).dtype == dace.dtypes.gpuStream_t), (
                "Only one sink node with should exist, which is a GPU stream AccessNode and it "
                "should have two ingoing edges as original graph consisted of two connected components.")

    # Test 2: We synchronize at the end of the state
    assert (isinstance(pre, dace.nodes.Tasklet) and f"{backend}StreamSynchronize(" in pre.code.as_string
            for pre in state.predecessors(node)), ("At then end of each state any used stream must be synchronized.")

    # Test 3: Check that we have memory copy tasklets (as we perform two "Main Memory -> GPU GLobal"
    # memory copies and two "GPU Global -> Main Memory" memory copies by applying the gpu tranformation)
    # and that they use the name of the in connector of the GPU stream in the copy call
    memcopy_tasklets = [
        n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet) and f"{backend}MemcpyAsync(" in n.code.as_string
    ]
    for tasklet in memcopy_tasklets:
        assert len(tasklet.in_connectors) == 2, ("Memcpy tasklets must have two connectors "
                                                 "corresponding to the GPU stream.")
        assert "__dace_current_stream" in tasklet.in_connectors

    sdfg.compile()

@pytest.mark.gpu
def test_numerical_correctness():
    """
    Test that verifies numerical correctness by comparing CPU and GPU program outputs.
    The test creates a simple computation (element-wise multiplication and addition),
    runs it on both CPU and GPU, and verifies that the results match within tolerance.
    """
    import numpy as np

    @dace.program
    def compute(A: dace.float32[128], B: dace.float32[128], C: dace.float32[128]):
        for i in dace.map[0:128:1]:
            C[i] = A[i] * 2.0 + B[i]

    # Create test data
    rng = np.random.default_rng(42)
    A = rng.random(128, dtype=np.float32)
    B = rng.random(128, dtype=np.float32)
    C_cpu = np.zeros(128, dtype=np.float32)
    C_gpu = np.zeros(128, dtype=np.float32)

    # Run on CPU
    sdfg_cpu = compute.to_sdfg()
    sdfg_cpu(A=A.copy(), B=B.copy(), C=C_cpu)

    # Run on GPU
    sdfg_gpu = compute.to_sdfg()
    sdfg_gpu.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg_gpu, {})
    sdfg_gpu(A=A.copy(), B=B.copy(), C=C_gpu)

    # Verify numerical correctness
    assert np.allclose(C_cpu, C_gpu, rtol=1e-5, atol=1e-7), (
        f"CPU and GPU results do not match. Max difference: {np.max(np.abs(C_cpu - C_gpu))}"
    )

    # Verify expected result
    expected = A * 2.0 + B
    assert np.allclose(C_cpu, expected, rtol=1e-5, atol=1e-7), (
        "CPU result does not match expected computation"
    )
    assert np.allclose(C_gpu, expected, rtol=1e-5, atol=1e-7), (
        "GPU result does not match expected computation"
    )


@pytest.mark.gpu
def test_numerical_correctness_complex():
    """
    Test numerical correctness for a more complex computation involving
    multiple operations and dependencies between array elements.
    """
    import numpy as np

    @dace.program
    def complex_compute(A: dace.float64[128], B: dace.float64[128], 
                       C: dace.float64[128], D: dace.float64[128]):
        # First map: C = A * B
        for i in dace.map[0:128:1]:
            C[i] = A[i] * B[i]
        
        # Second map: D = C + A (depends on result of first map)
        for i in dace.map[0:128:1]:
            D[i] = C[i] + A[i]

    # Create test data
    rng = np.random.default_rng(123)
    A = rng.random(128, dtype=np.float64)
    B = rng.random(128, dtype=np.float64)
    C_cpu = np.zeros(128, dtype=np.float64)
    D_cpu = np.zeros(128, dtype=np.float64)
    C_gpu = np.zeros(128, dtype=np.float64)
    D_gpu = np.zeros(128, dtype=np.float64)

    # Run on CPU
    sdfg_cpu = complex_compute.to_sdfg()
    sdfg_cpu(A=A.copy(), B=B.copy(), C=C_cpu, D=D_cpu)

    # Run on GPU
    sdfg_gpu = complex_compute.to_sdfg()
    sdfg_gpu.apply_gpu_transformations()
    gpu_stream_pipeline.apply_pass(sdfg_gpu, {})
    sdfg_gpu(A=A.copy(), B=B.copy(), C=C_gpu, D=D_gpu)

    # Verify numerical correctness for intermediate result C
    assert np.allclose(C_cpu, C_gpu, rtol=1e-12, atol=1e-14), (
        f"CPU and GPU results for C do not match. Max difference: {np.max(np.abs(C_cpu - C_gpu))}"
    )

    # Verify numerical correctness for final result D
    assert np.allclose(D_cpu, D_gpu, rtol=1e-12, atol=1e-14), (
        f"CPU and GPU results for D do not match. Max difference: {np.max(np.abs(D_cpu - D_gpu))}"
    )

    # Verify expected results
    expected_C = A * B
    expected_D = expected_C + A
    assert np.allclose(D_cpu, expected_D, rtol=1e-12, atol=1e-14), (
        "CPU result does not match expected computation"
    )
    assert np.allclose(D_gpu, expected_D, rtol=1e-12, atol=1e-14), (
        "GPU result does not match expected computation"
    )
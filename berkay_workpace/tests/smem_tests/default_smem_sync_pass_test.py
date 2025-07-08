import dace
import dace.sdfg.nodes as nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.shared_memory_synchronization import DefaultSharedMemorySync

import pytest
"""
Simple tests checking core functionality of the "DefaultSharedMemorySync" pass.
"""


@pytest.mark.gpu
def test_scalar_multiplic():
    """
    Constructs an SDFG that performs scalar multiplication on a vector.

    In this test, a sequential loop is placed inside the GPU kernel, reusing shared memory.
    As a result, the 'DefaultSharedMemorySync' pass should insert a "__syncthreads();"
    at the end of each iteration to ensure correctness.

    Note: This test is designed to evaluate where the 'DefaultSharedMemorySync' pass places
    synchronization tasklets. In this particular example, the inserted synchronizations are
    not strictly necessary and could be avoided with more advanced analysis, which is beyond
    the scope of this pass.
    """

    #----------------- Build test program/SDFG--------------------

    # Create SDFG and state
    sdfg = dace.SDFG("scalarMultiplication_smem")
    state = sdfg.add_state("main")

    # Add arrays
    sdfg.add_array("A", (128, ), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("scalar", dace.uint32)
    sdfg.add_array("S", (32, ),
                   dace.uint32,
                   storage=dace.dtypes.StorageType.GPU_Shared,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.Scope)

    # Add access nodes
    a_acc = state.add_read("A")
    a_store = state.add_write("A")
    scalar_acc = state.add_access("scalar")
    s_acc = state.add_access("S")

    # Sequential map (outermost)
    seq_map_entry, seq_map_exit = state.add_map(
        "seq_map",
        dict(k="0:4"),
        schedule=dace.dtypes.ScheduleType.Sequential,
    )

    # GPU Device map
    gpu_map_entry, gpu_map_exit = state.add_map(
        "gpu_map",
        dict(i="0:32:32"),
        schedule=dace.dtypes.ScheduleType.GPU_Device,
    )

    #  GPU TB map
    tb_map_entry, tb_map_exit = state.add_map(
        "tb",
        dict(j="0:32"),
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock,
    )

    # Add tasklets for A -> S -> B
    tasklet1 = state.add_tasklet("addMult",
                                 inputs={"__inp_A", "__inp_scalar"},
                                 outputs={"__out"},
                                 code="__out = __inp_A * __inp_scalar;",
                                 language=dace.dtypes.Language.CPP)

    tasklet2 = state.add_tasklet("store_to_global",
                                 inputs={"__inp"},
                                 outputs={"__out"},
                                 code="__out = __inp;",
                                 language=dace.dtypes.Language.CPP)

    # Edges

    # A and scalar to first map
    state.add_edge(a_acc, None, gpu_map_entry, None, dace.Memlet("A[0:128]"))
    state.add_edge(scalar_acc, None, gpu_map_entry, None, dace.Memlet("scalar[0]"))

    # Add both down to last map, the threadblock map
    state.add_edge(gpu_map_entry, None, seq_map_entry, None, dace.Memlet("A[0:128]"))
    state.add_edge(gpu_map_entry, None, seq_map_entry, None, dace.Memlet("scalar[0]"))
    state.add_edge(seq_map_entry, None, tb_map_entry, None, dace.Memlet("A[32 * k: 32 * (k+1)]"))
    state.add_edge(seq_map_entry, None, tb_map_entry, None, dace.Memlet("scalar[0]"))

    # connect to tasklets
    state.add_edge(tb_map_entry, None, tasklet1, "__inp_A", dace.Memlet("A[j + 32* k]"))
    state.add_edge(tb_map_entry, None, tasklet1, "__inp_scalar", dace.Memlet("scalar[0]"))
    state.add_edge(tasklet1, "__out", s_acc, None, dace.Memlet("S[j]"))
    state.add_edge(s_acc, None, tasklet2, "__inp", dace.Memlet("S[j]"))

    # connect to all map exit nodes and then back to A to store back
    state.add_edge(tasklet2, "__out", tb_map_exit, None, dace.Memlet("A[j + 32* k]"))
    state.add_edge(tb_map_exit, None, seq_map_exit, None, dace.Memlet("A[32 * k: 32 * (k+1)]"))
    state.add_edge(seq_map_exit, None, gpu_map_exit, None, dace.Memlet("A[0:128]"))
    state.add_edge(gpu_map_exit, None, a_store, None, dace.Memlet("A[0:128]"))

    sdfg.fill_scope_connectors()

    #----------------- Apply pass --------------------

    DefaultSharedMemorySync().apply_pass(sdfg, None)

    #----------------- Check correct insertion of sync tasklets --------------------

    # s_acc has a sync tasklet successor
    found = None
    for succ in state.successors(s_acc):
        if (hasattr(succ, "_label") and succ._label == "pre_sync_barrier" and isinstance(succ, nodes.Tasklet)
                and "__syncthreads();" in succ.code.code):
            found = succ
            break

    assert found is not None, "There should be a synchronization tasklet after the shared memory access"

    # smem is reused in seq map, so we need synchronization after each iteration
    found = None
    for pred in state.predecessors(seq_map_exit):
        if (hasattr(pred, "_label") and pred._label == "post_sync_barrier" and isinstance(pred, nodes.Tasklet)
                and "__syncthreads();" in pred.code.code):
            found = pred
            break

    assert found is not None, "There should be a synchronization tasklet after each iteration of the sequential map"


@pytest.mark.gpu
def test_scalar_multiplic_special():
    """
    Constructs an SDFG that performs scalar multiplication on a vector.

    Similar to 'test_scalar_multiplic()', but now, since the sequential map
    only iterates once, there is no post synchronization required and should be
    omitted (although having it would not lead to wrong computations).

    """

    #----------------- Build test program/SDFG--------------------

    # Create SDFG and state
    sdfg = dace.SDFG("scalarMultiplication_smem")
    state = sdfg.add_state("main")

    # Add arrays
    sdfg.add_array("A", (32, ), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("scalar", dace.uint32)
    sdfg.add_array("S", (32, ),
                   dace.uint32,
                   storage=dace.dtypes.StorageType.GPU_Shared,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.Scope)

    # Add access nodes
    a_acc = state.add_read("A")
    a_store = state.add_write("A")
    scalar_acc = state.add_access("scalar")
    s_acc = state.add_access("S")

    # Sequential map (outermost)
    seq_map_entry, seq_map_exit = state.add_map(
        "seq_map",
        dict(k="0:1"),
        schedule=dace.dtypes.ScheduleType.Sequential,
    )

    # GPU Device map
    gpu_map_entry, gpu_map_exit = state.add_map(
        "gpu_map",
        dict(i="0:32:32"),
        schedule=dace.dtypes.ScheduleType.GPU_Device,
    )

    #  GPU TB map
    tb_map_entry, tb_map_exit = state.add_map(
        "tb",
        dict(j="0:32"),
        schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock,
    )

    # Add tasklets for A -> S -> B
    tasklet1 = state.add_tasklet("addMult",
                                 inputs={"__inp_A", "__inp_scalar"},
                                 outputs={"__out"},
                                 code="__out = __inp_A * __inp_scalar;",
                                 language=dace.dtypes.Language.CPP)

    tasklet2 = state.add_tasklet("store_to_global",
                                 inputs={"__inp"},
                                 outputs={"__out"},
                                 code="__out = __inp;",
                                 language=dace.dtypes.Language.CPP)

    # Edges

    # A and scalar to first map
    state.add_edge(a_acc, None, gpu_map_entry, None, dace.Memlet("A[0:32]"))
    state.add_edge(scalar_acc, None, gpu_map_entry, None, dace.Memlet("scalar[0]"))

    # Add both down to last map, the threadblock map
    state.add_edge(gpu_map_entry, None, seq_map_entry, None, dace.Memlet("A[0:32]"))
    state.add_edge(gpu_map_entry, None, seq_map_entry, None, dace.Memlet("scalar[0]"))
    state.add_edge(seq_map_entry, None, tb_map_entry, None, dace.Memlet("A[32 * k: 32 * (k+1)]"))
    state.add_edge(seq_map_entry, None, tb_map_entry, None, dace.Memlet("scalar[0]"))

    # connect to tasklets
    state.add_edge(tb_map_entry, None, tasklet1, "__inp_A", dace.Memlet("A[j + 32* k]"))
    state.add_edge(tb_map_entry, None, tasklet1, "__inp_scalar", dace.Memlet("scalar[0]"))
    state.add_edge(tasklet1, "__out", s_acc, None, dace.Memlet("S[j]"))
    state.add_edge(s_acc, None, tasklet2, "__inp", dace.Memlet("S[j]"))

    # connect to all map exit nodes and then back to A to store back
    state.add_edge(tasklet2, "__out", tb_map_exit, None, dace.Memlet("A[j + 32* k]"))
    state.add_edge(tb_map_exit, None, seq_map_exit, None, dace.Memlet("A[32 * k: 32 * (k+1)]"))
    state.add_edge(seq_map_exit, None, gpu_map_exit, None, dace.Memlet("A[0:32]"))
    state.add_edge(gpu_map_exit, None, a_store, None, dace.Memlet("A[0:32]"))

    sdfg.fill_scope_connectors()

    #----------------- Apply pass --------------------

    DefaultSharedMemorySync().apply_pass(sdfg, None)

    #----------------- Check correct insertion of sync tasklets --------------------

    # s_acc has a sync tasklet successor
    found = None
    for succ in state.successors(s_acc):
        if (hasattr(succ, "_label") and succ._label == "pre_sync_barrier" and isinstance(succ, nodes.Tasklet)
                and "__syncthreads();" in succ.code.code):
            found = succ
            break

    assert found is not None, "There should be a synchronization tasklet after the shared memory access"

    # smem is NOT reused in seq map
    found = None
    for pred in state.predecessors(seq_map_exit):
        if (hasattr(pred, "_label") and pred._label == "post_sync_barrier" and isinstance(pred, nodes.Tasklet)
                and "__syncthreads();" in pred.code.code):
            found = pred
            break

    assert found is None, "The DefaultSharedMemorySync pass should not have inserted at the end of the sequential map body"


@pytest.mark.gpu
def test_scalar_multiplic_loopRegion():
    """
    Constructs an SDFG that performs scalar multiplication on a vector.

    Analogous to 'test_scalar_multiplic()', where a for loop instead of a sequential map
    is used.
    """

    #----------------- Build test program/SDFG--------------------

    sdfg = dace.SDFG("scalarMultiplication_smem")
    state = sdfg.add_state("main")

    # Arrays and access nodes
    sdfg.add_array("A", (128, ), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    sdfg.add_scalar("scalar", dace.uint32)
    a_acc = state.add_read("A")
    a_store = state.add_write("A")
    scalar_acc = state.add_access("scalar")

    # Device and thread-block maps
    gpu_map_entry, gpu_map_exit = state.add_map("gpu_map",
                                                dict(i="0:32:32"),
                                                schedule=dace.dtypes.ScheduleType.GPU_Device)
    tb_map_entry, tb_map_exit = state.add_map("tb", dict(j="0:32"), schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)

    # Nested SDFG setup
    inner_sdfg = dace.SDFG('nested_sdfg')
    nested = state.add_nested_sdfg(inner_sdfg, sdfg, inputs={'__inp_A', '__inp_scalar'}, outputs={'tmp_ret'})

    loopreg = LoopRegion("loop", "k < 4", "k", "k = 0", "k = (k + 1)", False, inner_sdfg)
    inner_sdfg.add_node(loopreg)
    inner_state = loopreg.add_state("use_smem")

    # Shared memory and result
    inner_sdfg.add_array("S", (32, ), dace.uint32, storage=dace.dtypes.StorageType.GPU_Shared, transient=True)
    inner_sdfg.add_scalar("tmp_ret", dace.uint32)
    s_acc = inner_state.add_access("S")
    ret = inner_state.add_write("tmp_ret")

    # Tasklets
    tasklet1 = inner_state.add_tasklet("assign_to_smem",
                                       inputs={},
                                       outputs={"__out1"},
                                       code="__out1 = __inp_A[j + 32 * k]",
                                       language=dace.dtypes.Language.CPP)
    tasklet2 = inner_state.add_tasklet("addMult",
                                       inputs={"__inp2"},
                                       outputs={"__out2"},
                                       code="__out2 = __inp2 * __inp_scalar;",
                                       language=dace.dtypes.Language.CPP)

    # Main SDFG edges
    state.add_edge(a_acc, None, gpu_map_entry, None, dace.Memlet("A[0:128]"))
    state.add_edge(scalar_acc, None, gpu_map_entry, None, dace.Memlet("scalar[0]"))
    state.add_edge(gpu_map_entry, None, tb_map_entry, None, dace.Memlet("A[0:128]"))
    state.add_edge(gpu_map_entry, None, tb_map_entry, None, dace.Memlet("scalar[0]"))
    state.add_edge(tb_map_entry, None, nested, "__inp_A", dace.Memlet("A[j : j + 97 : 32]"))
    state.add_edge(tb_map_entry, None, nested, "__inp_scalar", dace.Memlet("scalar[0]"))
    state.add_edge(nested, "tmp_ret", tb_map_exit, None, dace.Memlet("A[j : j + 97 : 32]"))
    state.add_edge(tb_map_exit, None, gpu_map_exit, None, dace.Memlet("A[0:128]"))
    state.add_edge(gpu_map_exit, None, a_store, None, dace.Memlet("A[0:128]"))

    # Inner SDFG edges
    inner_state.add_edge(tasklet1, "__out1", s_acc, None, dace.Memlet("S[j]"))
    inner_state.add_edge(s_acc, None, tasklet2, "__inp2", dace.Memlet("S[j]"))
    inner_state.add_edge(tasklet2, "__out2", ret, None, dace.Memlet("S[j]"))

    sdfg.fill_scope_connectors()

    #----------------- Apply pass --------------------

    DefaultSharedMemorySync().apply_pass(sdfg, None)

    #----------------- Check correct insertion of sync tasklets --------------------

    try:
        # there should be only one successor of the ret accessNode, which is a sync tasklet
        post_sync_tasklet = inner_state.successors(ret)[0]
        assert "__syncthreads();" in post_sync_tasklet.code.code, "Post synchronization tasklet is not correctly inserted"
    except:
        # Any other weird failures
        assert False, "Post synchronization tasklet is not correctly inserted"

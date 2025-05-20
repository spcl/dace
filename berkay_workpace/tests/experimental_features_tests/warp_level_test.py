import dace
import pytest
import cupy as cp

from IPython.display import Code
from dace.config import Config


####################### Testing correct mapping of indices to WarpIds ##################

# NOTE: Focus in these section is not on the tasklet (just used to have a simple 
# verification option) and the SDFG is not correct, dataFlow to warps includes 32 elements
# and not only 1 element. But there is no support for correct representation (yet). However,
# the construction of the warpIds is not affected by this. Correct SDFGs appear in the next 
# test section 
@pytest.mark.gpu
@pytest.mark.parametrize("start, end, stride", [
    (0, 32, 1),
    (3, 16, 1),
    (5, 17, 3)
])
def test_warp_map_single_TB(start, end, stride):
    @dace.program
    def simple_warp_map(A: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global):
        """
        1D check with different start, end and strides.
        """
        for i in dace.map[0:1024:1024] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:1024] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                for _ in dace.map[start:end:stride] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[j] = result


    sdfg = simple_warp_map.to_sdfg()

    A = cp.ones(1024, dtype=cp.uint32) 
    B = cp.zeros(1024, dtype=cp.uint32) 

    sdfg(A=A, B=B)

    expected = cp.full(1024, 0, dtype=cp.uint32)
    for tid in range(1024):
        warpId = tid // 32
        if warpId in range(start, end, stride):
            expected[tid] = 32

    cp.testing.assert_array_equal(B, expected)




@pytest.mark.gpu
@pytest.mark.parametrize("start, end, stride", [
    (2, 16, 6),
    (3, 15, 3)
])
def test_warp_map_multiple_TB(start, end, stride):
    @dace.program
    def multTB_warp_map(A: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global):
        """
        The case where we have more than one ThreadBlock.
        """
        for i in dace.map[0:1024:512] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:512] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                for _ in dace.map[start:end:stride] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[i + j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[i + j] = result


    sdfg = multTB_warp_map.to_sdfg()

    A = cp.ones(1024, dtype=cp.uint32) 
    B = cp.zeros(1024, dtype=cp.uint32) 

    sdfg(A=A, B=B)

    expected = cp.full(1024, 0, dtype=cp.uint32)
    for block_start in range(0, 1024, 512):
        for tid in range(512):
            warpId = tid // 32
            if warpId in range(start, end, stride):
                expected[block_start + tid] = 32

    cp.testing.assert_array_equal(B, expected)



@pytest.mark.gpu
@pytest.mark.parametrize("b1, e1, s1, b2, e2, s2", [
    (0, 4, 1, 0, 4, 1),
    (0, 3, 2, 0, 5, 3),
])
def test_warp_map_2D(b1, e1, s1, b2, e2, s2):
    @dace.program
    def multTB_warp_map_2D(A: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global):
        """
        Simple functionality check of 2D maps, focus is on 2D and less on multible TB.
        """
        for i in dace.map[0:1024:512] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:512] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                for k, l in dace.map[b1:e1:s1, b2:e2:s2] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[i + j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[i + j] = result


    sdfg = multTB_warp_map_2D.to_sdfg()

    A = cp.ones(1024, dtype=cp.uint32) 
    B = cp.zeros(1024, dtype=cp.uint32) 

    sdfg(A=A, B=B)

    # Check whether result is as expected
    expected = cp.full(1024, 0, dtype=cp.uint32)
    for block_start in range(0, 1024, 512):
        for tid in range(512):
            warpId = (tid // 32)
            if warpId >= e1 * e2:
                continue
            warpIdx = (warpId % e2 ) 
            warpIdy = (warpId // e2 ) % e1
            if (warpIdx - b2) % s2 == 0 and (warpIdy - b1) % s1 == 0:
                expected[block_start + tid] = 32

        
    cp.testing.assert_array_equal(B, expected)




@pytest.mark.gpu
@pytest.mark.parametrize("b1, e1, s1, b2, e2, s2, b3, e3, s3", [
    (0, 4, 1, 0, 4, 2, 0, 2, 1),
    (0, 3, 2, 1, 5, 3, 1, 2, 1),
])
def test_warp_map_3D(b1, e1, s1, b2, e2, s2, b3, e3, s3):
    @dace.program
    def warp_map_3D(A: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[1024] @ dace.dtypes.StorageType.GPU_Global):
        """
        Simple functionality check of 3D maps
        """
        for i in dace.map[0:1024:1024] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:1024] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                for k, l, m in dace.map[b1:e1:s1, b2:e2:s2, b3:e3:s3] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[i + j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[i + j] = result


    sdfg = warp_map_3D.to_sdfg()

    A = cp.ones(1024, dtype=cp.uint32) 
    B = cp.zeros(1024, dtype=cp.uint32) 

    sdfg(A=A, B=B)

    # Check whether result is as expected
    expected = cp.full(1024, 0, dtype=cp.uint32)
    for block_start in range(0, 1024, 1024):
        for tid in range(1024):
            warpId = (tid // 32)
            if warpId >= e1 * e2 * e3:
                continue
            warpIdx = warpId % e3 
            warpIdy = (warpId // e3 ) % e2
            warpIdz = (warpId // (e3 * e2) ) % e1
            if ((warpIdx - b3) % s3 == 0 and warpIdx >= b3 and
                (warpIdy - b2) % s2 == 0 and warpIdx >= b2 and
                (warpIdz - b1) % s1 == 0 and warpIdx >= b1):
                expected[block_start + tid] = 32

        
    cp.testing.assert_array_equal(B, expected)





@pytest.mark.gpu
@pytest.mark.parametrize("bs, ns", [(512, 1024), (1024, 2048)])
def test_symbolic_warp_map(bs, ns):

    BS = dace.symbol('BS')
    NS = dace.symbol('NS')

    START = dace.symbol('START')
    WS = dace.symbol('WS')
    STRIDE = dace.symbol('STRIDE')

    start = 2
    stride = 3
    ws = bs // 32
    @dace.program
    def symbolic_warp_map(A: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global):
        """
        Focus is in the use of symbolic variables in the MAP.
        """
        for i in dace.map[0:NS:BS] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:BS] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:

                for k in dace.map[START:WS:STRIDE] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[i + j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[i + j] = result


    sdfg = symbolic_warp_map.to_sdfg()

    A = cp.ones(ns, dtype=cp.uint32) 
    B = cp.zeros(ns, dtype=cp.uint32) 

    sdfg(A=A, B=B, START= start, WS=ws, STRIDE=stride, BS=bs, NS=ns)

    expected = cp.full(ns, 0, dtype=cp.uint32)
    for block_start in range(0, ns, bs):
        for tid in range(bs):
            warpId = tid // 32
            if warpId in range(start, ws, stride):
                expected[block_start + tid] = 32

    cp.testing.assert_array_equal(B, expected)







@pytest.mark.gpu
def test_dynamic_warpSize_warp_map():

    STRIDE = 3 # just smth else than 1, 1 is easy to pass
    BS = dace.symbol('BS')
    NS = dace.symbol('NS')

    bs = 1024
    ns = 2024
    @dace.program
    def symbolic_warp_map(A: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global, B: dace.uint32[NS] @ dace.dtypes.StorageType.GPU_Global):
        """
        What if warpSize is determined at runtime.
        """
        for i in dace.map[0:NS:BS] @ dace.dtypes.ScheduleType.GPU_Device:
            for j in dace.map[0:BS] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                ws = bs // 32
                for k in dace.map[0:ws:STRIDE] @ dace.dtypes.ScheduleType.GPU_Warp:
                    mask = 0xffffffff
                    value = A[i + j]
                    result = dace.define_local_scalar(dace.uint32)
                    with dace.tasklet(dace.Language.CPP):
                        inp_mask << mask
                        inp_value << value
                        out_result >> result
                        """
                        out_result = __reduce_add_sync(inp_mask, inp_value);
                        """
                    
                    B[i + j] = result


    sdfg = symbolic_warp_map.to_sdfg()

    A = cp.ones(ns, dtype=cp.uint32) 
    B = cp.zeros(ns, dtype=cp.uint32) 

    sdfg(A=A, B=B, BS=bs, NS=ns)

    expected = cp.full(ns, 0, dtype=cp.uint32)
    for block_start in range(0, ns, bs):
        for tid in range(bs):
            ws = bs // 32
            warpId = tid // 32
            if warpId in range(0, ws, STRIDE):
                expected[block_start + tid] = 32

    cp.testing.assert_array_equal(B, expected)

####################### Testing simple warplevel programs #################

@pytest.mark.gpu
def test_warp_reduce_add():
    """
    Best way to understand this is to copy paste it and
    to look at the sdfg. A simple explanation: It tests whether
    the most basic functionality of warp maps work and whether
    we can use "__reduce_add_sync(mask, value)" on by definining a
    custom tasklet.
    """

    # Generate framework
    sdfg = dace.SDFG("Warp_test_1")
    state = sdfg.add_state("main")

    # Generate access nodes
    a_dev = sdfg.add_array("A", (32,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    b_dev = sdfg.add_array("B", (32,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    a_acc = state.add_access("A")
    b_acc = state.add_access("B")


    # Generate maps, connect entries with access data
    gpu_map_entry, gpu_map_exit = state.add_map(name = "GPU_Map",
                                                ndrange = dict(i='0:32:32'),
                                                schedule = dace.dtypes.ScheduleType.GPU_Device)
    state.add_edge(a_acc, None, gpu_map_entry, None, dace.memlet.Memlet('A[0:32]'))


    tblock_map_entry, tblock_map_exit = state.add_map(name = "Block_Map",
                                                    ndrange = dict(j='0:32'),
                                                    schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock)
    state.add_edge(gpu_map_entry, None, tblock_map_entry, None, dace.memlet.Memlet('A[0:32]'))


    tasklet, warp_scope_entry, warp_scope_exit = state.add_mapped_tasklet(
        name='WarpLevel_Operation',
        map_ranges=dict(_='0:1'),
        inputs=dict(inp=dace.Memlet('A[0:32]', volume=32)),
        code=
"""
value = inp[j]
out = __reduce_add_sync(0xFFFFFFFF, value);
""",
        outputs=dict(out=dace.Memlet("B[j]")),
        schedule=dace.dtypes.ScheduleType.GPU_Warp
    )

    state.add_edge(tblock_map_entry, None, warp_scope_entry, None, dace.memlet.Memlet('A[0:32]'))

    # Connect Exit nodes
    state.add_edge(warp_scope_exit, None, tblock_map_exit, None, dace.memlet.Memlet('B[j]'))
    state.add_edge(tblock_map_exit, None, gpu_map_exit, None, dace.memlet.Memlet('B[j]'))
    state.add_edge(gpu_map_exit, None, b_acc, None, dace.memlet.Memlet('B[0:32]'))

    sdfg.fill_scope_connectors()

    A = cp.ones(32, dtype=cp.uint32) 
    B = cp.zeros(32, dtype=cp.uint32) 

    sdfg(A=A, B=B)

    all_32 = cp.full(32, 32, dtype=cp.uint32)
    cp.testing.assert_array_equal(B, all_32)



@pytest.mark.gpu
def test_warp_shfl_op():
    """
    Best way to understand this is to copy paste it and
    to look at the sdfg. A simple explanation: It tests now another
    warpLevel primitive, namely __shfl_down_sync and __shfl_up_sync.
    """
    sdfg = dace.SDFG("Warp_test_1")
    state = sdfg.add_state("main")

    # Generate access nodes
    a_dev = sdfg.add_array("A", (32,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    b_dev = sdfg.add_array("B", (32,), dace.uint32, dace.dtypes.StorageType.GPU_Global)
    a_acc = state.add_access("A")
    b_acc = state.add_access("B")


    # Generate maps, connect entries with access data
    gpu_map_entry, gpu_map_exit = state.add_map(name = "GPU_Map",
                                                ndrange = dict(i='0:32:32'),
                                                schedule = dace.dtypes.ScheduleType.GPU_Device)
    state.add_edge(a_acc, None, gpu_map_entry, None, dace.memlet.Memlet('A[0:32]'))


    tblock_map_entry, tblock_map_exit = state.add_map(name = "Block_Map",
                                                    ndrange = dict(j='0:32'),
                                                    schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock)
    state.add_edge(gpu_map_entry, None, tblock_map_entry, None, dace.memlet.Memlet('A[0:32]'))


    tasklet, warp_scope_entry, warp_scope_exit = state.add_mapped_tasklet(
        name='WarpLevel_Operation',
        map_ranges=dict(_='0:1'),
        inputs=dict(inp=dace.Memlet('A[0:32]', volume=32)),
        code=
"""
tid = j;
value = inp[tid];
up = __shfl_down_sync(0xFFFFFFFF, value, 16);
low = __shfl_up_sync(0xFFFFFFFF, value, 16);
if tid < 16:
    value = up;
else:
    value = low
out= value

""",
        outputs=dict(out=dace.Memlet("B[j]")),
        schedule=dace.dtypes.ScheduleType.GPU_Warp
    )

    state.add_edge(tblock_map_entry, None, warp_scope_entry, None, dace.memlet.Memlet('A[0:32]'))

    # Connect Exit nodes
    state.add_edge(warp_scope_exit, None, tblock_map_exit, None, dace.memlet.Memlet('B[j]'))
    state.add_edge(tblock_map_exit, None, gpu_map_exit, None, dace.memlet.Memlet('B[j]'))
    state.add_edge(gpu_map_exit, None, b_acc, None, dace.memlet.Memlet('B[0:32]'))

    sdfg.fill_scope_connectors()

    A = cp.array([0 if False else i for i in range(32)], dtype=cp.uint32)
    B = cp.zeros(32, dtype=cp.uint32)

    sdfg(A=A, B=B)

    expected = cp.array(cp.concatenate((A[16:32], A[0:16])))
    cp.testing.assert_array_equal(B,expected)






if __name__ == '__main__':
    
    # Warnings are ignored
    #test_warp_map(0, 32, 1)
    pytest.main(["-v", "-p", "no:warnings", __file__])

    # Use this if you want to see the warning
    # pytest.main(["-v", __file__])
import dace
import numpy as np
from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG


# SDFG APIs
def handcrafted_sdfg_scalar_add():
    sdfg = dace.SDFG('handcrafted_sdfg')
   

    ################OTHER################
    # other data nodes
    sdfg.add_symbol('Symbol', dace.int64)   # symbol can't be added in a state
    sdfg.add_constant('constant_bool', True)  # constant
 
    ################DEPRICATED################
    # data(scalar, symbol, array, constant, stream, transient) - everything is depricated
    sdfg.add_array('Array_normal', [2, 2], dace.int64, storage=dace.StorageType.Default, 
                   transient=False) # normal array
    sdfg.add_array('Array_transient', [2, 2], dace.int64, storage=dace.StorageType.Default, 
                   transient=True)  #Transiant
    sdfg.add_array('Array_onGPU', [2, 2], dace.int64, storage=dace.StorageType.GPU_Global, 
                   transient=False)  #on GPU
    # sdfg.add_stream('stream', dace.float32, transient=True, buffer_size=10)  # stream
    # sdfg.add_transient('transient', [2, 2], dace.int64)  # transient
    # sdfg.add_scalar('a_scalar', dace.int32)
   
    
    ############################################
    root = sdfg.add_state('top_level_state', is_start_block=True, is_start_state=True)
    ################USE THIS################
    A = root.add_access("Array_normal")
    B = root.add_access("Array_transient")
    C = root.add_access("Array_onGPU")
    # D = root.add_access("stream")
    # E = root.add_access("transient")
    # F = root.add_access("a_scalar")

    ################MEMLET################


    # # state
    middle =  sdfg.add_state('middle_level_state', is_start_block=False, is_start_state=False)
    exit = sdfg.add_state('bottom_level_state', is_start_block=False, is_start_state=False)

    # cfg
    sdfg.add_edge(root, middle, dace.InterstateEdge())
    sdfg.add_edge(middle, exit, dace.InterstateEdge())

        # dfg edges/ Memlets
    root.add_nedge(A, B, dace.Memlet("Array_normal[0]"))
    root.add_edge(B, None, C, None, dace.Memlet("Array_transient[0]"))
    print("Total edges", root.number_of_edges())
    # root.add_nedge(A, middle, dace.Memlet("Array_normal[0]"))

    # # tasklet
    # tasklet = root.add_tasklet('add',  {'tmp_A', 'tmp_B'}, {'tmp_C'}, 'tmp_C = tmp_A + tmp_B', language=dace.Language.Python)
    
    # # edges inside DFG/Memlet
    # root.add_edge(A, None, tasklet, "tmp_A", dace.Memlet("A[0]"))
    # root.add_edge(B, None, tasklet, "tmp_B", dace.Memlet("B[0]"))
    # root.add_edge(tasklet, "tmp_C", C, None, dace.Memlet("C[0]"))

    sdfg()  # uncomment for upstream dace to codegen
    code = sdfg.generate_code()[0].clean_code

def structure():
    sdfg = dace.SDFG('structure')
    state = sdfg.add_state('state')

    sdfg()
    code = sdfg.generate_code()[0].clean_code

def vector_add():
    sdfg = dace.SDFG('vector_add')
    #########GLOBAL VARIABLES#########
    # # data(vector add)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [10], dace.float64)  
    sdfg.add_array('C', [10], dace.float64)
    
    ###########STATE, CFG, GLOBAL DATA################
    # # add state
    state = sdfg.add_state('sum', is_start_block=True)
    a = state.add_read('A')
    b = state.add_read('B')
    c = state.add_write('C')

    ###########DFG################
    # Add nodes
    # # map
    add_entry, add_exit = state.add_map('add_map', dict(i='0:10'), schedule=dace.ScheduleType.Sequential)
    # # tasklet
    t1 = state.add_tasklet('add_scalar', {'_a', '_b'}, {'_c'}, '_c = _a + _b')

    # Add add_edge_pair(map mostly)
    state.add_edge_pair(add_entry, t1, a, dace.Memlet.simple(a, 'i'), internal_connector='_a')
    state.add_edge_pair(add_entry, t1, b, dace.Memlet.simple(b, 'i'), internal_connector='_b')
    state.add_edge_pair(add_exit, t1, c, dace.Memlet.simple(c, 'i'), internal_connector='_c')

    ###########CODEGEN################

    A = np.random.rand(10)
    B = np.random.rand(10)
    C = np.zeros(10)

    print(A)
    print(B)
    print(C)
    sdfg(A, B, C)
    print(C)    

def gpu_accessnode_test():
    sdfg = dace.SDFG('gpu_accessnode_test')
    #########GLOBAL VARIABLES#########
 
    # sdfg.add_scalar("scalarNode",  dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode1", dace.bool, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode2", dace.int32, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode3", dace.int64, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode4", dace.uint8, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode5", dace.uint64, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode6", dace.float16, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode7", dace.float32, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode8", dace.string, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_scalar("scalarNode9", dace.int8, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("write_to_scalar", dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    
    sdfg.add_array("arrayNode", [10], dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    # sdfg.add_stream("StreamNode", dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    
    # sdfg.add_scalar("B_scalar", dace.float64, storage=dace.StorageType.GPU_Global, transient=False)
    # sdfg.add_scalar("C_scalar", dace.float64, storage=dace.StorageType.GPU_Global, transient=False)
    # sdfg.add_constant('constant', 1)

    
    # ###########STATE, CFG, GLOBAL DATA################
    # # # add state
    state = sdfg.add_state('sum', is_start_block=True)
    
    # scalar_read = state.add_read('scalarNode')
    # scalar_read1 = state.add_read('scalarNode1')
    # scalar_read2 = state.add_read('scalarNode2')
    # scalar_read3 = state.add_read('scalarNode3')
    # scalar_read4 = state.add_read('scalarNode4')
    # scalar_read5 = state.add_read('scalarNode5')
    # scalar_read6 = state.add_read('scalarNode6')
    # scalar_read7 = state.add_read('scalarNode7')
    # scalar_read8 = state.add_read('scalarNode8')
    # scalar_read9 = state.add_read('scalarNode9')
    scalar_write = state.add_write('write_to_scalar')
    array_ = state.add_read('arrayNode')
    # stream_ = state.add_read('StreamNode')
    
    

    
    # b = state.add_read('B_scalar')
    # c = state.add_write('C_scalar')
    # state.add_edge(scalar_read, None, scalar_write, None, dace.Memlet(f"scalarNode[0]"))
    # state.add_edge(scalar_read1, None, scalar_write, None, dace.Memlet(f"scalarNode1[0]"))
    # state.add_edge(scalar_read2, None, scalar_write, None, dace.Memlet(f"scalarNode2[0]"))
    # state.add_edge(scalar_read3, None, scalar_write, None, dace.Memlet(f"scalarNode3[0]"))
    # state.add_edge(scalar_read4, None, scalar_write, None, dace.Memlet(f"scalarNode4[0]"))
    # state.add_edge(scalar_read5, None, scalar_write, None, dace.Memlet(f"scalarNode5[0]"))
    # state.add_edge(scalar_read6, None, scalar_write, None, dace.Memlet(f"scalarNode6[0]"))
    # state.add_edge(scalar_read7, None, scalar_write, None, dace.Memlet(f"scalarNode7[0]"))
    # state.add_edge(scalar_read8, None, scalar_write, None, dace.Memlet(f"scalarNode8[0]"))
    # state.add_edge(scalar_read9, None, scalar_write, None, dace.Memlet(f"scalarNode9[0]"))
    state.add_edge(array_, None, scalar_write, None, dace.Memlet(f"arrayNode[0]"))
    # state.add_edge(stream_, None, scalar_write, None, dace.Memlet(f"StreamNode[0]"))


    ###########CODEGEN################
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.zeros(1)
    print(A)
    print(B)
    print("Before", C)
    sdfg = sdfg(A)
    sdfg.apply_transformations(GPUTransformSDFG)
    print("After", C)    


def gpu_scalar_add():
    sdfg = dace.SDFG('gpu_scalar_add')
    #########GLOBAL VARIABLES#########
 
    sdfg.add_scalar("scalarNode",  dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode1", dace.bool, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode2", dace.int32, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode3", dace.int64, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode4", dace.uint8, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode5", dace.uint64, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode6", dace.float16, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode7", dace.float32, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode8", dace.string, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("scalarNode9", dace.int8, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_scalar("write_to_scalar", dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    
    sdfg.add_array("arrayNode", [10], dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    sdfg.add_stream("StreamNode", dace.float64, storage=dace.StorageType.IPU_Memory, transient=True)
    
    # sdfg.add_scalar("B_scalar", dace.float64, storage=dace.StorageType.GPU_Global, transient=False)
    # sdfg.add_scalar("C_scalar", dace.float64, storage=dace.StorageType.GPU_Global, transient=False)
    # sdfg.add_constant('constant', 1)

    
    # ###########STATE, CFG, GLOBAL DATA################
    # # # add state
    state = sdfg.add_state('sum', is_start_block=True)
    
    scalar_read = state.add_read('scalarNode')
    scalar_read1 = state.add_read('scalarNode1')
    scalar_read2 = state.add_read('scalarNode2')
    scalar_read3 = state.add_read('scalarNode3')
    scalar_read4 = state.add_read('scalarNode4')
    scalar_read5 = state.add_read('scalarNode5')
    scalar_read6 = state.add_read('scalarNode6')
    scalar_read7 = state.add_read('scalarNode7')
    scalar_read8 = state.add_read('scalarNode8')
    scalar_read9 = state.add_read('scalarNode9')
    scalar_write = state.add_write('write_to_scalar')
    array_ = state.add_read('arrayNode')
    stream_ = state.add_read('StreamNode')
    
    

    
    # b = state.add_read('B_scalar')
    # c = state.add_write('C_scalar')
    state.add_edge(scalar_read, None, scalar_write, None, dace.Memlet(f"scalarNode[0]"))
    state.add_edge(scalar_read1, None, scalar_write, None, dace.Memlet(f"scalarNode1[0]"))
    state.add_edge(scalar_read2, None, scalar_write, None, dace.Memlet(f"scalarNode2[0]"))
    state.add_edge(scalar_read3, None, scalar_write, None, dace.Memlet(f"scalarNode3[0]"))
    state.add_edge(scalar_read4, None, scalar_write, None, dace.Memlet(f"scalarNode4[0]"))
    state.add_edge(scalar_read5, None, scalar_write, None, dace.Memlet(f"scalarNode5[0]"))
    state.add_edge(scalar_read6, None, scalar_write, None, dace.Memlet(f"scalarNode6[0]"))
    state.add_edge(scalar_read7, None, scalar_write, None, dace.Memlet(f"scalarNode7[0]"))
    state.add_edge(scalar_read8, None, scalar_write, None, dace.Memlet(f"scalarNode8[0]"))
    state.add_edge(scalar_read9, None, scalar_write, None, dace.Memlet(f"scalarNode9[0]"))
    state.add_edge(array_, None, scalar_write, None, dace.Memlet(f"arrayNode[0]"))
    state.add_edge(stream_, None, scalar_write, None, dace.Memlet(f"StreamNode[0]"))
    

    # ###########DFG################
    # # Add nodes
    # # # map
    # # add_entry, add_exit = state.add_map('add_map', dict(i='0:31'), schedule=dace.ScheduleType.Default)
    # # # tasklet
    # t1 = state.add_tasklet('add_scalar', {'_a', '_b'}, {'_c'}, '_c = _a + _b')

    # Add add_edge_pair(map mostly)
    # state.add_edge_pair(add_entry, t1, a, dace.Memlet.simple(a, 'i'))
    # state.add_edge_pair(add_entry, t1, b, dace.Memlet.simple(b, 'i'))
    # state.add_edge_pair(add_exit, t1, c, dace.Memlet.simple(c, 'i'))

    # # Add memlet_path
    # state.add_memlet_path(a, t1, dst_conn='_a', memlet=dace.Memlet(f"A[i]"))
    # state.add_memlet_path(b, t1, dst_conn='_b', memlet=dace.Memlet(f"B[i]"))
    # state.add_memlet_path(t1, c, src_conn='_c', memlet=dace.Memlet(f"C[i]"))

    # # just add_edge
    # state.add_edge(a, None, t1, '_a', dace.Memlet(f"A_scalar"))
    # state.add_edge(b, None, t1, '_b', dace.Memlet(f"B_scalar"))
    # state.add_edge(t1, '_c', c, None, dace.Memlet(f"C_scalar"))

    
    # state.add_edge(a, None, t1, '_a', dace.Memlet(f"A[0]"))
    # state.add_edge(b, None, t1, '_b', dace.Memlet(f"B[0]"))
    # state.add_edge(t1, '_c', c, None, dace.Memlet(f"C[0]"))

    ###########CODEGEN################
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.zeros(1)
    print(A)
    print(B)
    print("Before", C)
    sdfg = sdfg(A)
    sdfg.apply_transformations(GPUTransformSDFG)
    print("After", C)    

def cpu_scalar_add():
    sdfg = dace.SDFG('cpu_scalar_add')
    #########GLOBAL VARIABLES#########
    # # data(vector add)

    # sdfg.add_array('A', [1], dace.float64)
    # sdfg.add_array('B', [1], dace.float64)
    # sdfg.add_array('C', [1], dace.float64)
    sdfg.add_scalar("A_scalar", dace.float64, storage=dace.StorageType.Default, transient=False)
    sdfg.add_scalar("B_scalar", dace.float64, storage=dace.StorageType.Default, transient=False)
    sdfg.add_scalar("C_scalar", dace.float64, storage=dace.StorageType.Default, transient=False)
    sdfg.add_constant('constant', 1)

    
    ###########STATE, CFG, GLOBAL DATA################
    # # add state
    state = sdfg.add_state('sum', is_start_block=True)
    a = state.add_read('A_scalar')
    b = state.add_read('B_scalar')
    c = state.add_write('C_scalar')

    ###########DFG################
    # Add nodes
    # # map
    # add_entry, add_exit = state.add_map('add_map', dict(i='0:31'), schedule=dace.ScheduleType.Default)
    # # tasklet
    t1 = state.add_tasklet('add_scalar', {'_a', '_b'}, {'_c'}, '_c = _a + _b')

    # Add add_edge_pair(map mostly)
    # state.add_edge_pair(add_entry, t1, a, dace.Memlet.simple(a, 'i'))
    # state.add_edge_pair(add_entry, t1, b, dace.Memlet.simple(b, 'i'))
    # state.add_edge_pair(add_exit, t1, c, dace.Memlet.simple(c, 'i'))

    # # Add memlet_path
    # state.add_memlet_path(a, t1, dst_conn='_a', memlet=dace.Memlet(f"A[i]"))
    # state.add_memlet_path(b, t1, dst_conn='_b', memlet=dace.Memlet(f"B[i]"))
    # state.add_memlet_path(t1, c, src_conn='_c', memlet=dace.Memlet(f"C[i]"))

    # just add_edge
    state.add_edge(a, None, t1, '_a', dace.Memlet(f"A_scalar"))
    state.add_edge(b, None, t1, '_b', dace.Memlet(f"B_scalar"))
    state.add_edge(t1, '_c', c, None, dace.Memlet(f"C_scalar"))

    
    # state.add_edge(a, None, t1, '_a', dace.Memlet(f"A[0]"))
    # state.add_edge(b, None, t1, '_b', dace.Memlet(f"B[0]"))
    # state.add_edge(t1, '_c', c, None, dace.Memlet(f"C[0]"))

    ###########CODEGEN################
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.zeros(1)
    print(A)
    print(B)
    print("Before", C)
    sdfg = sdfg(A, B, C)
    print("After", C)    

def only_state():
    sdfg = dace.SDFG('only_state')
    sdfg.add_constant('constant_variable', 1)
    sdfg.add_symbol('symbol_variable', dace.int64)
    sdfg.add_array('A_array', [1], dace.float64) #, storage=dace.StorageType.IPU_Tile_Local, transient=False)
    sdfg.add_array('B_array', [1], dace.float64)
    sdfg.add_array('C_array', [1], dace.float64)
    state1 = sdfg.add_state('state1' , is_start_state=True)
    a = state1.add_read('A_array')
    b = state1.add_read('B_array')
    c = state1.add_write('C_array')
    t = state1.add_tasklet('add', {'a', 'b'}, {'c'}, 'c = a + b')    
    state1.add_edge(a, None, t, 'a', dace.Memlet('A_array[0]'))
    state1.add_edge(b, None, t, 'b', dace.Memlet('B_array[0]'))
    state1.add_edge(t, 'c', c, None, dace.Memlet('C_array[0]'))
    
    # state2 = sdfg.add_state('state2')
    # state3 = sdfg.add_state('state3')
    # state4 = sdfg.add_state('state4')

    # # cfg/program::sequential
    # sdfg.add_edge(state1, state2, dace.InterstateEdge())
    # sdfg.add_edge(state2, state3, dace.InterstateEdge())
    # sdfg.add_edge(state3, state4, dace.InterstateEdge())


    sdfg(A, B, C)

#### Python
def add(A, B, C):
    C = A + B


    
def allocate_data(sdfg):
        
     # data
    sdfg.add_array('A', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    sdfg.add_array('B', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    # Add a C array
    sdfg.add_array('C', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    
    # add a _tmp1 accessnode with transient state, shape 1 and dtype int32
    sdfg.add_array('_tmp1', 
                   shape=[1],
                   dtype=dace.int32, 
                   storage=dace.StorageType.Register, 
                   location=None, 
                   transient=True, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=1)

    # me, mx = state.add_map('outer', dict(i='0:2'))
    # nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'a'}, {'b'})
    # state.add_memlet_path(rnode, me, nsdfg_node, dst_conn='a', memlet=dace.Memlet.simple('A', 'i'))
    # state.add_memlet_path(nsdfg_node, mx, wnode, src_conn='b', memlet=dace.Memlet.simple('A', 'i'))
    
def gpu_vector_add_python_copy():
    
      # # add a _tmp1 accessnode with transient state, shape 1 and dtype int32
    # sdfg.add_array('_tmp1_outer', 
    #                shape=[1],
    #                dtype=dace.int32, 
    #                storage=dace.StorageType.Register, 
    #                location=None, 
    #                transient=True, 
    #                strides=[1], 
    #                offset=[0], 
    #                lifetime=dace.AllocationLifetime.Scope, 
    #                debuginfo=None, total_size=1)
    def nested() -> dace.SDFG:
            # Inner SDFG
        nsdfg = dace.SDFG('nested')
        nsdfg.add_array('a', [1], dace.int32)
        nsdfg.add_array('b', [1], dace.int32)
        nsdfg.add_array('c', [1], dace.int32)
        nsdfg.add_transient('t', [1], dace.int32)

        # init state
        ninitstate = nsdfg.add_state()
        # a,b->t state
        nstate = nsdfg.add_state()
        irnode = nstate.add_read('a')
        irnodeb = nstate.add_read('b')
        task = nstate.add_tasklet('t1', {'inp1', 'inp2'}, {'out'}, 'out = inp1 + inp2')
        iwnode = nstate.add_write('t')
        nstate.add_edge(irnode, None, task, 'inp1', dace.Memlet.simple('a', '0'))
        nstate.add_edge(irnodeb, None, task, 'inp2', dace.Memlet.simple('b', '0'))
        nstate.add_edge(task, 'out', iwnode, None, dace.Memlet.simple('t', '0'))

        # t->c state
        first_state = nstate
        nstate = nsdfg.add_state()
        irnode = nstate.add_read('t')
        task = nstate.add_tasklet('t2', {'inp1'}, {'out1'}, 'out1 = inp1')
        iwnode = nstate.add_write('c')
        nstate.add_edge(irnode, None, task, 'inp1', dace.Memlet.simple('t', '0'))
        nstate.add_edge(task, 'out1', iwnode, None, dace.Memlet.simple('c', '0'))

        nsdfg.add_edge(ninitstate, first_state, dace.InterstateEdge())
        nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())
        return nsdfg
   
    
    ###############################################################
      # Outer SDFG
    sdfg = dace.SDFG('gpu_vector_add_python_copy')
     # data
    sdfg.add_array('A_outer', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    sdfg.add_array('B_outer', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    # Add a C array
    sdfg.add_array('C_outer', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.CPU_Heap, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    
    sdfg.add_symbol('i', dace.int32)
    
    # State machine
    initstate = sdfg.add_state("init")
    state = sdfg.add_state()
    rnode = state.add_read('A_outer')
    rnodeb = state.add_read('B_outer')
    wnode = state.add_write('C_outer')
    me, mx = state.add_map('map_parallelizn', dict(i='0:20'))
    nsdfg_node = state.add_nested_sdfg(nested(), None, {'a', 'b'}, {'c'}, schedule=dace.ScheduleType.Sequential)
    state.add_memlet_path(rnode, me, nsdfg_node, dst_conn='a', memlet=dace.Memlet.simple('A_outer', 'i'))
    state.add_memlet_path(rnodeb, me, nsdfg_node, dst_conn='b', memlet=dace.Memlet.simple('B_outer', 'i'))
    state.add_memlet_path(nsdfg_node, mx, wnode, src_conn='c', memlet=dace.Memlet.simple('C_outer', 'i'))
        
   # add state edges
    sdfg.add_edge(initstate, state, dace.InterstateEdge())

 ###########CODEGEN################
    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.zeros(20)
    print("A Values:", A)
    print("B Values:", B)
    print("C Values:", C)
    
    sdfg = sdfg(A, B, C)
    

# def gpu_vec_add_python():
            
#     @dace.program
#     def gpu_vector_add(A: dace.int32, B: dace.int32, C: dace.int32):
#         for i in dace.map[0:20]:       # parallelization construct
#             C[i] =  A[i] + B[i]

#     sdfg = gpu_vector_add.to_sdfg(simplify=False)   # compiled SDFG
#     sdfg.apply_transformations(GPUTransformSDFG)

#     # call with values
#     A = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
#     B = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
#     C = np.zeros((20), dtype=np.int32)  # 0,0,0,0,...
#     sdfg(A, B, C)

# main
if __name__ == "__main__":
    # handcrafted_sdfg_scalar_add()
    # structure()
    #add a,b,c values
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.zeros(1)
    # print (A)
    # print (B)
    # add(A, B, C)
    # print (C)
    # only_state()
    # print (C)
    # vector_add()
    # gpu_scalar_add()
    gpu_accessnode_test()
    # gpu_vector_add_python_copy()

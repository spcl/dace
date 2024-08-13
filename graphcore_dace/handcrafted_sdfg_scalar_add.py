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

def scalar_add():
    sdfg = dace.SDFG('scalar_add')
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
    sdfg(A, B, C)
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
    scalar_add()
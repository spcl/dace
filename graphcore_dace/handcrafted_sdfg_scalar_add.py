import dace
from dace.sdfg.propagation import propagate_states

# Handcrafted SDFG for scalar addition

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
   
if __name__ == "__main__":
    # handcrafted_sdfg_scalar_add()
    structure()
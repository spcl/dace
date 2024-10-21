import dace
import numpy as np


def copy_a_to_b():
    # Define the SDFG
    sdfg = dace.SDFG('copy_a_to_b')
    
    # Add arrays        
    sdfg.add_array('A', [1], dace.float64, storage=dace.StorageType.IPU_Memory)
    sdfg.add_array('C', [1], dace.float64, storage=dace.StorageType.IPU_Memory)

    # Add state
    state = sdfg.add_state('compute_state')
    
    # Add read and write nodes
    A_read = state.add_read('A')
    C_write = state.add_write('C')
    
    # add edge
    state.add_edge(A_read, None, C_write, None, dace.Memlet('A[0] -> C[0]'))
    
    ###############################################################
    # Runtime code
    # Initialize data
    A = np.ones(1, dtype=np.float64)
    C = np.zeros(1, dtype=np.float64)
    
    # PRINT BEFORE
    print("\nBefore")
    print("A:", A)
    print("C:", C)
    
    # Run the SDFG
    sdfg(A=A, C=C)
    
    # Print the result
    print ("\nAfter")
    print("A:", A)
    print("C:", C)

    ###############################################################
if __name__ == "__main__":
    copy_a_to_b()


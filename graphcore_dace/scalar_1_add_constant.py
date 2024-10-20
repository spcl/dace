import dace
import numpy as np


def array_add_constant_sdfg():
    # Define the SDFG
    sdfg = dace.SDFG('array_add_constant_sdfg')
    
    # Add arrays        
    sdfg.add_array('A', [1], dace.float64, storage=dace.StorageType.IPU_Memory, transient=False)
    sdfg.add_array('B', [1], dace.float64, storage=dace.StorageType.IPU_Memory, transient=False)
    sdfg.add_array('C', [1], dace.float64, storage=dace.StorageType.IPU_Memory, transient=False)
    
    # Add state
    state = sdfg.add_state('compute_state')
    
    # Add read and write nodes
    A_read = state.add_read('A')
    B_read = state.add_read('B')
    C_write = state.add_write('C')
    
    # Add map
    map_entry, map_exit = state.add_map('map', dict(i='0:1'), schedule=dace.ScheduleType.Sequential)
    
    # Add tasklet
    tasklet = state.add_tasklet('add_constant', {'a_in', 'b_in'}, {'c_out'}, 'c_out = a_in + b_in')
    
    # Connect nodes with memlets
    state.add_memlet_path(A_read, map_entry, tasklet, dst_conn='a_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(B_read, map_entry, tasklet, dst_conn='b_in', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(tasklet, map_exit, C_write, src_conn='c_out', memlet=dace.Memlet('C[i]'))
    
    # Runtime code
    # Initialize data
    A = np.ones(1, dtype=np.float64)
    B = np.ones(1, dtype=np.float64)
    C = np.zeros(1, dtype=np.float64)
    
    # Run the SDFG
    sdfg(A=A, B=B, C=C)
    
    # Print the result
    print("A:", A)
    print("B:", B)
    print("C:", C)

if __name__ == "__main__":
    array_add_constant_sdfg()


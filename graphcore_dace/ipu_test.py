import dace
import numpy as np
from dace.transformation.interstate.gpu_transform_sdfg import GPUTransformSDFG


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
   
def ipu_vector_add_python_copy():
    
    ###############################################################
      # Outer SDFG
    sdfg = dace.SDFG('gpu_vector_add_python_copy')
     # data
    sdfg.add_array('A_outer', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=20)
    sdfg.add_array('B_outer', 
                   shape=[20],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
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
                   storage=dace.StorageType.IPU_Memory, 
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
    me, mx = state.add_map('map_parallelizn', dict(i='0:20')) #, schedule=dace.ScheduleType.IPU_SCHEDULE)
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

def ipu_test1():
    nsdfg = dace.SDFG('ipu_test1')
     # data
    nsdfg.add_array('a', 
                   shape=[1],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=1)
    nsdfg.add_array('b', 
                   shape=[1],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=1)
    # Add a C array
    nsdfg.add_array('c', 
                   shape=[1],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=1)
    
    
    nsdfg.add_symbol('i', dace.int32)
    # nsdfg.add_transient('t', [1], dace.int32)
    nsdfg.add_array('t', 
                   shape=[1],
                   dtype=dace.int32, 
                   storage=dace.StorageType.IPU_Memory, 
                   location=None, 
                   transient=False, 
                   strides=[1], 
                   offset=[0], 
                   lifetime=dace.AllocationLifetime.Scope, 
                   debuginfo=None, total_size=1)
        

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
    # nstate = nsdfg.add_state()
    # irnode = nstate.add_read('t')
    # task = nstate.add_tasklet('t2', {'inp1'}, {'out1'}, 'out1 = inp1')
    # iwnode = nstate.add_write('c')
    # nstate.add_edge(irnode, None, task, 'inp1', dace.Memlet.simple('t', '0'))
    # nstate.add_edge(task, 'out1', iwnode, None, dace.Memlet.simple('c', '0'))

    nsdfg.add_edge(ninitstate, first_state, dace.InterstateEdge())
    # nsdfg.add_edge(first_state, nstate, dace.InterstateEdge())
    ###########CODEGEN################
    A = np.random.rand(20)
    B = np.random.rand(20)
    C = np.zeros(20)
    # codeobjects = nsdfg(A, B, C).generate_code()
    code = nsdfg(A, B, C).generate_code(recompile=False)[0].clean_code
    
   
# main
if __name__ == "__main__":
    ipu_test1()
    # nested()
    # ipu_vector_add_python_copy()

    
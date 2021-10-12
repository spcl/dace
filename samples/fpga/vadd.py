import dace
import numpy as np

from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState

N = dace.symbol("N")
N.set(1024)
veclen = 16

sdfg = dace.SDFG("vector_addition")

# TODO using the same data type breaks when halving the veclen
#vec_type = dace.vector(dace.float32, veclen)
sdfg.add_array("A", [N / veclen], dace.vector(dace.float32, veclen))
sdfg.add_array("B", [N / veclen], dace.vector(dace.float32, veclen))
sdfg.add_array("C", [N / veclen], dace.vector(dace.float32, veclen))

state = sdfg.add_state()
a = state.add_read("A")
b = state.add_read("B")
c = state.add_write("C")

c_entry, c_exit = state.add_map("compute_map", dict({'i': f'0:N//{veclen}'}), schedule=dace.ScheduleType.FPGA_Double)
tasklet = state.add_tasklet('vector_add_core', {'a', 'b'}, {'c'}, 'c = a + b')

state.add_memlet_path(a, c_entry, tasklet, memlet=dace.Memlet("A[i]"), dst_conn='a')
state.add_memlet_path(b, c_entry, tasklet, memlet=dace.Memlet("B[i]"), dst_conn='b')
state.add_memlet_path(tasklet, c_exit, c, memlet=dace.Memlet("C[i]"), src_conn='c')

A = np.random.rand(N.get()).astype(np.float32)
B = np.random.rand(N.get()).astype(np.float32)
C = np.zeros(N.get()).astype(np.float32)
expected = A + B

sdfg.specialize(dict(N=N))

# transformations
sdfg.apply_transformations(FPGATransformState)
sdfg.apply_transformations_repeated(StreamingMemory, dict(storage=dace.StorageType.FPGA_Local))

sdfg.save('aoeu.sdfg')

sdfg(A=A, B=B, C=C)

print (np.sum(np.abs(expected-C)))
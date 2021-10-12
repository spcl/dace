import dace
import numpy as np

from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState

N = dace.symbol("N")
N.set(10)
veclen = 2

sdfg = dace.SDFG("vector_addition")

vec_type = dace.vector(dace.float32, veclen)
sdfg.add_array("A", [N / veclen], vec_type)
sdfg.add_array("B", [N / veclen], vec_type)
sdfg.add_array("C", [N / veclen], vec_type)

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

# Add the rtl placeholder subgraph
#i_entry, i_exit = state.add_map('rtl_map', dict({'i': f'0:N//{veclen}'}))
#i_task = state.add_tasklet('rtl_core', {'a', 'b'}, {'c'},
#'''
#hls_core hls_core_0 (
#    .ap_clk(ap_aclk),
#    .ap_rst_n(ap_areset_n2),
#
#    .a_V_TVALID(s_axis_a_tvalid),
#    .a_V_TDATA(s_axis_a_)
#);
#''', language=dace.dtypes.Language.SystemVerilog)
#i_a = state.add_read('fpga_A_0')
#i_b = state.add_read('fpga_B_0')
#i_c = state.add_write('fpga_C_0')
#state.add_memlet_path(i_a, i_entry, i_task, memlet=dace.Memlet('fpga_A[0]'), dst_conn='a')
#state.add_memlet_path(i_b, i_entry, i_task, memlet=dace.Memlet('fpga_B[0]'), dst_conn='b')
#state.add_memlet_path(i_task, i_exit, i_c, memlet=dace.Memlet('fpga_C[0]'), src_conn='c')
sdfg.save('aoeu.sdfg')

sdfg(A=A, B=B, C=C)

print (np.sum(np.abs(expected-C)))
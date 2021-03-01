import dace
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState
from dace.transformation.dataflow import TrivialMapElimination
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('reciprog_pure')

# add state
state = sdfg.add_state('device_state')

# add parameter
veclen = 2
sdfg.add_constant('VECLEN', veclen)
sdfg.add_constant('DATA_WIDTH', 64)
sdfg.add_constant('WORD_WIDTH', 32)
sdfg.add_constant('RATIO', 2)

# add arrays
sdfg.add_array('A', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('C', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_A', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_C', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)

# add streams
sdfg.add_stream('A_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('C_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)

# add custom rtl tasklet
rtl_tasklet = state.add_tasklet('recip', {'inp'}, {'out'}, 'out = 1. / inp')

# add read and write tasklets
read_a = state.add_tasklet('read_a', {'inp'}, {'out'}, 'out = inp')
write_c = state.add_tasklet('write_c', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_a_entry, read_a_exit = state.add_map(
    'read_a_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)
write_c_entry, write_c_exit = state.add_map(
    'write_c_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)
compute_entry, compute_exit = state.add_map(
    'compute_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_a_inp = state.add_read('fpga_A')
read_a_out = state.add_write('A_stream')
state.add_memlet_path(read_a_inp,
                      read_a_entry,
                      read_a,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('fpga_A', 'i'))
state.add_memlet_path(read_a,
                      read_a_exit,
                      read_a_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('A_stream', '0'))

# add tasklet memlets
A = state.add_read('A_stream')
C = state.add_write('C_stream')
state.add_memlet_path(A,
                      compute_entry,
                      rtl_tasklet,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('A_stream', '0'))
state.add_memlet_path(rtl_tasklet,
                      compute_exit,
                      C,
                      src_conn='out',
                      memlet=dace.Memlet.simple('C_stream', '0'))

# add write_c memlets and access nodes
write_c_inp = state.add_read('C_stream')
write_c_out = state.add_write('fpga_C')
state.add_memlet_path(write_c_inp,
                      write_c_entry,
                      write_c,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('C_stream', '0'))
state.add_memlet_path(write_c,
                      write_c_exit,
                      write_c_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('fpga_C', 'i'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_a = copy_to_device.add_read('A')
dev_a = copy_to_device.add_write('fpga_A')
copy_to_device.add_memlet_path(cpu_a,
                               dev_a,
                               memlet=dace.Memlet.simple('A', '0:N//VECLEN'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_c = copy_to_host.add_read('fpga_C')
cpu_c = copy_to_host.add_write('C')
copy_to_host.add_memlet_path(dev_c,
                             cpu_c,
                             memlet=dace.Memlet.simple('C', '0:N//VECLEN'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    # 1024 kb
    N.set(262144 * 128)
    a = np.random.randint(1, 100, N.get()).astype(np.float32)
    c = np.zeros((N.get(), )).astype(np.float32)
    print (a.shape, c.shape)

    # show initial values
    print("a={}".format(a))

    # call program
    sdfg(A=a, C=c, N=N)

    # show result
    print("a={}, c={}".format(a, c))

    # check result
    for i in range(N.get() // veclen):
        assert abs(c[i] - (1. / a[i])) < .001

    print("Assert passed!")

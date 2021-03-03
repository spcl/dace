import dace
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState
from dace.transformation.dataflow import TrivialMapElimination
import numpy as np

# add symbol
N = dace.symbol('N')

# add sdfg
sdfg = dace.SDFG('axpy_pure')

# add state
state = sdfg.add_state('device_state')

# add parametr
veclen = 2
sdfg.add_constant('VECLEN', veclen)
sdfg.add_constant('DATA_WIDTH', 64)
sdfg.add_constant('WORD_WIDTH', 32)
sdfg.add_constant('RATIO', 2)

# add arrays
sdfg.add_scalar('a',
                dtype=dace.float32,
                storage=dace.StorageType.FPGA_Global)
sdfg.add_array('x', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('y', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('result', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               storage=dace.StorageType.CPU_Heap)
sdfg.add_array('fpga_x', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_y', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)
sdfg.add_array('fpga_result', [N // veclen],
               dtype=dace.vector(dace.float32, veclen),
               transient=True,
               storage=dace.StorageType.FPGA_Global)

# add streams
sdfg.add_stream('x_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('y_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)
sdfg.add_stream('result_stream',
                buffer_size=32,
                dtype=dace.vector(dace.float32, veclen),
                transient=True,
                storage=dace.StorageType.FPGA_Local)

# add custom rtl tasklet
rtl_tasklet = state.add_tasklet(name='rtl_tasklet',
                                inputs={'a_in', 'x_in', 'y_in'},
                                outputs={'result_out'},
                                code='result_out = a_in * x_in + y_in')

# add read and write tasklets
read_x = state.add_tasklet('read_x', {'inp'}, {'out'}, 'out = inp')
read_y = state.add_tasklet('read_y', {'inp'}, {'out'}, 'out = inp')
write_result = state.add_tasklet('write_result', {'inp'}, {'out'}, 'out = inp')

# add read and write maps
read_x_entry, read_x_exit = state.add_map(
    'read_x_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)
read_y_entry, read_y_exit = state.add_map(
    'read_y_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)
write_result_entry, write_result_exit = state.add_map(
    'write_result_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)
compute_entry, compute_exit = state.add_map(
    'compute_map',
    dict(i='0:N//VECLEN'),
    schedule=dace.ScheduleType.FPGA_Device)

# add read_a memlets and access nodes
read_x_inp = state.add_read('fpga_x')
read_x_out = state.add_write('x_stream')
state.add_memlet_path(read_x_inp,
                      read_x_entry,
                      read_x,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('fpga_x', 'i'))
state.add_memlet_path(read_x,
                      read_x_exit,
                      read_x_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('x_stream', '0'))

read_y_inp = state.add_read('fpga_y')
read_y_out = state.add_write('y_stream')
state.add_memlet_path(read_y_inp,
                      read_y_entry,
                      read_y,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('fpga_y', 'i'))
state.add_memlet_path(read_y,
                      read_y_exit,
                      read_y_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('y_stream', '0'))

# add tasklet memlets
a = state.add_read('a')
x = state.add_read('x_stream')
y = state.add_read('y_stream')
result = state.add_write('result_stream')
state.add_memlet_path(a,
                      compute_entry,
                      rtl_tasklet,
                      dst_conn='a_in',
                      memlet=dace.Memlet.simple('a', '0'))
state.add_memlet_path(x,
                      compute_entry,
                      rtl_tasklet,
                      dst_conn='x_in',
                      memlet=dace.Memlet.simple('x_stream', '0'))
state.add_memlet_path(y,
                      compute_entry,
                      rtl_tasklet,
                      dst_conn='y_in',
                      memlet=dace.Memlet.simple('y_stream', '0'))
state.add_memlet_path(rtl_tasklet,
                      compute_exit,
                      result,
                      src_conn='result_out',
                      memlet=dace.Memlet.simple('result_stream', '0'))

# add write_c memlets and access nodes
write_result_inp = state.add_read('result_stream')
write_result_out = state.add_write('fpga_result')
state.add_memlet_path(write_result_inp,
                      write_result_entry,
                      write_result,
                      dst_conn='inp',
                      memlet=dace.Memlet.simple('result_stream', '0'))
state.add_memlet_path(write_result,
                      write_result_exit,
                      write_result_out,
                      src_conn='out',
                      memlet=dace.Memlet.simple('fpga_result', 'i'))

# add copy to device state
copy_to_device = sdfg.add_state('copy_to_device')
cpu_x = copy_to_device.add_read('x')
cpu_y = copy_to_device.add_read('y')
dev_x = copy_to_device.add_write('fpga_x')
dev_y = copy_to_device.add_write('fpga_y')
copy_to_device.add_memlet_path(cpu_x,
                               dev_x,
                               memlet=dace.Memlet.simple('x', '0:N//VECLEN'))
copy_to_device.add_memlet_path(cpu_y,
                               dev_y,
                               memlet=dace.Memlet.simple('y', '0:N//VECLEN'))
sdfg.add_edge(copy_to_device, state, dace.InterstateEdge())

# add copy to host state
copy_to_host = sdfg.add_state('copy_to_host')
dev_result = copy_to_host.add_read('fpga_result')
cpu_result = copy_to_host.add_write('result')
copy_to_host.add_memlet_path(dev_result,
                             cpu_result,
                             memlet=dace.Memlet.simple('result', '0:N//VECLEN'))
sdfg.add_edge(state, copy_to_host, dace.InterstateEdge())

# validate sdfg
sdfg.validate()

######################################################################

if __name__ == '__main__':

    # init data structures
    N.set(16777216)
    a = np.random.rand(1)[0].astype(np.float32)
    x = np.random.rand(N.get()).astype(np.float32)
    y = np.random.rand(N.get()).astype(np.float32)
    result = np.zeros((N.get(), )).astype(np.float32)

    # show initial values
    print("a={}, x={}, y={}".format(a, x, y))

    # call program
    sdfg(a=a, x=x, y=y, result=result, N=N)

    # show result
    print("result={}".format(result))

    # check result
    expected = a * x + y
    diff = np.linalg.norm(expected - result) / N.get()
    print("Difference:", diff)
    exit(0 if diff <= 1e-5 else 1)

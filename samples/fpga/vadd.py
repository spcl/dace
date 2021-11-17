import click
import dace
import numpy as np
from dace.subsets import Range

from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState

def make_sdfg(N, V, double_pumped):
    sdfg = dace.SDFG(f"vector_addition_{N.get()}_{V.get()}_{'double' if double_pumped else 'single'}")

    vec_type = dace.vector(dace.float32, V.get())
    _, arr_a = sdfg.add_array("A", [N / V], vec_type)
    arr_a.location['bank'] = '0'
    arr_a.location["memorytype"] = 'hbm'
    _, arr_b = sdfg.add_array("B", [N / V], vec_type)
    arr_b.location['bank'] = '16'
    arr_b.location["memorytype"] = 'hbm'
    _, arr_c = sdfg.add_array("C", [N / V], vec_type)
    arr_c.location['bank'] = '28'
    arr_c.location["memorytype"] = 'hbm'

    state = sdfg.add_state()
    a = state.add_read("A")
    b = state.add_read("B")
    c = state.add_write("C")

    # TODO Do the division of the loop bound in codegen, rather in the SDFG. Or maybe in the future, it should actually be through the SDFG as that would be more transparent and less magic?
    c_entry, c_exit = state.add_map("compute_map", dict({'i': f'0:N//V'}),
        schedule=dace.ScheduleType.FPGA_Double if double_pumped else dace.ScheduleType.Default)
    tasklet = state.add_tasklet('vector_add_core', {'a', 'b'}, {'c'}, 'c = a + b')

    state.add_memlet_path(a, c_entry, tasklet, memlet=dace.Memlet("A[i]"), dst_conn='a')
    state.add_memlet_path(b, c_entry, tasklet, memlet=dace.Memlet("B[i]"), dst_conn='b')
    state.add_memlet_path(tasklet, c_exit, c, memlet=dace.Memlet("C[i]"), src_conn='c')

    sdfg.specialize(dict(N=N, V=V))

    # transformations
    sdfg.apply_transformations(FPGATransformState)
    sdfg.apply_transformations_repeated(StreamingMemory, dict(storage=dace.StorageType.FPGA_Local, buffer_size=32))


    if double_pumped:
        #c_entry.map.range = Range([(0,f'(N//(V//2))-1',1)])
        c_entry.map.range = Range([(0,f'N-1',1)])


    from dace.codegen.targets.fpga import is_fpga_kernel
    for s in sdfg.states():
        if is_fpga_kernel(sdfg, s):
            s.instrument = dace.InstrumentationType.FPGA
            
    return sdfg

@click.command()
@click.argument("size_n", type=int)
@click.argument("veclen", type=int)
@click.option("--double_pumped/--no-double_pumped",
              default=False,
              help="Whether to double pump the compute kernel")
def cli(size_n, veclen, double_pumped):
    N = dace.symbol("N")
    V = dace.symbol("V")
    N.set(size_n)
    V.set(veclen)

    A = np.random.rand(N.get()).astype(np.float32)
    B = np.random.rand(N.get()).astype(np.float32)
    C = np.zeros(N.get()).astype(np.float32)
    expected = A + B

    sdfg = make_sdfg(N, V, double_pumped)

    #sdfg.compile()
    sdfg(A=A, B=B, C=C)

    print (np.sum(np.abs(expected-C)))

if __name__ == '__main__':
    cli()

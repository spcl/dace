import dace
from dace.transformation.dataflow import StreamingMemory
from dace.transformation.interstate import FPGATransformState
from dace.transformation.dataflow import TrivialMapElimination
import numpy as np

# add sdfg
sdfg = dace.SDFG('add_fortytwo')

# add state
state = sdfg.add_state()

# add arrays
sdfg.add_array('A', [1], dtype=dace.int32)
sdfg.add_array('B', [1], dtype=dace.int32)

# add custom cpp tasklet
tasklet = state.add_tasklet('htsn', ['a'], ['b'], 'b = a+42')

# add input/output array
A = state.add_read('A')
B = state.add_write('B')

# add a simple map
mentry, mexit = state.add_map('aoeu_map', dict(i='0:1'))

# connect input/output array with the tasklet
state.add_memlet_path(A, mentry, tasklet,
    dst_conn='a', memlet=dace.Memlet.simple('A', '0'))
state.add_memlet_path(tasklet, mexit, B,
    src_conn='b', memlet=dace.Memlet.simple('B', '0'))

# validate sdfg
sdfg.validate()

# apply the transformations
sdfg.apply_transformations(FPGATransformState)
sdfg.apply_transformations_repeated(StreamingMemory, dict(storage=dace.StorageType.FPGA_Local))
sdfg.apply_transformations_repeated(TrivialMapElimination)
sdfg.save('_dacegraphs/program.sdfg')

######################################################################

if __name__ == '__main__':

    # init data structures
    a = np.random.randint(0, 100, 1).astype(np.int32)
    b = np.array([0]).astype(np.int32)

    # show initial values
    print("a={}, b={}".format(a, b))

    # call program
    sdfg(A=a, B=b)

    # show result
    print("a={}, b={}".format(a, b))

    # check result
    assert b == a+42

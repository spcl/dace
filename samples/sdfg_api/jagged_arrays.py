# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

sdfg = dace.SDFG('jagged')

# Number of arrays
num_arrays = dace.symbol('num_arrays')
# Total length
L = dace.symbol('L')

sdfg.add_array('JaggedArray', [L], dace.float32)
sdfg.add_array('JaggedOffsets', [num_arrays + 1], dace.int32)
sdfg.add_scalar('off_ind', dace.int32, transient=True)
sdfg.add_scalar('len_ind', dace.int32, transient=True)

state = sdfg.add_state()

read_jarr = state.add_read('JaggedArray')
read_joff = state.add_read('JaggedOffsets')
write_jarr = state.add_write('JaggedArray')
offT = state.add_access('off_ind')
lenT = state.add_access('len_ind')

me, mx = state.add_map('readlen', dict(i='0:num_arrays'))

indt = state.add_tasklet('indirection', {'offs'}, {'off', 'len'}, 'off = offs[i]; len = offs[i+1]')

ime, imx = state.add_map('addone', dict(j='off_ind:len_ind'))
ime.add_in_connector('off_ind')
ime.add_in_connector('len_ind')
task = state.add_tasklet('add1', {'a'}, {'b'}, 'b = a + 10*(i+1)')

state.add_memlet_path(read_joff,
                      me,
                      indt,
                      memlet=dace.Memlet.simple('JaggedOffsets', '0:num_arrays+1'),
                      dst_conn='offs')
state.add_memlet_path(read_jarr, me, ime, task, memlet=dace.Memlet.simple('JaggedArray', 'j'), dst_conn='a')
state.add_edge(indt, 'off', offT, None, dace.Memlet.simple('off_ind', '0'))
state.add_edge(indt, 'len', lenT, None, dace.Memlet.simple('len_ind', '0'))

state.add_edge(offT, None, ime, 'off_ind', dace.Memlet.simple('off_ind', '0'))
state.add_edge(lenT, None, ime, 'len_ind', dace.Memlet.simple('len_ind', '0'))

state.add_memlet_path(task, imx, mx, write_jarr, src_conn='b', memlet=dace.Memlet.simple('JaggedArray', 'j'))

# Validate correctness of initial SDFG
sdfg.validate()

# Fuses redundant states and removes unnecessary transient arrays
sdfg.simplify()

######################################
if __name__ == '__main__':
    print('Program start')

    arrs = 5
    nnz = np.random.randint(1, 10, size=arrs, dtype=np.int32)
    offs = np.ndarray([arrs + 1], dtype=np.int32)
    offs[0] = 0
    offs[1:] = np.cumsum(nnz)
    length = np.sum(nnz)

    jagged = np.random.rand(length).astype(np.float32)
    ref = jagged.copy()

    for i in range(arrs):
        print(jagged[offs[i]:offs[i + 1]])
        ref[offs[i]:offs[i + 1]] += 10 * (i + 1)

    sdfg(JaggedArray=jagged, JaggedOffsets=offs, L=length, num_arrays=arrs)

    for i in range(arrs):
        print(jagged[offs[i]:offs[i + 1]])

    diff = np.linalg.norm(ref - jagged)
    print('Difference:', diff)
    exit(0 if diff <= 1e-5 else 1)

# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Sample that shows dynamic ranges in maps by iterating over a jagged array. """
import dace
import numpy as np

# Create an empty SDFG
sdfg = dace.SDFG('jagged')

# Symbols:
num_arrays = dace.symbol('num_arrays')  # Number of arrays
L = dace.symbol('L')  # Total length

# Create a 1D version of the jagged array, as well as a number of offsets
sdfg.add_array('JaggedArray', [L], dace.float32)
sdfg.add_array('JaggedOffsets', [num_arrays + 1], dace.int32)

# Add scalars for indices to control the dynamic map ranges
sdfg.add_scalar('off_ind', dace.int32, transient=True)
sdfg.add_scalar('len_ind', dace.int32, transient=True)

# Create our main state state
state = sdfg.add_state()

# Read inputs
read_jarr = state.add_read('JaggedArray')
read_joff = state.add_read('JaggedOffsets')
write_jarr = state.add_write('JaggedArray')
offT = state.add_access('off_ind')
lenT = state.add_access('len_ind')

# Create a map that reads all the arrays. There are no dynamic inputs here, only the `num_arrays` symbol
me, mx = state.add_map('readlen', dict(i='0:num_arrays'))

# Create a tasklet that will read two indices (offset, length) to indirectly influence the internal map range
indt = state.add_tasklet('indirection', {'offs'}, {'off', 'len'}, 'off = offs[i]; len = offs[i+1]')

# Create a dynamic-range map: the map node itself is constructed the same way, however, any dynamic input has to have
# connectors with name that matches the symbols in the range. In this case, `begin` and `end`. Those are then connected
# to the map entry node.
ime, imx = state.add_map('addone', dict(j='begin:end'))
ime.add_in_connector('begin')
ime.add_in_connector('end')

# Memlets that lead into the map entry node
state.add_edge(offT, None, ime, 'begin', dace.Memlet('off_ind'))
state.add_edge(lenT, None, ime, 'end', dace.Memlet('len_ind'))

# Create tasklet and other memlets
task = state.add_tasklet('add1', {'a'}, {'b'}, 'b = a + 10*(i+1)')
state.add_memlet_path(read_joff,
                      me,
                      indt,
                      memlet=dace.Memlet(data='JaggedOffsets', subset='0:num_arrays+1'),
                      dst_conn='offs')
state.add_memlet_path(read_jarr, me, ime, task, memlet=dace.Memlet(data='JaggedArray', subset='j'), dst_conn='a')
state.add_edge(indt, 'off', offT, None, dace.Memlet(data='off_ind', subset='0'))
state.add_edge(indt, 'len', lenT, None, dace.Memlet(data='len_ind', subset='0'))

state.add_memlet_path(task, imx, mx, write_jarr, src_conn='b', memlet=dace.Memlet(data='JaggedArray', subset='j'))

# Validate correctness of initial SDFG
sdfg.validate()

# Fuses redundant states and removes unnecessary transient arrays
sdfg.simplify()

######################################
if __name__ == '__main__':
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

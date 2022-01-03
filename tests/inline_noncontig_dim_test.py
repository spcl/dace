# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace

# Construct SDFG
sdfg = dace.SDFG('noncontig')
state = sdfg.add_state()
sdfg.add_array('A', [2, 3, 4], dace.float32)
sdfg.add_array('B', [2, 3, 4], dace.float32)
A = state.add_read('A')
B = state.add_write('B')

# Construct nested SDFG
nsdfg = dace.SDFG('noncontig_internal')
nsdfg.add_array('aA', [2, 3, 4], dace.float32)
nsdfg.add_array('bB', [2, 3, 4], dace.float32)
s = nsdfg.add_state()
s.add_mapped_tasklet('dostuff',
                     dict(i='0:2', k='0:4'),
                     dict(a=dace.Memlet.simple('aA', 'i, 0, k')),
                     'b = a * 5',
                     dict(b=dace.Memlet.simple('bB', 'i, 0, k')),
                     external_edges=True)
########################

# Add nested SDFG to SDFG
map_entry, map_exit = state.add_map('elements', dict(j='0:3'))
nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'aA'}, {'bB'})
state.add_memlet_path(A, map_entry, nsdfg_node, dst_conn='aA', memlet=dace.Memlet.simple('A', '0:2, j, 0:4'))
state.add_memlet_path(nsdfg_node, map_exit, B, src_conn='bB', memlet=dace.Memlet.simple('B', '0:2, j, 0:4'))


def test():
    print('Nested SDFG with non-contiguous access test')

    input = np.random.rand(2, 3, 4).astype(np.float32)
    output = np.zeros(shape=(2, 3, 4), dtype=np.float32)

    sdfg(A=input, B=output)
    diff1 = np.linalg.norm(output - input * 5)
    print("Difference (without dataflow coarsening):", diff1)

    output = np.zeros(shape=(2, 3, 4), dtype=np.float32)

    sdfg.coarsen_dataflow()
    sdfg(A=input, B=output)
    diff2 = np.linalg.norm(output - input * 5)
    print("Difference:", diff2)

    print("==== Program end ====")
    assert (diff1 <= 1e-5 and diff2 <= 1e-5)


if __name__ == "__main__":
    test()

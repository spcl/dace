# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

sdfg = dace.SDFG('sfusion')

sdfg.add_array('A', [2], dace.float32)
sdfg.add_array('B', [2], dace.float32)


# Sample state contents
def mystate(state, src, dst):
    src_node = state.add_read(src)
    dst_node = state.add_write(dst)
    me, mx = state.add_map('aaa', dict(i='0:2'))
    tasklet = state.add_tasklet('aaa2', {'a'}, {'b'}, 'b = a')

    # input path (src->me->tasklet[a])
    state.add_memlet_path(src_node, me, tasklet, dst_conn='a', memlet=dace.Memlet.simple(src, 'i'))
    # output path (tasklet[b]->mx->dst)
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='b', memlet=dace.Memlet.simple(dst, 'i'))


state = sdfg.add_state('s0')
mystate(state, 'A', 'B')

state = sdfg.add_state('s1')
mystate(state, 'B', 'A')

state = sdfg.add_state('s2')
mystate(state, 'A', 'B')

state = sdfg.add_state('s3')
mystate(state, 'B', 'A')

# Loop over all states and connect them to each other programmatically
nodes = list(sdfg.nodes())
for i in range(len(nodes) - 1):
    sdfg.add_edge(nodes[i], nodes[i + 1], dace.InterstateEdge())

# Validate correctness of initial SDFG
sdfg.validate()

# Fuses redundant states and removes unnecessary transient arrays
sdfg.simplify()

######################################
if __name__ == '__main__':
    print('Program start')

    a = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)

    print(a, b)

    sdfg(A=a, B=b)

    print(b - a)

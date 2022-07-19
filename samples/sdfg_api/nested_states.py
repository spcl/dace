# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG API sample that showcases nested SDFG creation. """
import dace
import numpy as np

# Create outer SDFG
sdfg = dace.SDFG('nested_main')

# Add global array
sdfg.add_array('A', [2], dace.float32)


# Sample state contents
def mystate(state, src, dst):
    src_node = state.add_read(src)
    dst_node = state.add_write(dst)
    tasklet = state.add_tasklet('aaa2', {'a'}, {'b'}, 'b = a + 1')

    # input path (src->tasklet[a])
    state.add_memlet_path(src_node, tasklet, dst_conn='a', memlet=dace.Memlet(data=src, subset='0'))
    # output path (tasklet[b]->dst)
    state.add_memlet_path(tasklet, dst_node, src_conn='b', memlet=dace.Memlet(data=dst, subset='0'))


# Create nested SDFG
sub_sdfg = dace.SDFG('nested_sub')

# Declare arrays for nested SDFG (the names can differ from the top-level SDFG)
# In this case, we read only one element out of the full arrays
sub_sdfg.add_array('sA', [1], dace.float32)
sub_sdfg.add_transient('sB', [1], dace.float32)
sub_sdfg.add_array('sC', [1], dace.float32)

# Create nested states
state0 = sub_sdfg.add_state('subs0')
mystate(state0, 'sA', 'sB')
state1 = sub_sdfg.add_state('subs1')
mystate(state1, 'sB', 'sC')

sub_sdfg.add_edge(state0, state1, dace.InterstateEdge())

#############

# Create top-level SDFG
state = sdfg.add_state('s0')
me, mx = state.add_map('mymap', dict(k='0:2'))
# NOTE: The names of the inputs/outputs of the nested SDFG must match array
#       names above (lines 30, 32)!
nsdfg = state.add_nested_sdfg(sub_sdfg, sdfg, {'sA'}, {'sC'})
Ain = state.add_read('A')
Aout = state.add_write('A')

# Connect dataflow nodes
state.add_memlet_path(Ain, me, nsdfg, memlet=dace.Memlet(data='A', subset='k'), dst_conn='sA')
state.add_memlet_path(nsdfg, mx, Aout, memlet=dace.Memlet(data='A', subset='k'), src_conn='sC')
###

# Validate correctness of SDFG
sdfg.validate()

######################################
if __name__ == '__main__':
    a = np.random.rand(2).astype(np.float32)
    b = np.zeros([2])
    b[:] = a

    sdfg(A=a)

    print((b + 2) - a)

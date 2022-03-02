# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG API sample that showcases state machine creation and the `simplify` call, which will fuse them. """
import dace
import numpy as np

# Define a symbol to be used in the SDFG
T = dace.symbol('T')

# Create an empty SDFG
sdfg = dace.SDFG('cflow')

# Add global arrays and the symbol from above to the SDFG
sdfg.add_array('A', [2], dace.float32)
sdfg.add_array('B', [2], dace.float32)
sdfg.add_symbol('T', T.dtype)


# Sample state contents
def mystate(state, src, dst):
    # Create access nodes for reading and writing
    src_node = state.add_read(src)
    dst_node = state.add_write(dst)
    # Create a map in which a tasklet will reside
    me, mx = state.add_map('aaa', dict(i='0:2'))
    # Create the tasklet
    tasklet = state.add_tasklet('aaa2', {'a'}, {'b'}, 'b = a')

    # input path (src->me->tasklet[a])
    state.add_memlet_path(src_node, me, tasklet, dst_conn='a', memlet=dace.Memlet(data=src, subset='i'))
    # output path (tasklet[b]->mx->dst)
    state.add_memlet_path(tasklet, mx, dst_node, src_conn='b', memlet=dace.Memlet(data=dst, subset='i'))


# End state contents
def endstate(state):
    # Only reading A into the tasklet
    A = state.add_read('A')
    t = state.add_tasklet('endtask', {'a'}, {}, 'printf("done %f\\n", a)')
    state.add_edge(A, None, t, 'a', dace.Memlet(data='A', subset='0'))


# State construction
state0 = sdfg.add_state('s0')
mystate(state0, 'A', 'B')

# For an easier creation of loops, see the `sdfg.add_loop` helper function.
guard = sdfg.add_state('guard')

loopstate0 = sdfg.add_state('loops0')
mystate(loopstate0, 'A', 'B')

loopstate1 = sdfg.add_state('loops1')
mystate(loopstate1, 'B', 'A')

state2 = sdfg.add_state('s2')
endstate(state2)

# State connection (control flow)

# Note: dataflow (arrays) CAN affect control flow assignments and conditions,
#       but not the other way around (you cannot change an interstate variable
#       inside a state). The following code works as well:
#sdfg.add_edge(state0, guard, dace.InterstateEdge(assigments=dict('k', 'A[0]')))

# Loop initialization (k=0)
sdfg.add_edge(state0, guard, dace.InterstateEdge(assignments=dict(k='0')))

# Loop condition (k < T / k >= T)
sdfg.add_edge(guard, loopstate0, dace.InterstateEdge('k < T'))
sdfg.add_edge(guard, state2, dace.InterstateEdge('k >= T'))

# Loop incrementation (k++)
sdfg.add_edge(loopstate1, guard, dace.InterstateEdge(assignments=dict(k='k+1')))

# Loop-internal interstate edges
sdfg.add_edge(loopstate0, loopstate1, dace.InterstateEdge())

# Validate correctness of initial SDFG
sdfg.validate()

# Fuses redundant states and removes unnecessary transient arrays
sdfg.simplify()

######################################
if __name__ == '__main__':
    a = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)

    print(a, b)

    # Don't forget the symbols!
    sdfg(A=a, B=b, T=5)

    print(b - a)

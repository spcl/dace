# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np

sdfg = dp.SDFG('sta_test')
s0 = sdfg.add_state()
s1 = sdfg.add_state()
s2 = sdfg.add_state()

# Arrays
inp = s0.add_array('inp', [1], dp.float32)
A = s0.add_array('A', [1], dp.float32)
t = s0.add_tasklet('seta', {'a'}, {'b'}, 'b = a')
s0.add_edge(inp, None, t, 'a', dp.Memlet.from_array(inp.data, inp.desc(sdfg)))
s0.add_edge(t, 'b', A, None, dp.Memlet.from_array(A.data, A.desc(sdfg)))

A = s1.add_array('A', [1], dp.float32)
t = s1.add_tasklet('geta', {'a'}, {}, 'printf("ok %f\\n", a + 1)')
s1.add_edge(A, None, t, 'a', dp.Memlet.from_array(A.data, A.desc(sdfg)))

A = s2.add_array('A', [1], dp.float32)
t = s2.add_tasklet('geta', {'a'}, {}, 'printf("BAD %f\\n", a - 1)')
s2.add_edge(A, None, t, 'a', dp.Memlet.from_array(A.data, A.desc(sdfg)))

sdfg.add_edge(s0, s1, dp.InterstateEdge('A[0] > 3'))
sdfg.add_edge(s0, s2, dp.InterstateEdge('A[0] <= 3'))

if __name__ == '__main__':
    print('Toplevel array usage in interstate edge')
    input = np.ndarray([1], np.float32)
    input[0] = 10
    output = np.ndarray([1], np.float32)
    output[0] = 10

    sdfg(inp=input, A=output)

    exit(0)

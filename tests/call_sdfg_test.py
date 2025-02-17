# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

sdfg = dace.SDFG('internal')
sdfg.add_array('inp', [2], dace.float32)
state = sdfg.add_state()
t = state.add_tasklet('p', {'i'}, set(), 'printf("hello world %f\\n", i)')
r = state.add_read('inp')
state.add_edge(r, None, t, 'i', dace.Memlet.simple('inp', '1'))


@dace.program
def caller(A: dace.float32[4]):
    sdfg(inp=A[1:3])


def test():
    A = np.random.rand(4).astype(np.float32)
    caller(A)
    print('Should print', A[2])


if __name__ == '__main__':
    test()

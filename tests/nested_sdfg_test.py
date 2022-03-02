# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

N = dp.symbol('N')


@dp.program
def sdfg_internal(input: dp.float32, output: dp.float32[1]):
    @dp.tasklet
    def init():
        out >> output
        out = input

    for k in range(4):

        @dp.tasklet
        def do():
            oin << output
            out >> output
            out = oin * input


# Construct SDFG
mysdfg = SDFG('outer_sdfg')
state = mysdfg.add_state()
A = state.add_array('A', [N, N], dp.float32)
B = state.add_array('B', [N, N], dp.float32)

map_entry, map_exit = state.add_map('elements', [('i', '0:N'), ('j', '0:N')])
nsdfg = state.add_nested_sdfg(sdfg_internal.to_sdfg(), mysdfg, {'input'}, {'output'})

# Add edges
state.add_memlet_path(A, map_entry, nsdfg, dst_conn='input', memlet=Memlet.simple(A, 'i,j'))
state.add_memlet_path(nsdfg, map_exit, B, src_conn='output', memlet=Memlet.simple(B, 'i,j'))


def test():
    print('Nested SDFG test')
    # Externals (parameters, symbols)

    N.set(64)

    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([N, N], dp.float32)
    input[:] = np.random.rand(N.get(), N.get()).astype(dp.float32.type)
    output[:] = dp.float32(0)

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N.get() * N.get())
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
